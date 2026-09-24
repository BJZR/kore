// kore_tui.cpp - KORE, iteracion 3: interfaz de terminal (ANSI + termios, sin librerias)
// Compilar: g++ -std=c++20 -O2 -Wall -Wextra kore_tui.cpp -o kore_tui
// Uso:      ./kore_tui [--orch /tmp/kore-orch.sock]
//
// Chat minimalista: historial scrollable arriba (PgUp/PgDn/flechas), input
// abajo. Streaming en tiempo real. Comandos: /reset /help /quit.
// No requiere ncurses: renderiza con secuencias ANSI sobre el buffer alterno.

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <termios.h>
#include <unistd.h>

typedef std::vector<uint8_t> Buf;

static bool sock_write_all(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}

static bool sock_read_all(int fd, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}

static bool send_frame(int fd, char type, const void *data, uint32_t len) {
    char hdr[5] = {type,
                   (char)(len & 0xFF), (char)((len >> 8) & 0xFF),
                   (char)((len >> 16) & 0xFF), (char)((len >> 24) & 0xFF)};
    if (!sock_write_all(fd, hdr, 5)) return false;
    return len == 0 || sock_write_all(fd, data, len);
}

static bool recv_frame(int fd, char *type, Buf &payload) {
    char hdr[5];
    if (!sock_read_all(fd, hdr, 5)) return false;
    *type = hdr[0];
    uint32_t len = (uint32_t)(uint8_t)hdr[1] | ((uint32_t)(uint8_t)hdr[2] << 8) |
                   ((uint32_t)(uint8_t)hdr[3] << 16) | ((uint32_t)(uint8_t)hdr[4] << 24);
    if (len > (64u << 20)) return false;
    payload.resize(len);
    return len == 0 || sock_read_all(fd, payload.data(), len);
}

// ------------------------------------------------------------------ bloques del chat

enum BlkKind { B_BIENV, B_USER, B_AI, B_TOOL, B_OUT, B_SYS, B_ERR };

struct Block {
    BlkKind k;
    std::string text;
};

static std::vector<Block> blocks;
static size_t scroll = 0;                    // lineas de desplazamiento desde el final

static void push_block(BlkKind k, const std::string &text) {
    if (k == B_AI && !blocks.empty() && blocks.back().k == B_AI) {
        blocks.back().text += text;
        return;
    }
    blocks.push_back({k, text});
    scroll = 0;
}

static std::string clean_ctl(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char u = (unsigned char)c;
        if (u == '\t') out += "    ";
        else if (u < 0x20 || u == 0x7F) out += ' ';
        else out += c;
    }
    return out;
}

// Ancho de display de un caracter UTF-8 (2 si es CJK).
static int char_width(const std::string &s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) return 1;
    const unsigned char n = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0) return 1;
    uint32_t cp = c & (0xFF >> (n + 1));
    for (size_t k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    const bool cjk = (cp >= 0x1100 && cp <= 0x115F) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
                     (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0xF900 && cp <= 0xFAFF) ||
                     (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF00 && cp <= 0xFF60);
    return cjk ? 2 : 1;
}

// Parte una linea en trozos UTF-8 de <= width columnas.
static void wrap(const std::string &src, int width, std::vector<std::string> &out) {
    if (src.empty()) { out.push_back(""); return; }
    std::string line;
    int w = 0;
    for (size_t i = 0; i < src.size();) {
        if (src[i] == '\n') { out.push_back(line); line.clear(); w = 0; i++; continue; }
        const unsigned char c = (unsigned char)src[i];
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + n > src.size()) break;
        const int cw = [&]() {
            uint32_t cp = 0;
            size_t k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            if (k == 1) cp = c;
            else {
                cp = c & (0xFF >> (k + 1));
                for (size_t j = 1; j < k; j++) cp = (cp << 6) | ((unsigned char)src[i + j] & 0x3F);
            }
            const bool cjk = (cp >= 0x1100 && cp <= 0x115F) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
                             (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0xF900 && cp <= 0xFAFF) ||
                             (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF00 && cp <= 0xFF60);
            return cjk ? 2 : 1;
        }();
        if (w + cw > width && !line.empty()) { out.push_back(line); line.clear(); w = 0; }
        line.append(src, i, n);
        w += cw;
        i += n;
    }
    out.push_back(line);
}

static const char *block_color(BlkKind k) {
    switch (k) {
        case B_USER:  return "\x1b[1;37m";
        case B_AI:    return "\x1b[0;37m";
        case B_TOOL:  return "\x1b[0;36m";
        case B_OUT:   return "\x1b[0;33m";
        case B_ERR:   return "\x1b[1;31m";
        default:      return "\x1b[2;90m";
    }
}

static const char *block_prefix(BlkKind k) {
    switch (k) {
        case B_USER:  return "tu> ";
        case B_OUT:   return "  | ";
        case B_SYS:   return "* ";
        case B_ERR:   return "! ";
        default:      return "";
    }
}

// ------------------------------------------------------------------ pantalla

static const char *INPUT_PROMPT = "kore> ";

struct RenderedLine {
    const Block *b;
    std::string text;
};

static int rows = 24, cols = 80;

static void get_size() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 4 && ws.ws_col > 8) {
        rows = ws.ws_row;
        cols = ws.ws_col;
    }
}

static void render(const std::string &input, int cursor) {
    get_size();
    std::vector<RenderedLine> lines;
    for (const Block &b : blocks) {
        std::vector<std::string> wl;
        wrap(clean_ctl(b.text), cols, wl);
        if (!wl.empty()) {
            wl[0] = std::string(block_prefix(b.k)) + wl[0];
            for (size_t j = 1; j < wl.size(); j++)
                wl[j] = (b.k == B_TOOL ? "  > " : "    ") + wl[j];
        }
        for (const std::string &l : wl) lines.push_back({&b, l});
    }
    const size_t total = lines.size();
    if (scroll > total) scroll = total;
    const int body_h = rows > 3 ? rows - 2 : 1;
    const size_t vis = total > (size_t)body_h ? (size_t)body_h : total;
    const size_t start_line = scroll >= vis ? scroll - vis : 0;

    int in_width = 0;
    for (size_t i = 0; i < input.size() && (int)i < cursor;) {
        const int w = char_width(input, i);
        in_width += w;
        i += (unsigned char)input[i] < 0x80 ? 1 : (unsigned char)input[i] < 0xE0 ? 2 : 3;
    }

    std::string out;
    out += "\x1b[H\x1b[2J";
    size_t shown = 0;
    for (size_t i = start_line; i < total && shown < (size_t)body_h; i++, shown++) {
        out += "\x1b[K";
        out += block_color(lines[i].b->k);
        out += lines[i].text;
        out += "\x1b[0m\r\n";
    }
    for (; shown < (size_t)body_h; shown++) out += "\x1b[K\x1b[0m\r\n";
    out += "\x1b[K\x1b[2;90m";
    out += INPUT_PROMPT;
    out += "\x1b[0m";
    out += input;
    out += "\x1b[K\x1b[0m";
    out += "\x1b[" + std::to_string(rows) + ";1H";
    out += "\x1b[" + std::to_string((int)strlen(INPUT_PROMPT) + in_width + 1) + "G";
    write(STDOUT_FILENO, out.data(), out.size());
}

// ------------------------------------------------------------------ input

static std::string input;
static int cursor = 0;

static void input_append(const std::string &bytes) {
    input.insert((size_t)cursor, bytes);
    cursor += (int)bytes.size();
}

static void input_backspace() {
    if (cursor <= 0) return;
    int i = cursor - 1;
    while (i > 0 && ((unsigned char)input[i] & 0xC0) == 0x80) i--;
    input.erase(i, (size_t)(cursor - i));
    cursor = i;
}

static void input_left() {
    if (cursor > 0) {
        cursor--;
        while (cursor > 0 && ((unsigned char)input[cursor] & 0xC0) == 0x80) cursor--;
    }
}

static void input_right() {
    if (cursor < (int)input.size()) {
        cursor++;
        while (cursor < (int)input.size() && ((unsigned char)input[cursor] & 0xC0) == 0x80) cursor++;
    }
}

// ------------------------------------------------------------------ terminal

static termios g_saved;
static bool g_raw = false;

static void term_enter() {
    tcgetattr(STDIN_FILENO, &g_saved);
    termios t = g_saved;
    t.c_lflag &= ~(ICANON | ECHO);
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
    g_raw = true;
    write(STDOUT_FILENO, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H", 15);
}

static void term_leave() {
    write(STDOUT_FILENO, "\x1b[?25h\x1b[?1049l\x1b[0m", 13);
    if (g_raw) tcsetattr(STDIN_FILENO, TCSANOW, &g_saved);
}

static int connect_unix(const char *path) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    return fd;
}

static volatile sig_atomic_t g_quit = 0;
static void on_int(int) { g_quit = 1; }

int main(int argc, char **argv) {
    std::string orch_path = "/tmp/kore-orch.sock";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--orch" && has) orch_path = argv[++i];
        else { fprintf(stderr, "argumento no reconocido: %s\n", a.c_str()); return 1; }
    }

    const int fd = connect_unix(orch_path.c_str());
    if (fd < 0) {
        fprintf(stderr, "error: no se puede conectar al Orchestrator en %s\n", orch_path.c_str());
        return 1;
    }

    term_enter();
    signal(SIGINT, on_int);
    push_block(B_BIENV, "KORE TUI (conectado). /help para comandos.");

    const struct timeval tv = {0, 200000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char type = 0;
    int running = 1;
    while (running && !g_quit) {
        render(input, cursor);
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(0, &rfds);
        FD_SET(fd, &rfds);
        struct timeval sel = {0, 50000};
        const int n = select(fd + 1, &rfds, nullptr, nullptr, &sel);
        if (n < 0) { if (errno == EINTR) continue; break; }

        if (FD_ISSET(fd, &rfds)) {
            Buf pay;
            if (!recv_frame(fd, &type, pay)) {
                push_block(B_ERR, "desconectado del Orchestrator");
                running = 0;
                break;
            }
            std::string text((const char *)pay.data(), pay.size());
            switch (type) {
                case 'M': break;
                case 'S': push_block(B_SYS, text); break;
                case 'U': push_block(B_USER, text); break;
                case 'A': push_block(B_AI, text); break;
                case 'T': push_block(B_TOOL, text); break;
                case 'O': push_block(B_OUT, text); break;
                case 'E': if (!blocks.empty() && blocks.back().k == B_AI) blocks.back().text += "\n"; break;
                case 'R': push_block(B_SYS, "conversacion reiniciada"); break;
                case 'X': push_block(B_ERR, text); break;
                default: break;
            }
        }

        if (FD_ISSET(0, &rfds)) {
            unsigned char b[64];
            const ssize_t n = read(0, b, sizeof(b));
            if (n <= 0) running = 0;
            for (ssize_t i = 0; i < n; i++) {
                const unsigned char c = b[i];
                if (c == 3 || c == 4) { running = 0; break; }
                if (c == '\r' || c == '\n') {
                    const std::string msg = input;
                    cursor = 0;
                    input.clear();
                    if (msg == "/quit" || msg == "/exit") { running = 0; break; }
                    if (msg == "/reset") { send_frame(fd, 'R', nullptr, 0); continue; }
                    if (msg == "/help") {
                        push_block(B_SYS, "/reset reinicia la conversacion | /quit salir | PgUp/PgDn y flechas para scroll");
                        continue;
                    }
                    if (!msg.empty()) send_frame(fd, 'P', msg.data(), (uint32_t)msg.size());
                    continue;
                }
                if (c == 0x7f || c == 0x08) { input_backspace(); continue; }
                if (c == 0x1b && i + 2 < n && b[i + 1] == '[') {
                    const unsigned char k = b[i + 2];
                    if (k == 'A') { scroll++; }
                    else if (k == 'B') { if (scroll > 0) scroll--; }
                    else if (k == 'C') { input_right(); }
                    else if (k == 'D') { input_left(); }
                    else if (k == '5' && i + 3 < n && b[i + 3] == '~') { scroll += 5; }
                    else if (k == '6' && i + 3 < n && b[i + 3] == '~') { scroll = scroll > 5 ? scroll - 5 : 0; }
                    i += (k == '5' || k == '6') ? 3 : 2;
                    continue;
                }
                if (c >= 0x20) input_append(std::string(1, (char)c));
            }
        }
    }

    term_leave();
    close(fd);
    printf("\r\nbye\r\n");
    return 0;
}