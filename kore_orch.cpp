// kore_orch.cpp - KORE, iteracion 3: Orchestrator (bucle ReAct + servidor de estado)
// Compilar: g++ -std=c++20 -O2 -pthread -Wall -Wextra kore_orch.cpp -o kore_orch
// Uso:      ./kore_orch [--core /tmp/kore.sock] [--state /tmp/kore-orch.sock] [--system "prompt"]
//
// Se conecta a KORE-Core (cliente del protocolo de tramas) y expone un socket
// de estado para la TUI. Protocolo de salida (Orchestrator -> interfaz):
//   'M' meta   'S' estado del agente   'U' turno de usuario
//   'A' fragmento de asistente (streaming)   'T' llamada a herramienta
//   'O' salida de la herramienta   'E' fin de turno   'X' error
// Entrada (interfaz -> Orchestrator): 'P' prompt, 'R' reset, 'Q' estado.
//
// Bucle ReAct: mientras llegan tokens, acumula en un buffer con ventana de
// mirada previa. Si detecta "[ACTION: <tool> <arg>]" completo, pausa, ejecuta
// la herramienta (fork/execve), reinyecta el resultado en Core como turno de
// usuario via 'I' y continua generando.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
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

static int listen_unix(const char *path) {
    unlink(path);
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    if (listen(fd, 4) < 0) { close(fd); return -1; }
    return fd;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------ herramientas

struct ToolResult {
    bool ok = false;
    bool timed_out = false;
    std::string out;
};

// Lanza "<tool> <arg>" con fork/execve (shell del usuario o python3 -c) y captura stdout+stderr.
static const char *shell_path = "/bin/sh";
static const char *shell_name = "sh";
static void init_shell() {
    const char *s = getenv("SHELL");
    if (s && s[0] && access(s, X_OK) == 0) {
        shell_path = s;
        const char *b = strrchr(s, '/');
        shell_name = b ? b + 1 : s;
    }
}
static ToolResult exec_tool(const std::string &tool, const std::string &arg, size_t max_out) {
    ToolResult r;
    int pfd[2];
    if (pipe(pfd) < 0) { r.out = "error: no se pudo crear el pipe"; return r; }
    fcntl(pfd[0], F_SETFL, O_NONBLOCK);
    pid_t pid = fork();
    if (pid < 0) { close(pfd[0]); close(pfd[1]); r.out = "error: fork fallo"; return r; }
    if (pid == 0) {
        dup2(pfd[1], 1);
        dup2(pfd[1], 2);
        close(pfd[0]);
        close(pfd[1]);
        if (tool == "python")      execl("/usr/bin/python3", "python3", "-c", arg.c_str(), (char *)0);
        else if (tool == "python3") execl("/usr/bin/python3", "python3", "-c", arg.c_str(), (char *)0);
        else                       execl(shell_path, shell_name, "-c", arg.c_str(), (char *)0);
        _exit(127);
    }
    close(pfd[1]);
    const double deadline = now_s() + 30.0;
    char tmp[4096];
    for (;;) {
        if (now_s() > deadline) {
            kill(pid, SIGKILL);
            r.timed_out = true;
            break;
        }
        struct pollfd p = {pfd[0], POLLIN | POLLHUP, 0};
        const int pr = poll(&p, 1, 250);
        if (pr < 0 && errno == EINTR) continue;
        if (pr > 0 && (p.revents & POLLIN)) {
            for (;;) {
                const ssize_t n = read(pfd[0], tmp, sizeof(tmp));
                if (n > 0) { r.out.append(tmp, (size_t)n); if (r.out.size() >= max_out) continue; }
                else break;
            }
        }
        int status = 0;
        const pid_t wr = waitpid(pid, &status, WNOHANG);
        if (wr == pid) {
            for (;;) {
                const ssize_t n = read(pfd[0], tmp, sizeof(tmp));
                if (n <= 0) break;
                r.out.append(tmp, (size_t)n);
            }
            r.ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
            break;
        }
        if (wr < 0) { r.ok = false; break; }
    }
    close(pfd[0]);
    if (r.timed_out) {
        int status;
        waitpid(pid, &status, 0);
        r.out += "\n[tool timeout]";
    }
    if (r.out.size() > max_out) {
        r.out = r.out.substr(0, max_out);
        r.out += "\n[tool out recortada]";
    }
    return r;
}

// ------------------------------------------------------------------ bucle ReAct

struct Turn {
    std::string buf;        // texto bruto del asistente en curso
    size_t emit_upto = 0;   // bytes ya enviados a la interfaz
    size_t pend = std::string::npos;   // donde abre "[ACTION:" aun sin ']'
    size_t pend_since = 0;
};

static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Envia un fragmento de asistente a la interfaz.
static void emit_buffer(int ui, Turn &t) {
    if (t.emit_upto >= t.buf.size()) return;
    const std::string piece = t.buf.substr(t.emit_upto);
    t.emit_upto = t.buf.size();
    send_frame(ui, 'A', piece.data(), (uint32_t)piece.size());
}

static void emit_status(int ui, const std::string &s) {
    send_frame(ui, 'S', s.data(), (uint32_t)s.size());
}

// Detecta "[ACTION: <tool> <arg>]" en t.buf. Devuelve true si proceso una accion.
static bool scan_action(Turn &t, int ui, int core) {
    const size_t limit = t.buf.size();
    for (;;) {
        const std::string open = "[ACTION:";
        size_t s = std::string::npos;
        size_t from = t.emit_upto < t.pend ? t.emit_upto : 0;
        const size_t search_from = (t.pend != std::string::npos) ? t.pend : from;
        s = t.buf.find(open, search_from);
        if (s == std::string::npos) { t.pend = std::string::npos; break; }

        const size_t e = t.buf.find(']', s + open.size());
        if (e == std::string::npos) {
            if (t.pend == s) {
                if (limit - s > 512) t.pend = std::string::npos;   // marcador demasiado largo: ignoralo
                return false;
            }
            t.pend = s;
            t.pend_since = limit;
            return false;
        }
        if (e > limit) return false;
        const std::string body = trim(t.buf.substr(s + open.size(), e - s - open.size()));
        t.pend = std::string::npos;
        if (body.empty()) { t.buf.erase(0, e + 1); t.emit_upto = 0; continue; }

        const size_t sp = body.find(' ');
        std::string tool = body, arg;
        if (sp != std::string::npos) { tool = body.substr(0, sp); arg = trim(body.substr(sp + 1)); }
        tool = trim(tool);

        if (s > t.emit_upto) {                     // texto antes de la accion
            const std::string pre = t.buf.substr(t.emit_upto, s - t.emit_upto);
            send_frame(ui, 'A', pre.data(), (uint32_t)pre.size());
        }
        const std::string show = "# " + tool + (arg.empty() ? "" : " " + arg);
        send_frame(ui, 'T', show.data(), (uint32_t)show.size());
        emit_status(ui, "agente: ejecutando " + tool + "...");

        const ToolResult res = exec_tool(tool, arg, 3000);
        const std::string out = res.ok || !res.out.empty()
            ? (res.timed_out ? res.out : res.out)
            : "(sin salida)";
        send_frame(ui, 'O', out.data(), (uint32_t)out.size());

        if (core >= 0) {                           // reinyecta el resultado y sigue
            send_frame(core, 'I', out.data(), (uint32_t)out.size());
            char type = 0;
            Buf pay;
            if (!recv_frame(core, &type, pay) || type != 'I') {
                const std::string m = "core no acepto la reinyeccion de la herramienta";
                send_frame(ui, 'X', m.data(), (uint32_t)m.size());
                return false;
            }
        }
        t.buf.erase(0, e + 1);                     // el resto tras ']' se descarta: el modelo
        t.emit_upto = 0;                           // CONTINUARA desde el resultado inyectado
        if (t.buf.empty()) break;
    }
    return true;
}

// ------------------------------------------------------------------ turno completo

// Ejecuta un turno: envia el prompt a Core y hace streaming de tokens a la
// interfaz, intercalando el bucle ReAct (detectar accion -> ejecutar -> inyectar).
static void run_turn(int ui, int core, const std::string &user_text) {
    send_frame(ui, 'U', user_text.data(), (uint32_t)user_text.size());
    emit_status(ui, "agente: pensando...");

    std::string payload = "C";
    payload += user_text;
    if (!send_frame(core, 'P', payload.data(), (uint32_t)payload.size())) {
        const std::string m = "core no disponible";
        send_frame(ui, 'X', m.data(), (uint32_t)m.size());
        return;
    }
    char type = 0;
    Buf pay;
    if (!recv_frame(core, &type, pay) || type == 'X') {
        const std::string m(type == 'X' ? std::string((char *)pay.data(), pay.size()) : std::string("core no respondio"));
        send_frame(ui, 'X', m.data(), (uint32_t)m.size());
        return;
    }

    Turn t;
    for (;;) {
        send_frame(core, 'T', nullptr, 0);
        if (!recv_frame(core, &type, pay)) {
            const std::string m = "core desconectado durante la generacion";
            send_frame(ui, 'X', m.data(), (uint32_t)m.size());
            return;
        }
        if (type == 'X') {
            send_frame(ui, 'X', pay.data(), (uint32_t)pay.size());
            return;
        }
        if (type == 'E') {
            if (t.pend != std::string::npos) {      // marcador abierto sin cerrar: muestra el texto
                t.pend = std::string::npos;
                if (t.emit_upto < t.buf.size()) emit_buffer(ui, t);
            } else if (t.emit_upto < t.buf.size()) {
                emit_buffer(ui, t);
            }
            send_frame(ui, 'E', nullptr, 0);
            emit_status(ui, "agente: listo");
            return;
        }
        if (type == 'T') {
            if (pay.size() < 4) {                              // Core envia [id u32] + fragmento
                const std::string m = "trama TOKEN malformada de core";
                send_frame(ui, 'X', m.data(), (uint32_t)m.size());
                return;
            }
            t.buf += std::string((char *)pay.data() + 4, pay.size() - 4);
            if (!scan_action(t, ui, core)) {
                const size_t safe = t.buf.size() > 48 ? t.buf.size() - 48 : 0;
                if (safe > t.emit_upto) {
                    const std::string piece = t.buf.substr(t.emit_upto, safe - t.emit_upto);
                    t.emit_upto = safe;
                    send_frame(ui, 'A', piece.data(), (uint32_t)piece.size());
                }
            }
        }
    }
}

// ------------------------------------------------------------------ programa principal

static void usage(const char *prog) {
    fprintf(stderr,
        "uso: %s [--core /tmp/kore.sock] [--state /tmp/kore-orch.sock] [--system \"...\"]\n", prog);
}

int main(int argc, char **argv) {
    std::string core_path = "/tmp/kore.sock";
    std::string state_path = "/tmp/kore-orch.sock";
    std::string system_msg =
        "You are KORE, an autonomous assistant with access to tools.\n"
        "When you need real data or want to run code, request an action on its own line EXACTLY as:\n"
        "[ACTION: sh <bash command or script>]\n"
        "[ACTION: python <python3 source code>]\n"
        "The tool then executes and its output is provided to you. Use that output to answer.\n"
        "Never wrap tools in quotes or backticks; the brackets delimit the entire request.\n"
        "If no tool is needed, just answer directly.";
    init_shell();
    fprintf(stderr, "shell del agente: %s (%s)\n", shell_name, shell_path);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--core" && has)  core_path = argv[++i];
        else if (a == "--state" && has) state_path = argv[++i];
        else if (a == "--system" && has) system_msg = argv[++i];
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 1; }
    }

    const int core = connect_unix(core_path.c_str());
    if (core < 0) {
        fprintf(stderr, "error: no se puede conectar a KORE-Core en %s (¿corre con --serve?)\n", core_path.c_str());
        return 1;
    }
    char type = 0;
    Buf pay;
    recv_frame(core, &type, pay);                    // 'M' meta
    if (type != 'M') { fprintf(stderr, "error: protocolo de core inesperado\n"); return 1; }
    if (!send_frame(core, 'R', system_msg.data(), (uint32_t)system_msg.size()) ||
        !recv_frame(core, &type, pay) || type != 'R') {
        fprintf(stderr, "error: core no acepto el system message\n");
        return 1;
    }
    fprintf(stderr, "Orchestrator: conectado a %s, estado en %s\n", core_path.c_str(), state_path.c_str());

    const int srv = listen_unix(state_path.c_str());
    if (srv < 0) { perror("listen_unix"); return 1; }
    fprintf(stderr, "Orchestrator: sirviendo estado en %s\n", state_path.c_str());

    for (;;) {
        const int ui = accept(srv, nullptr, nullptr);
        if (ui < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        const std::string meta = "kore-orch listo";
        send_frame(ui, 'M', meta.data(), (uint32_t)meta.size());
        emit_status(ui, "agente: listo");
        fprintf(stderr, "Orchestrator: interfaz conectada\n");

        for (;;) {
            if (!recv_frame(ui, &type, pay)) break;
            if (type == 'P' && !pay.empty()) {
                const std::string prompt((const char *)pay.data(), pay.size());
                run_turn(ui, core, prompt);
            } else if (type == 'R') {
                send_frame(ui, 'R', nullptr, 0);
                send_frame(core, 'R', system_msg.data(), (uint32_t)system_msg.size());
                recv_frame(core, &type, pay);
                emit_status(ui, "agente: conversacion reiniciada");
            } else if (type == 'Q') {
                send_frame(ui, 'S', "agente: listo", 14);
            } else {
                const std::string m = "comando desconocido";
                send_frame(ui, 'X', m.data(), (uint32_t)m.size());
            }
        }
        close(ui);
        fprintf(stderr, "Orchestrator: interfaz desconectada\n");
    }
    close(srv);
    unlink(state_path.c_str());
    close(core);
    return 0;
}