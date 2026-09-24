// kore_client.cpp - KORE, iteracion 2: cliente IPC del servidor --serve
// Compilar: g++ -std=c++20 -O2 -Wall -Wextra kore_client.cpp -o kore_client
// Uso:      ./kore_client /tmp/kore.sock "pregunta" [opciones]
//   -r        texto crudo, sin plantilla ChatML
//   -n N      maximo de tokens a pedir (10000)
//   -s "txt"  reinicia la conversacion con ese system message
//
// Envia PROMPT y va tirando de TOKEN hasta que llega END, imprimiendo el
// streaming. El mismo cliente se puede usar para reconectar mientras el
// servidor conserva la conversacion (caché KV viva).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

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

static bool recv_frame(int fd, char *type, std::vector<uint8_t> &payload) {
    char hdr[5];
    if (!sock_read_all(fd, hdr, 5)) return false;
    *type = hdr[0];
    uint32_t len = (uint32_t)(uint8_t)hdr[1] | ((uint32_t)(uint8_t)hdr[2] << 8) |
                   ((uint32_t)(uint8_t)hdr[3] << 16) | ((uint32_t)(uint8_t)hdr[4] << 24);
    if (len > (64u << 20)) return false;
    payload.resize(len);
    return len == 0 || sock_read_all(fd, payload.data(), len);
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 0 = ok, 1 = error.
static int run_client(int fd, const std::string &prompt,
                      bool raw, int max_tokens, const std::string &system_msg) {
    char type = 0;
    std::vector<uint8_t> pay;
    if (recv_frame(fd, &type, pay) && type == 'M')
        fprintf(stderr, "[servidor] %.*s\n", (int)pay.size(), (const char *)pay.data());

    if (!system_msg.empty()) {
        if (!send_frame(fd, 'R', system_msg.data(), (uint32_t)system_msg.size())) return 1;
        if (!recv_frame(fd, &type, pay) || type != 'R') return 1;
    }

    std::string payload;
    payload += raw ? 'R' : 'C';
    payload += prompt;
    if (!send_frame(fd, 'P', payload.data(), (uint32_t)payload.size())) return 1;
    if (!recv_frame(fd, &type, pay)) return 1;
    if (type == 'X') {
        fprintf(stderr, "\n[error] %.*s\n", (int)pay.size(), (const char *)pay.data());
        return 1;
    }
    if (type != 'P') return 1;

    const double t0 = now_ms();
    int n = 0;
    for (;;) {
        if (n >= max_tokens) { send_frame(fd, 'A', nullptr, 0); break; }
        if (!send_frame(fd, 'T', nullptr, 0)) return 1;
        if (!recv_frame(fd, &type, pay)) return 1;
        if (type == 'T') {
            if (pay.size() >= 4) {
                fwrite(pay.data() + 4, 1, pay.size() - 4, stdout);
                fflush(stdout);
                ++n;
            }
        } else if (type == 'E') {
            break;
        } else if (type == 'X') {
            fprintf(stderr, "\n[error] %.*s\n", (int)pay.size(), (const char *)pay.data());
            return 1;
        } else {
            fprintf(stderr, "[protocolo] respuesta inesperada '%c'\n", type);
            return 1;
        }
    }
    const double ms = now_ms() - t0;
    fprintf(stderr, "\n[generados %d tokens en %.0f ms, %.1f tok/s]\n", n, ms,
            n > 0 ? (double)n / (ms / 1000.0) : 0.0);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s /ruta/socket \"pregunta\" [-r] [-n N] [-s \"system\"]\n", argv[0]);
        return 1;
    }
    const char *sock_path = argv[1];
    const std::string prompt = argv[2];
    bool raw = false;
    int max_tokens = 10000;
    std::string system_msg;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        const bool has_val = i + 1 < argc;
        if (a == "-n" && has_val)             max_tokens = atoi(argv[++i]);
        else if (a == "-r" || a == "--raw")   raw = true;
        else if ((a == "-s" || a == "--system") && has_val) system_msg = argv[++i];
        else { fprintf(stderr, "argumento no reconocido: %s\n", a.c_str()); return 1; }
    }
    if (max_tokens < 1) max_tokens = 1;

    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "error: no se puede conectar a %s (¿esta kore --serve corriendo?)\n", sock_path);
        close(fd);
        return 1;
    }

    const int rc = run_client(fd, prompt, raw, max_tokens, system_msg);
    close(fd);
    return rc;
}