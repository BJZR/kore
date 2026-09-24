// kore_tok.cpp - KORE, paso 2: tokenizer BPE de bytes (Qwen2.5) leido del GGUF
// Compilar: g++ -std=c++20 -O2 -Wall -Wextra kore_tok.cpp -o kore_tok
// Uso:      ./kore_tok modelo.gguf "texto a tokenizar"
//           ./kore_tok --spans "texto"     (solo el pre-tokenizador, sin modelo)
//
// Reutiliza el lector de kore_gguf.cpp (copiado aqui para seguir en un solo archivo).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------- PARTE 1
// Lector GGUF con chequeo de limites (igual que en el paso 1).

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
    bool err;
} Reader;

typedef struct {
    const char *ptr;   // NO termina en '\0'
    uint64_t len;
} Str;

enum {
    T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL,
    T_STR, T_ARR, T_U64, T_I64, T_F64
};

static bool rd_bytes(Reader *r, void *dst, size_t n) {
    if (r->err || n > r->size - r->pos) {
        if (!r->err)
            fprintf(stderr, "error: lectura fuera de rango en offset %zu\n", r->pos);
        r->err = true;
        return false;
    }
    memcpy(dst, r->base + r->pos, n);
    r->pos += n;
    return true;
}

static uint32_t rd_u32(Reader *r) { uint32_t v = 0; rd_bytes(r, &v, 4); return v; }
static uint64_t rd_u64(Reader *r) { uint64_t v = 0; rd_bytes(r, &v, 8); return v; }

static void rd_skip(Reader *r, uint64_t n) {
    if (r->err || n > r->size - r->pos) {
        if (!r->err)
            fprintf(stderr, "error: salto fuera de rango en offset %zu\n", r->pos);
        r->err = true;
        return;
    }
    r->pos += (size_t)n;
}

static Str rd_str(Reader *r) {
    Str s = {nullptr, 0};
    uint64_t len = rd_u64(r);
    if (r->err) return s;
    if (len > r->size - r->pos) {
        fprintf(stderr, "error: string de %llu bytes excede el archivo\n",
                (unsigned long long)len);
        r->err = true;
        return s;
    }
    s.ptr = (const char *)(r->base + r->pos);
    s.len = len;
    r->pos += (size_t)len;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL:  return 1;
        case T_U16: case T_I16:             return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default:                            return 0;
    }
}

// Salta un valor de metadato sin interpretarlo.
static void skip_value(Reader *r, uint32_t t) {
    if (t == T_STR) {
        rd_str(r);
    } else if (t == T_ARR) {
        uint32_t et = rd_u32(r);
        uint64_t n = rd_u64(r);
        if (r->err) return;
        if (et == T_STR) {
            for (uint64_t i = 0; i < n && !r->err; i++) rd_str(r);
        } else if (scalar_size(et) != 0 && n <= (r->size - r->pos) / scalar_size(et)) {
            rd_skip(r, n * scalar_size(et));
        } else {
            fprintf(stderr, "error: array invalido\n");
            r->err = true;
        }
    } else if (scalar_size(t) != 0) {
        rd_skip(r, scalar_size(t));
    } else {
        fprintf(stderr, "error: tipo de metadato desconocido: %u\n", t);
        r->err = true;
    }
}

// Lee un array de strings copiandolos a std::string.
static void read_str_array(Reader *r, std::vector<std::string> &out) {
    uint32_t et = rd_u32(r);
    uint64_t n = rd_u64(r);
    if (r->err) return;
    if (et != T_STR || n > r->size - r->pos) {   // cada string ocupa >= 8 bytes
        fprintf(stderr, "error: array de strings invalido\n");
        r->err = true;
        return;
    }
    out.reserve((size_t)n);
    for (uint64_t i = 0; i < n && !r->err; i++) {
        Str s = rd_str(r);
        if (!r->err) out.emplace_back(s.ptr, s.len);
    }
}

static void read_i32_array(Reader *r, std::vector<int32_t> &out) {
    uint32_t et = rd_u32(r);
    uint64_t n = rd_u64(r);
    if (r->err) return;
    if (et != T_I32 || n > (r->size - r->pos) / 4) {
        fprintf(stderr, "error: array de int32 invalido\n");
        r->err = true;
        return;
    }
    out.resize((size_t)n);
    rd_bytes(r, out.data(), (size_t)n * 4);
}

// Vocabulario BPE tal como lo guarda el GGUF.
struct Vocab {
    std::vector<std::string> tokens;                  // id -> texto (bytes ya mapeados a "unicode visible")
    std::vector<int32_t> types;                       // 1 normal, 3 control, 4 definido por usuario, 5 sin uso...
    std::unordered_map<std::string, int32_t> tok2id;  // texto -> id
    std::unordered_map<std::string, int32_t> rank;    // "izq der" -> prioridad de fusion (menor = antes)
    std::vector<int32_t> specials;                    // ids especiales, del mas largo al mas corto
    int32_t eos = -1, bos = -1;
};

static bool load_vocab(const char *path, Vocab &v) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return false; }
    size_t size = (size_t)st.st_size;
    void *map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap"); return false; }

    Reader r = {(const uint8_t *)map, size, 0, false};
    char magic[4] = {0};
    rd_bytes(&r, magic, 4);
    if (r.err || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "error: no es un archivo GGUF\n");
        munmap(map, size);
        return false;
    }
    rd_u32(&r);                       // version
    rd_u64(&r);                       // n_tensors
    uint64_t n_kv = rd_u64(&r);

    std::vector<std::string> merges;
    std::string model, pre;
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        Str k = rd_str(&r);
        uint32_t t = rd_u32(&r);
        if (r.err) break;
        std::string key(k.ptr, k.len);
        if (key == "tokenizer.ggml.tokens" && t == T_ARR)          read_str_array(&r, v.tokens);
        else if (key == "tokenizer.ggml.merges" && t == T_ARR)     read_str_array(&r, merges);
        else if (key == "tokenizer.ggml.token_type" && t == T_ARR) read_i32_array(&r, v.types);
        else if (key == "tokenizer.ggml.model" && t == T_STR)      { Str s = rd_str(&r); model.assign(s.ptr, s.len); }
        else if (key == "tokenizer.ggml.pre" && t == T_STR)        { Str s = rd_str(&r); pre.assign(s.ptr, s.len); }
        else if (key == "tokenizer.ggml.eos_token_id" && t == T_U32) v.eos = (int32_t)rd_u32(&r);
        else if (key == "tokenizer.ggml.bos_token_id" && t == T_U32) v.bos = (int32_t)rd_u32(&r);
        else skip_value(&r, t);
    }
    munmap(map, size);                // todo lo necesario ya esta copiado en std::string
    if (r.err) return false;

    if (v.tokens.empty() || merges.empty()) {
        fprintf(stderr, "error: el GGUF no trae vocabulario/merges BPE\n");
        return false;
    }
    if (model != "gpt2" || pre != "qwen2")
        fprintf(stderr, "aviso: tokenizer '%s'/'%s'; este codigo esta hecho para gpt2/qwen2\n",
                model.c_str(), pre.c_str());
    if (v.types.size() != v.tokens.size()) v.types.assign(v.tokens.size(), 1);

    v.tok2id.reserve(v.tokens.size() * 2);
    for (size_t i = 0; i < v.tokens.size(); i++) v.tok2id.emplace(v.tokens[i], (int32_t)i);
    v.rank.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); i++) v.rank.emplace(merges[i], (int32_t)i);

    for (size_t i = 0; i < v.tokens.size(); i++)
        if ((v.types[i] == 3 || v.types[i] == 4) && !v.tokens[i].empty())
            v.specials.push_back((int32_t)i);
    std::sort(v.specials.begin(), v.specials.end(), [&](int32_t a, int32_t b) {
        return v.tokens[a].size() > v.tokens[b].size();
    });
    return true;
}

// ---------------------------------------------------------------- PARTE 2
// Tablas byte <-> "unicode visible" de GPT-2 y fusion BPE de una palabra.
// Cada byte 0..255 se representa como un caracter imprimible: los bytes
// "comodos" (33-126, 161-172, 174-255) se representan a si mismos y el resto
// (controles, espacio...) se desplazan a U+0100 en adelante. Por eso el
// espacio (0x20) aparece en el vocabulario como 'G con punto' (U+0120).

static std::string g_b2u[256];                        // byte -> UTF-8 del caracter visible
static std::unordered_map<uint32_t, uint8_t> g_u2b;   // codepoint -> byte

static void utf8_append(std::string &s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
}

// Lee el codepoint UTF-8 que empieza en s[i]; devuelve cuantos bytes ocupa.
// Secuencia invalida -> U+FFFD consumiendo 1 byte.
static size_t utf8_next(const std::string &s, size_t i, uint32_t *cp) {
    unsigned char c = (unsigned char)s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) { *cp = 0xFFFD; return 1; }
    uint32_t v = n == 1 ? c : (uint32_t)(c & (0xFF >> (n + 1)));
    for (size_t k = 1; k < n; k++) {
        unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (cc & 0x3F);
    }
    *cp = v;
    return n;
}

static void init_byte_tables() {
    bool keep[256] = {false};
    for (int b = 33; b <= 126; b++)  keep[b] = true;
    for (int b = 161; b <= 172; b++) keep[b] = true;
    for (int b = 174; b <= 255; b++) keep[b] = true;
    uint32_t n = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = keep[b] ? (uint32_t)b : 256 + n++;
        utf8_append(g_b2u[b], cp);
        g_u2b[cp] = (uint8_t)b;
    }
}

// BPE de una "palabra" (ya separada por el pre-tokenizador): parte de un
// simbolo por byte y fusiona siempre el par adyacente de menor rank. En empate
// gana el de mas a la izquierda, igual que la implementacion de referencia.
static void bpe_word(const Vocab &v, const std::string &word, std::vector<int32_t> &out) {
    std::vector<std::string> sym;
    sym.reserve(word.size());
    for (unsigned char b : word) sym.push_back(g_b2u[b]);

    std::string key;
    while (sym.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t best = 0;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            key.assign(sym[i]);
            key += ' ';
            key += sym[i + 1];
            auto it = v.rank.find(key);
            if (it != v.rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == INT32_MAX) break;      // ya no hay fusiones aplicables
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + (long)best + 1);
    }

    for (const std::string &s : sym) {
        auto it = v.tok2id.find(s);
        if (it != v.tok2id.end()) { out.push_back(it->second); continue; }
        // No deberia pasar: caer a bytes sueltos (los 256 existen en el vocabulario).
        for (size_t i = 0; i < s.size();) {
            uint32_t c;
            size_t n = utf8_next(s, i, &c);
            auto b = v.tok2id.find(s.substr(i, n));
            if (b != v.tok2id.end()) out.push_back(b->second);
            else fprintf(stderr, "aviso: simbolo sin id (U+%04X)\n", c);
            i += n;
        }
    }
}

// ---------------------------------------------------------------- PARTE 3
// Pre-tokenizador de Qwen2, escrito a mano (sin libreria de regex). Equivale a:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Las clases \p{L} y \p{N} de Unicode se APROXIMAN por rangos (sin tablas
// completas): exactas en ASCII, Latin-1, cirilico, griego, CJK, etc.; pueden
// fallar en alfabetos poco comunes (p. ej. signos vocalicos indios).

static bool in(uint32_t c, uint32_t a, uint32_t b) { return c >= a && c <= b; }

static bool is_space(uint32_t c) {
    return in(c, 9, 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           in(c, 0x2000, 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}

static bool is_num(uint32_t c) {
    return in(c, '0', '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || in(c, 0xBC, 0xBE) ||
           in(c, 0x660, 0x669) || in(c, 0x6F0, 0x6F9) || in(c, 0x966, 0x96F) ||
           c == 0x2070 || in(c, 0x2074, 0x2079) || in(c, 0x2080, 0x2089) ||
           in(c, 0x2150, 0x2189) || in(c, 0x2460, 0x249B) || in(c, 0xFF10, 0xFF19);
}

static bool is_letter(uint32_t c) {
    if (c < 0x80) return in(c, 'a', 'z') || in(c, 'A', 'Z');
    if (is_space(c) || is_num(c)) return false;
    if (in(c, 0x80, 0xBF)) return c == 0xAA || c == 0xB5 || c == 0xBA;   // Latin-1: solo estas son letras
    if (c == 0xD7 || c == 0xF7) return false;                            // x y division
    if (in(c, 0x300, 0x36F)) return false;                              // marcas combinantes
    if (in(c, 0x2000, 0x2BFF)) return false;                             // puntuacion, simbolos, flechas, matematicas
    if (in(c, 0x3000, 0x303F)) return false;                             // puntuacion CJK
    if (in(c, 0xE000, 0xF8FF)) return false;                             // uso privado
    if (in(c, 0xFE00, 0xFE0F) || in(c, 0xFE30, 0xFE6F)) return false;    // selectores de variacion, compatibilidad
    if (in(c, 0xFF00, 0xFF0F) || in(c, 0xFF1A, 0xFF20) ||
        in(c, 0xFF3B, 0xFF40) || in(c, 0xFF5B, 0xFF65)) return false;    // puntuacion de ancho completo
    if (c == 0xFFFD || in(c, 0x1F000, 0x1FAFF)) return false;            // invalido, emoji
    return true;
}

struct Span { size_t b, e; };   // rango de BYTES [b, e) dentro del texto original

static std::vector<Span> pretokenize(const std::string &s) {
    std::vector<uint32_t> cp;      // codepoints
    std::vector<size_t> off;       // offset en bytes de cada codepoint (+ el final)
    for (size_t i = 0; i < s.size();) {
        uint32_t c;
        size_t n = utf8_next(s, i, &c);
        cp.push_back(c);
        off.push_back(i);
        i += n;
    }
    off.push_back(s.size());
    const size_t N = cp.size();

    auto is_nl = [&](size_t i) { return cp[i] == '\n' || cp[i] == '\r'; };
    auto is_sym = [&](size_t i) { return !is_space(cp[i]) && !is_letter(cp[i]) && !is_num(cp[i]); };

    std::vector<Span> out;
    size_t i = 0;
    while (i < N) {
        size_t e = 0;              // fin (en codepoints) del token; 0 = ninguna alternativa encajo

        // 1) contracciones: 's 't 're 've 'm 'll 'd (sin distinguir mayusculas)
        if (cp[i] == '\'' && i + 1 < N) {
            uint32_t a = cp[i + 1] | 0x20;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                e = i + 2;
            } else if (i + 2 < N) {
                uint32_t b = cp[i + 2] | 0x20;
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = i + 3;
            }
        }

        // 2) [^\r\n\p{L}\p{N}]? \p{L}+   (un prefijo opcional + letras)
        if (!e) {
            size_t j = i;
            if (!is_letter(cp[i]) && !is_nl(i) && !is_num(cp[i]) && i + 1 < N && is_letter(cp[i + 1]))
                j = i + 1;
            if (is_letter(cp[j])) {
                while (j < N && is_letter(cp[j])) j++;
                e = j;
            }
        }

        // 3) un solo digito
        if (!e && is_num(cp[i])) e = i + 1;

        // 4) ' '? [^\s\p{L}\p{N}]+ [\r\n]*   (signos de puntuacion / simbolos)
        if (!e) {
            size_t j = i;
            if (cp[j] == ' ') j++;
            if (j < N && is_sym(j)) {
                while (j < N && is_sym(j)) j++;
                while (j < N && is_nl(j)) j++;
                e = j;
            }
        }

        // 5-7) espacios en blanco
        if (!e && is_space(cp[i])) {
            size_t k = i;
            while (k < N && is_space(cp[k])) k++;         // k = fin de la racha de espacios
            size_t last_nl = 0;
            bool has_nl = false;
            for (size_t m = i; m < k; m++)
                if (is_nl(m)) { last_nl = m; has_nl = true; }
            if (has_nl)          e = last_nl + 1;         // \s*[\r\n]+
            else if (k == N)     e = k;                   // \s+(?!\S) al final del texto
            else if (k - i >= 2) e = k - 1;               // deja 1 espacio para la palabra siguiente
            else                 e = k;                   // \s+
        }

        if (!e) e = i + 1;                                // seguridad: nunca bucle infinito
        out.push_back({off[i], off[e]});
        i = e;
    }
    return out;
}

// ---------------------------------------------------------------- PARTE 3b
// encode / decode

static void encode_plain(const Vocab &v, const std::string &text, std::vector<int32_t> &ids) {
    for (const Span &sp : pretokenize(text))
        bpe_word(v, text.substr(sp.b, sp.e - sp.b), ids);
}

// Convierte texto a ids. Con parse_special, las cadenas como "<|im_start|>" se
// convierten directamente en su id de token especial (ChatML), sin pasar por BPE.
static std::vector<int32_t> encode(const Vocab &v, const std::string &text, bool parse_special) {
    std::vector<int32_t> ids;
    size_t seg = 0, i = 0;     // [seg, i) = tramo de texto normal pendiente
    while (i < text.size()) {
        int32_t hit = -1;
        if (parse_special) {
            for (int32_t id : v.specials) {            // ya ordenados: el mas largo primero
                const std::string &t = v.tokens[id];
                if (text.compare(i, t.size(), t) == 0) { hit = id; break; }
            }
        }
        if (hit < 0) { i++; continue; }
        if (i > seg) encode_plain(v, text.substr(seg, i - seg), ids);
        ids.push_back(hit);
        i += v.tokens[hit].size();
        seg = i;
    }
    if (text.size() > seg) encode_plain(v, text.substr(seg), ids);
    return ids;
}

// ids -> bytes originales (deshace el mapeo byte<->unicode visible).
static std::string decode(const Vocab &v, const std::vector<int32_t> &ids) {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= v.tokens.size()) continue;
        const std::string &t = v.tokens[(size_t)id];
        for (size_t i = 0; i < t.size();) {
            uint32_t c;
            size_t n = utf8_next(t, i, &c);
            auto it = g_u2b.find(c);
            if (it != g_u2b.end()) out += (char)it->second;
            else out.append(t, i, n);
            i += n;
        }
    }
    return out;
}

// ---------------------------------------------------------------- PARTE 4
int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--spans") == 0) {   // depuracion del pre-tokenizador
        for (const Span &sp : pretokenize(argv[2])) printf("%zu %zu\n", sp.b, sp.e);
        return 0;
    }
    if (argc != 3) {
        fprintf(stderr, "uso: %s modelo.gguf \"texto\"\n       %s --spans \"texto\"\n", argv[0], argv[0]);
        return 1;
    }

    init_byte_tables();
    Vocab v;
    if (!load_vocab(argv[1], v)) return 1;
    printf("vocab: %zu tokens, %zu merges, %zu especiales, bos=%d eos=%d\n\n",
           v.tokens.size(), v.rank.size(), v.specials.size(), v.bos, v.eos);

    std::string text = argv[2];
    std::vector<int32_t> ids = encode(v, text, true);
    printf("%zu tokens:\n", ids.size());
    for (int32_t id : ids)
        printf("  %7d  %s\n", id, v.tokens[(size_t)id].c_str());

    std::string back = decode(v, ids);
    printf("\nroundtrip: %s\n", back == text ? "OK" : "FALLO");
    return back == text ? 0 : 2;
}
