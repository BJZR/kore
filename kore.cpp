// kore.cpp - KORE: motor de inferencia de Qwen2.5 en C++ puro (un solo archivo)
// Compilar: g++ -std=c++20 -O2 -pthread -Wall -Wextra kore.cpp -o kore
// Uso:      ./kore modelo.gguf "tu pregunta"             (chat ChatML, generacion en streaming)
//           ./kore modelo.gguf "texto" --raw             (sin plantilla de chat)
//           ./kore modelo.gguf --ids 151644,872,198      (depuracion: logits del ultimo token)
//           ./kore modelo.gguf --serve /tmp/kore.sock    (iteracion 2: servidor IPC)
// Opciones: -n N (tokens a generar, 256)  --ctx N (contexto, 2048)  --threads N  --temp T (0 = greedy)
//           --system "..."  --seed N  --float (matvec exacto en float, lento: para depurar)  --dump f.f32
//
// Partes: 1-5 = cargador GGUF, descuantizacion y kernels Q4_K/Q6_K (pasos 1, 3 y 4)
//         TOK = tokenizer BPE (paso 2)   |   6 = forward de Qwen2   |   7 = programa principal
//         8 = servidor IPC sobre Unix Domain Socket (protocolo PROMPT/TOKEN/RESET/ABORT)

#include <algorithm>
#include <functional>
#include <chrono>
#include <immintrin.h>
#include <random>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

// ---------------------------------------------------------------- PARTE 1
// Lector GGUF con chequeo de limites (igual que en los pasos 1 y 2).

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
static float rd_f32(Reader *r) { uint32_t u = rd_u32(r); float f; memcpy(&f, &u, 4); return f; }

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

// Elementos por bloque y bytes por bloque de cada tipo ggml (0 = desconocido).
static bool type_info(uint32_t t, const char **name, uint64_t *blk, uint64_t *bytes) {
    switch (t) {
        case 0:  *name = "F32";  *blk = 1;   *bytes = 4;   return true;
        case 1:  *name = "F16";  *blk = 1;   *bytes = 2;   return true;
        case 2:  *name = "Q4_0"; *blk = 32;  *bytes = 18;  return true;
        case 3:  *name = "Q4_1"; *blk = 32;  *bytes = 20;  return true;
        case 6:  *name = "Q5_0"; *blk = 32;  *bytes = 22;  return true;
        case 7:  *name = "Q5_1"; *blk = 32;  *bytes = 24;  return true;
        case 8:  *name = "Q8_0"; *blk = 32;  *bytes = 34;  return true;
        case 10: *name = "Q2_K"; *blk = 256; *bytes = 84;  return true;
        case 11: *name = "Q3_K"; *blk = 256; *bytes = 110; return true;
        case 12: *name = "Q4_K"; *blk = 256; *bytes = 144; return true;
        case 13: *name = "Q5_K"; *blk = 256; *bytes = 176; return true;
        case 14: *name = "Q6_K"; *blk = 256; *bytes = 210; return true;
        case 15: *name = "Q8_K"; *blk = 256; *bytes = 292; return true;
        case 30: *name = "BF16"; *blk = 1;   *bytes = 2;   return true;
        default: return false;
    }
}

enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_K = 12, GGML_Q6_K = 14, GGML_BF16 = 30 };

// Un tensor apunta directamente al archivo mapeado (cero copias).
struct Tensor {
    uint32_t type = 0;
    uint32_t n_dims = 0;
    uint64_t dims[4] = {1, 1, 1, 1};   // dims[0] es la dimension CONTIGUA (orden ggml)
    uint64_t nbytes = 0;
    const uint8_t *data = nullptr;     // nullptr si el tipo es desconocido
};

struct Hparams {
    uint32_t n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0;
    uint32_t n_ctx = 0, n_vocab = 0;
    float rope_base = 0, rms_eps = 0;
};

struct Model {
    Hparams hp;
    std::unordered_map<std::string, Tensor> tensors;   // busqueda por nombre
    size_t file_size = 0;
    uint64_t data_start = 0;
};

// ---------------------------------------------------------------- PARTE 2
// Carga del modelo: hiperparametros + tabla de tensores por nombre.

static bool load_model(const char *path, Model &m) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return false; }
    size_t size = (size_t)st.st_size;
    void *map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap"); return false; }   // el mapeo vive hasta salir del proceso
    m.file_size = size;

    Reader r = {(const uint8_t *)map, size, 0, false};
    char magic[4] = {0};
    rd_bytes(&r, magic, 4);
    if (r.err || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "error: no es un archivo GGUF\n");
        return false;
    }
    uint32_t version = rd_u32(&r);
    uint64_t n_tensors = rd_u64(&r);
    uint64_t n_kv = rd_u64(&r);
    if (version < 2 || version > 3)
        fprintf(stderr, "aviso: version GGUF %u no probada\n", version);

    // --- Metadatos: solo los que necesita el motor; el resto se salta.
    uint64_t alignment = 32;
    Hparams &hp = m.hp;
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        Str k = rd_str(&r);
        uint32_t t = rd_u32(&r);
        if (r.err) break;
        std::string key(k.ptr, k.len);
        if (key == "general.alignment" && t == T_U32) {
            uint32_t a = rd_u32(&r);
            if (a != 0) alignment = a;
        }
        else if (key == "qwen2.block_count" && t == T_U32)                    hp.n_layer = rd_u32(&r);
        else if (key == "qwen2.embedding_length" && t == T_U32)               hp.n_embd = rd_u32(&r);
        else if (key == "qwen2.feed_forward_length" && t == T_U32)            hp.n_ff = rd_u32(&r);
        else if (key == "qwen2.attention.head_count" && t == T_U32)           hp.n_head = rd_u32(&r);
        else if (key == "qwen2.attention.head_count_kv" && t == T_U32)        hp.n_head_kv = rd_u32(&r);
        else if (key == "qwen2.context_length" && t == T_U32)                 hp.n_ctx = rd_u32(&r);
        else if (key == "qwen2.rope.freq_base" && t == T_F32)                 hp.rope_base = rd_f32(&r);
        else if (key == "qwen2.attention.layer_norm_rms_epsilon" && t == T_F32) hp.rms_eps = rd_f32(&r);
        else if (key == "tokenizer.ggml.tokens" && t == T_ARR && r.size - r.pos >= 12) {
            uint64_t n;
            memcpy(&n, r.base + r.pos + 4, 8);              // tras el tipo del elemento viene la cantidad
            hp.n_vocab = (uint32_t)n;
            skip_value(&r, t);
        }
        else skip_value(&r, t);
    }

    // --- Tabla de tensores (todavia sin punteros: el inicio de datos se conoce al final)
    struct Info { std::string name; Tensor t; uint64_t offset; };
    std::vector<Info> infos;
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        Info in;
        Str name = rd_str(&r);
        uint32_t n_dims = rd_u32(&r);
        if (r.err) break;
        if (n_dims == 0 || n_dims > 4) {
            fprintf(stderr, "error: n_dims=%u invalido\n", n_dims);
            return false;
        }
        in.name.assign(name.ptr, name.len);
        in.t.n_dims = n_dims;
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            in.t.dims[d] = rd_u64(&r);
            nelem *= in.t.dims[d];
        }
        in.t.type = rd_u32(&r);
        in.offset = rd_u64(&r);
        if (r.err) break;
        const char *tn; uint64_t blk, bpb;
        if (type_info(in.t.type, &tn, &blk, &bpb) && nelem % blk == 0)
            in.t.nbytes = nelem / blk * bpb;
        infos.push_back(std::move(in));
    }
    if (r.err) return false;

    m.data_start = (r.pos + alignment - 1) / alignment * alignment;
    if (m.data_start > size) { fprintf(stderr, "error: archivo truncado\n"); return false; }
    uint64_t avail = size - m.data_start;
    for (Info &in : infos) {
        if (in.t.nbytes != 0) {
            if (in.offset > avail || in.t.nbytes > avail - in.offset) {
                fprintf(stderr, "error: el tensor %s se sale del archivo (descarga incompleta?)\n",
                        in.name.c_str());
                return false;
            }
            in.t.data = (const uint8_t *)map + m.data_start + in.offset;
        }
        m.tensors.emplace(in.name, in.t);
    }
    return true;
}

// ---------------------------------------------------------------- PARTE 3
// Descuantizacion escalar de referencia (mismo esquema que ggml).

// Estas funciones auxiliares se usan dentro de los kernels AVX2: DEBEN expandirse en linea. Si el compilador
// las deja como llamadas (pasa con -O2), corren con codificacion SSE antigua mientras los registros YMM
// estan sucios y cada llamada paga la penalizacion de transicion SSE<->AVX (kernel ~10x mas lento).
#define KORE_INLINE static inline __attribute__((always_inline))

KORE_INLINE float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                                   // +-0
        } else {                                           // subnormal: normalizar
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x400));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (man << 13);            // inf / NaN
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

KORE_INLINE float rd_f16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return f16_to_f32(h);
}

// Q4_K: bloque de 144 bytes = 256 valores.
//   [0..1] d (f16)  [2..3] dmin (f16)  [4..15] 8 escalas y 8 minimos de 6 bits  [16..143] 256 nibbles
// valor = d*escala*nibble - dmin*minimo, en 8 sub-bloques de 32 valores.
KORE_INLINE void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void dequant_block_q4_K(const uint8_t *b, float *y) {
    const float d = rd_f16(b);
    const float dmin = rd_f16(b + 2);
    const uint8_t *scales = b + 4;
    const uint8_t *q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, mn;
        get_scale_min_k4(is + 0, scales, &sc, &mn);
        const float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, scales, &sc, &mn);
        const float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0x0F) - m1;   // nibbles bajos: primeros 32
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;     // nibbles altos: siguientes 32
        q += 32;
        is += 2;
    }
}

// Q6_K: bloque de 210 bytes = 256 valores.
//   [0..127] 4 bits bajos  [128..191] 2 bits altos  [192..207] 16 escalas int8  [208..209] d (f16)
// valor = d * escala * (q6 - 32), en 16 sub-bloques de 16 valores.
static void dequant_block_q6_K(const uint8_t *b, float *y) {
    const uint8_t *ql = b;
    const uint8_t *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    const float d = rd_f16(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0x0F)      | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4)        | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l]      = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// Descuantiza una fila completa (dims[0] elementos) de un tensor a float32.
static bool get_row(const Tensor &t, uint64_t row, std::vector<float> &out) {
    const char *tn; uint64_t blk, bpb;
    if (!t.data || !type_info(t.type, &tn, &blk, &bpb)) return false;
    const uint64_t n = t.dims[0];
    const uint64_t n_rows = t.dims[1] * t.dims[2] * t.dims[3];
    if (row >= n_rows || n % blk != 0) return false;
    const uint8_t *p = t.data + row * (n / blk * bpb);
    out.resize((size_t)n);
    switch (t.type) {
        case GGML_F32:  memcpy(out.data(), p, (size_t)n * 4); return true;
        case GGML_F16:  for (uint64_t i = 0; i < n; i++) out[i] = rd_f16(p + 2 * i); return true;
        case GGML_BF16:
            for (uint64_t i = 0; i < n; i++) {
                uint32_t u = (uint32_t)(p[2 * i] | (p[2 * i + 1] << 8)) << 16;
                memcpy(&out[i], &u, 4);
            }
            return true;
        case GGML_Q4_K: for (uint64_t b = 0; b < n / 256; b++) dequant_block_q4_K(p + b * 144, &out[b * 256]); return true;
        case GGML_Q6_K: for (uint64_t b = 0; b < n / 256; b++) dequant_block_q6_K(p + b * 210, &out[b * 256]); return true;
        default:
            fprintf(stderr, "tipo %s todavia no soportado\n", tn);
            return false;
    }
}


// ---------------------------------------------------------------- PARTE 4
// Activaciones en Q8_K y productos punto escalares de referencia.
// Igual que ggml: el vector x se cuantiza a bloques Q8_K (256 int8 + escala) y el
// producto con los pesos Q4_K/Q6_K se hace casi todo en aritmetica entera.

struct BlockQ8K {
    float d;             // escala del bloque
    int8_t qs[256];      // valores cuantizados
    int16_t bsums[16];   // suma de qs en cada grupo de 16 (para los minimos / sesgos)
};
static_assert(sizeof(BlockQ8K) == 292, "BlockQ8K debe medir 292 bytes");

static void quantize_row_q8_K(const float *x, BlockQ8K *y, int64_t n) {
    for (int64_t i = 0; i < n / 256; i++, x += 256) {
        float amax = 0, maxv = 0;
        for (int j = 0; j < 256; j++) {
            float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; maxv = x[j]; }
        }
        if (amax == 0) {
            y[i].d = 0;
            memset(y[i].qs, 0, 256);
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const float iscale = -127.0f / maxv;             // el valor de mayor modulo pasa a -127
        for (int j = 0; j < 256; j++) {
            int v = (int)std::nearbyint(iscale * x[j]);  // redondeo al par (modo por defecto)
            y[i].qs[j] = (int8_t)(v > 127 ? 127 : v);
        }
        for (int j = 0; j < 16; j++) {
            int s = 0;
            for (int l = 0; l < 16; l++) s += y[i].qs[16 * j + l];
            y[i].bsums[j] = (int16_t)s;
        }
        y[i].d = 1.0f / iscale;
    }
}

// Producto punto de una fila Q4_K (n valores) con x en Q8_K. Referencia escalar.
//   suma = d*dx * sum_j esc[j]*(sum q4*q8)  -  dmin*dx * sum_j min[j]*(sum q8)
static float dot_q4_K_q8_K_ref(int64_t n, const uint8_t *w, const BlockQ8K *y) {
    float sumf = 0;
    for (int64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = w + i * 144;
        const float d = rd_f16(b) * y[i].d;
        const float dmin = rd_f16(b + 2) * y[i].d;
        const uint8_t *scales = b + 4;
        const uint8_t *q = b + 16;
        int32_t isum = 0, imin = 0;
        for (int c = 0; c < 4; c++) {                    // 4 trozos de 64 valores = 2 sub-bloques de 32
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(2 * c, scales, &sc0, &m0);
            get_scale_min_k4(2 * c + 1, scales, &sc1, &m1);
            const int8_t *q8 = y[i].qs + 64 * c;
            int32_t lo = 0, hi = 0;
            for (int l = 0; l < 32; l++) {
                lo += (q[32 * c + l] & 0x0F) * q8[l];        // nibbles bajos -> primeros 32 valores
                hi += (q[32 * c + l] >> 4) * q8[32 + l];     // nibbles altos -> siguientes 32
            }
            isum += sc0 * lo + sc1 * hi;
            imin += m0 * (y[i].bsums[4 * c] + y[i].bsums[4 * c + 1]) +
                    m1 * (y[i].bsums[4 * c + 2] + y[i].bsums[4 * c + 3]);
        }
        sumf += d * (float)isum - dmin * (float)imin;
    }
    return sumf;
}

// Producto punto de una fila Q6_K con x en Q8_K. Referencia escalar.
static float dot_q6_K_q8_K_ref(int64_t n, const uint8_t *w, const BlockQ8K *y) {
    float sumf = 0;
    for (int64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = w + i * 210;
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        const float d = rd_f16(b + 208) * y[i].d;
        int8_t a[256];                                   // q6 - 32, en el orden de los valores
        for (int h = 0; h < 2; h++) {
            const uint8_t *l_ = ql + 64 * h, *h_ = qh + 32 * h;
            for (int l = 0; l < 32; l++) {
                a[128 * h + l]      = (int8_t)(((l_[l] & 0x0F)      | (((h_[l] >> 0) & 3) << 4)) - 32);
                a[128 * h + l + 32] = (int8_t)(((l_[l + 32] & 0x0F) | (((h_[l] >> 2) & 3) << 4)) - 32);
                a[128 * h + l + 64] = (int8_t)(((l_[l] >> 4)        | (((h_[l] >> 4) & 3) << 4)) - 32);
                a[128 * h + l + 96] = (int8_t)(((l_[l + 32] >> 4)   | (((h_[l] >> 6) & 3) << 4)) - 32);
            }
        }
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {                   // 16 sub-bloques de 16 valores, una escala cada uno
            int32_t s = 0;
            for (int l = 0; l < 16; l++) s += a[16 * j + l] * y[i].qs[16 * j + l];
            isum += sc[j] * s;
        }
        sumf += d * (float)isum;
    }
    return sumf;
}

// ---------------------------------------------------------------- PARTE 5
// Productos punto AVX2 y matvec. Las funciones AVX2 llevan target("avx2,fma"), asi que se
// compilan siempre (sin -march=native) y se eligen en ejecucion segun la CPU.

#if defined(__x86_64__)
#define KORE_HAVE_AVX2 1
#define KORE_AVX2 __attribute__((target("avx2,fma")))

static bool cpu_has_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

KORE_AVX2 static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// Q4_K: nibbles (0..15, sin signo) x q8 (con signo) con _mm256_maddubs_epi16 -> sumas de pares en int16;
// _mm256_madd_epi16 con la escala del sub-bloque (int16) las pondera y las pasa a int32 en un solo paso.
// Las 8 escalas y 8 minimos de 6 bits se extraen de una vez con mascaras de 32 bits (como ggml);
// el termino de los minimos usa las sumas por grupo (bsums) y tambien va en SIMD.
KORE_AVX2 static inline float hsum128(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}

KORE_AVX2 static float dot_q4_K_q8_K_avx2(int64_t n, const uint8_t *w, const BlockQ8K *y) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    __m256 acc = _mm256_setzero_ps();
    __m128 accm = _mm_setzero_ps();
    for (int64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = w + i * 144;
        const float d = rd_f16(b) * y[i].d;
        const float dmin = rd_f16(b + 2) * y[i].d;

        // desempaquetar: bytes 0..7 = 8 escalas, bytes 8..15 = 8 minimos
        uint32_t u[4];
        memcpy(u, b + 4, 12);
        u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = u[1] & kmask1;
        u[1] = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
        u[2] = uaux;
        u[0] &= kmask1;
        const __m256i sm16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)u));   // [0..7] escalas, [8..15] minimos
        alignas(32) int16_t sm[16];
        _mm256_store_si256((__m256i *)sm, sm16);

        // minimos: sum_j min[j] * (bsums[2j] + bsums[2j+1])
        const __m256i bs = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(sm16, 1), q8s);     // 4 x int32
        accm = _mm_fmadd_ps(_mm_set1_ps(-dmin), _mm_cvtepi32_ps(prod), accm);

        const uint8_t *q4 = b + 16;
        const int8_t *q8 = y[i].qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int c = 0; c < 4; c++) {
            const __m256i bits = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i ql = _mm256_and_si256(bits, m4);
            const __m256i qh = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            const __m256i q8l = _mm256_loadu_si256((const __m256i *)q8);
            const __m256i q8h = _mm256_loadu_si256((const __m256i *)(q8 + 32));
            q8 += 64;
            const __m256i pl = _mm256_maddubs_epi16(ql, q8l);        // 16 x int16 (<= 3810)
            const __m256i ph = _mm256_maddubs_epi16(qh, q8h);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_set1_epi16(sm[2 * c]), pl));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_set1_epi16(sm[2 * c + 1]), ph));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256(acc) + hsum128(accm);
}

// Q6_K: se reconstruyen los 6 bits (0..63, sin signo) de 4 grupos de 32 valores por trozo de 128;
// el "-32" se aplica al final con las sumas por grupo: sum(sc*(q6-32)*q8) = sum(sc*q6*q8) - 32*sum(sc*bsums).
KORE_AVX2 static float dot_q6_K_q8_K_avx2(int64_t n, const uint8_t *w, const BlockQ8K *y) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m3 = _mm256_set1_epi8(0x03);
    __m256 acc = _mm256_setzero_ps();
    float summ = 0;
    for (int64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = w + i * 210;
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        const float d = rd_f16(b + 208) * y[i].d;
        int32_t ibias = 0;
        for (int j = 0; j < 16; j++) ibias += sc[j] * y[i].bsums[j];
        summ -= d * 32.0f * (float)ibias;

        const int8_t *q8 = y[i].qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {
            const __m256i qlA = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i qlB = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);
            ql += 64;
            qh += 32;
            __m256i g[4];
            g[0] = _mm256_or_si256(_mm256_and_si256(qlA, m4),
                       _mm256_slli_epi16(_mm256_and_si256(qhv, m3), 4));
            g[1] = _mm256_or_si256(_mm256_and_si256(qlB, m4),
                       _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m3), 4));
            g[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlA, 4), m4),
                       _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m3), 4));
            g[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlB, 4), m4),
                       _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m3), 4));
            for (int k = 0; k < 4; k++) {
                const __m256i q8v = _mm256_loadu_si256((const __m256i *)q8);
                q8 += 32;
                const __m256i p = _mm256_maddubs_epi16(g[k], q8v);      // 16 x int16
                // mitad baja = sub-bloque de 16 con escala sc[8h+2k]; mitad alta = el siguiente
                const __m256i sv = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_set1_epi16(sc[8 * h + 2 * k])),
                    _mm_set1_epi16(sc[8 * h + 2 * k + 1]), 1);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sv, p));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256(acc) + summ;
}
#else
#define KORE_HAVE_AVX2 0
static bool cpu_has_avx2() { return false; }
static float dot_q4_K_q8_K_avx2(int64_t n, const uint8_t *w, const BlockQ8K *y) { return dot_q4_K_q8_K_ref(n, w, y); }
static float dot_q6_K_q8_K_avx2(int64_t n, const uint8_t *w, const BlockQ8K *y) { return dot_q6_K_q8_K_ref(n, w, y); }
#endif

// out[r] = fila r del tensor . x, para r en [r0, r1). x ya viene cuantizado a Q8_K.
static bool matvec_rows(const Tensor &t, const BlockQ8K *xq, float *out, uint64_t r0, uint64_t r1, bool avx2) {
    if (!t.data || (t.type != GGML_Q4_K && t.type != GGML_Q6_K) || t.dims[0] % 256 != 0) return false;
    const int64_t n = (int64_t)t.dims[0];
    const uint64_t row_bytes = (uint64_t)(n / 256) * (t.type == GGML_Q4_K ? 144 : 210);
    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *w = t.data + r * row_bytes;
        if (t.type == GGML_Q4_K) out[r] = avx2 ? dot_q4_K_q8_K_avx2(n, w, xq) : dot_q4_K_q8_K_ref(n, w, xq);
        else                     out[r] = avx2 ? dot_q6_K_q8_K_avx2(n, w, xq) : dot_q6_K_q8_K_ref(n, w, xq);
    }
    return true;
}

// Reparte las filas en trozos contiguos, uno por hilo. Cada fila es independiente, asi que el
// resultado es identico al de 1 hilo. (Un motor mas afinado usaria un pool de hilos persistente.)
static void parallel_rows(uint64_t rows, unsigned nthreads, const std::function<void(uint64_t, uint64_t)> &fn) {
    if (nthreads <= 1 || rows < nthreads) { fn(0, rows); return; }
    std::vector<std::thread> th;
    const uint64_t chunk = (rows + nthreads - 1) / nthreads;
    for (unsigned k = 1; k < nthreads; k++) {                 // el hilo actual hace el trozo 0
        const uint64_t r0 = k * chunk, r1 = std::min<uint64_t>(rows, r0 + chunk);
        if (r0 >= r1) break;
        th.emplace_back([&fn, r0, r1] { fn(r0, r1); });
    }
    fn(0, std::min<uint64_t>(rows, chunk));
    for (std::thread &x : th) x.join();
}

static void matvec_mt(const Tensor &t, const BlockQ8K *xq, float *out, bool avx2, unsigned nthreads) {
    const uint64_t rows = t.dims[1] * t.dims[2] * t.dims[3];
    parallel_rows(rows, nthreads, [&](uint64_t r0, uint64_t r1) { matvec_rows(t, xq, out, r0, r1, avx2); });
}

// ---------------------------------------------------------------- PARTE 5b
// Matmul cuantizado batcheado para el prefill. Procesa B activaciones a la vez
// contra la MISMA fila de pesos: los bloques Q4_K/Q6_K se leen de la caché UNA
// vez en vez de una vez por token (el prefill token a token releia los ~4.7 GB de
// pesos por cada token; con B de un golpe los lee 1 vez).
// Salida con layout por columnas: out[r + b*rows] = fila r . activacion b.
#define KORE_BMAX 64

#if KORE_HAVE_AVX2
KORE_AVX2 static float hd_dot_avx2(const float *a, const float *b, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float hd_dot(const float *a, const float *b, int n, bool avx2) {
    if (avx2) return hd_dot_avx2(a, b, n);
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

KORE_AVX2 static void hd_fma_avx2(float *o, const float *v, float p, int n) {
    const __m256 pv = _mm256_set1_ps(p);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 ov = _mm256_loadu_ps(o + i);
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(pv, _mm256_loadu_ps(v + i), ov));
    }
    for (; i < n; i++) o[i] += p * v[i];
}

static void hd_fma(float *o, const float *v, float p, int n, bool avx2) {
    if (avx2) { hd_fma_avx2(o, v, p, n); return; }
    for (int i = 0; i < n; i++) o[i] += p * v[i];
}
#endif



// ---------------------------------------------------------------- PARTE 5c
// Matmul batcheado con activaciones EN FLOAT: producto "externo" (rank-1) por fila.
// El camino int Q4/Q8 (maddubs) relee la fila de pesos por cada columna del lote
// (~273 MB/s a B=64); aqui los pesos se descuantizan en bloques de 256 columnas y
// se acumulan con FMA por difusion sobre el lote. Las claves de cache:
//   - el bloque de pesos de una fila se lee del DRAM una sola vez;
//   - el bloque de activaciones transpuestas (256*B8 floats) queda en L2 y se
//     reutiliza en el barrido de todas las filas;
//   - cada hilo acumula su rango de filas en un slice contiguo (sin RMW disperso).
// x viene fila-mayor [B][n]; se transpone a columnas [n][B8] en xT. Salida:
// out[r + b*rows] = fila r . activacion b (col-major, igual que el resto).
#if KORE_HAVE_AVX2
// Acumula un bloque de C columnas sobre 2 filas a la vez: cada load de xT se
// comparte entre ambas filas (mitad de trafico de carga que la version de 1 fila).
KORE_AVX2 static void rank1_2row_acc_avx2(const float *wf0, const float *wf1, const float *xT,
                                          int C, int B, int B8, float *acc0, float *acc1) {
    const int ng = (B + 7) >> 3;
    __m256 a0[8], a1[8];
    for (int g = 0; g < ng; g++) {
        a0[g] = _mm256_loadu_ps(acc0 + 8 * g);
        a1[g] = _mm256_loadu_ps(acc1 + 8 * g);
    }
    for (int j = 0; j < C; j++) {
        const __m256 wb0 = _mm256_broadcast_ss(&wf0[(size_t)j]);
        const __m256 wb1 = _mm256_broadcast_ss(&wf1[(size_t)j]);
        const float *xr = xT + (size_t)j * B8;
        for (int g = 0; g < ng; g++) {
            const __m256 xv = _mm256_loadu_ps(xr + 8 * g);
            a0[g] = _mm256_fmadd_ps(wb0, xv, a0[g]);
            a1[g] = _mm256_fmadd_ps(wb1, xv, a1[g]);
        }
    }
    for (int g = 0; g < ng; g++) {
        _mm256_storeu_ps(acc0 + 8 * g, a0[g]);
        _mm256_storeu_ps(acc1 + 8 * g, a1[g]);
    }
}

KORE_AVX2 static void rank1_1row_acc_avx2(const float *wf0, const float *xT,
                                          int C, int B, int B8, float *acc0) {
    const int ng = (B + 7) >> 3;
    __m256 a0[8];
    for (int g = 0; g < ng; g++) a0[g] = _mm256_loadu_ps(acc0 + 8 * g);
    for (int j = 0; j < C; j++) {
        const __m256 wb0 = _mm256_broadcast_ss(&wf0[(size_t)j]);
        const float *xr = xT + (size_t)j * B8;
        for (int g = 0; g < ng; g++)
            a0[g] = _mm256_fmadd_ps(wb0, _mm256_loadu_ps(xr + 8 * g), a0[g]);
    }
    for (int g = 0; g < ng; g++) _mm256_storeu_ps(acc0 + 8 * g, a0[g]);
}
#endif

static void matmul_batch_float(const Tensor &t, const float *x, int B,
                               float *xT, float *out, bool avx2, unsigned nthreads) {
    const int n = (int)t.dims[0];
    const uint64_t rows = t.dims[1] * t.dims[2] * t.dims[3];
    const uint64_t rb = (uint64_t)(n / 256) * (t.type == GGML_Q4_K ? 144 : 210);
    const int B8 = (B + 7) & ~7;
    const int nblk = n / 256;
    const bool q4 = t.type == GGML_Q4_K;

    for (int j = 0; j < n; j++)
        for (int b = 0; b < B; b++)
            xT[(size_t)j * B8 + b] = x[(size_t)b * n + j];

    parallel_rows(rows, nthreads, [&](uint64_t r0, uint64_t r1) {
        std::vector<float> acc((size_t)(r1 - r0) * B8, 0.0f);
        float wf[512];
        for (int i = 0; i < nblk; i++) {                 // bloques de columnas (xT en L2)
            const float *xTb = xT + (size_t)i * 256 * B8;
            const uint64_t rp = (r1 - r0) & ~(uint64_t)1;
            uint64_t r = r0;
            for (; r < r0 + rp; r += 2) {                // pares de filas comparten los loads de xT
                float *ar0 = acc.data() + (size_t)(r - r0) * B8;
                float *ar1 = ar0 + B8;
                const uint8_t *w0 = t.data + r * rb, *w1 = t.data + (r + 1) * rb;
                if (q4) {
                    dequant_block_q4_K(w0 + i * 144, wf); dequant_block_q4_K(w1 + i * 144, wf + 256);
                } else {
                    dequant_block_q6_K(w0 + i * 210, wf); dequant_block_q6_K(w1 + i * 210, wf + 256);
                }
#if KORE_HAVE_AVX2
                if (avx2) { rank1_2row_acc_avx2(wf, wf + 256, xTb, 256, B, B8, ar0, ar1); continue; }
#endif
                for (int bb = 0; bb < B; bb++) {
                    double s0 = 0, s1 = 0;
                    for (int j = 0; j < 256; j++) {
                        const float xv = xTb[(size_t)j * B8 + bb];
                        s0 += (double)wf[j] * xv;
                        s1 += (double)wf[256 + j] * xv;
                    }
                    ar0[bb] += (float)s0;
                    ar1[bb] += (float)s1;
                }
            }
            if (r < r1) {                                // fila impar restante
                float *ar = acc.data() + (size_t)(r - r0) * B8;
                const uint8_t *w = t.data + r * rb;
                if (q4) dequant_block_q4_K(w + i * 144, wf);
                else    dequant_block_q6_K(w + i * 210, wf);
#if KORE_HAVE_AVX2
                if (avx2) { rank1_1row_acc_avx2(wf, xTb, 256, B, B8, ar); continue; }
#endif
                for (int bb = 0; bb < B; bb++) {
                    double s = 0;
                    for (int j = 0; j < 256; j++) s += (double)wf[j] * xTb[(size_t)j * B8 + bb];
                    ar[bb] += (float)s;
                }
            }
        }
        for (uint64_t r = r0; r < r1; r++)
            for (int bb = 0; bb < B; bb++)
                out[(size_t)bb * rows + r] = acc[(size_t)(r - r0) * B8 + bb];
    });
}

// Version exacta (sin cuantizar activaciones): descuantiza cada fila a float y multiplica en double.
// Muy lenta; sirve para separar errores de logica de errores de cuantizacion.
static void matvec_float(const Tensor &t, const float *x, float *out, unsigned nthreads) {
    const uint64_t rows = t.dims[1] * t.dims[2] * t.dims[3];
    const size_t n = (size_t)t.dims[0];
    parallel_rows(rows, nthreads, [&](uint64_t r0, uint64_t r1) {
        std::vector<float> row;
        for (uint64_t r = r0; r < r1; r++) {
            get_row(t, r, row);
            double s = 0;
            for (size_t i = 0; i < n; i++) s += (double)row[i] * x[i];
            out[r] = (float)s;
        }
    });
}


// ================================================================ TOKENIZER (paso 2)
// Usa el lector GGUF de las partes 1-5.

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

// ---------------------------------------------------------------- TOK 2
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

// ---------------------------------------------------------------- TOK 3
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

// ---------------------------------------------------------------- TOK 3b
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


// ---------------------------------------------------------------- PARTE 6
// Forward de Qwen2 para UN token con cache KV (prefill = token a token; sin batch todavia).
// Por capa:  x += Wo * Atencion(RoPE(Wq*n(x)+bq), RoPE(Wk*n(x)+bk), Wv*n(x)+bv)
//            x += Wdown * ( silu(Wgate*n(x)) * (Wup*n(x)) )       con n() = RMSNorm.
// Al final:  logits = Wout * RMSNorm(x). GQA: la cabeza q h usa la cabeza kv h / (n_head / n_head_kv).
// RoPE estilo NEOX: rota los pares (i, i + head_dim/2).

struct Layer {
    const Tensor *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;
    const float *attn_norm, *ffn_norm, *bq, *bk, *bv;
};

struct Engine {
    Hparams hp;
    int head_dim = 0, kv_dim = 0, n_rep = 0, ctx = 0, n_vocab = 0;
    unsigned nth = 1;
    bool avx2 = false, use_float = false;
    const Tensor *tok_embd = nullptr, *out_w = nullptr;
    const float *out_norm = nullptr;
    std::vector<Layer> layers;
    std::vector<float> inv_freq, rope_c, rope_s;
    std::vector<float> kcache, vcache;                 // [capa][pos][kv_dim]
    std::vector<float> emb, x, xb, q, k, v, att, o, gate, up, scores, logits;
    std::vector<BlockQ8K> xq_e, xq_f;                  // activaciones cuantizadas: dim E y dim FF

    // Buffers del prefill batcheado (PARTE 5b): [bsz][dim] por capa.
    int bsz = KORE_BMAX;
    std::vector<float> scratch;                        // fila temporal (token_embd -> bemb)
    std::vector<float> bemb, bx, bxb, bq, batt, bo;    // [bsz][E]
    std::vector<float> bk, bv;                         // [bsz][kv_dim]
    std::vector<float> bgate, bup;                     // [bsz][FF]
    std::vector<float> bscores;                        // [bsz][ctx]
    std::vector<float> bc_rope, bs_rope;               // cos/sen por posicion del lote
    std::vector<float> bxt;                            // activaciones transpuestas [n][B8] de la capa
};

static bool engine_init(Engine &e, const Model &m, int ctx_max, unsigned nth, bool use_float) {
    e.hp = m.hp;
    const Hparams &h = e.hp;
    if (h.n_layer == 0 || h.n_head == 0 || h.n_head_kv == 0 || h.n_embd % h.n_head != 0 ||
        h.n_head % h.n_head_kv != 0 || h.rope_base <= 0) {
        fprintf(stderr, "error: hiperparametros qwen2.* ausentes o invalidos\n");
        return false;
    }
    if (h.n_embd % 256 != 0 || h.n_ff % 256 != 0) {
        fprintf(stderr, "error: embd y ffn deben ser multiplos de 256\n");
        return false;
    }
    e.head_dim = (int)(h.n_embd / h.n_head);
    if (e.head_dim % 2 != 0) { fprintf(stderr, "error: head_dim impar\n"); return false; }
    e.kv_dim = (int)h.n_head_kv * e.head_dim;
    e.n_rep = (int)(h.n_head / h.n_head_kv);
    e.ctx = h.n_ctx ? (int)std::min<uint32_t>(h.n_ctx, (uint32_t)ctx_max) : ctx_max;
    e.nth = std::max(1u, nth);
    e.avx2 = KORE_HAVE_AVX2 && cpu_has_avx2();
    e.use_float = use_float;

    bool ok = true;
    auto get = [&](const std::string &n) -> const Tensor * {
        auto it = m.tensors.find(n);
        if (it == m.tensors.end() || !it->second.data) { fprintf(stderr, "error: falta el tensor '%s'\n", n.c_str()); ok = false; return nullptr; }
        return &it->second;
    };
    auto wt = [&](const std::string &n) -> const Tensor * {       // pesos de matvec: Q4_K o Q6_K
        const Tensor *t = get(n);
        if (t && t->type != GGML_Q4_K && t->type != GGML_Q6_K) {
            fprintf(stderr, "error: '%s' no es Q4_K/Q6_K (solo esos tipos estan soportados)\n", n.c_str());
            ok = false;
        }
        return t;
    };
    auto ft = [&](const std::string &n) -> const float * {        // normas y sesgos: F32
        const Tensor *t = get(n);
        if (t && t->type != GGML_F32) { fprintf(stderr, "error: '%s' no es F32\n", n.c_str()); ok = false; return nullptr; }
        return t ? (const float *)t->data : nullptr;
    };

    e.tok_embd = get("token_embd.weight");
    e.out_w = wt("output.weight");
    e.out_norm = ft("output_norm.weight");
    for (uint32_t l = 0; l < h.n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer L;
        L.wq = wt(p + "attn_q.weight");      L.bq = ft(p + "attn_q.bias");
        L.wk = wt(p + "attn_k.weight");      L.bk = ft(p + "attn_k.bias");
        L.wv = wt(p + "attn_v.weight");      L.bv = ft(p + "attn_v.bias");
        L.wo = wt(p + "attn_output.weight");
        L.attn_norm = ft(p + "attn_norm.weight");
        L.ffn_norm = ft(p + "ffn_norm.weight");
        L.w_gate = wt(p + "ffn_gate.weight");
        L.w_up = wt(p + "ffn_up.weight");
        L.w_down = wt(p + "ffn_down.weight");
        e.layers.push_back(L);
    }
    if (!ok) return false;
    e.n_vocab = (int)e.out_w->dims[1];
    if ((int)e.tok_embd->dims[1] != e.n_vocab) { fprintf(stderr, "error: token_embd y output tienen distinto vocabulario\n"); return false; }

    const int half = e.head_dim / 2;
    e.inv_freq.resize((size_t)half);
    for (int i = 0; i < half; i++) e.inv_freq[(size_t)i] = (float)std::pow((double)h.rope_base, -2.0 * i / e.head_dim);
    e.rope_c.resize((size_t)half);
    e.rope_s.resize((size_t)half);

    const size_t E = h.n_embd, FF = h.n_ff;
    e.kcache.assign((size_t)h.n_layer * e.ctx * e.kv_dim, 0.0f);
    e.vcache.assign((size_t)h.n_layer * e.ctx * e.kv_dim, 0.0f);
    e.emb.resize(E); e.x.resize(E); e.xb.resize(E); e.q.resize(E); e.att.resize(E); e.o.resize(E);
    e.k.resize((size_t)e.kv_dim); e.v.resize((size_t)e.kv_dim);
    e.gate.resize(FF); e.up.resize(FF);
    e.scores.resize((size_t)e.ctx);
    e.logits.resize((size_t)e.n_vocab);
    e.xq_e.resize(E / 256);
    e.xq_f.resize(FF / 256);

    e.bsz = std::min(KORE_BMAX, e.ctx);
    const size_t bsz = (size_t)e.bsz;
    e.scratch.resize(E);
    e.bemb.resize(bsz * E); e.bx.resize(bsz * E); e.bxb.resize(bsz * E);
    e.bq.resize(bsz * E); e.batt.resize(bsz * E); e.bo.resize(bsz * E);
    e.bk.resize(bsz * (size_t)e.kv_dim); e.bv.resize(bsz * (size_t)e.kv_dim);
    e.bgate.resize(bsz * FF); e.bup.resize(bsz * FF);
    e.bscores.resize(bsz * (size_t)e.ctx);
    e.bc_rope.resize(bsz * (size_t)half); e.bs_rope.resize(bsz * (size_t)half);
    e.bxt.resize(((bsz + 7) & ~7) * FF);
    return true;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void rope_neox(float *vec, int n_heads, int hd, const float *c, const float *s) {
    const int half = hd / 2;
    for (int h = 0; h < n_heads; h++) {
        float *p = vec + (size_t)h * hd;
        for (int i = 0; i < half; i++) {
            const float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * c[i] - x1 * s[i];
            p[i + half] = x0 * s[i] + x1 * c[i];
        }
    }
}

// out = W * x  (x en float; xq = x ya cuantizado a Q8_K)
static void mv(Engine &e, const Tensor &t, const float *x, const BlockQ8K *xq, float *out) {
    if (e.use_float) matvec_float(t, x, out, e.nth);
    else             matvec_mt(t, xq, out, e.avx2, e.nth);
}

static void quant(const Engine &e, const float *x, BlockQ8K *xq, int n) {
    if (!e.use_float) quantize_row_q8_K(x, xq, n);
}

// Procesa `token` en la posicion `pos`. Si want_logits, deja los logits en e.logits.
static bool forward(Engine &e, int token, int pos, bool want_logits) {
    const Hparams &h = e.hp;
    const int E = (int)h.n_embd, FF = (int)h.n_ff, hd = e.head_dim, kvd = e.kv_dim;
    if (token < 0 || token >= e.n_vocab || pos < 0 || pos >= e.ctx) return false;

    if (!get_row(*e.tok_embd, (uint64_t)token, e.emb)) return false;
    memcpy(e.x.data(), e.emb.data(), (size_t)E * sizeof(float));

    const int half = hd / 2;
    for (int i = 0; i < half; i++) {
        const double a = (double)pos * e.inv_freq[(size_t)i];
        e.rope_c[(size_t)i] = (float)std::cos(a);
        e.rope_s[(size_t)i] = (float)std::sin(a);
    }
    const float inv_sqrt = 1.0f / std::sqrt((float)hd);

    for (uint32_t l = 0; l < h.n_layer; l++) {
        const Layer &L = e.layers[l];

        // --- atencion
        rmsnorm(e.xb.data(), e.x.data(), L.attn_norm, E, h.rms_eps);
        quant(e, e.xb.data(), e.xq_e.data(), E);
        mv(e, *L.wq, e.xb.data(), e.xq_e.data(), e.q.data());
        mv(e, *L.wk, e.xb.data(), e.xq_e.data(), e.k.data());
        mv(e, *L.wv, e.xb.data(), e.xq_e.data(), e.v.data());
        for (int i = 0; i < E; i++) e.q[(size_t)i] += L.bq[i];
        for (int i = 0; i < kvd; i++) { e.k[(size_t)i] += L.bk[i]; e.v[(size_t)i] += L.bv[i]; }
        rope_neox(e.q.data(), (int)h.n_head, hd, e.rope_c.data(), e.rope_s.data());
        rope_neox(e.k.data(), (int)h.n_head_kv, hd, e.rope_c.data(), e.rope_s.data());

        float *kc = e.kcache.data() + ((size_t)l * e.ctx) * kvd;     // cache de esta capa: [pos][kv_dim]
        float *vc = e.vcache.data() + ((size_t)l * e.ctx) * kvd;
        memcpy(kc + (size_t)pos * kvd, e.k.data(), (size_t)kvd * sizeof(float));
        memcpy(vc + (size_t)pos * kvd, e.v.data(), (size_t)kvd * sizeof(float));

        for (int hh = 0; hh < (int)h.n_head; hh++) {
            const float *qh = e.q.data() + (size_t)hh * hd;
            const int kvh = hh / e.n_rep;
            float *sc = e.scores.data();
            float mx = -INFINITY;
            for (int t = 0; t <= pos; t++) {
                const float *kt = kc + (size_t)t * kvd + (size_t)kvh * hd;
                float s = 0;
                for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
                s *= inv_sqrt;
                sc[t] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            const float inv = 1.0f / sum;
            float *oh = e.att.data() + (size_t)hh * hd;
            for (int i = 0; i < hd; i++) oh[i] = 0;
            for (int t = 0; t <= pos; t++) {
                const float p = sc[t] * inv;
                const float *vt = vc + (size_t)t * kvd + (size_t)kvh * hd;
                for (int i = 0; i < hd; i++) oh[i] += p * vt[i];
            }
        }
        quant(e, e.att.data(), e.xq_e.data(), E);
        mv(e, *L.wo, e.att.data(), e.xq_e.data(), e.o.data());
        for (int i = 0; i < E; i++) e.x[(size_t)i] += e.o[(size_t)i];

        // --- feed-forward SwiGLU
        rmsnorm(e.xb.data(), e.x.data(), L.ffn_norm, E, h.rms_eps);
        quant(e, e.xb.data(), e.xq_e.data(), E);
        mv(e, *L.w_gate, e.xb.data(), e.xq_e.data(), e.gate.data());
        mv(e, *L.w_up, e.xb.data(), e.xq_e.data(), e.up.data());
        for (int i = 0; i < FF; i++) {
            const float g = e.gate[(size_t)i];
            e.gate[(size_t)i] = g / (1.0f + std::exp(-g)) * e.up[(size_t)i];
        }
        quant(e, e.gate.data(), e.xq_f.data(), FF);
        mv(e, *L.w_down, e.gate.data(), e.xq_f.data(), e.o.data());
        for (int i = 0; i < E; i++) e.x[(size_t)i] += e.o[(size_t)i];
    }

    if (want_logits) {
        rmsnorm(e.xb.data(), e.x.data(), e.out_norm, E, h.rms_eps);
        quant(e, e.xb.data(), e.xq_e.data(), E);
        mv(e, *e.out_w, e.xb.data(), e.xq_e.data(), e.logits.data());
    }
    return true;
}

// Prefill batcheado (PARTE 5b): procesa `nb` tokens contiguos en las posiciones
// [pos0, pos0+nb) a la vez. La atencion es causal: cada token del lote solo mira
// a las posiciones anteriores mas las suyas propias. Deja los logits en e.logits
// solo si want_logits (debe ser entonces el ultimo bloque del prompt).
static bool forward_batch(Engine &e, const int *toks, int nb, int pos0, bool want_logits) {
    const Hparams &h = e.hp;
    const int E = (int)h.n_embd, FF = (int)h.n_ff, hd = e.head_dim, kvd = e.kv_dim;
    const int half = hd / 2;
    if (nb <= 0 || nb > e.bsz || pos0 < 0 || pos0 + nb > e.ctx) return false;
    if (e.use_float) {                       // modo exacto: token a token (referencia)
        for (int z = 0; z < nb; z++)
            if (!forward(e, toks[z], pos0 + z, want_logits && z + 1 == nb)) return false;
        return true;
    }
    for (int z = 0; z < nb; z++) {
        if (toks[z] < 0 || toks[z] >= e.n_vocab) return false;
        if (!get_row(*e.tok_embd, (uint64_t)toks[z], e.scratch)) return false;
        memcpy(e.bemb.data() + (size_t)z * E, e.scratch.data(), (size_t)E * sizeof(float));
        memcpy(e.bx.data() + (size_t)z * E, e.scratch.data(), (size_t)E * sizeof(float));
    }

    for (int z = 0; z < nb; z++) {           // cos/seno de rotacion por posicion del lote
        const double ph = (double)(pos0 + z);
        float *cc = e.bc_rope.data() + (size_t)z * half;
        float *ss = e.bs_rope.data() + (size_t)z * half;
        for (int i = 0; i < half; i++) {
            const double a = ph * e.inv_freq[(size_t)i];
            cc[i] = (float)std::cos(a);
            ss[i] = (float)std::sin(a);
        }
    }
    const float inv_sqrt = 1.0f / std::sqrt((float)hd);

    float *const xt = e.bxt.data();
    for (uint32_t l = 0; l < h.n_layer; l++) {
        const Layer &L = e.layers[l];
        float *kc = e.kcache.data() + ((size_t)l * e.ctx) * kvd;     // [pos][kv_dim]
        float *vc = e.vcache.data() + ((size_t)l * e.ctx) * kvd;

        // --- atencion: norm + QKV en lote (matmul rank-1 en float, PARTE 5c)
        for (int z = 0; z < nb; z++)
            rmsnorm(e.bxb.data() + (size_t)z * E, e.bx.data() + (size_t)z * E, L.attn_norm, E, h.rms_eps);
        matmul_batch_float(*L.wq, e.bxb.data(), nb, xt, e.bq.data(), e.avx2, e.nth);
        matmul_batch_float(*L.wk, e.bxb.data(), nb, xt, e.bk.data(), e.avx2, e.nth);
        matmul_batch_float(*L.wv, e.bxb.data(), nb, xt, e.bv.data(), e.avx2, e.nth);
        for (int z = 0; z < nb; z++) {
            float *q = e.bq.data() + (size_t)z * E;
            float *k = e.bk.data() + (size_t)z * kvd;
            float *v = e.bv.data() + (size_t)z * kvd;
            const float *bc = e.bc_rope.data() + (size_t)z * half;
            const float *bs = e.bs_rope.data() + (size_t)z * half;
            for (int i = 0; i < E; i++) q[i] += L.bq[i];
            for (int i = 0; i < kvd; i++) { k[i] += L.bk[i]; v[i] += L.bv[i]; }
            rope_neox(q, (int)h.n_head, hd, bc, bs);
            rope_neox(k, (int)h.n_head_kv, hd, bc, bs);
            memcpy(kc + (size_t)(pos0 + z) * kvd, k, (size_t)kvd * sizeof(float));
            memcpy(vc + (size_t)(pos0 + z) * kvd, v, (size_t)kvd * sizeof(float));
        }

        // --- atencion batcheada, con mascara causal dentro del lote
        for (int hh = 0; hh < (int)h.n_head; hh++) {
            const int kvh = hh / e.n_rep;
            for (int z = 0; z < nb; z++) {
                const int lim = pos0 + z;
                const float *qh = e.bq.data() + (size_t)z * E + (size_t)hh * hd;
                float *sc = e.bscores.data() + (size_t)z * e.ctx;
                float mx = -INFINITY;
                for (int t = 0; t <= lim; t++) {
                    const float *kt = kc + (size_t)t * kvd + (size_t)kvh * hd;
                    const float s = hd_dot(qh, kt, hd, e.avx2) * inv_sqrt;
                    sc[t] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0;
                for (int t = 0; t <= lim; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                const float inv = 1.0f / sum;
                float *oh = e.batt.data() + (size_t)z * E + (size_t)hh * hd;
                memset(oh, 0, (size_t)hd * sizeof(float));
                for (int t = 0; t <= lim; t++)
                    hd_fma(oh, vc + (size_t)t * kvd + (size_t)kvh * hd, sc[t] * inv, hd, e.avx2);
            }
        }

        // --- wo: atencion -> residuo
        matmul_batch_float(*L.wo, e.batt.data(), nb, xt, e.bo.data(), e.avx2, e.nth);
        for (int z = 0; z < nb; z++) {
            float *x = e.bx.data() + (size_t)z * E, *o = e.bo.data() + (size_t)z * E;
            for (int i = 0; i < E; i++) x[i] += o[i];
        }

        // --- feed-forward SwiGLU en lote
        for (int z = 0; z < nb; z++)
            rmsnorm(e.bxb.data() + (size_t)z * E, e.bx.data() + (size_t)z * E, L.ffn_norm, E, h.rms_eps);
        matmul_batch_float(*L.w_gate, e.bxb.data(), nb, xt, e.bgate.data(), e.avx2, e.nth);
        matmul_batch_float(*L.w_up, e.bxb.data(), nb, xt, e.bup.data(), e.avx2, e.nth);
        for (int z = 0; z < nb; z++) {
            float *g = e.bgate.data() + (size_t)z * FF;
            const float *u = e.bup.data() + (size_t)z * FF;
            for (int i = 0; i < FF; i++) {
                const float gg = g[i];
                g[i] = gg / (1.0f + std::exp(-gg)) * u[i];
            }
        }
        matmul_batch_float(*L.w_down, e.bgate.data(), nb, xt, e.bo.data(), e.avx2, e.nth);
        for (int z = 0; z < nb; z++) {
            float *x = e.bx.data() + (size_t)z * E, *o = e.bo.data() + (size_t)z * E;
            for (int i = 0; i < E; i++) x[i] += o[i];
        }
    }

    if (want_logits) {                       // solo el ultimo token del lote
        rmsnorm(e.xb.data(), e.bx.data() + (size_t)(nb - 1) * E, e.out_norm, E, h.rms_eps);
        quant(e, e.xb.data(), e.xq_e.data(), E);
        mv(e, *e.out_w, e.xb.data(), e.xq_e.data(), e.logits.data());
    }
    return true;
}

// Elige el siguiente token: greedy si temp <= 0; si no, muestreo con temperatura sobre los top_k mejores.
static int sample_token(const std::vector<float> &logits, float temp, int top_k, std::mt19937 &rng) {
    const int n = (int)logits.size();
    if (temp <= 0) return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; i++) idx[(size_t)i] = i;
    const int k = std::min(top_k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
    std::vector<double> p((size_t)k);
    const double mx = logits[(size_t)idx[0]];
    double sum = 0;
    for (int i = 0; i < k; i++) { p[(size_t)i] = std::exp((logits[(size_t)idx[(size_t)i]] - mx) / temp); sum += p[(size_t)i]; }
    double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
    for (int i = 0; i < k; i++) { r -= p[(size_t)i]; if (r <= 0) return idx[(size_t)i]; }
    return idx[(size_t)k - 1];
}

// ---------------------------------------------------------------- PARTE 8
// Iteracion 2: servidor IPC sobre Unix Domain Socket. Tramas [tipo][len LE32][payload].
//   Cliente -> servidor:  'P' PROMPT  payload = ['C' ChatML | 'R' crudo] + texto
//                         'T' TOKEN   sin payload (saca el siguiente token)
//                         'R' RESET   payload opcional = nuevo system message
//                         'A' ABORT   sin payload (cancela la generacion)
//                         'I' INJECT  payload = texto: lo inserta como turno de
//                                       usuario (sin system) y sigue generando
//   Servidor -> cliente:  'M' META informacion del modelo
//                         'P'/'I' acuse de que el prompt fue aceptado
//                         'T' TOKEN  payload = [id u32 LE] + bytes del fragmento
//                         'E' fin de generacion  payload = estado (0 fin, 1 ctx lleno, 2 sin gen)
//                         'R'/'A' acuses  |  'X' ERROR  payload = mensaje
// La sesion (caché KV + chat) persiste entre conexiones: el cliente puede
// reconectarse y seguir tirando del turno en curso (modelo push asincrono).

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
    return len==0 || sock_write_all(fd, data, len);
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

// Estado de una conversacion: caché KV viva + historial de tokens + RNG.
struct ChatSession {
    Engine e;
    Vocab v;
    std::vector<int32_t> conv;          // todos los tokens ya volcados en la caché KV
    std::mt19937 rng;
    float temp = 0.0f;
    std::string system_msg = "You are a helpful assistant.";
    bool generating = false;
};

static void build_chat_message(const Vocab &v, const std::string &system_msg,
                               const std::string &user_text, std::vector<int32_t> &ids,
                               bool with_system) {
    auto special = [&](const char *s) -> int32_t {
        auto it = v.tok2id.find(s);
        return it == v.tok2id.end() ? -1 : it->second;
    };
    const int32_t im_start = special("<|im_start|>"), im_end = special("<|im_end|>");
    auto text = [&](const std::string &s) {
        std::vector<int32_t> t = encode(v, s, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    if (im_start >= 0 && im_end >= 0) {
        if (with_system) { ids.push_back(im_start); text("system\n" + system_msg); ids.push_back(im_end); text("\n"); }
        ids.push_back(im_start); text("user\n" + user_text);    ids.push_back(im_end); text("\n");
        ids.push_back(im_start); text("assistant\n");
    } else {
        text(user_text);
    }
}

static void chat_ids(const Vocab &v, const std::string &system_msg,
                     const std::string &user_text, std::vector<int32_t> &ids) {
    build_chat_message(v, system_msg, user_text, ids, true);
}

static void chat_ids_user_only(const Vocab &v, const std::string &user_text, std::vector<int32_t> &ids) {
    build_chat_message(v, "", user_text, ids, false);
}

// Vuelca los ids en la caché KV APENDIENDO a la conversacion actual (la KV
// persiste entre turnos) y deja la sesion generando.
static bool prefill(ChatSession &s, const std::vector<int32_t> &ids, std::string *err) {
    const int base = (int)s.conv.size();
    if (ids.empty() || base + (int)ids.size() >= s.e.ctx) {
        if (err) *err = "prompt demasiado largo para el contexto";
        return false;
    }
    const int B = s.e.bsz;
    for (size_t i = 0; i < ids.size(); i += (size_t)B) {
        const int nb = (int)std::min<size_t>((size_t)B, ids.size() - i);
        const bool last = i + (size_t)nb == ids.size();
        if (!forward_batch(s.e, (const int *)(ids.data() + i), nb, base + (int)i, last)) {
            if (err) *err = "error en el forward del prompt";
            return false;
        }
    }
    s.conv.insert(s.conv.end(), ids.begin(), ids.end());
    s.generating = true;
    return true;
}

// Un paso de generacion: samplea de los logits actuales, devuelve el token y
// deja los logits del siguiente. Devuelve 1 si hay token, 0 si termino, -1 si fallo.
static int gen_step(ChatSession &s, int *token, std::string *piece, bool *ctx_full) {
    *ctx_full = false;
    if (!s.generating) return 0;
    *token = sample_token(s.e.logits, s.temp, 40, s.rng);
    if (*token == s.v.eos || *token == s.v.bos) { s.generating = false; return 0; }
    if ((int)s.conv.size() >= s.e.ctx) { s.generating = false; *ctx_full = true; return 0; }
    *piece = decode(s.v, {*token});
    const int pos = (int)s.conv.size();
    s.conv.push_back(*token);
    if (!forward(s.e, *token, pos, true)) return -1;
    return 1;
}

static void serve(const char *sock_path, ChatSession &s) {
    unlink(sock_path);
    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0) { perror("socket"); return; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); close(lfd); return; }
    if (listen(lfd, 4) < 0) { perror("listen"); close(lfd); return; }
    fprintf(stderr, "KORE-Core: servidor IPC escuchando en %s (ctx %d, temp %.2f)\n",
            sock_path, s.e.ctx, (double)s.temp);

    char meta[160];
    snprintf(meta, sizeof(meta), "kore pid=%d ctx=%d temp=%.2f", (int)getpid(), s.e.ctx, (double)s.temp);

    for (;;) {
        int cfd = accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept"); break;
        }
        send_frame(cfd, 'M', meta, (uint32_t)strlen(meta));
        for (;;) {
            char type = 0;
            std::vector<uint8_t> pay;
            if (!recv_frame(cfd, &type, pay)) break;
            if (type == 'P' && !pay.empty()) {
                const bool chat = pay[0] == 'C';
                std::string text((const char *)pay.data() + 1, pay.size() - 1);
                std::vector<int32_t> ids;
                if (chat) chat_ids(s.v, s.system_msg, text, ids);
                else      ids = encode(s.v, text, true);
                std::string err;
                if (prefill(s, ids, &err)) send_frame(cfd, 'P', nullptr, 0);
                else {
                    s.generating = false;
                    send_frame(cfd, 'X', err.data(), (uint32_t)err.size());
                }
            } else if (type == 'T') {
                int tok = 0; std::string piece; bool full = false;
                const int r = gen_step(s, &tok, &piece, &full);
                if (r == 1) {
                    uint8_t hdr[4] = {(uint8_t)(tok & 0xFF), (uint8_t)((tok >> 8) & 0xFF),
                                      (uint8_t)((tok >> 16) & 0xFF), (uint8_t)((tok >> 24) & 0xFF)};
                    std::vector<uint8_t> frame(hdr, hdr + 4);
                    frame.insert(frame.end(), piece.begin(), piece.end());
                    send_frame(cfd, 'T', frame.data(), (uint32_t)frame.size());
                } else if (r == 0) {
                    const char st = full ? (char)1 : (char)(s.generating ? 2 : 0);
                    send_frame(cfd, 'E', &st, 1);
                } else {
                    const std::string m = "error en el forward";
                    send_frame(cfd, 'X', m.data(), (uint32_t)m.size());
                }
            } else if (type == 'R') {
                s.conv.clear();
                s.generating = false;
                if (!pay.empty()) s.system_msg.assign((const char *)pay.data(), pay.size());
                send_frame(cfd, 'R', nullptr, 0);
            } else if (type == 'A') {
                s.generating = false;
                send_frame(cfd, 'A', nullptr, 0);
            } else if (type == 'I' && !pay.empty()) {
                const std::string text((const char *)pay.data(), pay.size());
                std::vector<int32_t> ids;
                chat_ids_user_only(s.v, text, ids);
                std::string err;
                if (prefill(s, ids, &err)) send_frame(cfd, 'I', nullptr, 0);
                else { s.generating = false; send_frame(cfd, 'X', err.data(), (uint32_t)err.size()); }
            } else {
                const std::string m = "comando desconocido";
                send_frame(cfd, 'X', m.data(), (uint32_t)m.size());
            }
        }
        close(cfd);
        fprintf(stderr, "KORE-Core: cliente desconectado\n");
    }
    close(lfd);
    unlink(sock_path);
}

// ---------------------------------------------------------------- PARTE 7
// Programa principal: chat ChatML en streaming, o modo depuracion con ids.

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void usage(const char *prog) {
    fprintf(stderr,
        "uso: %s modelo.gguf \"pregunta\" [opciones]\n"
        "     %s modelo.gguf --ids 1,2,3 [--dump logits.f32]\n"
        "     %s modelo.gguf --serve /tmp/kore.sock [opciones]   (servidor IPC, iteracion 2)\n"
        "opciones: -n N  --ctx N  --threads N  --temp T  --seed N  --system \"...\"\n"
        "          --raw  --float  --serve SOCKET\n", prog, prog, prog);
}

int main(int argc, char **argv) {
    const char *path = nullptr;
    std::string prompt, system_msg = "You are a helpful assistant.", ids_arg, dump_path, serve_path;
    bool have_prompt = false, raw = false, use_float = false;
    int n_predict = 256, ctx = 2048;
    unsigned nth = std::max(1u, std::thread::hardware_concurrency());
    float temp = 0.0f;
    uint32_t seed = (uint32_t)std::chrono::system_clock::now().time_since_epoch().count();

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        const bool has_val = i + 1 < argc;
        if (a == "-n" && has_val)              n_predict = atoi(argv[++i]);
        else if (a == "--ctx" && has_val)      ctx = atoi(argv[++i]);
        else if (a == "--threads" && has_val)  nth = (unsigned)std::max(1, atoi(argv[++i]));
        else if (a == "--temp" && has_val)     temp = (float)atof(argv[++i]);
        else if (a == "--seed" && has_val)     seed = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (a == "--system" && has_val)   system_msg = argv[++i];
        else if (a == "--ids" && has_val)      ids_arg = argv[++i];
        else if (a == "--dump" && has_val)     dump_path = argv[++i];
        else if (a == "--serve" && has_val)    serve_path = argv[++i];
        else if (a == "--raw")                 raw = true;
        else if (a == "--float")               use_float = true;
        else if (!path && a[0] != '-')         path = argv[i];
        else if (!have_prompt && a[0] != '-')  { prompt = a; have_prompt = true; }
        else { fprintf(stderr, "argumento no reconocido: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (!path || ctx < 8 || n_predict < 0 ||
        (serve_path.empty() && !have_prompt && ids_arg.empty())) { usage(argv[0]); return 1; }

    Model m;
    if (!load_model(path, m)) return 1;
    Engine e;
    if (!engine_init(e, m, ctx, nth, use_float)) return 1;
    fprintf(stderr, "modelo: %u capas, embd %u, %d hilos, contexto %d, AVX2 %s%s\n", m.hp.n_layer, m.hp.n_embd,
            (int)e.nth, e.ctx, e.avx2 ? "si" : "no (escalar)", e.use_float ? ", MODO FLOAT EXACTO" : "");

    init_byte_tables();
    Vocab v;
    if (!load_vocab(path, v)) return 1;
    if ((int)v.tokens.size() != e.n_vocab)
        fprintf(stderr, "aviso: el tokenizer tiene %zu tokens y el modelo %d (normal en Qwen: el modelo trae relleno)\n",
                v.tokens.size(), e.n_vocab);

    // ---- modo servidor IPC (iteracion 2)
    if (!serve_path.empty()) {
        ChatSession s;
        s.e = std::move(e);
        s.v = std::move(v);
        s.temp = temp;
        s.rng = std::mt19937(seed);
        s.system_msg = system_msg;
        serve(serve_path.c_str(), s);
        return 0;
    }

    // ---- modo depuracion: ids -> logits del ultimo token
    if (!ids_arg.empty()) {
        std::vector<int> ids;
        for (char *tok = strtok(&ids_arg[0], ","); tok; tok = strtok(nullptr, ",")) ids.push_back(atoi(tok));
        if (ids.empty() || (int)ids.size() > e.ctx) { fprintf(stderr, "error: --ids vacio o mas largo que el contexto\n"); return 1; }
        for (size_t i = 0; i < ids.size(); i += (size_t)e.bsz) {
            const int nb = (int)std::min<size_t>((size_t)e.bsz, ids.size() - i);
            if (!forward_batch(e, (const int *)(ids.data() + i), nb, (int)i, i + (size_t)nb == ids.size()))
                { fprintf(stderr, "error: token invalido\n"); return 1; }
        }
        std::vector<int> order((size_t)e.n_vocab);
        for (int i = 0; i < e.n_vocab; i++) order[(size_t)i] = i;
        const int top = std::min(5, e.n_vocab);
        std::partial_sort(order.begin(), order.begin() + top, order.end(),
                          [&](int a, int b) { return e.logits[(size_t)a] > e.logits[(size_t)b]; });
        for (int i = 0; i < top; i++) printf("top%d: id=%d logit=%.6f\n", i + 1, order[(size_t)i], (double)e.logits[(size_t)order[(size_t)i]]);
        if (!dump_path.empty()) {
            FILE *f = fopen(dump_path.c_str(), "wb");
            if (!f) { perror("fopen"); return 1; }
            fwrite(e.logits.data(), 4, e.logits.size(), f);
            fclose(f);
        }
        return 0;
    }

    // ---- tokenizacion del prompt
    std::vector<int32_t> ids;
    if (raw) {
        ids = encode(v, prompt, true);
    } else {                                  // ChatML; el texto del usuario NO interpreta tokens especiales
        auto special = [&](const char *s) -> int32_t {
            auto it = v.tok2id.find(s);
            return it == v.tok2id.end() ? -1 : it->second;
        };
        const int32_t im_start = special("<|im_start|>"), im_end = special("<|im_end|>");
        if (im_start < 0 || im_end < 0) { fprintf(stderr, "error: el vocabulario no tiene <|im_start|>/<|im_end|>\n"); return 1; }
        auto text = [&](const std::string &s) { std::vector<int32_t> t = encode(v, s, false); ids.insert(ids.end(), t.begin(), t.end()); };
        ids.push_back(im_start); text("system\n" + system_msg); ids.push_back(im_end); text("\n");
        ids.push_back(im_start); text("user\n" + prompt);       ids.push_back(im_end); text("\n");
        ids.push_back(im_start); text("assistant\n");
    }
    if (ids.empty() || (int)ids.size() >= e.ctx) { fprintf(stderr, "error: prompt de %zu tokens no cabe en el contexto (%d)\n", ids.size(), e.ctx); return 1; }

    // ---- prefill (batcheado, parte 5b)
    double t0 = now_ms();
    int pos = (int)ids.size();
    for (size_t i = 0; i < ids.size(); i += (size_t)e.bsz) {
        const int nb = (int)std::min<size_t>((size_t)e.bsz, ids.size() - i);
        if (!forward_batch(e, (const int *)(ids.data() + i), nb, (int)i, i + (size_t)nb == ids.size()))
            { fprintf(stderr, "error en el forward (token %d)\n", ids[(size_t)i]); return 1; }
    }
    const double t_prefill = now_ms() - t0;

    // ---- generacion en streaming
    std::mt19937 rng(seed);
    int next = sample_token(e.logits, temp, 40, rng);
    int n_gen = 0;
    t0 = now_ms();
    for (; n_gen < n_predict; n_gen++) {
        if (next == v.eos || next == v.bos) break;
        const std::string piece = decode(v, {next});
        fwrite(piece.data(), 1, piece.size(), stdout);
        fflush(stdout);
        if (pos >= e.ctx) { fprintf(stderr, "\n[contexto lleno]\n"); break; }
        if (!forward(e, next, pos++, true)) break;
        next = sample_token(e.logits, temp, 40, rng);
    }
    const double t_gen = now_ms() - t0;
    printf("\n");
    fprintf(stderr, "\nprompt: %zu tokens en %.0f ms (%.2f tok/s)   generados: %d tokens en %.0f ms (%.2f tok/s)\n",
            ids.size(), t_prefill, (double)ids.size() / (t_prefill / 1000.0),
            n_gen, t_gen, n_gen > 0 ? (double)n_gen / (t_gen / 1000.0) : 0.0);
    return 0;
}
