// kore_kernels.cpp - KORE, paso 4: productos matriz-vector Q4_K / Q6_K (escalar + AVX2)
// Compilar: g++ -std=c++20 -O2 -pthread -Wall -Wextra kore_kernels.cpp -o kore_kernels
// Uso:      ./kore_kernels modelo.gguf tensor       (precision + velocidad de ese tensor)
//   ej.:    ./kore_kernels modelo.gguf blk.0.ffn_gate.weight
//
// Incluye el cargador y la descuantizacion del paso 3 (partes 1-3) y anade:
//   parte 4: cuantizacion de activaciones a Q8_K + productos punto escalares de referencia
//   parte 5: productos punto AVX2 + matvec
//   parte 6: comprobacion de precision y benchmark
// Los kernels escalares son la referencia; los AVX2 deben coincidir con ellos.

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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

static bool matvec(const Tensor &t, const BlockQ8K *xq, float *out, bool avx2) {
    return matvec_rows(t, xq, out, 0, t.dims[1] * t.dims[2] * t.dims[3], avx2);
}

// Reparte las filas en trozos contiguos, uno por hilo. Cada fila es independiente, asi que el
// resultado es identico al de 1 hilo. (Un motor real usara un pool de hilos persistente.)
static void matvec_mt(const Tensor &t, const BlockQ8K *xq, float *out, bool avx2, unsigned nthreads) {
    const uint64_t rows = t.dims[1] * t.dims[2] * t.dims[3];
    if (nthreads <= 1 || rows < nthreads) { matvec(t, xq, out, avx2); return; }
    std::vector<std::thread> th;
    const uint64_t chunk = (rows + nthreads - 1) / nthreads;
    for (unsigned k = 0; k < nthreads; k++) {
        const uint64_t r0 = k * chunk, r1 = std::min<uint64_t>(rows, r0 + chunk);
        if (r0 >= r1) break;
        th.emplace_back([&, r0, r1] { matvec_rows(t, xq, out, r0, r1, avx2); });
    }
    for (std::thread &x : th) x.join();
}

// ---------------------------------------------------------------- PARTE 6
// Comprobacion de precision y benchmark sobre un tensor real.

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "uso: %s modelo.gguf tensor\n", argv[0]);
        return 1;
    }
    Model m;
    if (!load_model(argv[1], m)) return 1;
    auto it = m.tensors.find(argv[2]);
    if (it == m.tensors.end()) { fprintf(stderr, "error: no existe el tensor '%s'\n", argv[2]); return 1; }
    const Tensor &t = it->second;
    const char *tn = "?"; uint64_t blk, bpb;
    type_info(t.type, &tn, &blk, &bpb);
    const int64_t n = (int64_t)t.dims[0];
    const uint64_t rows = t.dims[1] * t.dims[2] * t.dims[3];
    std::vector<float> ref_out(rows), avx_out(rows);

    // x aleatorio (normal, semilla fija) cuantizado a Q8_K
    std::vector<float> x((size_t)n);
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float &v : x) v = nd(rng);
    std::vector<BlockQ8K> xq((size_t)(n / 256));
    quantize_row_q8_K(x.data(), xq.data(), n);

    const bool use_avx2 = KORE_HAVE_AVX2 && cpu_has_avx2();
    printf("%s  %s  [%lld x %llu]  AVX2/FMA en esta CPU: %s\n", argv[2], tn, (long long)n,
           (unsigned long long)rows, use_avx2 ? "si" : "NO");
    if (!use_avx2) {
        fprintf(stderr, "error: esta CPU/compilacion no tiene AVX2+FMA; no hay nada que comparar\n");
        return 1;
    }

    double xnorm = 0;
    for (float v : x) xnorm += (double)v * v;
    xnorm = std::sqrt(xnorm);

    // 1) escalar: una pasada de calentamiento (trae el archivo a la cache de paginas: la primera
    //    lectura del mmap puede ir a disco) y despues el mejor tiempo de 3.
    if (!matvec(t, xq.data(), ref_out.data(), false)) { fprintf(stderr, "tipo no soportado\n"); return 1; }
    double t_ref = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_ms();
        matvec(t, xq.data(), ref_out.data(), false);
        t_ref = std::min(t_ref, now_ms() - t0);
    }

    const uint64_t n_check = rows < 256 ? rows : 256;
    double max_q8err = 0;
    std::vector<float> row;
    for (uint64_t r = 0; r < n_check; r++) {
        get_row(t, r, row);
        double dot = 0, wn = 0;
        for (int64_t i = 0; i < n; i++) { dot += (double)row[(size_t)i] * x[(size_t)i]; wn += (double)row[(size_t)i] * row[(size_t)i]; }
        double denom = std::sqrt(wn) * xnorm;
        if (denom > 0) max_q8err = std::max(max_q8err, std::fabs(dot - ref_out[r]) / denom);
    }
    printf("precision (escalar Q8_K vs float, %llu filas): max |dif| / (|w||x|) = %.3g\n",
           (unsigned long long)n_check, max_q8err);

    // 2) AVX2: tiempo (mejor de 5) + coincidencia con el escalar en TODAS las filas
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now_ms();
        matvec(t, xq.data(), avx_out.data(), true);
        best = std::min(best, now_ms() - t0);
    }
    double max_dif = 0, max_abs = 0;
    for (uint64_t r = 0; r < rows; r++) {
        max_dif = std::max(max_dif, (double)std::fabs(avx_out[r] - ref_out[r]));
        max_abs = std::max(max_abs, (double)std::fabs(ref_out[r]));
    }
    printf("AVX2 vs escalar (%llu filas): max |dif| = %.3g  (max |valor| = %.3g)  -> %s\n",
           (unsigned long long)rows, max_dif, max_abs, max_dif <= 1e-4 * max_abs + 1e-6 ? "OK" : "FALLO");

    const double mb = (double)(t.nbytes) / (1024.0 * 1024.0);
    printf("tiempo, 1 hilo, mejor de 3/5 con cache caliente:  escalar %.2f ms   AVX2 %.2f ms   (%.1fx)\n", t_ref, best, t_ref / best);
    printf("AVX2, 1 hilo:  %.2f GB/s de pesos leidos (%.1f MiB)\n", (double)t.nbytes / 1e9 / (best / 1000.0), mb);

    // 3) multihilo: mismo resultado exacto que con 1 hilo
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (const char *e = getenv("KORE_THREADS")) hw = (unsigned)std::max(1, atoi(e));   // forzar el numero de hilos
    std::vector<float> mt_out(rows);
    for (unsigned nt : {2u, 4u, hw}) {
        if (nt > hw || (nt == hw && (hw == 2 || hw == 4))) continue;
        double bmt = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now_ms();
            matvec_mt(t, xq.data(), mt_out.data(), true, nt);
            bmt = std::min(bmt, now_ms() - t0);
        }
        bool same = memcmp(mt_out.data(), avx_out.data(), rows * sizeof(float)) == 0;
        printf("AVX2, %2u hilos: %.2f ms  %.2f GB/s  (%.1fx sobre 1 hilo)  resultado identico: %s\n",
               nt, bmt, (double)t.nbytes / 1e9 / (bmt / 1000.0), best / bmt, same ? "si" : "NO");
        if (!same) return 2;
    }
    return max_dif <= 1e-4 * max_abs + 1e-6 ? 0 : 2;
}
