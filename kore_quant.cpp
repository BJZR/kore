// kore_quant.cpp - KORE, paso 3: carga de tensores por nombre + descuantizacion Q4_K / Q6_K
// Compilar: g++ -std=c++20 -O2 -Wall -Wextra kore_quant.cpp -o kore_quant
// Uso:      ./kore_quant modelo.gguf                        (hiperparametros + validacion Qwen2)
//           ./kore_quant modelo.gguf tensor fila [out.f32]  (descuantiza una fila y muestra estadisticas)
//   ej.:    ./kore_quant modelo.gguf token_embd.weight 151644
//
// Version escalar de referencia: primero correcta y verificable; los kernels
// AVX2 del paso 4 se comprobaran contra estas funciones.

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

static float f16_to_f32(uint16_t h) {
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

static float rd_f16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return f16_to_f32(h);
}

// Q4_K: bloque de 144 bytes = 256 valores.
//   [0..1] d (f16)  [2..3] dmin (f16)  [4..15] 8 escalas y 8 minimos de 6 bits  [16..143] 256 nibbles
// valor = d*escala*nibble - dmin*minimo, en 8 sub-bloques de 32 valores.
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
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
// Validacion de la arquitectura Qwen2 y programa principal.

static int expect(const Model &m, const std::string &name, std::initializer_list<uint64_t> dims) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) { printf("  FALTA         %s\n", name.c_str()); return 1; }
    const Tensor &t = it->second;
    bool ok = t.n_dims == dims.size() && t.data != nullptr;
    size_t i = 0;
    for (uint64_t d : dims) { if (i < 4 && t.dims[i] != d) ok = false; i++; }
    if (!ok) { printf("  DIMS/TIPO MAL %s\n", name.c_str()); return 1; }
    return 0;
}

// Comprueba que estan TODOS los tensores que necesita el forward de Qwen2
// con las dimensiones que implican los hiperparametros.
static int validate_qwen2(const Model &m) {
    const Hparams &h = m.hp;
    if (h.n_head == 0 || h.n_embd % h.n_head != 0) { printf("  hiperparametros invalidos\n"); return 1; }
    const uint64_t E = h.n_embd, FF = h.n_ff, V = h.n_vocab;
    const uint64_t KV = (uint64_t)h.n_head_kv * (E / h.n_head);     // 4 * 128 = 512
    int bad = 0;
    bad += expect(m, "token_embd.weight", {E, V});
    bad += expect(m, "output_norm.weight", {E});
    bad += expect(m, "output.weight", {E, V});
    for (uint32_t l = 0; l < h.n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        bad += expect(m, p + "attn_norm.weight", {E});
        bad += expect(m, p + "attn_q.weight", {E, E});
        bad += expect(m, p + "attn_q.bias", {E});
        bad += expect(m, p + "attn_k.weight", {E, KV});
        bad += expect(m, p + "attn_k.bias", {KV});
        bad += expect(m, p + "attn_v.weight", {E, KV});
        bad += expect(m, p + "attn_v.bias", {KV});
        bad += expect(m, p + "attn_output.weight", {E, E});
        bad += expect(m, p + "ffn_norm.weight", {E});
        bad += expect(m, p + "ffn_gate.weight", {E, FF});
        bad += expect(m, p + "ffn_up.weight", {E, FF});
        bad += expect(m, p + "ffn_down.weight", {FF, E});
    }
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc == 3 || argc > 5) {
        fprintf(stderr, "uso: %s modelo.gguf\n     %s modelo.gguf tensor fila [out.f32]\n", argv[0], argv[0]);
        return 1;
    }
    Model m;
    if (!load_model(argv[1], m)) return 1;

    if (argc == 2) {
        const Hparams &h = m.hp;
        printf("capas=%u  embd=%u  ffn=%u  cabezas=%u  kv=%u  vocab=%u  ctx=%u\n",
               h.n_layer, h.n_embd, h.n_ff, h.n_head, h.n_head_kv, h.n_vocab, h.n_ctx);
        printf("rope_base=%g  rms_eps=%g  tensores=%zu  inicio_datos=%llu\n",
               (double)h.rope_base, (double)h.rms_eps, m.tensors.size(),
               (unsigned long long)m.data_start);
        int bad = validate_qwen2(m);
        if (bad) { printf("\n%d problemas.\n", bad); return 2; }
        printf("Arquitectura Qwen2 OK: %zu tensores esperados presentes, con las dimensiones correctas.\n",
               (size_t)h.n_layer * 12 + 3);
        return 0;
    }

    auto it = m.tensors.find(argv[2]);
    if (it == m.tensors.end()) { fprintf(stderr, "error: no existe el tensor '%s'\n", argv[2]); return 1; }
    const Tensor &t = it->second;
    const char *tn = "?"; uint64_t blk, bpb;
    type_info(t.type, &tn, &blk, &bpb);
    char *end;
    uint64_t row = strtoull(argv[3], &end, 10);
    if (*end != '\0') { fprintf(stderr, "error: fila invalida\n"); return 1; }

    std::vector<float> v;
    if (!get_row(t, row, v)) { fprintf(stderr, "error: no se pudo leer la fila %llu\n", (unsigned long long)row); return 1; }

    double sum = 0, sq = 0;
    float mn = v[0], mx = v[0];
    for (float x : v) { sum += x; sq += (double)x * x; if (x < mn) mn = x; if (x > mx) mx = x; }
    printf("%s  %s  [%llu x %llu]  fila %llu (%zu valores)\n", argv[2], tn,
           (unsigned long long)t.dims[0], (unsigned long long)t.dims[1],
           (unsigned long long)row, v.size());
    printf("min=%.6g  max=%.6g  media=%.6g  norma2=%.6g\n", (double)mn, (double)mx,
           sum / (double)v.size(), std::sqrt(sq));
    printf("primeros:");
    for (size_t i = 0; i < 8 && i < v.size(); i++) printf(" %.6g", (double)v[i]);
    printf("\n");

    if (argc == 5) {
        FILE *f = fopen(argv[4], "wb");
        if (!f) { perror("fopen"); return 1; }
        fwrite(v.data(), 4, v.size(), f);
        fclose(f);
    }
    return 0;
}
