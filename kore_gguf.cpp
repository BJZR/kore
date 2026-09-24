// kore_gguf.cpp - KORE, paso 1: lector GGUF v3 (mmap, sin dependencias)
// Compilar: g++ -std=c++20 -O2 -Wall -Wextra kore_gguf.cpp -o kore_gguf
// Uso:      ./kore_gguf Qwen2.5-7B-Instruct-abliterated-v2-Q4_K_M.gguf
//
// Asume host little-endian (x86-64), igual que el propio formato GGUF.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------- PARTE 1
// Cursor sobre el archivo mapeado. Toda lectura se valida contra el tamano
// del archivo; si algo falla, se marca err y las lecturas devuelven 0.

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
    bool err;
} Reader;

typedef struct {
    const char *ptr;   // NO termina en '\0': usar siempre ptr + len
    uint64_t len;
} Str;

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

// Avanza n bytes sin copiar.
static void rd_skip(Reader *r, uint64_t n) {
    if (r->err || n > r->size - r->pos) {
        if (!r->err)
            fprintf(stderr, "error: salto fuera de rango en offset %zu\n", r->pos);
        r->err = true;
        return;
    }
    r->pos += (size_t)n;
}

// String GGUF: uint64 longitud + bytes (sin terminador).
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

// ---------------------------------------------------------------- PARTE 2
// Metadatos: tipos de valor GGUF e impresion de cada par clave/valor.

enum {
    T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL,
    T_STR, T_ARR, T_U64, T_I64, T_F64
};

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL:  return 1;
        case T_U16: case T_I16:             return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default:                            return 0;
    }
}

static void print_scalar(Reader *r, uint32_t t) {
    uint64_t raw = 0;                       // little-endian: los n bytes bajos
    if (!rd_bytes(r, &raw, scalar_size(t))) return;
    switch (t) {
        case T_U8:   printf("%u", (unsigned)(uint8_t)raw); break;
        case T_I8:   printf("%d", (int)(int8_t)raw); break;
        case T_U16:  printf("%u", (unsigned)(uint16_t)raw); break;
        case T_I16:  printf("%d", (int)(int16_t)raw); break;
        case T_U32:  printf("%u", (unsigned)(uint32_t)raw); break;
        case T_I32:  printf("%d", (int)(int32_t)raw); break;
        case T_U64:  printf("%llu", (unsigned long long)raw); break;
        case T_I64:  printf("%lld", (long long)raw); break;
        case T_BOOL: printf("%s", raw ? "true" : "false"); break;
        case T_F32: {
            uint32_t u = (uint32_t)raw; float f;
            memcpy(&f, &u, 4);
            printf("%g", (double)f);
            break;
        }
        case T_F64: {
            double d;
            memcpy(&d, &raw, 8);
            printf("%g", d);
            break;
        }
    }
}

// Los strings largos (p. ej. tokenizer.chat_template) se recortan a 80 chars.
static void print_string(Str s) {
    const uint64_t MAXC = 80;
    uint64_t n = s.len < MAXC ? s.len : MAXC;
    putchar('"');
    for (uint64_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s.ptr[i];
        if (c == '\n')                  fputs("\\n", stdout);
        else if (c < 32 || c == 127)    putchar('?');
        else                            putchar(c);
    }
    putchar('"');
    if (s.len > MAXC) printf("... (%llu bytes)", (unsigned long long)s.len);
}

// Los arrays (vocabulario, merges...) solo se resumen: tipo y cantidad.
// Se saltan sin copiar; el tokenizer BPE los leera en el paso 2.
static void print_value(Reader *r, uint32_t t) {
    if (t == T_STR) {
        print_string(rd_str(r));
    } else if (t == T_ARR) {
        uint32_t et = rd_u32(r);
        uint64_t n = rd_u64(r);
        if (r->err) return;
        printf("[array tipo=%u, %llu elementos]", et, (unsigned long long)n);
        if (et == T_STR) {
            for (uint64_t i = 0; i < n && !r->err; i++) rd_str(r);
        } else if (scalar_size(et) != 0) {
            if (n > (r->size - r->pos) / scalar_size(et)) {
                fprintf(stderr, "\nerror: array excede el archivo\n");
                r->err = true;
                return;
            }
            rd_skip(r, n * scalar_size(et));
        } else {
            fprintf(stderr, "\nerror: arrays anidados no soportados\n");
            r->err = true;
        }
    } else if (scalar_size(t) != 0) {
        print_scalar(r, t);
    } else {
        fprintf(stderr, "\nerror: tipo de metadato desconocido: %u\n", t);
        r->err = true;
    }
}

// ---------------------------------------------------------------- PARTE 3
// Tensores: nombre, dimensiones, tipo de cuantizacion, offset y tamano.

// Elementos por bloque y bytes por bloque de cada tipo ggml.
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

#define MAX_TYPES 64

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "uso: %s modelo.gguf\n", argv[0]);
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return 1; }
    size_t size = (size_t)st.st_size;
    void *map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    Reader r = {(const uint8_t *)map, size, 0, false};

    // --- Cabecera
    char magic[4] = {0};
    rd_bytes(&r, magic, 4);
    if (r.err || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "error: no es un archivo GGUF\n");
        return 1;
    }
    uint32_t version = rd_u32(&r);
    uint64_t n_tensors = rd_u64(&r);
    uint64_t n_kv = rd_u64(&r);
    printf("Archivo : %s (%.2f GiB)\n", argv[1], (double)size / (1024.0 * 1024 * 1024));
    printf("Version : %u   Tensores: %llu   Metadatos: %llu\n\n", version,
           (unsigned long long)n_tensors, (unsigned long long)n_kv);
    if (version < 2 || version > 3)
        fprintf(stderr, "aviso: version GGUF %u no probada (se esperaba 2 o 3)\n", version);

    // --- Metadatos
    printf("== Metadatos ==\n");
    uint64_t alignment = 32;                 // valor por defecto de GGUF
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        Str key = rd_str(&r);
        uint32_t type = rd_u32(&r);
        if (r.err) break;
        printf("  %.*s = ", (int)key.len, key.ptr);
        if (key.len == 17 && memcmp(key.ptr, "general.alignment", 17) == 0 &&
            type == T_U32 && r.size - r.pos >= 4) {
            uint32_t a;
            memcpy(&a, r.base + r.pos, 4);
            if (a != 0) alignment = a;
        }
        print_value(&r, type);
        printf("\n");
    }

    // --- Tensores
    printf("\n== Tensores ==\n");
    uint64_t type_count[MAX_TYPES] = {0};
    uint64_t type_bytes[MAX_TYPES] = {0};
    uint64_t max_end = 0;                    // mayor offset+tamano visto
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        Str name = rd_str(&r);
        uint32_t n_dims = rd_u32(&r);
        if (r.err) break;
        if (n_dims == 0 || n_dims > 4) {
            fprintf(stderr, "error: n_dims=%u invalido en tensor %llu\n",
                    n_dims, (unsigned long long)i);
            r.err = true;
            break;
        }
        uint64_t dims[4] = {1, 1, 1, 1};
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            dims[d] = rd_u64(&r);
            nelem *= dims[d];
        }
        uint32_t ttype = rd_u32(&r);
        uint64_t offset = rd_u64(&r);
        if (r.err) break;

        const char *tname = "?";
        uint64_t blk = 0, bpb = 0, nbytes = 0;
        if (type_info(ttype, &tname, &blk, &bpb) && nelem % blk == 0)
            nbytes = nelem / blk * bpb;
        else
            tname = "???";
        if (ttype < MAX_TYPES) { type_count[ttype]++; type_bytes[ttype] += nbytes; }
        if (offset + nbytes > max_end) max_end = offset + nbytes;

        printf("  %-44.*s %-5s [", (int)name.len, name.ptr, tname);
        for (uint32_t d = 0; d < n_dims; d++)
            printf(d ? " x %llu" : "%llu", (unsigned long long)dims[d]);
        printf("]  off=%llu  %.2f MiB\n", (unsigned long long)offset,
               (double)nbytes / (1024.0 * 1024));
    }

    if (r.err) {
        fprintf(stderr, "\nERROR: el archivo esta corrupto o truncado.\n");
        munmap(map, size);
        return 1;
    }

    // --- Inicio de los datos: fin de la tabla de tensores alineado
    uint64_t data_start = (r.pos + alignment - 1) / alignment * alignment;
    printf("\n== Resumen ==\n");
    printf("  alineacion       : %llu\n", (unsigned long long)alignment);
    printf("  inicio de datos  : %llu\n", (unsigned long long)data_start);
    for (uint32_t t = 0; t < MAX_TYPES; t++) {
        const char *tn; uint64_t b, p;
        if (type_count[t] && type_info(t, &tn, &b, &p))
            printf("  %-5s : %4llu tensores, %.2f GiB\n", tn,
                   (unsigned long long)type_count[t],
                   (double)type_bytes[t] / (1024.0 * 1024 * 1024));
        else if (type_count[t])
            printf("  tipo %u : %4llu tensores (tamano desconocido)\n", t,
                   (unsigned long long)type_count[t]);
    }
    if (data_start + max_end > size)
        printf("  AVISO: los tensores llegan al byte %llu pero el archivo mide %zu"
               " (descarga incompleta?)\n",
               (unsigned long long)(data_start + max_end), size);
    else
        printf("  Los tensores caben en el archivo: OK\n");

    munmap(map, size);
    return 0;
}
