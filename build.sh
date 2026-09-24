#!/usr/bin/env sh
# build.sh - KORE (iteracion 4): compila todas las herramientas en ./bin y opcionalmente instala.
#   ./build.sh                 compila en ./bin
#   ./build.sh --install DIR   compila y copia los binarios a DIR/bin
#   ./build.sh --uninstall DIR elimina los binarios que KORE copio a DIR/bin
#   ./build.sh --clean         borra bin/ y los binarios sueltos de la raiz
set -e
cd "$(dirname "$0")"

CXX="${CXX:-g++}"
FLAGS="-std=c++20 -O3 -pthread -Wall -Wextra"
TOOLS="kore kore_gguf kore_kernels kore_quant kore_tok kore_client kore_orch kore_tui"
INSTALLED="kore kore_gguf kore_kernels kore_quant kore_tok kore_client kore_orch kore_tui"

# --- deteccion de CPU (los kernels AVX2/FMA se eligen en runtime; esto es solo informativo)
if [ -r /proc/cpuinfo ]; then
    AVX2=$(grep -o -m1 avx2 /proc/cpuinfo || echo no)
    FMA=$(grep -o -m1 fma /proc/cpuinfo || echo no)
    echo "CPU: AVX2=$AVX2 FMA=$FMA  (sin AVX2 se usara el camino escalar, mas lento)"
fi

if [ "$1" = "--clean" ]; then
    rm -rf bin
    for t in $TOOLS; do rm -f "$t"; done
    echo "limpiado: bin/ y binarios sueltos eliminados"
    exit 0
fi

if [ "$1" = "--uninstall" ]; then
    [ -z "$2" ] && { echo "uso: ./build.sh --uninstall DIR" >&2; exit 1; }
    removed=0; missing=0
    for t in $INSTALLED; do
        if [ -e "$2/bin/$t" ]; then rm -f "$2/bin/$t"; removed=$((removed + 1));
        else missing=$((missing + 1)); fi
    done
    rmdir "$2/bin" 2>/dev/null || true
    echo "desinstalado: $removed binarios eliminados de $2/bin${missing:+ ($missing no estaban)}"
    exit 0
fi

mkdir -p bin
echo "compilando $([ "$1" = --install ] && echo "e instalando en $2/bin" || echo "en bin/")..."
for t in $TOOLS; do
    echo "  $t..."
    $CXX $FLAGS "$t.cpp" -o "bin/$t"
done

if [ "$1" = "--install" ]; then
    [ -z "$2" ] && { echo "uso: ./build.sh --install DIR" >&2; exit 1; }
    install -d "$2/bin"
    for t in $TOOLS; do install -m 755 "bin/$t" "$2/bin/"; done
    echo "instalado en $2/bin"
fi

echo "listo:"
for t in $TOOLS; do echo "  bin/$t"; done