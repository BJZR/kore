# KORE

**Motor LLM de código abierto en C++20 puro** (single-file, sin dependencias externas) que corre modelos GGUF de la familia **Qwen2** directamente con kernels AVX2/FMA propios, junto a un ecosistema completo: CLI, WebUI, API-style-OpenAI, agente ReAct y red IPC Core/Orchestrator/TUI.

Todo está hecho "desde cero": lectura del GGUF, tokenizer, cuantización, forward (atención con RoPE-GQA + SwiGLU), prefill por lotes, caché KV y servidores HTTP/Unix-socket. No usa llama.cpp ni ninguna biblioteca.

---

## Características

| | |
|---|---|
| **Motor** | C++20 single-file, `-O3`, kernels bloqueados AVX2/FMA con fallback escalar. |
| **Modelos** | GGUF Qwen2 (2/3/7B, Q4_K_M, etc.), contexto extendible hasta 32k (Qwen nativo). |
| **CLI** | Generación en streaming, `--temp 0` determinista, `--ids`/`--dump` para depurar. |
| **WebUI** | Chat SSE en `http://localhost:8080`, un solo cliente generando a la vez. |
| **API OpenAI** | `/v1/models`, `/v1/chat/completions` (streaming + `usage`), `/v1/embeddings` (motor propio, dim = `n_embd`). |
| **Agente ReAct** | Tools `sh` y `python` in-process (fork/execve, timeout 30 s), integrado en WebUI y por curl. |
| **Memoria larga** | Resumen automático: al acercarse al límite de contexto, la propia máquina resume la historia y continúa. |
| **IPC** | Core (`--serve`) + Orchestrator (`kore_orch`) + TUI (`kore_tui`). |
| **Control** | `./kore.sh`: menú interactivo y subcomandos scriptables para todo lo anterior. |

---

## Requisitos

- Linux (o cualquier Unix con sockets).
- `g++` con soporte **C++20** (GCC ≥ 11), `make` opcional.
- CPU con **AVX2 + FMA** recomendada (las kernels escalares son ~3× más lentas).
- ~6 GB de RAM libre (modelo Q4_K_M 7B ≈ 4,8 GB + KV 896 MiB a ctx 8192).
- Un modelo **GGUF Qwen2** (se detecta como `*.gguf` en la raíz, p. ej. `KORE_MODEL=...`).

> Descarga un modelo, por ejemplo: `Qwen2.5-7B-Instruct-Uncensored.Q4_K_M.gguf` (HuggingFace, «gguf» de `unsloth` o `bartowski`).

---

## Compilación

```bash
./build.sh                 # compila las 8 herramientas en ./bin/ y renueva ./kore
./build.sh --install DIR   # copia los binarios a DIR/bin
./build.sh --uninstall DIR # elimina los binarios instalados
./build.sh --clean         # borra bin/ y binarios sueltos
```

Compilar solo el motor:

```bash
g++ -std=c++20 -O3 -pthread -Wall -Wextra kore.cpp -o kore
```

Las 8 herramientas: `kore`, `kore_client`, `kore_orch`, `kore_tui`, `kore_gguf`, `kore_kernels`, `kore_quant`, `kore_tok`.

---

## Uso rápido (CLI)

```bash
./kore modelo.gguf "¿Qué es la entropía y por qué es importante en termodinámica?" -n 20 --temp 0
```

Opciones principales:

| Opción | Descripción |
|---|---|
| `-n N` | tokens a generar (por defecto 256). |
| `--temp T` | temperatura; `0` = greedy/determinista. |
| `--ctx N` | contexto (por defecto **8192**, KV ≈ 896 MiB). |
| `--threads N` | hilos (por defecto los de la CPU). |
| `--system "…"` | mensaje de sistema (ChatML). |
| `--raw` | sin ChatML (tokens crudos). |
| `--ids 1,2,3` | prefill de tokens concretos → imprime top-5 logits (verificación numérica). |
| `--dump logits.f32` | vuelca logits en binario. |
| `--float` | forward en fp32 exacto (análisis). |

---

## Centro de control: `./kore.sh`

Menú interactivo (whiptail, con fallback textual) y subcomandos para todo:

```bash
./kore.sh                     # menú
./kore.sh ask "prompt"        # pregunta rápida por CLI (streaming)
./kore.sh web start|stop|restart|status|logs
./kore.sh agent "instrucción" # agente ReAct (arranca el servidor si hace falta)
./kore.sh api "prompt"        # prueba la API OpenAI (/v1/*)
./kore.sh ipc start|stop|status   # Core + Orchestrator
./kore.sh tui                 # TUI sobre la red IPC
./kore.sh build [--install|--uninstall|--clean]
./kore.sh status|config|stop-all
```

Configuración por entorno o `./kore.conf` (antes de usar el script):

```bash
KORE_MODEL=modelo.gguf
KORE_HOST=127.0.0.1
KORE_PORT=8080
KORE_CTX=8192
KORE_THREADS=4
KORE_TEMP=0.3
KORE_N=256
KORE_SYSTEM="Eres un asistente útil."
KORE_SOCK_CORE=/tmp/kore.sock
KORE_SOCK_ORCH=/tmp/kore-orch.sock
```

PIDs y logs viven en `./run/`.

---

## WebUI + API OpenAI

```bash
./kore.sh web start          # o: ./kore modelo.gguf --web 127.0.0.1:8080
```

Abre `http://127.0.0.1:8080`: chat con streaming SSE, control de temperatura, modo agente y reinicio de conversación.

### Endpoints

| Endpoint | Descripción |
|---|---|
| `GET /` | WebUI (HTML/JS autocontenido). |
| `GET /api/state` | estado: modelo, ctx, hilos, AVX2, temp. |
| `POST /api/chat` | chat SSE: `{prompt, temp, n, agent}`. Eventos `chunk`, `tool`, `meta`, `done`; `409` si otra generación está en curso. |
| `POST /api/reset` | reinicia la conversación (la KV se reconstruye). |
| `GET /v1/models` | lista el modelo. |
| `POST /v1/chat/completions` | API OpenAI (no-stream con `usage`, o `stream: true` SSE estilo OpenAI + `[DONE]`). |
| `POST /v1/embeddings` | embeddings con motor propio (dim = `n_embd`, normalizados L2). |

### Ejemplos

```bash
# chat + agente (ReAct)
curl -s -N -X POST http://localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Calcula 25*4 con python y dime el resultado.","temp":0,"n":40,"agent":true}'

# OpenAI-compatible
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"2+2"}],"max_tokens":64,"temperature":0}'

curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' -d '{"input":"hola"}'
```

---

## Agente ReAct

El agente usa el protocolo `[ACTION: tool arg]` ejecutado con `sh` o `python` in-process. El resultado se reinyecta al modelo para que continúe:

```bash
./kore.sh agent "Lista los archivos de /tmp con sh."
# [herramienta]
# $ sh ls -1 /tmp
# …
```

Tools: `sh <comando o script>` y `python <código>` (python3). Timeout de ejecución de 30 s y salida limitada a 4 KiB. Si el modelo no pide herramienta, responde directamente.

---

## Memoria larga (resumen automático)

Cuando la conversación se acerca al límite de contexto, el propio motor genera **un resumen de la historia reciente**, rehidrata la caché KV (system + resumen + cola de tokens recientes) y continúa. El evento SSE `meta` informa a la UI. Ejemplo de log:

```
[condense] 1262 -> 269 tokens (resumen 48)
```

Tras compactar, el modelo conserva los datos clave de la conversación (nombres, temas, hechos).

---

## IPC: Core / Orchestrator / TUI

```bash
./kore.sh ipc start     # Core (serve /tmp/kore.sock) + Orchestrator (state /tmp/kore-orch.sock)
./kore.sh tui           # TUI de terminal
```

| Proceso | Comando |
|---|---|
| Core | `./kore modelo.gguf --serve /tmp/kore.sock` — motor + búfer de conversación. |
| Orchestrator | `./kore_orch --core /tmp/kore.sock --state /tmp/kore-orch.sock` — agenta la conversación y ejecuta herramientas. |
| TUI | `./kore_tui --orch /tmp/kore-orch.sock` — interfaz de terminal. |
| Cliente | `./kore_client /tmp/kore.sock "pregunta" [-n N] [-s "system"]` — cliente del protocolo. |

También hay utilidades de inspección: `kore_gguf modelo.gguf` (info de tensores), `kore_quant` (dumps), `kore_tok` (tokenizar/segmentar, `--spans`) y `kore_kernels` (pruebas de kernels). Los scripts `stub_core.py` y `probe_orch.py` sirven para probar el protocolo IPC sin motor.

---

## Arquitectura del motor

- **Lectura GGUF**: cabecera/`Metadata`, tensores (f32/f16/Q4_K_M) mapeados a memoria (`mmap`).
- **Tokenizer**: BPE con merges, ChatML (`<|im_start|>`/`<|im_end|>`), Qwen.
- **Forward**: bloque por capa — RMSNorm → Cuantización emb → matvec `Wqkvo (-Wgate/-Wup/-Wdown)` → RoPE-NEOX (GQA) → atención softmax sobre la caché KV → SwiGLU. Kernels AVX2/FMA bloqueados con fallback escalar; opción `--float` fp32 exacto.
- **Prefill por lotes**: `forward_batch` reduce el coste del prompt (`-O3` + batching ≈ 49 % más prefill que `-O2`).
- **Caché KV**: `[capa][pos][kv_dim]` en fp32, ≈ 112 KiB/token (8192 ≈ 896 MiB).
- **Embeddings**: engine propio reutilizando el bloque (contexto corto, `want_logits=false`), salida `rmsnorm + L2`.

---

## Verificación post-cambio

```bash
./kore modelo.gguf "¿Qué es la entropía y por qué es importante en termodinámica?" -n 20 --temp 0
# → "La entropía es un concepto fundamental en la termodinámica y la física en…"

./kore modelo.gguf "X" --ids 151644,872,198 -n 3 --temp 0   # determinista (top-5 logits)

curl -s -N -X POST http://127.0.0.1:8085/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Calcula 25*4 con python y dime el resultado.","temp":0,"n":40,"agent":true}'
# → {"type":"tool","text":"$ python print(25*4)\n100\n"} y respuesta "100"
```

La validación numérica del matmul por lotes se hace comparando contra una referencia en `double` (get_row + dot).

---

## Notas y límites

- Un solo cliente genera a la vez (protección por `mutex`); el resto espera o recibe `409`.
- El resumen automático tarda varios minutos en CPUs modestas (re-prefill con atención `O(posión)`).
- `--temp 0` es greedy; el muestreo con temperaturas altas puede escoger ramas educadas/rechazo del instruct.
- Sin GPU: todo es CPU.