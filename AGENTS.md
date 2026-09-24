# KORE

Motor LLM plein-c++ standalone (sin dependencias) + IPC Core/Orchestrator/TUI + WebUI (iteración 4).

## Build
- Canonical: `./build.sh` compila las 8 herramientas en `./bin/` (mismo compilador que el motor: `-std=c++20 -O3 -pthread -Wall -Wextra`; −O3 da ~49% más prefill que −O2).
- `./build.sh --install DIR` copia a `DIR/bin`; `./build.sh --uninstall DIR` los elimina; `./build.sh --clean` borra `bin/`.
- Compilar solo el motor: `g++ -std=c++20 -O3 -pthread -Wall -Wextra kore.cpp -o kore`.

## Uso rápido
- `./kore modelo.gguf "prompt" -n 30 --temp 0`
- WebUI (iteración 4): `./kore modelo.gguf --web 127.0.0.1:8080` → abre `http://127.0.0.1:8080` (chat con streaming SSE, un solo cliente generando a la vez; endpoint `POST /api/chat`).
- Servidor IPC: `./kore modelo.gguf --serve /tmp/kore.sock` (un solo cliente a la vez).
- ReAct: `./kore_orch --core /tmp/kore.sock` y `./kore_tui --orch /tmp/kore-orch.sock`.
- Regresión/benchmark: `--ids N --temp 0` (determinista), scripts `stub_core.py`/`probe_orch.py`.

## Verificación post-cambio de kernels
`./kore modelo.gguf "¿Qué es la entropía y por qué es importante en termodinámica?" -n 20 --temp 0`
→ debe emitir "La entropía es un concepto fundamental en la termodinámica y la física en…".
Validación numérica del matmul batch: comparar contra referencia en double (get_row+dot).