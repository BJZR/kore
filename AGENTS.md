# KORE

Motor LLM plein-c++ standalone (sin dependencias) + IPC Core/Orchestrator/TUI.

## Build
- Motor: `g++ -std=c++20 -O3 -pthread -Wall -Wextra kore.cpp -o kore`  (−O3 canónico; ~49% más prefill que −O2).
- Herramientas: `kore_gguf`, `kore_kernels`, `kore_quant`, `kore_tok`, `kore_client`, `kore_orch`, `kore_tui`.

## Uso rápido
- `./kore modelo.gguf "prompt" -n 30 --temp 0`
- Servidor IPC: `./kore modelo.gguf --serve /tmp/kore.sock` (un solo cliente a la vez).
- ReAct: `./kore_orch --core /tmp/kore.sock` y `./kore_tui --orch /tmp/kore-orch.sock`.
- Regresión/benchmark: `--ids N --temp 0` (determinista), scripts `stub_core.py`/`probe_orch.py`.

## Verificación post-cambio de kernels
`./kore modelo.gguf "¿Qué es la entropía y por qué es importante en termodinámica?" -n 20 --temp 0`
→ debe emitir "La entropía es un concepto fundamental en la termodinámica y la física en…".
Validación numérica del matmul batch: comparar contra referencia en double (get_row+dot).