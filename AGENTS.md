# KORE

Motor LLM plein-c++ standalone (sin dependencias) + IPC Core/Orchestrator/TUI + WebUI (iteración 4) + API OpenAI + Agente ReAct + resumen automático (iteración 6).

## Build
- Canonical: `./build.sh` compila las 8 herramientas en `./bin/` (mismo compilador que el motor: `-std=c++20 -O3 -pthread -Wall -Wextra`; −O3 da ~49% más prefill que −O2).
- `./build.sh --install DIR` copia a `DIR/bin`; `./build.sh --uninstall DIR` los elimina; `./build.sh --clean` borra `bin/`.
- Compilar solo el motor: `g++ -std=c++20 -O3 -pthread -Wall -Wextra kore.cpp -o kore`.

## Centro de control (`./kore.sh`, sin dependencias)
- `./kore.sh` → menú interactivo (whiptail; fallback textual). Subcomandos scriptables: `ask [texto]`, `web start|stop|restart|status|logs`, `agent [texto]`, `api [prompt]`, `ipc start|stop|status`, `tui`, `build [--install|--uninstall|--clean]`, `status`, `config`, `stop-all`.
- Config por entorno o `./kore.conf`: `KORE_MODEL KORE_HOST KORE_PORT KORE_CTX KORE_THREADS KORE_TEMP KORE_N KORE_SYSTEM KORE_SOCK_CORE KORE_SOCK_ORCH`; logs/pids en `./run/`.
- Detección automática del `*.gguf`; un solo proceso de servidor (pidfile + SSL probe); el agente/API requieren el servidor en marcha.

## Uso rápido
- `./kore modelo.gguf "prompt" -n 30 --temp 0` (contexto por defecto 8192, KV ≈ 896 MiB; `--ctx N` para menos; Qwen2.5 soporta 32k nativo).
- WebUI (iteración 4/6): `./kore modelo.gguf --web 127.0.0.1:8080` → abre `http://127.0.0.1:8080`.
  - chat con streaming SSE, un solo cliente generando a la vez; `POST /api/chat` (`temp`, `n`, `agent: true` = agente ReAct con tools `sh`/`python` in-process).
  - al acercarse al límite de contexto, `/api/chat` condensa la conversación automáticamente (resumen generado por la propia máquina + cola de 192 tokens; evento `meta`; tarda varios minutos en esta CPU).
  - API OpenAI stateless: `GET /v1/models`, `POST /v1/chat/completions` (stream SSE estilo OpenAI + `usage`), `POST /v1/embeddings` (engine propio, dim = n_embd).
- Servidor IPC: `./kore modelo.gguf --serve /tmp/kore.sock` (un solo cliente a la vez).
- ReAct: `./kore_orch --core /tmp/kore.sock` y `./kore_tui --orch /tmp/kore-orch.sock`.
- Regresión/benchmark: `--ids N --temp 0` (determinista), scripts `stub_core.py`/`probe_orch.py`.

## Verificación post-cambio de kernels
`./kore modelo.gguf "¿Qué es la entropía y por qué es importante en termodinámica?" -n 20 --temp 0`
→ debe emitir "La entropía es un concepto fundamental en la termodinámica y la física en…".
Validación numérica del matmul batch: comparar contra referencia en double (get_row+dot).
Agente: `curl -s -N -X POST http://127.0.0.1:8085/api/chat -H 'Content-Type: application/json' -d '{"prompt":"Calcula 25*4 con python y dime el resultado.","temp":0,"n":40,"agent":true}'` → esperar `{"type":"tool","text":"$ python print(25*4)\\n100\\n"}` y luego la respuesta "100".