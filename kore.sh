#!/usr/bin/env bash
# ============================================================================
#  KORE Control Center
#  Panel de mandos para el motor LLM: CLI, WebUI, API OpenAI, agente ReAct,
#  IPC (Core/Orchestrator/TUI), build/install y estado del sistema.
#
#  Uso:
#    ./kore.sh                     # menú interactivo (whiptail; texto si no)
#    ./kore.sh <comando> [args]    # modo scriptable
#
#  Comandos:
#    ask  [texto]        pregunta rápida por CLI (streaming)
#    agent [texto]       agente ReAct (necesita el servidor en marcha)
#    web  start|stop|restart|status|logs
#    ipc  start|stop|logs|tui     triada Core/Orchestrator/TUI
#    api  [prompt]       prueba la API OpenAI (servidor en marcha)
#    build [--install DIR | --uninstall DIR | --clean]
#    status              proceso + puertos + disco
#    config              muestra la configuración efectiva
#
#  Configuración (por prioridad: entorno > ./kore.conf > valores por defecto):
#    KORE_MODEL KORE_HOST KORE_PORT KORE_CTX KORE_THREADS KORE_TEMP KORE_N
#    KORE_SYSTEM KORE_LOG  KORE_SOCK_CORE KORE_SOCK_ORCH KORE_API_PORT
# ============================================================================
set -uo pipefail

KORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$KORE_DIR/bin"
RUN="${KORE_LOG:-$KORE_DIR/run}"
R0="\033[0m"; C_GRN="\033[1;32m"; C_YEL="\033[1;33m"; C_RED="\033[1;31m"; C_CYA="\033[1;36m"

# ---------------- configuración -------------------------------------------------
if [[ -f "$KORE_DIR/kore.conf" ]]; then source "$KORE_DIR/kore.conf"; fi
MODEL="${KORE_MODEL:-}"
HOST="${KORE_HOST:-127.0.0.1}"
PORT="${KORE_PORT:-8080}"
API_PORT="${KORE_API_PORT:-$PORT}"
CTX="${KORE_CTX:-8192}"
THREADS="${KORE_THREADS:-}"
TEMP="${KORE_TEMP:-0.3}"
N="${KORE_N:-256}"
SYSTEM="${KORE_SYSTEM:-}"
SOCK_CORE="${KORE_SOCK_CORE:-/tmp/kore.sock}"
SOCK_ORCH="${KORE_SOCK_ORCH:-/tmp/kore-orch.sock}"
VERBOSE="${KORE_VERBOSE:-0}"

model_detect() {
    if [[ -n "$MODEL" ]]; then return 0; fi
    local gguf
    gguf="$(find "$KORE_DIR" -maxdepth 1 -name '*.gguf' | sort | head -1)"
    if [[ -z "$gguf" ]]; then
        echo -e "${C_RED}no hay modelo .gguf en $KORE_DIR${R0} (define KORE_MODEL)" >&2
        exit 2
    fi
    MODEL="$gguf"
}

t() { [[ -n "${TERM:-}" ]] && [[ "$TERM" != "dumb" ]] && command -v tput >/dev/null 2>&1; }
ok()   { echo -e "${C_GRN}●${R0} $*"; }
inf()  { echo -e "${C_CYA}•${R0} $*"; }
warn() { echo -e "${C_YEL}!${R0} $*" >&2; }
die()  { echo -e "${C_RED}✖${R0} $*" >&2; exit 1; }

pid_alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }
curl_probe() { curl -s -m 2 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null | grep -q '200'; }

spawn() { mkdir -p "$RUN"/log; }

# ---------------- compilación ----------------------------------------------------
do_build() {
    spawn
    if [[ $# -gt 0 ]]; then
        case "$1" in
            --install) shift; (cd "$KORE_DIR" && ./build.sh --install "$1") ;;
            --uninstall) shift; (cd "$KORE_DIR" && ./build.sh --uninstall "$1") ;;
            --clean) (cd "$KORE_DIR" && ./build.sh --clean) ;;
            *) die "build: opción inválida ($1)";;
        esac
        return $?
    fi
    if [[ -x "$BIN/kore" ]] && [[ "$BIN/kore" -nt "$KORE_DIR"/kore.cpp ]]; then
        inf "binarios al día; usa 'build --clean' para recompilar desde cero."
        return 0
    fi
    ok "compilando ($(cd "$KORE_DIR" && ./build.sh >/dev/null 2>&1 && echo ok || echo 'FALLO'))"
    [[ -x "$BIN/kore" ]] || die "el build falló"
}

# ---------------- CLI / agente ----------------------------------------------------
threads_opt() { [[ -n "$THREADS" ]] && echo "--threads $THREADS"; }
system_opt() { [[ -n "$SYSTEM" ]] && echo "--system $SYSTEM"; }

ask_prompt() {   # streaming directo del motor (sin servidor)
    local text="$*"
    if [[ -z "$text" ]]; then
        read -r -p "Prompt: " text || true
    fi
    [[ -z "$text" ]] && die "prompt vacío"
    model_detect; do_build
    echo -e "${C_CYA}▸ modelo:${R0} $(basename "$MODEL")  ${C_CYA}ctx:${R0} $CTX  ${C_CYA}temp:${R0} $TEMP"
    "$BIN/kore" "$MODEL" "$text" -n "$N" --temp "$TEMP" --ctx "$CTX" $(threads_opt) $(system_opt)
}

agent_turn() {   # agente ReAct vía /api/chat del servidor
    local text="$*" base="http://$HOST:$PORT"
    [[ -z "$text" ]] && { read -r -p "Agente: " text || true; }
    [[ -z "$text" ]] && die "prompt vacío"
    ensure_server
    curl -s -N -X POST "$base/api/chat" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys
d={"prompt":sys.argv[1],"temp":'"$TEMP"',"n":'"$N"',"agent":True}
print(json.dumps(d,ensure_ascii=False))' "$text")" \
        | python3 -u -c 'import sys,re,json
for ln in sys.stdin:
    if ln.startswith("data: "):
        try: o = json.loads(ln[6:])
        except Exception: continue
        t = o.get("type")
        if t == "chunk":       sys.stdout.write(o.get("text","")); sys.stdout.flush()
        elif t == "tool":      print("\n\x1b[1;36m[herramienta]\x1b[0m\n" + o.get("text",""))
        elif t == "meta":      print("\n\x1b[1;33m"+o.get("text","")+"\x1b[0m")
        elif t == "done":      print("\n\x1b[2m["+str(o.get("tokens"))+" tok, "+str(round(o.get("ms",0)/1000))+" s]\x1b[0m")
        elif t == "error":     print("\n\x1b[1;31merror: "+o.get("text","")+"\x1b[0m")'
    echo
}

# ---------------- servidor WebUI + OpenAI --------------------------------------------
SERVER_PID="$RUN/server.pid"; SERVER_LOG="$RUN/log/server.log"
WEB_URL="http://$HOST:$PORT"

server_is_alive() { pid_alive "$SERVER_PID" && curl_probe "$WEB_URL/"; }

ensure_server() {
    if server_is_alive; then return 0; fi
    warn "servidor no levantado; arrancándolo en $WEB_URL ..."
    server_start >/dev/null
}

server_start() {
    model_detect; do_build; spawn
    if server_is_alive; then warn "ya en marcha en $WEB_URL (pid $(cat "$SERVER_PID"))"; return 0; fi
    nohup "$BIN/kore" "$MODEL" --web "$HOST:$PORT" --ctx "$CTX" $(threads_opt) >>"$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID"
    inf "esperando a que el modelo cargue ..."
    for _ in $(seq 1 240); do
        curl_probe "$WEB_URL/" && { ok "WebUI  -> $WEB_URL"; ok "OpenAI -> $WEB_URL/v1/models (o $WEB_URL/api/chat para chat+agente)"; return 0; }
        sleep 1
    done
    die "el servidor no respondió en 240 s (log: $SERVER_LOG)"
}

server_stop() {
    if pid_alive "$SERVER_PID"; then
        kill "$(cat "$SERVER_PID")" 2>/dev/null
        for _ in $(seq 1 20); do pid_alive "$SERVER_PID" || break; sleep 0.2; done
        if pid_alive "$SERVER_PID"; then kill -9 "$(cat "$SERVER_PID")" 2>/dev/null; fi
        ok "servidor detenido"
    else
        warn "el servidor no está en marcha"
    fi
    rm -f "$SERVER_PID"
}

server_status() {
    model_detect
    if server_is_alive; then
        ok "servidor EN MARCHA  $WEB_URL   (pid $(cat "$SERVER_PID") · ctx $CTX)"
        inf "modelo: $(basename "$MODEL")"
    else
        warn "servidor PARADO"
    fi
}

api_cmd() {   # prueba rápida de la API OpenAI
    local prompt="${1:-2+2}"
    ensure_server
    echo -e "${C_CYA}GET /v1/models${R0}"
    curl -s "$WEB_URL/v1/models" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || curl -s "$WEB_URL/v1/models"
    echo; echo -e "${C_CYA}POST /v1/chat/completions${R0}  \"$prompt\""
    curl -s -X POST "$WEB_URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":96,\"temperature\":0}" \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print("[usage]", d.get("usage"))'
}

# ---------------- IPC: Core / Orchestrator / TUI -------------------------------------
CORE_PID="$RUN/core.pid"; CORE_LOG="$RUN/log/core.log"
ORCH_PID="$RUN/orch.pid"; ORCH_LOG="$RUN/log/orch.log"

ipc_start() {
    model_detect; do_build; spawn
    if ! pid_alive "$CORE_PID"; then
        nohup "$BIN/kore" "$MODEL" --serve "$SOCK_CORE" $(threads_opt) >>"$CORE_LOG" 2>&1 &
        echo $! > "$CORE_PID"
        inf "Core iniciado (serve $SOCK_CORE)"
    else warn "Core ya en marcha (pid $(cat "$CORE_PID"))"; fi
    sleep 1
    if ! pid_alive "$ORCH_PID"; then
        nohup "$BIN/kore_orch" --core "$SOCK_CORE" --state "$SOCK_ORCH" $(system_opt) >>"$ORCH_LOG" 2>&1 &
        echo $! > "$ORCH_PID"
        inf "Orchestrator iniciado (state $SOCK_ORCH)"
    else warn "Orchestrator ya en marcha (pid $(cat "$ORCH_PID"))"; fi
    sleep 1
    ok "IPC listo. Lanza la TUI con:  ${C_CYA}./kore.sh tui${R0}"
}

ipc_stop() {
    for f in "$ORCH_PID" "$CORE_PID"; do
        if pid_alive "$f"; then kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; fi
    done
    rm -f "$SOCK_CORE" "$SOCK_ORCH"
    ok "IPC detenido"
}

tui_cmd() {
    [[ -x "$BIN/kore_tui" ]] || do_build
    if ! pid_alive "$ORCH_PID"; then ipc_start; sleep 1; fi
    exec "$BIN/kore_tui" --orch "$SOCK_ORCH"
}

ipc_status() {
    pid_alive "$CORE_PID" && ok "Core   en marcha (pid $(cat "$CORE_PID"))"   || warn "Core   parado"
    pid_alive "$ORCH_PID" && ok "Orch   en marcha (pid $(cat "$ORCH_PID"))"   || warn "Orch   parado"
}

# ---------------- estado ----------------------------------------------------
cmd_status() {
    model_detect
    echo -e "${C_GRN}══ KORE status ════════════════════════════════════════${R0}"
    inf "modelo: $(basename "$MODEL")"
    server_status
    ipc_status
    inf "bin/ : $(ls "$BIN"/kore* 2>/dev/null | wc -l) herramientas (build: $([[ -x "$BIN/kore" ]] && echo compilado || echo 'FALTA compilar'))"
    inf "RAM  : $(free -m | awk '/Mem:/{printf "%d MiB libres / %d MiB", $7, $2}')"
}

cmd_config() {
    echo -e "${C_GRN}══ Configuración efectiva ══════════════════════════════${R0}"
    printf '%-18s %s\n' KORE_MODEL   "${MODEL:-<auto-detect>}"
    printf '%-18s %s\n' KORE_HOST    "$HOST"
    printf '%-18s %s\n' KORE_PORT    "$PORT"
    printf '%-18s %s\n' KORE_CTX     "$CTX"
    printf '%-18s %s\n' KORE_THREADS "${THREADS:-auto}"
    printf '%-18s %s\n' KORE_TEMP    "$TEMP"
    printf '%-18s %s\n' KORE_N       "$N"
    printf '%-18s %s\n' KORE_SYSTEM  "${SYSTEM:-<ninguno>}"
    printf '%-18s %s\n' KORE_LOG     "$RUN"
    printf '%-18s %s\n' sockets     "$SOCK_CORE | $SOCK_ORCH"
}

# ---------------- menú interactivo -----------------------------------------------
menu_text() {
    echo -e "${C_GRN}● KORE Control Center${R0} — $(basename "${MODEL:-<sin modelo>}")"
    while true; do
        echo
        echo "  1) ask       pregunta rápida (CLI, streaming)"
        echo "  2) web       servidor WebUI + API OpenAI + agente"
        echo "  3) agent     turno de agente ReAct"
        echo "  4) ipc       Core / Orchestrator / TUI"
        echo "  5) api       probar la API OpenAI (/v1/*)"
        echo "  6) status    estado del sistema"
        echo "  7) build     compilar / instalar"
        echo "  8) config    configuración efectiva"
        echo "  0) salir"
        read -r -p "  opción [1-8/0]: " r || break
        case "$r" in
            1) ask_prompt "" ;; 2) server_menu ;; 3) agent_turn "" ;;
            4) ipc_menu ;; 5) api_cmd "" ;; 6) cmd_status ;;
            7) do_build ;; 8) cmd_config ;; 0|q) echo "adiós"; return 0 ;;
            *) warn "opción inválida" ;;
        esac
    done
}

server_menu() {
    local c
    c=$(whiptail --title "KORE · Servidor Web + OpenAI" --menu "Acción sobre $WEB_URL" 14 56 5 \
        "start"  "Arrancar el servidor WebUI + OpenAI" \
        "stop"   "Detener" \
        "restart" "Reiniciar" \
        "status" "Comprobar" \
        "logs"   "Ver el log en directo" 3>&1 1>&2 2>&3) || return
    case "$c" in
        start) server_start ;; stop) server_stop ;;
        restart) server_stop; server_start ;;
        status) server_status ;; logs) server_logs ;;
    esac
}

ipc_menu() {
    local c
    c=$(whiptail --title "KORE · IPC" --menu "Core / Orchestrator / TUI" 14 56 4 \
        "start" "Arrancar Core + Orchestrator" \
        "tui"   "Abrir la TUI (modo terminal)" \
        "stop"  "Detener triada" \
        "status" "Comprobar" 3>&1 1>&2 2>&3) || return
    case "$c" in
        start) ipc_start ;; tui) tui_cmd ;; stop) ipc_stop ;; status) ipc_status ;;
    esac
}

server_logs() { spawn; ([[ -f "$SERVER_LOG" ]] && tail -f "$SERVER_LOG") || die "sin log todavía"; }

menu_main() {
    local c
    if ! command -v whiptail >/dev/null 2>&1; then menu_text; return 0; fi
    while true; do
        c=$(whiptail --title "KORE Control Center" --menu "Panel de mandos · $(basename "${MODEL:-<sin modelo>}")" 18 60 9 \
            "ask"     "Pregunta rápida por CLI (streaming)" \
            "web"     "Servidor WebUI + API OpenAI + agente" \
            "agent"   "Turno de agente ReAct" \
            "api"     "Probar la API OpenAI (/v1/*)" \
            "ipc"     "Core / Orchestrator / TUI" \
            "build"   "Compilar instaladores" \
            "status"  "Estado del sistema" \
            "config"  "Configuración efectiva" \
            "quit"    "Salir" 3>&1 1>&2 2>&3) || break
        case "$c" in
            ask) ask_prompt "$(whiptail --title KORE --inputbox "Prompt:" 8 60 3>&1 1>&2 2>&3 || true)" ;;
            web) server_menu ;;
            agent) agent_turn "$(whiptail --title "KORE · Agente" --inputbox "Instrucción:" 8 60 3>&1 1>&2 2>&3 || true)" ;;
            api) api_cmd "" ;;
            ipc) ipc_menu ;;
            build) do_build ;;
            status) cmd_status ;;
            config) cmd_config ;;
            quit) echo "adiós"; return 0 ;;
        esac
        echo; read -r -p "pulsa ENTER para continuar ..." _ || true
    done
}

# ---------------- despacho ---------------------------------------------------
main() {
    spawn
    case "${1:-menu}" in
        menu|--menu|-h|--help)                menu_main ;;
        ask)                                  shift; ask_prompt "$@" ;;
        agent)                                shift; agent_turn "$@" ;;
        web)                                  shift; case "${1:-status}" in
                                                start) server_start ;; stop) server_stop ;;
                                                restart) server_stop; server_start ;;
                                                status) server_status ;; logs) server_logs ;;
                                                *) die "web: uso start|stop|restart|status|logs" ;; esac ;;
        ipc)                                  shift; case "${1:-status}" in
                                                start) ipc_start ;; stop) ipc_stop ;;
                                                status) ipc_status ;; logs) spawn; tail -f "$CORE_LOG" ;;
                                                *) die "ipc: uso start|stop|status|logs" ;; esac ;;
        tui)                                  tui_cmd ;;
        api)                                  shift; api_cmd "${1:-}" ;;
        build)                                shift; do_build "$@" ;;
        status)                               cmd_status ;;
        config)                               cmd_config ;;
        stop-all)                             server_stop; ipc_stop ;;
        *) echo "uso: ./kore.sh [comando]  (sin comando = menú; usa ./kore.sh --help)" >&2; exit 1 ;;
    esac
}

main "$@"