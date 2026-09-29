#!/usr/bin/env bash
# ============================================================================
#  kore_backup.sh - Guarda el proyecto KORE en GitHub (repo "kore", cuenta BJZR).
#
#  Uso:  ./kore_backup.sh
#
#  Solo te pide la clave (API key/personal access token) de GitHub la primera
#  vez; la guarda en ~/.config/kore-github-token (fuera del repo, chmod 600).
#  Tambien puedes pre-fijarla con:  KORE_GITHUB_TOKEN=... ./kore_backup.sh
#
#  Al ejecutarlo:
#    1. valida el token y que pertenezca a BJZR
#    2. crea el repo remoto "kore" si aun no existe
#    3. commitea los cambios (respeta .gitignore) y hace push a
#       https://github.com/BJZR/kore  (rama main)
#
#  El .gitignore ya excluye binarios, bin/, run/, *.gguf, *.log, *.sock...
#  y el token NUNCA se guarda dentro del repo.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

OWNER="BJZR"
REPO="kore"
TK_FILE="${KORE_TOKEN_FILE:-$HOME/.config/kore-github-token}"

# ---------------- clave ------------------------------------------------------
TK="${KORE_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -z "$TK" && -r "$TK_FILE" ]]; then
  TK="$(command cat "$TK_FILE")"
fi
if [[ -z "$TK" ]]; then
  printf 'API key de GitHub (no se muestra): ' >&2
  read -r -s TK || true
  printf '\n' >&2
  if [[ -z "$TK" ]]; then
    printf 'error: sin token.\nUsa KORE_GITHUB_TOKEN=... o escribe la clave.\n' >&2
    exit 1
  fi
  printf 'Guardar la clave en %s para proximas veces? [S/n]: ' "$TK_FILE" >&2
  read -r -n1 SAVE || true
  printf '\n' >&2
  if [[ -z "$SAVE" || "$SAVE" == [sSyY] ]]; then
    mkdir -p "$(dirname "$TK_FILE")"
    umask 077
    printf '%s' "$TK" > "$TK_FILE"
    chmod 600 "$TK_FILE"
  fi
fi
export KORE_GIT_TK="$TK"   # para el credential.helper del push (nunca en argv/git config)

# ---------------- validar token y cuenta --------------------------------------
LOGIN="$(curl -sf -H "Authorization: token $TK" https://api.github.com/user \
         | sed -n 's/.*"login"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1 || true)"
if [[ -z "$LOGIN" ]]; then
  printf 'error: token invalido o sin permiso (¿tienes scope repo?).\n' >&2
  exit 1
fi
if [[ "$LOGIN" != "$OWNER" ]]; then
  printf 'error: el token es de "%s", no de "%s".\n' "$LOGIN" "$OWNER" >&2
  exit 1
fi
printf 'token OK (cuenta: %s)\n' "$LOGIN"

# ---------------- reforzar ignores (idempotente) ------------------------------
append_ignore() {
  command grep -aqF "$1" .gitignore 2>/dev/null || printf '%s\n' "$1" >> .gitignore
}
append_ignore '*.token'
append_ignore '*.openai_key'
append_ignore 'texto.txt'           # artefactos de pruebas del agente

# ---------------- asegurar repo remoto ----------------------------------------
if ! curl -sf -H "Authorization: token $TK" "https://api.github.com/repos/$OWNER/$REPO" >/dev/null; then
  DESC="$(sed -n 's/^#*[[:space:]]*//p' README.md | grep -am1 . | sed 's/"/\\"/g')"
  printf 'creando repo remoto %s/%s...\n' "$OWNER" "$REPO"
  code="$(curl -s -o /tmp/kore_gh_resp.json -w '%{http_code}' \
          -X POST -H "Authorization: token $TK" -H 'Content-Type: application/json' \
          -d "{\"name\":\"$REPO\",\"description\":\"$DESC\"}" \
          https://api.github.com/user/repos)"
  if [[ "$code" != "201" ]]; then
    printf 'error: no pude crear el repo (HTTP %s):\n' "$code" >&2
    command cat /tmp/kore_gh_resp.json >&2; printf '\n' >&2
    exit 1
  fi
fi

# ---------------- remote (sin token en .git/config) ----------------------------
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/$OWNER/$REPO.git"
else
  git remote add origin "https://github.com/$OWNER/$REPO.git"
fi

# ---------------- commit + push ------------------------------------------------
git add -A
MSG=""
if git diff --cached --quiet; then
  printf 'sin cambios nuevos que subir\n'
else
  MSG="backup: $(date '+%Y-%m-%d %H:%M')"
  MSG_GUARDADA="$MSG"
  git commit -m "$MSG" >/dev/null
  printf 'commit: %s\n' "$MSG"
fi

git -c credential.helper='!f(){ printf "username=x-access-token\npassword=%s\n" "$KORE_GIT_TK"; };f' \
    push -u origin main

printf '\nListo. Repo publicado en:  https://github.com/%s/%s\n' "$OWNER" "$REPO"