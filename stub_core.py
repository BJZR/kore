#!/usr/bin/env python3
# stub_core.py - KORE: imita a KORE-Core para depurar el Orchestrator sin el modelo.
# La respuesta es una cola de fragmentos: hasta la accion, luego el resultado se
# reinyecta ('I') y la cola continua. Cuando la cola se vacia, se envia 'E'.
import os
import socket
import struct

def send_frame(s, typ, data=b""):
    s.sendall(struct.pack("<cI", typ.encode(), len(data)) + data)

def recv_frame(s):
    hdr = b""
    while len(hdr) < 5:
        c = s.recv(5 - len(hdr))
        if not c:
            return None, None
        hdr += c
    typ, ln = struct.unpack("<cI", hdr)
    data = b""
    while len(data) < ln:
        c = s.recv(ln - len(data))
        if not c:
            return None, None
        data += c
    return typ.decode(), data

SOCK = "/tmp/kore-stub.sock"
try:
    os.unlink(SOCK)
except FileNotFoundError:
    pass

srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(1)
print("stub core listo", flush=True)

conn, _ = srv.accept()
send_frame(conn, "M", b"stub core v1")
log = lambda m: print("[core]", m, flush=True)

HEAD = [
    "Simulacion: un agente ",
    "que pide herramientas con ",
    "[ACTION: sh echo hello-from-stub]",
    " y ",
    "despues informa.\n",
]
TAIL = [
    "Continuacion tras la herramienta: ",
    "stdout = hello-from-stub.\n",
]

pending = []
turn = 0
while True:
    typ, data = recv_frame(conn)
    if typ is None:
        log("cliente desconectado")
        break
    if typ == "R":
        log(f"RESET ({len(data or b'')} bytes)")
        send_frame(conn, "R")
    elif typ == "P":
        turn += 1
        log(f"PROMPT turno {turn}")
        send_frame(conn, "P")
        pending = list(HEAD)
    elif typ == "I":
        data = data or b""
        log(f"INJECT ({len(data)} B): {data.decode(errors='replace')[:60]!r}")
        send_frame(conn, "I")
        pending.extend(TAIL)
    elif typ == "T":
        if pending:
            piece = pending.pop(0)
            log(f"TOKEN -> {piece.strip()!r}")
            send_frame(conn, "T", struct.pack("<I", 0) + piece.encode())
        else:
            log("TOKEN -> fin de turno")
            send_frame(conn, "E", b"\x00")
    else:
        log(f"comando inesperado {typ!r}")
        send_frame(conn, "X", b"comando desconocido")