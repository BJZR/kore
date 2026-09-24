#!/usr/bin/env python3
# probe_orch.py - KORE iteracion 3: sondeo del Orchestrator (protocolo de tramas)
# Usa: python3 probe_orch.py "pregunta" [opciones]
import os, socket, struct, sys, time

SOCK = os.environ.get("KORE_ORCH_SOCK", "/tmp/kore-orch.sock")

def send_frame(s, typ, data=b""):
    s.sendall(struct.pack("<cI", typ.encode(), len(data)) + data)

def recv_frame(s):
    hdr = s.recv(5)
    if len(hdr) < 5:
        return None, None
    typ, ln = struct.unpack("<cI", hdr)
    data = b""
    while len(data) < ln:
        chunk = s.recv(ln - len(data))
        if not chunk:
            return None, None
        data += chunk
    return typ.decode(), data

def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "Hola"
    only_events = any(a == "--events" for a in sys.argv[2:])
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    t0 = time.time()
    send_frame(s, "P", q.encode())
    n_tok = 0
    while True:
        typ, data = recv_frame(s)
        now = time.time() - t0
        if typ is None:
            print(f"[IO] desconectado a los {now:.1f}s")
            break
        if typ == "M":
            if not only_events: print(f"[meta] {data.decode(errors='replace')}")
        elif typ == "U":
            print(f"\n[user] {data.decode(errors='replace')}")
        elif typ == "A":
            if not only_events:
                sys.stdout.write(data.decode(errors="replace"))
                sys.stdout.flush()
            n_tok += 1
        elif typ == "T":
            sys.stdout.write(f"\n[herramienta] {data.decode(errors='replace')}\n")
            sys.stdout.flush()
        elif typ == "O":
            sys.stdout.write(f"[salida] {data.decode(errors='replace')}\n")
            sys.stdout.flush()
        elif typ == "S":
            print(f"[estado] {data.decode(errors='replace')}")
        elif typ == "E":
            print(f"\n[fin] {n_tok} fragmentos de asistente en {time.time()-t0:.1f}s")
            break
        elif typ == "X":
            print(f"[error] {data.decode(errors='replace')}")
            break
    s.close()

if __name__ == "__main__":
    main()