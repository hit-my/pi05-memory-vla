"""omp_checker.py ROOT MODEL TOKEN [WORKERS=4] [PORT=18780]: ledger checker backed by a model reached through the local `omp`
client, kept warm as persistent `omp --mode rpc` processes (about 3.5 s per check instead of 30 s per CLI start).
HTTP on 127.0.0.1:PORT, POST /<TOKEN>/judge with the same JSON as checker_glm2.py -> {"p_yes","seconds"}.
With env SSH_PW set (and CHECKER_TUNNEL_HOST / CHECKER_TUNNEL_USER / CHECKER_TUNNEL_PORT), also opens a reverse tunnel so that
127.0.0.1:PORT on the evaluation server reaches this service.
The prompt is the tested "task information + describe first" one (glm_full.p_info_obs)."""
import base64, io, json, os, queue, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image, ImageDraw

R, MODEL, TOKEN = sys.argv[1], sys.argv[2], sys.argv[3]
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 4; PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 18780
os.environ.setdefault("GLM_KEY", "unused"); sys.argv = [sys.argv[0], R, "info_obs"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_full  # noqa: E402  (prompt text only)
STATS = {"calls": 0, "errors": 0, "seconds": []}


class Worker:
    def __init__(self): self.start()

    def start(self):
        self.p = subprocess.Popen(["omp", "--mode", "rpc", "--model", MODEL, "--thinking", "low", "--no-tools", "--no-session", "--no-context-files", "--no-skills", "--no-rules", "--no-lsp",
                                   "--no-title", "--cwd", R + r"\omp_work", "--system-prompt", "You are a careful visual checker. Answer with JSON only."],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.q = queue.Queue(); self.fresh = True
        threading.Thread(target=lambda p=self.p, q=self.q: [q.put(l) for l in p.stdout], daemon=True).start()

    def send(self, obj): self.p.stdin.write(json.dumps(obj) + "\n"); self.p.stdin.flush()

    def ask(self, text, jpeg_b64, timeout=90):
        if self.p.poll() is not None: self.start()
        while not self.q.empty(): self.q.get_nowait()
        if not self.fresh: self.send({"type": "new_session"})
        self.fresh = False
        self.send({"type": "prompt", "message": text, "images": [{"type": "image", "data": jpeg_b64, "mimeType": "image/jpeg"}]})
        t0 = time.time(); out = ""
        while time.time() - t0 < timeout:
            try: ev = json.loads(self.q.get(timeout=1))
            except queue.Empty: continue
            except Exception: continue
            if ev.get("type") == "message_end" and ev.get("message", {}).get("role") == "assistant":
                out += "".join(part.get("text", "") for part in ev["message"].get("content", []) if part.get("type") == "text")
            if ev.get("type") == "agent_end": return out
        self.p.kill(); raise TimeoutError("omp gave no answer")


POOL = queue.Queue()
for _ in range(WORKERS): POOL.put(Worker())


def sheet(q):
    im = Image.new("RGB", (448 * 3 + 16, 448 + 28), "white"); d = ImageDraw.Draw(im)
    for i, (k, lab) in enumerate((("ref", "FRONT CAMERA - when this step started"), ("now", "FRONT CAMERA - now"), ("wrist", "WRIST CAMERA - now"))):
        im.paste(Image.open(io.BytesIO(base64.b64decode(q[k]))).convert("RGB").resize((448, 448), Image.BICUBIC), (i * 456, 28)); d.text((i * 456 + 6, 8), lab, fill="black")
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()


def judge(q):
    case = {"instruction": q["instruction"], "step": q["step"], "k": q.get("k", ""), "checklist": q.get("checklist") or [], "seconds": float(q.get("seconds", 0))}
    text = glm_full.p_info_obs2(case); t0 = time.time(); STATS["calls"] += 1; w = POOL.get()
    try:
        txt = w.ask(text, sheet(q)); j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]); conf = min(1.0, max(0.5, float(j.get("confidence", 0.8))))
        res = {"p_yes": conf if j["verdict"] == "done" else 1.0 - conf, "reason": str(j.get("target_now", ""))[:200]}
    except Exception as error:
        STATS["errors"] += 1; res = {"p_yes": 1.0, "api_error": repr(error)[:200]}  # no verdict = do not touch the ledger
    finally: POOL.put(w)
    res["seconds"] = round(time.time() - t0, 2); STATS["seconds"] = (STATS["seconds"] + [res["seconds"]])[-200:]
    return res


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != f"/{TOKEN}/judge": self.send_response(403); self.end_headers(); return
        try: out = json.dumps(judge(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))).encode(); code = 200
        except Exception as error: out = json.dumps({"error": repr(error)}).encode(); code = 500
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def do_GET(self):
        s = sorted(STATS["seconds"]); out = json.dumps({"calls": STATS["calls"], "errors": STATS["errors"], "median_seconds": s[len(s) // 2] if s else None}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a): pass


def tunnel():
    import paramiko
    while True:
        try:
            c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(os.environ["CHECKER_TUNNEL_HOST"], port=int(os.environ.get("CHECKER_TUNNEL_PORT", "22")), username=os.environ["CHECKER_TUNNEL_USER"], password=os.environ["SSH_PW"], timeout=20, look_for_keys=False, allow_agent=False)
            tr = c.get_transport(); tr.set_keepalive(20); tr.request_port_forward("127.0.0.1", PORT)
            print("TUNNEL_UP", flush=True)
            while tr.is_active():
                ch = tr.accept(5)
                if ch is None: continue
                def pump(ch=ch):
                    s = socket.create_connection(("127.0.0.1", PORT))
                    def copy(a, b):
                        try:
                            while True:
                                d = a.recv(65536)
                                if not d: break
                                b.sendall(d)
                        except Exception: pass
                        finally:
                            try: b.shutdown(socket.SHUT_WR) if isinstance(b, socket.socket) else b.shutdown_write()
                            except Exception: pass
                    t = threading.Thread(target=copy, args=(ch, s), daemon=True); t.start(); copy(s, ch); t.join(); ch.close(); s.close()
                threading.Thread(target=pump, daemon=True).start()
        except Exception as error: print("TUNNEL_ERROR", repr(error)[:200], flush=True)
        time.sleep(10)


if os.environ.get("SSH_PW"): threading.Thread(target=tunnel, daemon=True).start()
print("CHECKER_READY", PORT, MODEL, flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
