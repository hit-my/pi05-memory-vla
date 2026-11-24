"""policy_agent.py (study, evaluation side only): an external checker maintains the ledger next to the policy.

A separate vision-language model (checker_server.py, no simulator state, not trained on this benchmark) is asked
  * at every tick the policy writes ("Subgoal:done k"): has step k really been completed?  (step-start image, image now, wrist now)
  * every STUDY_AGENT_PERIOD policy calls without a tick: is the most recently ticked step still true?
A negative verdict removes the tick of that step (and of any later step) from the memory the policy reads, and sets
Task:Incomplete, so the policy sees the step as not done and can redo it.

Asynchrony: the checker does not block the policy or the arm. The simulation is not real time, so the latency is modelled
in policy calls: a verdict asked at call n takes effect in the memory of call n + STUDY_AGENT_DELAY (one call = 20 control
steps). Exception: when the policy declares the whole task complete, the stop waits for the verdict of that last tick.

Env: STUDY_AGENT=1 enables; STUDY_AGENT_URL (default http://127.0.0.1:18777); STUDY_AGENT_TH tick threshold on P(yes);
STUDY_AGENT_TH_AUDIT periodic threshold; STUDY_AGENT_PERIOD (calls, 0 = no periodic check); STUDY_AGENT_DELAY (calls);
STUDY_AGENT_LOG directory for per-process jsonl logs. Imported by policy_ens.py of this code copy."""
import base64, io, json, os, re, time, urllib.request

import numpy as np
from PIL import Image

import policy as base

ON = os.environ.get("STUDY_AGENT") == "1"
URL = os.environ.get("STUDY_AGENT_URL", "http://127.0.0.1:18777")
TH = float(os.environ.get("STUDY_AGENT_TH", "0.5")); TH_AUDIT = float(os.environ.get("STUDY_AGENT_TH_AUDIT", "0.3"))
PERIOD = int(os.environ.get("STUDY_AGENT_PERIOD", "8")); DELAY = int(os.environ.get("STUDY_AGENT_DELAY", "2"))
LOGDIR = os.environ.get("STUDY_AGENT_LOG")
WAIT = int(os.environ.get("STUDY_AGENT_WAIT", "0"))  # policy calls between a tick and its (first) check
TWOLOOK = set(filter(None, os.environ.get("STUDY_AGENT_TWOLOOK", "").split(",")))  # step types that need a second negative look LAG2 calls later before a tick is removed
LAG2 = int(os.environ.get("STUDY_AGENT_LAG2", "3"))
VERBS = set(filter(None, os.environ.get("STUDY_AGENT_VERBS", "").split(",")))  # step types to check (empty = all)
# --- physical recovery after a tick is removed (policy_agent5) ---
#   STUDY_REC=""     ledger only (as policy_agent3)
#   STUDY_REC=open   the wrapper opens the gripper for one chunk (arm still), then the policy continues
#   STUDY_REC=back   opens the gripper, then drives the arm back along its own recorded path (end-effector poses of the
#                    earlier policy calls, no simulator state) to the pose STUDY_REC_BACK calls before the false tick,
#                    not earlier than the start of that step, and gives the policy the memory it had there
# Only for "pick"/"open" steps; a "pick" whose fingers are neither fully closed nor open (may hold the object) is left alone.
REC = os.environ.get("STUDY_REC", ""); REC_BACK = int(os.environ.get("STUDY_REC_BACK", "5")); REC_GAIN = float(os.environ.get("STUDY_REC_GAIN", "1.5"))
REC_VERBS = {"pick", "open"}
REC_ANCHOR = os.environ.get("STUDY_REC_ANCHOR", "tick")  # tick | close
_infer = base.JointMemoryPolicy.infer
S = {}  # per-process episode state (one policy server = one evaluation worker = one episode at a time)


def _log(**kw):
    if LOGDIR:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, f"{os.getpid()}.jsonl"), "a") as f: f.write(json.dumps(kw) + "\n")


def _b64(rgb):
    buf = io.BytesIO(); Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


def _ask(instruction, step, ref, now, wrist, **extra):
    body = json.dumps({"instruction": instruction, "step": step, "ref": _b64(ref), "now": _b64(now), "wrist": _b64(wrist), **extra}).encode()
    t0 = time.time()
    for attempt in range(4):  # the checker may sit behind a tunnel that drops for a moment
        try:
            with urllib.request.urlopen(urllib.request.Request(URL + "/judge", body, {"Content-Type": "application/json"}), timeout=120) as r:
                return float(json.loads(r.read())["p_yes"]), time.time() - t0
        except Exception:
            if attempt == 3: raise
            time.sleep(5)


def _items(memory):
    line = next((l for l in memory.split("\n") if l.startswith("Subgoal:")), "")
    return {int(k): (mark == "x", text.strip()) for mark, k, text in re.findall(r"\[([ x])\](\d+) ([^;\n]*)", line)}


def _lines(memory):
    line = next((l for l in memory.split("\n") if l.startswith("Subgoal:")), "")[8:]
    return [x.strip() for x in line.split(";")]


def _untick(memory, k):
    lines = memory.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("Subgoal:"): lines[i] = re.sub(r"\[x\](\d+) ", lambda m: ("[ ]" if int(m.group(1)) >= k else "[x]") + m.group(1) + " ", l)
        elif l.startswith("Task:"): lines[i] = "Task:Incomplete"
    return "\n".join(lines)


def _start_recovery(k, tick_call, n, st, prev):
    step = _items(prev).get(k, (False, ""))[1]; verb = step.split()[0] if step else ""; gap = float(abs(st[6] - st[7]))
    if verb not in REC_VERBS: _log(event="rec_skip", call=n, item=k, step=step, why="step type", gap=round(gap, 4)); return
    if verb == "pick" and 0.02 <= gap <= 0.065: _log(event="rec_skip", call=n, item=k, step=step, why="fingers may hold something", gap=round(gap, 4)); return
    tgt, way, hold = None, [], None
    if REC == "back":
        start_call = max(1, S["ref_call"].get(k, 0) + 1)
        anchor = tick_call
        if REC_ANCHOR == "close":  # count back from the call at which the fingers started to close (finger opening fell by > 1 cm), not from the tick, which follows it by a varying delay
            gaps = {i: abs(s[6] - s[7]) for i, s in S["traj"].items()}
            anchor = next((j for j in range(min(tick_call, n), start_call, -1) if j in gaps and j - 1 in gaps and gaps[j - 1] - gaps[j] > 0.01), None)
            if anchor is None: anchor = tick_call - 3
        tgt = max(start_call, anchor - REC_BACK)
        if tgt < n - 1:
            way = [i for i in range(n - 2, tgt, -2) if i in S["traj"]] + [tgt]; hold = _untick(S["mem"][tgt], k) if isinstance(S["mem"].get(tgt), str) else None
    S["rec"] = {"k": k, "phase": "open", "way": way, "tgt": tgt, "calls": 0, "hold": hold, "max": 2 * len(way) + 6}
    _log(event="rec_start", call=n, item=k, step=step, mode=REC, anchor=REC_ANCHOR, gap=round(gap, 4), tick_call=tick_call, target_call=tgt, waypoints=len(way))


def _recover_step(result, st, n, prev, raw_rgb):
    from scipy.spatial.transform import Rotation as R
    rec = S["rec"]; rec["calls"] += 1; shape = np.asarray(result["actions"]).shape
    a = np.zeros(shape, np.float32); a[:, 6] = -1.0; done = False; dist = None
    if rec["phase"] == "open":
        rec["phase"] = "move"; done = not rec["way"]
    else:
        way = rec["way"]
        while len(way) > 1 and np.linalg.norm(S["traj"][way[0]][:3] - st[:3]) < 0.03: way.pop(0)
        wp = S["traj"][way[0]]; d = wp[:3] - st[:3]; dist = float(np.linalg.norm(d)); steps = shape[0]
        a[:, :3] = np.clip(d / (0.05 * steps) * REC_GAIN, -1, 1)
        rv = (R.from_rotvec(wp[3:6]) * R.from_rotvec(st[3:6]).inv()).as_rotvec()
        a[:, 3:6] = np.clip(rv / (0.5 * steps) * REC_GAIN, -0.5, 0.5)
        done = (len(way) == 1 and dist < 0.03) or rec["calls"] > rec["max"]
    out = {**result, "actions": a, "next_memory_text": rec["hold"] or prev, "memory_valid": True, "sequence_complete": False}
    if done:
        _log(event="rec_end", call=n, item=rec["k"], calls=rec["calls"], final_dist=None if dist is None else round(dist, 3), gap=round(float(abs(st[6] - st[7])), 4))
        S["rec"] = None; S["ref"] = np.asarray(raw_rgb).copy(); S["last_tick_call"] = n; S["checks"] = []; S["pending"] = []; S["last_k"] = None
    return out


def infer(self, main_rgb, wrist_rgb, raw_rgb, state, instruction, previous_memory_text, previous_latents=None):
    if not ON: return _infer(self, main_rgb, wrist_rgb, raw_rgb, state, instruction, previous_memory_text, previous_latents)
    prev = previous_memory_text
    if not isinstance(prev, str) or "[" not in prev.split("\n")[1 if prev.count("\n") else 0]:  # no checklist yet: a new episode
        episode = S.get("episode", 0) + 1
        S.clear(); S.update(n=0, ref=np.asarray(raw_rgb).copy(), ref_of={}, ref_call={}, checks=[], last_ok_call=0, pending=[], last_tick_call=0, last_k=None, episode=episode, traj={}, mem={}, rec=None)
        _log(event="episode", episode=episode, instruction=instruction)
    S["n"] += 1; n = S["n"]
    st = np.asarray(state, np.float64).reshape(-1); S["traj"][n] = st.copy(); started = None
    try:
        due = [p for p in S["pending"] if p[0] <= n]; S["pending"] = [p for p in S["pending"] if p[0] > n]
        for _, k, why, tick_call in sorted(due, key=lambda p: p[1]):
            if _items(prev).get(k, (False, ""))[0]:
                prev = _untick(prev, k); S["ref"] = S["ref_of"].get(k, S["ref"]); S["last_k"] = None; S["checks"] = [c for c in S["checks"] if c["k"] < k]
                _log(event="untick", call=n, item=k, why=why)
                if started is None: started = (k, tick_call)
        S["mem"][n] = prev
        if started and REC and not S["rec"]: _start_recovery(started[0], started[1], n, st, prev)
        if S["rec"]: prev = _untick(prev, S["rec"]["k"])  # ticks written while the wrapper is moving the arm do not count
    except Exception as error: _log(event="error", where="apply", call=n, error=repr(error))
    result = _infer(self, main_rgb, wrist_rgb, raw_rgb, state, instruction, prev, previous_latents)
    if S.get("rec"):
        try: return _recover_step(result, st, n, prev, raw_rgb)
        except Exception as error:
            _log(event="error", where="recover", call=n, error=repr(error)); S["rec"] = None
    try:
        text = result.get("memory_text") or ""; sub = next((l for l in text.split("\n") if l.startswith("Subgoal:")), "")[8:]
        ticked_now = [int(x) for x in re.findall(r"\d+", sub.partition(" reveal ")[0])] if result.get("memory_valid") is True and sub.startswith("done") else []
        revealed = {int(k): t.strip() for k, t in re.findall(r"(\d+)=([^;]*)", sub.partition(" reveal ")[2])}
        items = _items(prev)
        for k in ticked_now:
            S["last_k"] = None  # a newer tick: the older step is no longer the one to audit
            step = revealed.get(k) or items.get(k, (False, ""))[1]
            if not step or step == "?" or "drawer:" in step: continue  # observation items are not physical steps
            if VERBS and step.split()[0] not in VERBS: continue        # step types the checker is not trusted on are left alone
            S["checks"].append({"k": k, "step": step, "ref": S["ref"], "tick_call": n, "due": n + WAIT, "ref_call": S["last_tick_call"], "checklist": _lines(prev)})
        if ticked_now: S["ref"] = np.asarray(raw_rgb).copy(); S["last_tick_call"] = n
        now_items = _items(result.get("next_memory_text") or prev)
        for chk in [c for c in S["checks"] if n >= c["due"]]:  # look WAIT calls after the tick: the policy ticks when the gripper closes, the outcome follows
            S["checks"].remove(chk); k, step = chk["k"], chk["step"]
            if not now_items.get(k, (False, ""))[0]: continue
            p, sec = _ask(instruction, step, chk["ref"], raw_rgb, wrist_rgb, k=k, checklist=chk["checklist"], seconds=n - chk["ref_call"])
            S["ref_of"][k] = chk["ref"]; S["ref_call"][k] = chk["ref_call"]
            _log(event="tick_check", call=n, tick_call=chk["tick_call"], item=k, step=step, p_yes=p, seconds=round(sec, 2), reject=p < TH, second=bool(chk.get("second")))
            if p < TH and step.split()[0] in TWOLOOK and not chk.get("second"): S["checks"].append({**chk, "due": n + LAG2, "second": True})  # e.g. a drawer opened to look inside is shut again a moment later: done at either look counts
            elif p < TH: S["pending"].append((n + DELAY, k, f"tick p={p:.2f}", chk["tick_call"]))
            elif not any(c["k"] > k for c in S["checks"]) and max(now_items and [i for i, v in now_items.items() if v[0]] or [0]) == k: S["last_k"] = (k, step); S["last_ok_call"] = n
        if PERIOD and S["last_k"] and not S["pending"] and not S["checks"] and n > S["last_ok_call"] and (n - S["last_ok_call"]) % PERIOD == 0:
            k, step = S["last_k"]; p, sec = _ask(instruction, step, S["ref_of"].get(k, S["ref"]), raw_rgb, wrist_rgb, k=k, checklist=_lines(_untick(prev, k)), seconds=n - S["ref_call"].get(k, 0))
            _log(event="audit", call=n, item=k, step=step, p_yes=p, seconds=round(sec, 2), reject=p < TH_AUDIT)
            if p < TH_AUDIT: S["pending"].append((n + DELAY, k, f"audit p={p:.2f}", n))
        if result.get("sequence_complete") is True and (S["checks"] or S["pending"]):
            result = {**result, "sequence_complete": False}; _log(event="hold_stop", call=n)  # the stop waits for the verdict
    except Exception as error:  # the checker must never end an episode
        _log(event="error", where="check", call=n, error=repr(error))
    return result


base.JointMemoryPolicy.infer = infer
