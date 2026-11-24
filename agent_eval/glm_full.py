"""glm_full.py ROOT VARIANT [REPEAT=1]: run GLM 5.3 Flash (no thinking) on the full offline set (230 tick events with known truth)
and cache the raw answers in ROOT/agent/full_<VARIANT>_r<i>.json. Scoring is done separately (glm_score.py) so rules can be
iterated without new API calls. Variants are prompt builders in PROMPTS."""
import base64, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

R, VAR = sys.argv[1], sys.argv[2]; REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 1
KEY = os.environ.get("GLM_KEY", ""); SET = os.environ.get("CASE_SET", "full_set")
try: man = json.load(open(R + "\\" + SET + r"\manifest.json", encoding="utf-8"))
except OSError: man = []  # imported for its prompts only (checker services)
HEAD = ("You are a visual checker for a robot's task checklist. The image shows three simulator views side by side, labelled at the top: "
        "left = front camera at an earlier reference time, middle = front camera NOW, right = wrist camera NOW (camera on the gripper; the two dark blocks at the bottom are the gripper fingers). "
        "Scene: a Franka arm at a table with objects such as cartons, bottles, pans, baskets, mugs, a drawer cabinet, a microwave. Low resolution.\n")
GLOSS = ("Objects in this simulator: milk = red-and-white carton with the word \"Milk\"; orange juice = carton with an orange picture; tomato sauce = small red can/bottle with a tomato label; "
         "butter = small yellow box; cookies = small box; chocolate (pudding) = small brown box; wine = tall dark bottle. The drawer cabinet has three drawers (top, middle, bottom): tell them apart by height. "
         "The task names a specific object: a different object being handled does NOT complete the step.\n")
CRIT = {"pick": "\"pick up X\" is done only if X itself is held between the gripper fingers and lifted off its support: in the wrist view X fills the space between the two fingers, and in the front view X is no longer standing where it was in the left image. If X is still standing at its old place, or the gripper is merely above/next to it, it is NOT done.",
        "place": "\"place X into/on Y\" is done only if X is now resting in/on Y and the gripper has let go. If X is still in the gripper, still at its old place, or a different object was put there, it is NOT done.",
        "open": "\"open ...\" is done only if that specific drawer/door is clearly open in the NOW views (drawer pulled out by a visible amount; microwave door swung wide open, not just touched or slightly ajar). Check that it is the named drawer (top/middle/bottom).",
        "close": "\"close ...\" is done only if that specific drawer/door is fully shut in the NOW views.",
        "pour": "\"pour X ... (first/second time)\" cannot be seen directly in one frame. Count it as done if X is held by the gripper at or near the target container (the pouring motion happens between the two front views); count it as NOT done only if X is not in the gripper or is far from the container."}
FIELDS = {"pick": "\"target\": \"the object to pick\", \"target_still_at_old_place\": \"yes\"/\"no\"/\"unclear\" (is the target still standing where it was in the left view?), "
                  "\"held_in_gripper\": \"target\"/\"other_object\"/\"nothing\"/\"unclear\" (what is between the fingers in the wrist view?), \"gripper_near_target\": \"yes\"/\"no\"",
          "place": "\"target\": \"the object to place\", \"destination\": \"where it must go\", \"target_at_destination\": \"yes\"/\"no\"/\"unclear\", "
                   "\"held_in_gripper\": \"target\"/\"other_object\"/\"nothing\"/\"unclear\", \"object_at_destination_is\": \"target\"/\"other_object\"/\"nothing\"/\"unclear\"",
          "open": "\"target\": \"the drawer/door to open\", \"target_state_then\": \"open\"/\"closed\"/\"unclear\" (left view), \"target_state_now\": \"wide_open\"/\"slightly_open\"/\"closed\"/\"unclear\", "
                  "\"other_part_open\": \"yes\"/\"no\" (is a different drawer/door open instead?)",
          "close": "\"target\": \"the drawer/door to close\", \"target_state_now\": \"open\"/\"closed\"/\"unclear\"",
          "pour": "\"target\": \"the object to pour\", \"held_in_gripper\": \"target\"/\"other_object\"/\"nothing\"/\"unclear\", \"near_container\": \"yes\"/\"no\"/\"unclear\""}


def base_text(c):
    verb = c["step"].split()[0]
    return (HEAD + GLOSS + f"Overall task: \"{c['instruction']}\".\nThe robot's checklist before this claim ([x] = already finished, [ ] = not yet; \"?\" = not decided yet):\n" + "\n".join(c["checklist"]) +
            f"\nThe robot's own memory has just claimed that step {c['k']} is finished: \"{c['step']}\". The left view shows the scene when the robot started working on this step, {c['seconds']:.0f} seconds ago.\n"
            f"Criterion: {CRIT.get(verb, 'Decide from the visible outcome.')}\n"), verb


def p_info_obs(c):
    t, _ = base_text(c)
    return t + ("Answer with JSON only, fields in this order: {\"target\": \"which object/drawer/door this step is about\", \"target_then\": \"where/how it is in the left view\", "
                "\"target_now\": \"where/how it is in the NOW views\", \"gripper_now\": \"what is between the fingers in the wrist view\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


def p_attr(c):
    t, verb = base_text(c)
    return t + ("First report what you see as fixed fields, then the verdict. Use \"unclear\" whenever the image does not show it clearly; do not guess.\n"
                "Answer with JSON only, fields in this order: {" + FIELDS.get(verb, "\"target\": \"...\"") + ", \"evidence\": \"one short sentence\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


def p_strict(c):
    t, verb = base_text(c)
    return t + ("Important: the robot's claim is usually right. Answer \"not_done\" ONLY if the NOW views show clear, positive evidence that the step did not happen "
                "(for example the target object visibly still standing at its old place with an empty gripper, or the named drawer/door visibly still shut). "
                "If the views are ambiguous, too dark, or the target is hidden, answer \"done\".\n"
                "Answer with JSON only, fields in this order: {" + FIELDS.get(verb, "\"target\": \"...\"") + ", \"evidence\": \"one short sentence\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


GLOSS2 = ("Do NOT identify objects by colour or by what you think a product looks like: the packages in this simulator do not look like the real products, and you do not know which box is which. "
          "Identify the step's target by change: it is the object (or drawer/door) that the gripper was working on, i.e. the thing that moved or changed between the left view and the NOW views. "
          "The drawer cabinet has three drawers (top, middle, bottom); the wrist view alone does not tell which one is open.\n")


def p_v4(c):
    verb = c["step"].split()[0]
    t = (HEAD + GLOSS2 + f"Overall task: \"{c['instruction']}\".\nThe robot's checklist before this claim ([x] = already finished, [ ] = not yet; \"?\" = not decided yet):\n" + "\n".join(c["checklist"]) +
         f"\nThe robot's own memory has just claimed that step {c['k']} is finished: \"{c['step']}\". The left view shows the scene when the robot started working on this step, {c['seconds']:.0f} seconds ago.\n"
         f"Criterion: {CRIT.get(verb, 'Decide from the visible outcome.')}\n")
    return t + ("Important: the robot's claim is usually right. Answer \"not_done\" ONLY if the NOW views show clear, positive evidence that the step did not happen "
                "(for example: nothing at all has moved since the left view and the gripper is empty; or the drawer/door the gripper is at is visibly still shut). "
                "If the views are ambiguous, too dark, or the relevant part is hidden, answer \"done\". Never answer \"not_done\" because an object's colour or label looks wrong.\n"
                "Answer with JSON only, fields in this order: {\"what_changed\": \"what moved or changed between the left view and now\", \"gripper_now\": \"empty\"/\"holding_object\"/\"unclear\", "
                "\"evidence\": \"one short sentence\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


PROMPTS = {"info_obs": p_info_obs, "attr": p_attr, "strict": p_strict, "v4": p_v4}

def p_v5(c):
    """step-type specific: pick -> v4; open / place -> dedicated neutral prompts; close / pour are not checked (returns None)"""
    verb = c["step"].split()[0]
    if verb == "pick": return p_v4(c)
    if verb not in ("open", "place"): return None
    t = (HEAD + GLOSS2 + f"Overall task: \"{c['instruction']}\".\nThe robot's checklist before this claim ([x] = already finished, [ ] = not yet; \"?\" = not decided yet):\n" + "\n".join(c["checklist"]) +
         f"\nThe robot's own memory has just claimed that step {c['k']} is finished: \"{c['step']}\". The left view shows the scene when the robot started working on this step, {c['seconds']:.0f} seconds ago.\n")
    if verb == "open":
        return t + ("How to judge an \"open\" step:\n"
                    "- Drawer: in the front view an open drawer sticks out of the cabinet front as a box-shaped tray (the cabinet outline is no longer a flat block); in the wrist view you look down into the tray and see its inside (often with a small object in it). A closed cabinet shows only flat dark fronts with handles. Compare the cabinet in the left view and in the NOW front view: has its outline changed?\n"
                    "- Microwave door: when open, the door panel stands out from the microwave body like a wall at an angle and the bright/hollow inside of the oven is visible; when closed the front is one flat face. A hand merely touching the handle is NOT open.\n"
                    "- It also counts as NOT done if the part that is open is a different one from the one named (for drawers judge top/middle/bottom by height in the front view only).\n"
                    "Report honestly; use \"unclear\" when the views do not show it.\n"
                    "Answer with JSON only, fields in this order: {\"target\": \"the drawer/door named in the step\", \"front_view_changed\": \"yes\"/\"no\"/\"unclear\" (did the cabinet or microwave outline change between left and NOW front views?), "
                    "\"target_state_now\": \"clearly_open\"/\"slightly_open\"/\"closed\"/\"unclear\", \"open_part\": \"top\"/\"middle\"/\"bottom\"/\"microwave\"/\"none\"/\"unclear\" (which part is open now), "
                    "\"evidence\": \"one short sentence\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")
    return t + ("How to judge a \"place X into/on Y\" step: find the object the gripper was carrying when this step started (in the left view it is in or near the gripper). The step is done only if that object now rests in/on Y and the gripper has let go of it. "
                "It is NOT done if the gripper still holds it, if it was dropped somewhere else, or if the gripper never carried anything and nothing new has appeared in/on Y.\n"
                "Report honestly; use \"unclear\" when the views do not show it.\n"
                "Answer with JSON only, fields in this order: {\"destination\": \"Y\", \"carried_object_then\": \"what the gripper carried in the left view (or nothing)\", \"new_object_at_destination\": \"yes\"/\"no\"/\"unclear\" (is there now an object in/on Y that was not there in the left view?), "
                "\"gripper_now\": \"empty\"/\"holding_object\"/\"unclear\", \"evidence\": \"one short sentence\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


PROMPTS["v5"] = p_v5
def p_info_obs2(c):
    """describe-first prompt without the (wrong) appearance glossary: objects are identified by what changed"""
    verb = c["step"].split()[0]
    t = (HEAD + GLOSS2 + f"Overall task: \"{c['instruction']}\".\nThe robot's checklist before this claim ([x] = already finished, [ ] = not yet; \"?\" = not decided yet):\n" + "\n".join(c["checklist"]) +
         f"\nThe robot's own memory has just claimed that step {c['k']} is finished: \"{c['step']}\". The left view shows the scene when the robot started working on this step, {c['seconds']:.0f} seconds ago.\n"
         f"Criterion: {CRIT.get(verb, 'Decide from the visible outcome.')}\n")
    return t + ("Answer with JSON only, fields in this order: {\"target\": \"which object/drawer/door this step is about\", \"target_then\": \"where/how it is in the left view\", "
                "\"target_now\": \"where/how it is in the NOW views\", \"gripper_now\": \"what is between the fingers in the wrist view\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


PROMPTS["info_obs2"] = p_info_obs2

def ask(c, temperature):
    if PROMPTS[VAR](c) is None: return {"id": c["id"], "verdict": "skip"}
    img = base64.b64encode(open(R + "\\" + SET + rf"\cases\{c['id']}.jpg", "rb").read()).decode()
    body = {"model": "glm-5.3-flash", "max_tokens": 500, "temperature": temperature, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}, {"type": "text", "text": PROMPTS[VAR](c)}]}]}
    err = ""
    for attempt in range(5):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://open.bigmodel.cn/api/paas/v4/chat/completions", json.dumps(body).encode(), {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r: d = json.loads(r.read())
            txt = d["choices"][0]["message"]["content"]; j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            return {"id": c["id"], **j, "seconds": round(time.time() - t0, 2)}
        except Exception as e: err = repr(e)[:200]; time.sleep(2 + 3 * attempt)
    return {"id": c["id"], "verdict": "error", "error": err}


if __name__ == "__main__":
    for i in range(REPEAT):
        path = R + rf"\agent\{SET}_{VAR}_r{i}.json"
        if os.path.exists(path): continue
        with ThreadPoolExecutor(5) as ex: out = list(ex.map(lambda c: ask(c, 0 if i == 0 else 0.7), man))
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(VAR, "run", i, "errors", sum(o["verdict"] == "error" for o in out), "median seconds", sorted(o.get("seconds", 0) for o in out)[len(out) // 2], flush=True)
