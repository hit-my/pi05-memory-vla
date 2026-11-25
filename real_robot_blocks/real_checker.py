"""The vision-language checker used by real_agent.py: one question about one claimed step, answered from three images.

It is an off-the-shelf model behind an OpenAI-compatible chat endpoint. It is not trained on this task and sees nothing
of the policy. Configuration comes from the environment only (never put a key in a file or on a command line):
    CHECKER_API_URL   chat-completions URL of the service
    CHECKER_MODEL     model name
    CHECKER_API_KEY   key
    CHECKER_EXTRA     optional JSON merged into the request body (e.g. to switch a "thinking" mode off)

    python real_checker.py --case CASE.json      # one saved case (see --save-dir of real_deploy_agent.py): prints the answer

The answer is P(step done) for a tick, and P(a pink block is visible where the cup stood) for an observation.
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

HEAD = ("You are a visual checker for a robot's task checklist. The image shows three camera views side by side, labelled at the top: "
        "left = top camera at an earlier reference time, middle = top camera NOW, right = wrist camera NOW (camera on the gripper; the gripper fingers are at the edge of that view). "
        "Scene: a robot arm at a table, seen from above. On the right, three upturned cups stand in a column: the FAR cup is the highest in the image, the MIDDLE cup is in the middle, "
        "the NEAR cup is the lowest. On the left, three numbered blocks (2, 1, 0 from top to bottom) and a plate. A small pink block may lie under a cup; it is visible only while that cup is lifted.\n")
CRITERIA = {
    "lift": "\"lift the X cup\" is done only if that cup is held by the gripper and off the table: in the top view it is no longer standing at its place in the column, and the wrist view shows the cup between the fingers. If the cup still stands where it was, or the gripper closed beside or above it, it is NOT done.",
    "put": "\"put the X cup back\" is done only if that cup stands upside down at its place in the column again and the gripper has let go of it. If the gripper still holds the cup, or the cup lies somewhere else, it is NOT done.",
    "pick": "\"pick up block N\" is done only if the block with that number is held by the gripper and off the table: its place in the column of blocks is empty and the wrist view shows a block between the fingers. If the block still stands where it was, or a block with a different number was taken, it is NOT done.",
    "place": "\"place the block on the plate\" is done only if a numbered block now lies on the plate and the gripper has let go of it. If the gripper still holds the block, or the plate is empty, it is NOT done.",
}
ANSWER = ("Important: the robot's claim is usually right. Answer \"not_done\" ONLY if the NOW views show clear, positive evidence that the step did not happen. "
          "If the views are ambiguous or the relevant part is hidden by the arm, answer \"done\".\n"
          "Answer with JSON only, fields in this order: {\"what_changed\": \"what moved between the left view and now\", \"gripper_now\": \"empty\"/\"holding_object\"/\"unclear\", "
          "\"evidence\": \"one short sentence\", \"verdict\": \"done\" or \"not_done\", \"confidence\": 0.5-1.0}")


def tick_prompt(case):
    verb = case["step"].split()[0]
    return (HEAD + f"Overall task: \"{case['instruction']}\".\nThe robot's checklist before this claim ([x] = already finished, [ ] = not yet; \"?\" = not decided yet):\n"
            + "\n".join(case["checklist"]) +
            f"\nThe robot's own memory has just claimed that step {case['k']} is finished: \"{case['step']}\". "
            f"The left view shows the scene when the robot started working on this step, {case['seconds']:.0f} seconds ago.\n"
            f"Criterion: {CRITERIA.get(verb, 'Decide from the visible outcome.')}\n" + ANSWER)


def observation_prompt(case):
    place = case["step"].split(" cup:")[0]
    return (HEAD + f"The robot has lifted the {place.upper()} cup to look under it and has written in its memory: \"{case['step']}\".\n"
            "Look at the NOW views, at the spot on the table where that cup stood. Is a small pink block lying there?\n"
            "If the spot is hidden by the arm or by the cup, answer \"unclear\".\n"
            "Answer with JSON only, fields in this order: {\"spot_visible\": \"yes\"/\"no\", \"evidence\": \"one short sentence\", "
            "\"pink_block\": \"yes\"/\"no\"/\"unclear\", \"confidence\": 0.5-1.0}")


def compose(case, side=336):
    """The three views in one image, each with its label."""
    names = (("ref", "top camera, earlier"), ("now", "top camera, NOW"), ("wrist", "wrist camera, NOW"))
    sheet = Image.new("RGB", (3 * side + 16, side + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (key, label) in enumerate(names):
        view = Image.fromarray(np.ascontiguousarray(case[key], np.uint8)).convert("RGB").resize((side, side), Image.BICUBIC)
        sheet.paste(view, (i * (side + 8), 22))
        draw.text((i * (side + 8) + 4, 5), label, fill="black")
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def probability(kind, answer):
    confidence = min(1.0, max(0.5, float(answer.get("confidence", 0.5))))
    if kind == "observation":
        seen = str(answer.get("pink_block", "unclear")).lower()
        return 0.5 if seen not in ("yes", "no") else (confidence if seen == "yes" else 1.0 - confidence)
    return confidence if str(answer.get("verdict", "done")).lower() == "done" else 1.0 - confidence


class VLMChecker:
    def __init__(self, save_dir=None, attempts=4, timeout=60.0):
        self.url, self.model = os.environ["CHECKER_API_URL"], os.environ["CHECKER_MODEL"]
        self.key = os.environ["CHECKER_API_KEY"]
        self.extra = json.loads(os.environ.get("CHECKER_EXTRA", "{}"))
        self.save_dir, self.attempts, self.timeout, self.count = save_dir, attempts, timeout, 0
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def judge(self, case):
        prompt = observation_prompt(case) if case["kind"] == "observation" else tick_prompt(case)
        image = compose(case)
        body = {"model": self.model, "max_tokens": 400, "temperature": 0, **self.extra,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
                    {"type": "text", "text": prompt}]}]}
        started, error = time.time(), None
        for attempt in range(self.attempts):
            try:
                request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                    text = json.loads(reply.read())["choices"][0]["message"]["content"]
                answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
                p = probability(case["kind"], answer)
                self._save(case, image, prompt, answer, p, time.time() - started)
                return p
            except Exception as failure:
                error = failure
                time.sleep(1.0 + 2.0 * attempt)
        raise RuntimeError(f"checker failed after {self.attempts} attempts: {error!r}")

    def _save(self, case, image, prompt, answer, p, seconds):
        if not self.save_dir:
            return
        self.count += 1
        stem = os.path.join(self.save_dir, f"{self.count:04d}_{case['kind']}_item{case['k']}")
        with open(stem + ".jpg", "wb") as f:
            f.write(image)
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump({"kind": case["kind"], "k": case["k"], "step": case["step"], "checklist": case["checklist"], "seconds": case["seconds"],
                       "instruction": case["instruction"], "prompt": prompt, "answer": answer, "p": p, "checker_seconds": round(seconds, 2)}, f, ensure_ascii=False, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="a .json saved by --save-dir (its .jpg must lie next to it)")
    args = parser.parse_args()
    saved = json.load(open(args.case, encoding="utf-8"))
    sheet = np.asarray(Image.open(args.case[:-5] + ".jpg").convert("RGB"))
    side = (sheet.shape[1] - 16) // 3
    views = [sheet[22:22 + side, i * (side + 8):i * (side + 8) + side] for i in range(3)]
    case = {**{k: saved[k] for k in ("kind", "k", "step", "checklist", "seconds", "instruction")}, "ref": views[0], "now": views[1], "wrist": views[2]}
    print("p =", VLMChecker().judge(case))


if __name__ == "__main__":
    main()
