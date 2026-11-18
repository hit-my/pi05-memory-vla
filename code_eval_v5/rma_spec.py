"""RoboMemArena pilot tasks: prompts (eval_common.TASK_PROMPTS), tracked objects, and subtask parsing.

Objects are the things the arm carries. Each has a text noun for the memory (colour + noun, colour by
order: blue, orange, purple) and a detector prompt. Containers (drawers, microwave, basket) are not tracked;
their steps live in the checklist.
"""
import re

TASKS = {
    1: {"folder": "1_cookies_tomato_basket_dataset",
        "prompt": "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.",
        "objects": [("cookies", ("cookies",), ("cookie box.", "box.")), ("sauce", ("tomato", "sauce"), ("ketchup bottle.", "small bottle.", "bottle."))]},
    4: {"folder": "4_drawer_butter_dataset",
        "prompt": "Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the drawer that already contains an object.",
        "objects": [("butter", ("butter",), ("butter.", "small box."))]},
    8: {"folder": "8_pick_chocolate_in_frypan_pour_tomato_sauce_twice_dataset",
        "prompt": "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
        "objects": [("chocolate", ("chocolate",), ("chocolate bar.", "small box.")), ("sauce", ("tomato", "sauce"), ("ketchup bottle.", "small bottle.", "bottle."))]},
    10: {"folder": "10_pour_wine_bottle_into_mug_dataset",
         "prompt": "Pour wine into the mug twice.",
         "objects": [("bottle", ("wine", "bottle"), ("wine bottle.", "small bottle.", "bottle."))]},
    20: {"folder": "20_cookies_chocolate_microwave_dataset",
         "prompt": "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.",
         "objects": [("cookies", ("cookies",), ("cookie box.", "box.")), ("chocolate", ("chocolate",), ("chocolate bar.", "small box."))]},
    25: {"folder": "25_butter_cream_dataset",
         "prompt": "First pick and place butter from plate1 to plate2, then pick and place cream from plate1 to plate2.",
         "objects": [("butter", ("butter",), ("butter.", "small box.")), ("cream", ("cream",), ("cream cheese.", "small box."))]},
}

# All 26 tasks (full-data run). Prompts are eval_common.TASK_PROMPTS; objects are the carried items in prompt
# order. Keys are matched against the LeRobot subtask texts ("pick up the tomato sauce", ...).
_BOX, _BOTTLE = ("small box.", "box."), ("small bottle.", "bottle.")
_OBJ = {"cookies": ("cookies", ("cookies",), ("cookie box.",) + _BOX), "sauce": ("sauce", ("tomato", "sauce"), ("ketchup bottle.",) + _BOTTLE),
        "butter": ("butter", ("butter",), ("butter.",) + _BOX), "popcorn": ("popcorn", ("popcorn",), ("popcorn box.",) + _BOX),
        "cream": ("cream", ("cream",), ("cream cheese.",) + _BOX), "chocolate": ("chocolate", ("chocolate",), ("chocolate bar.",) + _BOX),
        "bottle": ("bottle", ("wine", "bottle"), ("wine bottle.",) + _BOTTLE), "milk": ("milk", ("milk",), ("milk carton.",) + _BOX),
        "juice": ("juice", ("orange", "juice"), ("juice carton.",) + _BOX)}
_OBJECTS26 = {1: "cookies sauce", 2: "butter popcorn", 3: "cream chocolate", 4: "butter", 5: "butter", 6: "sauce", 7: "sauce",
              8: "chocolate sauce", 9: "butter sauce", 10: "bottle", 11: "cookies butter", 12: "cookies chocolate",
              13: "cookies butter", 14: "cookies chocolate", 15: "butter milk", 16: "milk", 17: "butter chocolate",
              18: "chocolate butter", 19: "sauce milk juice", 20: "cookies chocolate", 21: "butter chocolate",
              22: "sauce cookies", 23: "cream popcorn", 24: "cookies popcorn", 25: "butter cream", 26: "chocolate cream"}
_PROMPTS26 = {
    1: "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.",
    2: "Pick and place butter into the basket, then pick and place popcorn into the same basket.",
    3: "Pick and place cream into the basket, then pick and place chocolate into the same basket.",
    4: "Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the drawer that already contains an object.",
    5: "Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the empty drawer.",
    6: "Pour tomato sauce over cookies twice and place the sauce bottle into the bowl drainer.",
    7: "Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.",
    8: "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    9: "Put butter into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    10: "Pour wine into the mug twice.",
    11: "Put cookies into the top drawer, then put butter into the middle drawer.",
    12: "Put cookies into the middle drawer and then put chocolate into the same drawer.",
    13: "Put cookies into the middle drawer and then put butter into the same drawer.",
    14: "Put cookies into the top drawer and put chocolate into another drawer.",
    15: "Pick and place butter into the frypan, then pour milk over it twice.",
    16: "Pick milk from the table, pour it into the mug twice, then place the milk container into the bowl drainer.",
    17: "Put butter into the middle drawer and then put chocolate into the same drawer.",
    18: "Pick and place chocolate and butter from cabinet1 to cabinet2, respectively.",
    19: "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2, in that order.",
    20: "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.",
    21: "Put butter into the microwave and then put chocolate into the location where the butter was placed.",
    22: "Pour tomato sauce over cookies twice, then put the cookies into the microwave.",
    23: "Put cream into the microwave and then put popcorn into the location where the cream was placed.",
    24: "Put cookies into the microwave and then put popcorn into the location where the cookies were placed.",
    25: "First pick and place butter from plate1 to plate2, then pick and place cream from plate1 to plate2.",
    26: "Pick and place chocolate and cream from plate1 to plate2, respectively."}
for _t in range(1, 27):
    if _t not in TASKS:
        TASKS[_t] = {"folder": None, "prompt": _PROMPTS26[_t], "objects": [_OBJ[o] for o in _OBJECTS26[_t].split()]}
    assert TASKS[_t]["prompt"] == _PROMPTS26[_t], _t
    assert [o[0] for o in TASKS[_t]["objects"]] == _OBJECTS26[_t].split(), _t
# Clicks for the other 20 tasks (rma_clicks.py): the end effector at each object's first grasp, projected with
# the agentview camera and snapped to the changed blob where the object stood (median of 100 seeds); set by eye
# where that failed (cookies in 11-14, task 6 sauce, task 26). Tasks 18 and 19 shuffle the objects on the
# cabinet per seed, so a fixed click there is often the wrong object (known limitation).
_CLICKS20 = ({2: [(125, 145), (96, 144)], 3: [(125, 145), (96, 144)], 5: [(99, 161)], 6: [(122, 132)], 7: [(127, 142)],
               9: [(99, 160), (103, 120)], 11: [(79, 177), (99, 161)], 12: [(79, 177), (87, 119)], 13: [(79, 177), (99, 161)],
               14: [(79, 177), (87, 119)], 15: [(99, 161), (105, 129)], 16: [(105, 126)], 17: [(99, 161), (87, 119)],
               18: [(108, 107), (104, 112)], 19: [(94, 104), (101, 104), (97, 110)], 21: [(114, 169), (99, 164)],
               22: [(98, 137), (100, 169)], 23: [(114, 169), (99, 169)], 24: [(114, 169), (99, 169)], 26: [(95, 165), (104, 167)]})
COLOURS = ("blue", "orange", "purple")
MIN_PIXELS = 20  # these objects are small at 256 px; LIBERO-Mem bowls used 80
MAX_AREA = 900  # detector boxes larger than this are tables, pans or baskets, not the carried object


# First-frame clicks (x, y in the 256-px agentview), one per object, read by eye from seed-100 frames. The layout
# varies only slightly between seeds. This is the only human input; afterwards tracking is RGB only.
CLICKS = {8: [(106, 128), (103, 152)], 10: [(158, 119)], 20: [(114, 169), (99, 164)], 4: [(99, 161)],
          1: [(139, 132), (119, 136)], 25: [(95, 168), (105, 167)]}
# Identities were checked against the end-effector position at the grasp in the demonstrations (proprioception).
CLICKS.update(_CLICKS20)
assert all(len(CLICKS[t]) == len(TASKS[t]["objects"]) for t in range(1, 27))


def detector_objects(task_id):
    if task_id in CLICKS:
        return [{"point": list(p)} for p in CLICKS[task_id]]
    return [{"prompts": list(prompts), "max_area": MAX_AREA} for _, _, prompts in TASKS[task_id]["objects"]]


def registry(task_id):
    """Renderer registry: one entry per tracked object; 'bowl' is the renderer's category for drawn tracks."""
    return {str(k + 1): {"category": "bowl", "segmentation_id": k + 1, "physical_name": noun}
            for k, (noun, _, _) in enumerate(TASKS[task_id]["objects"])}


def names(task_id):
    return [f"{COLOURS[k]} {noun}" for k, (noun, _, _) in enumerate(TASKS[task_id]["objects"])]


def parse_subtask(filename):
    """pick_wine_bottle_0_seed100_task10.hdf5 -> ('pick wine bottle', 0, 100)"""
    m = re.match(r"(.+)_(\d+)_seed(\d+)_task(\d+)\.hdf5$", filename)
    return m.group(1).replace("_", " "), int(m.group(2)), int(m.group(3))


def subtask_object(task_id, text):
    """Index of the object a subtask handles, or None (drawer, microwave)."""
    # the object named first in the text is the one handled ("pour the tomato sauce over the chocolate")
    words = text.lower().split()
    best = None
    for k, (_, keys, _) in enumerate(TASKS[task_id]["objects"]):
        hits = [i for i, w in enumerate(words) if w in keys]
        if hits and (best is None or hits[0] < best[0]):
            best = (hits[0], k)
    return None if best is None else best[1]
