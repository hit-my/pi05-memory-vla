"""Constants for the real-robot dataset wwccww/yx_blocks (bimanual Dobot, only the right arm moves; ACT-style hdf5).

Task: three upturned cups stand in a column; under some of them is a small pink block. The robot lifts each cup in
turn (far, middle, near), looks, puts it back, then picks the numbered block (0, 1 or 2) equal to the number of pink
blocks it saw and places it on the plate. What was under a cup is hidden again once the cup is back, so the count
has to be carried in memory.

Everything a deployment must reproduce exactly is defined here, once (same layout as real_robot/real_spec.py).
"""
import numpy as np

TASK_ID = 28  # RoboMemArena uses 1..26, yx_bowls is 27
PROMPT = "Lift each cup to check for a block under it, then place the block whose number equals the count on the plate."
FPS = 30

# ---- images -------------------------------------------------------------------------------------------------
# top camera 640x480 -> the right 480x480 (numbered blocks, plate, cups; the idle left gripper is cut off) -> 256 -> 224
TOP_CROP = (160, 0, 480)        # x0, y0, side in the 640x480 "top" image
WRIST_CROP = (80, 0, 480)       # right wrist camera: centre square
SOURCE = 256


def crop_resize(frame, crop):
    """uint8 [480,640,3] camera frame -> uint8 [256,256,3]. The ONE function used for the dataset and at deployment."""
    from PIL import Image
    x0, y0, side = crop
    patch = np.ascontiguousarray(np.asarray(frame, np.uint8)[y0:y0 + side, x0:x0 + side])
    if patch.shape != (side, side, 3):
        raise ValueError(f"camera frame {np.asarray(frame).shape} does not contain crop {crop}")
    return np.asarray(Image.fromarray(patch).resize((SOURCE, SOURCE), Image.BILINEAR))


def model_image(rgb256):
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb256, np.uint8)).resize((224, 224), Image.BILINEAR))

# ---- state / action -----------------------------------------------------------------------------------------
# hdf5 order (observations/qpos and action): left_joint_0..5, left_gripper, right_joint_0..5, right_gripper.
RIGHT = slice(7, 14)
GRIPPER = 13
# The gripper READING is about 0.94 open, 0.67 around a cup, 0.55 around a numbered block, so the grasps are cut from
# the gripper COMMAND (action[13]: about 0.93 open, 0 closed), which is unambiguous. Labels only; not used at deployment.
COMMAND_CLOSED_BELOW = 0.4


def state8(state14):
    s = np.asarray(state14, np.float32)
    return np.concatenate([s[..., RIGHT], s[..., GRIPPER:GRIPPER + 1]], axis=-1)


def action7(action14):
    return np.asarray(action14, np.float32)[..., RIGHT]


def action14(action7_row, left_hold):
    return np.concatenate([np.asarray(left_hold, np.float32), np.asarray(action7_row, np.float32)])


# ---- objects ------------------------------------------------------------------------------------------------
# Tracked objects, in this order (= entity slots of the model, so the training code needs MAX_ENTITIES = 6):
#   0 far cup, 1 middle cup, 2 near cup   (top to bottom in the cropped top image = the order they are lifted)
#   3 block "0", 4 block "1", 5 block "2" (the numbered blocks stand in a column: 2 on top, 1, 0 at the bottom)
# As everywhere in this code base the Action line names an object by the colour of its drawn track (renderer palette:
# blue, orange, purple, green, magenta, ochre), not by its real colour.
OBJECTS = ("cup", "cup", "cup", "block", "block", "block")
COLOURS = ("blue", "orange", "purple", "green", "magenta", "ochre")
NAMES = tuple(f"{c} {n}" for c, n in zip(COLOURS, OBJECTS))
PLACES = ("far", "middle", "near")          # how the checklist names the cups
BLOCK_OBJECT = {0: 3, 1: 4, 2: 5}           # number on the block -> object index
DETECTOR_CONFIDENCE = 0.03
# A cup seen from above is a weak match for "cup" alone (all three found in 29 of 47 first frames), so several prompts
# are asked and the first frame keeps the three candidates that form the column of that kind.
# kind, prompts, area range at 256 px, column rule (x range of the centres, spread, vertical gaps)
KINDS = (
    ("cup", ("cup", "blue cup", "green cup", "pink cup"), (200, 2500),
     {"x": (95.0, 256.0), "x_spread": 60.0, "gap": (20.0, 85.0), "gap_difference": 40.0}),
    ("block", ("cube", "block", "red cube", "blue cube", "green cube"), (250, 1400),
     {"x": (5.0, 100.0), "x_spread": 45.0, "gap": (25.0, 75.0), "gap_difference": 30.0}),
)
MIN_PIXELS = (120, 120, 120, 100, 100, 100)   # fewer visible pixels = not seen (a cup is 600..1500 px, a block about 700)
MAX_PIXELS = 3500
# ---- checklist ----------------------------------------------------------------------------------------------
# Per cup i: "lift the <place> cup", then the observation "<place> cup: has block | empty" (its own item, written when
# the spot under the cup is in view), then "put the <place> cup back". Then "pick up block <count>" (not known before
# it starts: "?" in the checklist until then) and "place the block on the plate".
OBSERVED = {True: "has block", False: "empty"}
PICK_TEXT = "pick up block {}"
PLACE_TEXT = "place the block on the plate"
VIEW_AFTER_SEEN = 3             # has block: the observation is written this many frames after the pink block is first seen
VIEW_EMPTY = 40                 # empty: frames after the close command (the block, when there is one, is seen 20..51
                                # frames after it, median 29); capped at half the hold for short lifts
PICK_MARGIN = 8                 # frames after the close command before "pick up block" counts as done
PLACE_MARGIN = 5                # frames after the open command before a "put back" / "place" counts as done
SEEN_PIXELS = 300               # pink pixels in the 640x480 top image that count as "the block is in view" (a block is 500..1100)

# Validation episodes: two per answer (the last ones of each group), not the last five (those all have the answer 0).
VAL_EPISODES = (15, 16, 32, 33, 48, 49)

MOTION_PIXELS = 6.0             # as real_robot: minimum displacement over half a 20-frame window, 256-px pixels


def items_for(has_block):
    """has_block: three bools (far, middle, near) -> (items without observations, observations [(k, text)])."""
    items, observed = [], []
    for place, has in zip(PLACES, has_block):
        observed.append((len(items), f"{place} cup: {OBSERVED[bool(has)]}"))
        items += [f"lift the {place} cup", f"put the {place} cup back"]
    items += [PICK_TEXT.format(int(sum(has_block))), PLACE_TEXT]
    return items, observed


def pink_mask(rgb):
    """Pixels of a small pink block in a top-camera frame (int16 or uint8 RGB). Labels only."""
    im = np.asarray(rgb).astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 150) & (g > 95) & (g < 135) & (b - g > 6) & (r - g > 38)


def registry():
    return {str(k + 1): {"category": "bowl", "segmentation_id": k + 1, "physical_name": n} for k, n in enumerate(OBJECTS)}


def episode_name(index):
    return f"t{TASK_ID:02d}_seed{index}"
