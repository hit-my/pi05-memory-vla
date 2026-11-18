"""Constants for the real-robot dataset wwccww/yx_bowls (bimanual Dobot, only the right arm moves).

Everything a deployment must reproduce exactly is defined here, once:
  * which camera goes into which model image slot and how it is cropped,
  * which 7 of the 14 state/action dimensions the model sees,
  * the object order (-> overlay colour and name) and the checklist of the task.
"""
import numpy as np

TASK_ID = 27  # RoboMemArena uses 1..26; the real task gets the next free id
PROMPT = "Swap the 2 bowls on their plates using the empty plate."
FPS = 30

# ---- images -------------------------------------------------------------------------------------------------
# top camera 640x480 -> square crop around the three plates -> 256x256 (tracker / overlay source) -> 224 (model)
TOP_CROP = (210, 140, 320)      # x0, y0, side in the 640x480 top image
# right wrist camera 640x480 -> centre square -> 256 -> 224 (no anisotropic squash)
WRIST_CROP = (80, 0, 480)
SOURCE = 256                    # the tracker and the overlay renderer work on 256x256 frames


def crop_resize(frame, crop):
    """uint8 [480,640,3] camera frame -> uint8 [256,256,3]. The ONE function used for the dataset and at deployment."""
    from PIL import Image
    x0, y0, side = crop
    patch = np.ascontiguousarray(np.asarray(frame, np.uint8)[y0:y0 + side, x0:x0 + side])
    if patch.shape != (side, side, 3):
        raise ValueError(f"camera frame {np.asarray(frame).shape} does not contain crop {crop}")
    return np.asarray(Image.fromarray(patch).resize((SOURCE, SOURCE), Image.BILINEAR))


def model_image(rgb256):
    """256 -> 224 for the wrist and clean front model inputs (PIL BILINEAR, as in the simulator dataset)."""
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb256, np.uint8)).resize((224, 224), Image.BILINEAR))

# ---- state / action -----------------------------------------------------------------------------------------
# Dataset order: left_joint_0..5, left_gripper, right_joint_0..5, right_gripper. The left arm never moves in the
# 50 demonstrations, so the model controls the right arm only and the training code stays 7-dimensional.
RIGHT = slice(7, 14)
GRIPPER = 13                    # right gripper in the 14-vector: about 0.94 open, 0.01 closed
CLOSED_BELOW = 0.3


def state8(state14):
    """[N,14] -> [N,8]: right joints (6), right gripper, right gripper again (the code's state has 8 entries)."""
    s = np.asarray(state14, np.float32)
    return np.concatenate([s[..., RIGHT], s[..., GRIPPER:GRIPPER + 1]], axis=-1)


def action7(action14):
    """[N,14] -> [N,7]: absolute joint targets of the right arm (6) + right gripper target."""
    return np.asarray(action14, np.float32)[..., RIGHT]


def action14(action7_row, left_hold):
    """Deployment: put a predicted right-arm action back into the robot's 14-vector; the left arm keeps left_hold [7]."""
    return np.concatenate([np.asarray(left_hold, np.float32), np.asarray(action7_row, np.float32)])


# ---- objects and checklist ----------------------------------------------------------------------------------
# Object order = left-to-right position of the bowl in the first cropped top frame, largest x first:
#   object 0 = the bowl on the RIGHT plate at the start (moved first in every demonstration) -> "blue bowl"
#   object 1 = the bowl on the LEFT plate at the start                                        -> "orange bowl"
OBJECTS = ("bowl", "bowl")
COLOURS = ("blue", "orange")
NAMES = tuple(f"{c} {n}" for c, n in zip(COLOURS, OBJECTS))
DETECTOR_PROMPT = "beige bowl"   # plain "bowl" finds nothing from this top view; "beige bowl" finds both
MIN_PIXELS = 150                # a bowl covers about 1500 px at 256; fewer visible pixels = not drawn / occluded

ITEMS = ("pick up the right bowl",
         "place the right bowl on the spare plate",
         "pick up the left bowl",
         "place the left bowl on the right plate",
         "pick up the bowl on the spare plate",
         "place that bowl on the left plate")
ITEM_OBJECT = (0, 0, 1, 1, 0, 0)  # the object each item handles
PICK_MARGIN = 8                 # frames after the gripper reads closed before "pick" counts as done (bowl lifted)
PLACE_MARGIN = 5                # frames after the gripper reads open before "place" counts as done

# Action-line direction words come from the bowl's motion in the cropped top image (no end-effector pose in the data):
# image x+ = "right", x- = "left", image y- (away from the camera's bottom edge) = "forward", y+ = "backward".
MOTION_PIXELS = 6.0             # minimum displacement over half a 20-frame window, in 256-px image pixels


def registry():
    """Renderer registry, same shape as rma_spec.registry (category 'bowl' = a drawn track)."""
    return {str(k + 1): {"category": "bowl", "segmentation_id": k + 1, "physical_name": n} for k, n in enumerate(OBJECTS)}


def episode_name(index):
    return f"t{TASK_ID:02d}_seed{index}"
