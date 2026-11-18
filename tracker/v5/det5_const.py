"""constants of the v5 first-frame selection for tasks 18/19 (no torch import; det5.py / track5.py hold the same values)"""
T0 = {18: 2, 19: 10}
RECHECK = {18: 6}
CROP = {18: (56, 66, 152, 162), 19: (56, 66, 152, 162)}
PROMPTS = ["box", "bottle", "carton", "small object", "food package", "chocolate box", "butter", "milk carton", "juice carton", "ketchup bottle", "chocolate bar", "brown box", "dark box"]
THR = 0.15
