"""Round 1 of the encoder comparison: 5000-update runs, all with failure rows, differing only as named."""
import json
from pathlib import Path

here = Path(__file__).parent
ROOT = "/pfs/pfs-7jnepv/lgd/pi05-abc"
VARIANTS = {"A5": ("A", None), "C5": ("C", None), "P": ("C", f"{ROOT}/latents_P"), "Q": ("B", f"{ROOT}/latents_P")}
for name, (base, latents) in VARIANTS.items():
    contract = json.loads((here / f"contract_{base}_fail.json").read_text(encoding="utf-8"))
    contract["training"]["optimizer_steps"] = 5000
    contract["training"]["checkpoint_interval"] = 5000
    contract["training"]["validation_interval"] = 5000
    if latents:
        contract["dataset"]["latents_dir"] = latents
    contract["training"]["wandb"]["name"] = f"pi05-variant-{name}"
    contract["project"] = f"pi05_variant_{name}"
    (here / f"contract_{name}.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(name, contract["model"]["memory_variant"], contract["dataset"].get("latents_dir"), contract["training"]["align_loss_weight"], contract["training"]["verb_loss_weight"])
