"""contract_X_fail.json = contract_X.json plus the injected-failure rows and their latent tables.

Same seed, steps, learning rate and loss weights, so only the data differs from the first A/B/C runs."""
import json
from pathlib import Path

here = Path(__file__).parent
for group in "ABC":
    contract = json.loads((here / f"contract_{group}.json").read_text(encoding="utf-8"))
    contract["dataset"]["failure_index"] = "failures/train.jsonl"
    if group != "A":
        contract["dataset"]["latents_dir"] = f"xxxx/pi05-abc/latents_fail/{group}"
    contract["training"]["wandb"]["name"] = f"pi05-ledger-{group}-fail-10k"
    contract["project"] = f"{contract.get('project', 'pi05')}_fail"
    contract["memory"]["failure_rows"] = ("Injected empty-hand and drop failures on training episodes; true labels; "
                                          "valid_action_steps=0, so they train the memory and never the action branch.")
    (here / f"contract_{group}_fail.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(group, contract["dataset"]["failure_index"], contract["dataset"].get("latents_dir"))
