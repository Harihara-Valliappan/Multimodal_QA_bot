#!/usr/bin/env python3
"""Pick the LoRA learning rate on VALIDATION, then write the A7 configs.

Rule, fixed before seeing results: higher normalized val accuracy wins;
a tie goes to 2e-5 so the existing PEFT-0.17 baselines stay comparable.
The test split is not read.
"""
import json, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[2]
scores = {}
for tag in ("lr2e5", "lr1e4"):
    rep = root / f"reports/a6_lora021_{tag}_val_report.json"
    if not rep.is_file():
        sys.exit(f"missing {rep}: A6 did not finish")
    scores[tag] = json.loads(rep.read_text())["results"]["metrics"]["normalized"]["overall"]
winner = "lr1e4" if scores["lr1e4"] > scores["lr2e5"] else "lr2e5"
print(f"val normalized: {scores} -> winner {winner}")

base = (root / f"configs/a6_lora_{winner}.yaml").read_text()
arms = {
    "lora_temporal": "  enabled: true\n  kernel_size: 3\n  learning_rate: 1.0e-3\n  freeze_lora: false\n",
    "temporal_only": "  enabled: true\n  kernel_size: 3\n  learning_rate: 1.0e-3\n  freeze_lora: true\n",
}
for arm, block in arms.items():
    s = base.replace(f"experiment_name: a6_lora_peft021_{winner}", f"experiment_name: a7_{arm}")
    s = (f"# A7 ablation arm: {arm} (our temporal adapter, models/temporal.py).\n"
         f"# LoRA learning rate = A6 winner on VAL ({winner}; scores {scores}).\n"
         "# Temporal adapter has its own LR (1e-3): its gate starts at zero.\n#\n") + s
    s += "\ntemporal:\n" + block
    (root / f"configs/a7_{arm}.yaml").write_text(s)
    print("wrote", f"configs/a7_{arm}.yaml")
(root / "reports/a6_lr_selection.json").write_text(json.dumps({"scores": scores, "winner": winner}, indent=2))
