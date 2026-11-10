"""Phase 20 — writes the 6 training configs (A, B x seeds 11-13) from the data manifests, per phase20/CRITERIA_20.md."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
m = json.load(open(os.path.join(ROOT, "phase20/manifests/sft_v6a_manifest.json")))
rp = json.load(open(os.path.join(ROOT, "phase20/manifests/replay_20b_manifest.json")))
base = json.load(open(os.path.join(ROOT, "phase17/configs/sft_v5_s13.json")))
for variant in ("A", "B"):
    for seed in (11, 12, 13):
        c = json.loads(json.dumps(base))
        c["name"] = f"Phase 20 variant {variant} ({'detemplate' if variant == 'A' else 'detemplate + gentle'}) seed {seed}"
        c["data"].update(train=m["files"]["train"]["file"], train_sha256=m["files"]["train"]["sha256"],
                         val=m["files"]["val"]["file"], val_sha256=m["files"]["val"]["sha256"])
        if variant == "B":
            c["data"]["replay"] = {"file": rp["file"], "sha256": rp["sha256"]}
            c["training"].update(epochs=1, learning_rate=1e-4, min_lr=1e-5)
        c["training"]["seed"] = seed
        c["out_dir"] = f"phase20/runs/{variant}_s{seed}"
        json.dump(c, open(os.path.join(ROOT, f"phase20/configs/{variant}_s{seed}.json"), "w"), indent=2)
        print(variant, seed, c["training"]["epochs"], c["training"]["learning_rate"], "replay" in c["data"])
