"""Phase 17D — writes phase17/configs/foundation_v4x_base.json from the Foundation v4 config: new training file (hash
from the 17D tokens manifest), warmup 300 steps to 3e-4, cosine to 3e-5, start from the v4 step-5,722 weights; all else
unchanged. max_steps is filled in by phase17/gate_17d.py from the measured budget."""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import project_path
c = json.load(open(project_path("phase17/configs/foundation_v4.json")))
t = json.load(open(project_path("phase17/manifests17d/tokens_manifest.json")))["joined"]["train_17d"]
c["name"] = "Foundation v4 extended (Phase 17D continued pretraining, 70% FineWeb-Edu / 30% FineWeb)"
c["data"]["train_bin"], c["data"]["train_sha256"] = t["file"], t["sha256"]
c["out_dir"] = "phase17/runs/foundation_v4x"
c["init_checkpoint"] = {"file": "phase17/runs/foundation_v4/snapshots/model_step_05722.pt",
                        "sha256": "3c5d02dd3f8ef070623b057828efa8856f0ec7ae630cb1b382011eb47b498851"}
c["training"].update(max_steps=0, learning_rate=3e-4, min_lr=3e-5, warmup_steps=300, snapshot_every_steps=2000)
json.dump(c, open(project_path("phase17/configs/foundation_v4x_base.json"), "w"), indent=2)
print(json.dumps(c, indent=1))
