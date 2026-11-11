"""Phase 23 — a local harness copy of a Qwen model dir (files symlinked) whose generation_config.json keeps only the token ids,
as Phase 18 did, so shipped sampling / repetition / length defaults cannot apply.  Usage: python phase23/make_local_model.py SRC DST"""
import json, os, sys
src, dst = (os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:3])
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f == "generation_config.json":
        g = json.load(open(os.path.join(src, f)))
        json.dump({k: g[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id") if k in g}, open(os.path.join(dst, f), "w"), indent=1)
    elif not os.path.exists(os.path.join(dst, f)):
        os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(dst, f))
print(sorted(os.listdir(dst)))
