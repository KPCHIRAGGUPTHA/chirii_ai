"""
Phase 26D — DPO with LoRA from v0.3.1 (phase26d/CRITERIA_26D.md, "Start model and training"). Written here (trl is not
installed); the loss is the standard sigmoid DPO loss:
    L = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )
with log-probabilities summed over the reply tokens (and the end-of-turn token) only. Policy = v0.3.1 + LoRA; reference = v0.3.1
(the same model with the adapter disabled; reference log-probs computed once before training, no grad).
LoRA r 8, alpha 16, dropout 0.05, q/k/v/o/gate/up/down; beta 0.1; AdamW lr 5e-6, 10% linear warmup then cosine; clip 1.0;
1 epoch; 1 pair per micro-batch x 8 accumulation; bf16 base with float32 LoRA weights under bf16 autocast; gradient checkpointing.
Qwen chat template with its default system prompt. Seed sets LoRA init and pair order. Nothing is printed except numbers.
Usage:  python phase26d/train_dpo_26d.py --seed 11      -> phase26d/runs/d_s{seed}/adapter/ and metrics.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase26d.common26d import BASE, START, sha  # noqa: E402

MANIFEST = os.path.join(ROOT, "phase26d/manifests/pairs_26d_manifest.json")
BETA, LR, ACCUM, WARMUP_FRAC, MAX_LEN = 0.1, 5e-6, 8, 0.10, 1024
R, ALPHA = 8, 16
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, prompt, reply):
    msgs = [{"role": "user", "content": prompt}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": reply}], tokenize=False)
    assert full.startswith(prefix)
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    f = tok(full.rstrip("\n"), add_special_tokens=False)["input_ids"]
    assert len(f) <= MAX_LEN, "pair over the length limit (should have been dropped at build)"
    return f, len(p)


def seq_logp(model, ids, start, device):
    """Sum of log p(token) over positions >= start (the reply)."""
    x = torch.tensor([ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits[:, :-1].float()
    tgt = x[:, 1:]
    lp = torch.gather(F.log_softmax(logits, -1), 2, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, start - 1:].sum()


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    import transformers

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(11, 12, 13))
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, f"phase26d/runs/d_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    m = json.load(open(MANIFEST))["pairs"]
    data = {}
    for split in ("train", "val"):
        f = m["files"][split]
        assert sha(f["file"]) == f["sha256"], f["file"]
        data[split] = [json.loads(l) for l in open(os.path.join(ROOT, f["file"]), encoding="utf-8")]
    assert sha(os.path.join(START[0], "model.safetensors")) == START[1], "start model changed"
    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1])
    enc = {s: [(encode(tok, r["prompt"], r["chosen"]), encode(tok, r["prompt"], r["rejected"]), r["part"]) for r in rows] for s, rows in data.items()}
    random.Random(a.seed).shuffle(enc["train"])
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, START[0]), dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    cfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def ref_logps(items):
        model.eval()
        out = []
        with model.disable_adapter():
            for (cw, sw), (cl, sl), _ in items:
                out.append((float(seq_logp(model, cw, sw, device)), float(seq_logp(model, cl, sl, device))))
        model.train()
        return out

    t0 = time.time()
    ref = {s: ref_logps(enc[s]) for s in enc}
    ref_seconds = round(time.time() - t0)

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        loss, acc, marg, by = 0.0, 0, 0.0, {}
        for ((cw, sw), (cl, sl), part), (rw, rl) in zip(enc[split], ref[split]):
            z = BETA * ((float(seq_logp(model, cw, sw, device)) - rw) - (float(seq_logp(model, cl, sl, device)) - rl))
            l = -float(F.logsigmoid(torch.tensor(z)))
            loss += l; acc += z > 0; marg += z
            b = by.setdefault(part, [0, 0]); b[0] += z > 0; b[1] += 1
        model.train()
        n = len(enc[split])
        return {"loss": round(loss / n, 4), "reward_acc": round(acc / n, 4), "reward_margin": round(marg / n, 4),
                "reward_acc_by_part": {k: round(v[0] / v[1], 4) for k, v in by.items()}}

    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    steps = math.ceil(len(enc["train"]) / ACCUM)
    warm = max(1, round(WARMUP_FRAC * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log = {"seed": a.seed, "start": START, "base": BASE, "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
           "train_pairs": len(enc["train"]), "val_pairs": len(enc["val"]), "steps": steps, "warmup_steps": warm, "beta": BETA, "lr": LR,
           "accum": ACCUM, "micro_pairs": 1, "max_len": MAX_LEN, "lora": {"r": R, "alpha": ALPHA, "dropout": 0.05, "targets": TARGETS},
           "trainable_params": sum(p.numel() for p in params), "data_sha256": m["files"]["train"]["sha256"], "ref_seconds": ref_seconds, "log": []}
    log["val_start"] = evaluate("val")
    print("val at start", log["val_start"], flush=True)
    model.train()
    t0 = time.time()
    step, micro, run_l, run_a, run_m = 0, 0, 0.0, 0, 0.0
    for ((cw, sw), (cl, sl), _), (rw, rl) in zip(enc["train"], ref["train"]):
        pw = seq_logp(model, cw, sw, device)
        pl = seq_logp(model, cl, sl, device)
        z = BETA * ((pw - rw) - (pl - rl))
        loss = -F.logsigmoid(z) / ACCUM
        loss.backward()
        run_l += float(loss) * ACCUM; run_a += float(z) > 0; run_m += float(z)
        micro += 1
        if micro % ACCUM == 0 or micro == len(enc["train"]):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == steps:
                k = micro - (log["log"][-1]["micro"] if log["log"] else 0)
                log["log"].append({"step": step, "micro": micro, "loss": round(run_l / k, 4), "reward_acc": round(run_a / k, 4), "reward_margin": round(run_m / k, 4),
                                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)})
                print(f"step {step}/{steps} " + json.dumps(log["log"][-1]), flush=True)
                run_l, run_a, run_m = 0.0, 0, 0.0
    log["optimizer_steps"] = step
    log["train_seconds"] = round(time.time() - t0)
    log["val_end"] = evaluate("val")
    print("val at end", log["val_end"], flush=True)
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    json.dump(log, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
