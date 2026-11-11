"""
Phase 24 — fine-tune v2: LoRA on Qwen2.5-1.5B-Instruct (phase24/CRITERIA_24.md). Same trainer as Phase 23B; lr 5e-5, Phase 24 data.

LoRA r=16, alpha=32, dropout 0.05 on all linear layers of the blocks (q/k/v/o/gate/up/down); base frozen (bf16);
LoRA weights in float32 under bf16 autocast; AdamW lr 5e-5, cosine with 3% warmup; 1 epoch; micro-batch 4 x 8 accumulation (8 x 4 ran out of memory)
= 32; max length 1,024; loss on the assistant reply (and its end-of-turn token) only, using Qwen's own chat template with
its default system prompt (the same template the runtime uses). Seed sets LoRA init and data order.

Usage:  python phase24/train_lora_v2.py --seed 11
Writes phase24/runs/lora_s{seed}/adapter/ and metrics.json.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

BASE = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
MANIFEST = os.path.join(ROOT, "phase24/manifests/sft_24_manifest.json")
MAX_LEN, MICRO, ACCUM, LR, WARMUP_FRAC = 1024, 4, 8, 5e-5, 0.03  # micro 8 x 4 ran out of GPU memory (152k-word logits); same effective batch 32
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def encode(tok, rows):
    out, dropped = [], 0
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["reply"]}], tokenize=False)
        assert full.startswith(prefix)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        f_ids = tok(full.rstrip("\n"), add_special_tokens=False)["input_ids"]  # ends with <|im_end|>
        if len(f_ids) > MAX_LEN:
            dropped += 1
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        out.append((f_ids, labels))
    return out, dropped


def batches(examples, pad_id):
    for i in range(0, len(examples), MICRO):
        chunk = examples[i:i + MICRO]
        n = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        yield ids, lab, att


@torch.no_grad()
def val_loss(model, examples, pad_id, device):
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, att in batches(examples, pad_id):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        k = int((lab[:, 1:] != -100).sum())
        tot += float(out.loss) * k
        n += k
    model.train()
    return tot / max(1, n)


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    import transformers

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, f"phase24/runs/lora_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    m = json.load(open(MANIFEST))
    rows = {}
    for split in ("train", "val"):
        f = os.path.join(ROOT, m["files"][split]["file"])
        assert sha(f) == m["files"][split]["sha256"], f
        rows[split] = [json.loads(l) for l in open(f, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1])
    train, dropped = encode(tok, rows["train"])
    val, _ = encode(tok, rows["val"])
    random.Random(a.seed).shuffle(train)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(BASE[0], revision=BASE[1], dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for n_, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    steps = math.ceil(len(train) / (MICRO * ACCUM))
    warm = max(1, round(WARMUP_FRAC * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log = {"seed": a.seed, "base": BASE, "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
           "train_rows": len(train), "dropped_over_max_len": dropped, "val_rows": len(val), "steps": steps, "warmup_steps": warm,
           "trainable_params": trainable, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": TARGETS},
           "lr": LR, "micro_batch": MICRO, "accum": ACCUM, "max_len": MAX_LEN, "data_sha256": m["files"]["train"]["sha256"], "log": []}
    log["val_loss_start"] = round(val_loss(model, val, pad, device), 4)
    print("val loss at start", log["val_loss_start"], flush=True)
    model.train()
    t0 = time.time()
    step, micro_i, running = 0, 0, 0.0
    for ids, lab, att in batches(train, pad):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss / ACCUM
        loss.backward()
        running += float(loss)
        micro_i += 1
        if micro_i % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == steps:
                log["log"].append({"step": step, "loss": round(running / 20 if step % 20 == 0 else running, 4), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)})
                print(f"step {step}/{steps} loss {log['log'][-1]['loss']:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                running = 0.0
    if micro_i % ACCUM:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
    log["optimizer_steps"] = step
    log["val_loss_end"] = round(val_loss(model, val, pad, device), 4)
    log["train_seconds"] = round(time.time() - t0)
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    json.dump(log, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print("val loss at end", log["val_loss_end"], "seconds", log["train_seconds"], flush=True)


if __name__ == "__main__":
    main()
