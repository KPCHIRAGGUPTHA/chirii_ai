"""Phase 10A tests: model v3, trainer v3 and the evaluation."""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

from model import MiniGPT, MiniGPTConfig
from phase10.common import project_path, read_text_lf, sha256_text
from phase10.eval_v3 import OLD_TEXT, OLD_TEXT_SHA256, tokenize_old_text
from phase10.model_v3 import MiniGPTV3
from phase10.train_v3 import DEFAULT_TRAINING, compile_model, load_config, train, truncate_jsonl
from phase9.train_v2 import build_param_groups, get_learning_rate

FULL_MODEL = dict(vocab_size=16384, block_size=512, n_layer=8, n_head=8, n_embd=512, dropout=0.0, bias=True)
TINY_MODEL = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True)
TRAINER = project_path("phase10/train_v3.py")


class TestModel(unittest.TestCase):
    def test_matches_the_original_attention_given_the_same_weights(self):
        cfg = MiniGPTConfig(**TINY_MODEL)
        torch.manual_seed(0)
        old = MiniGPT(cfg).eval()
        new = MiniGPTV3(cfg).eval()
        new.load_state_dict({k: v for k, v in old.state_dict().items() if not k.endswith(".attn.bias")})
        x = torch.randint(0, 64, (3, 16))
        self.assertLess((old(x, x)[0] - new(x, x)[0]).abs().max().item(), 1e-5)

    def test_attention_is_causal(self):
        model = MiniGPTV3(MiniGPTConfig(**TINY_MODEL)).eval()
        a = torch.randint(0, 64, (1, 16))
        b = a.clone()
        b[0, 10] = (a[0, 10] + 1) % 64
        out_a, out_b = model(a, a)[0], model(b, b)[0]
        self.assertTrue(torch.equal(out_a[0, :10], out_b[0, :10]))
        self.assertFalse(torch.equal(out_a[0, 10:], out_b[0, 10:]))

    def test_embeddings_are_tied(self):
        model = MiniGPTV3(MiniGPTConfig(**TINY_MODEL))
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)

    def test_full_model_has_the_expected_parameter_count(self):
        with torch.device("meta"):
            model = MiniGPTV3(MiniGPTConfig(**FULL_MODEL))
        self.assertEqual(model.get_num_params(), 33_870_848)

    def test_weight_decay_skips_biases_norms_and_embeddings(self):
        model = MiniGPTV3(MiniGPTConfig(**TINY_MODEL))
        decay_group, no_decay_group = build_param_groups(model, DEFAULT_TRAINING["weight_decay"])
        self.assertEqual(decay_group["weight_decay"], 0.1)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        no_decay = {id(p) for p in no_decay_group["params"]}
        for name, param in model.named_parameters():
            should_skip = name.endswith(".bias") or "ln_" in name or "wte" in name or "wpe" in name
            self.assertEqual(id(param) in no_decay, should_skip, name)


class TestSettings(unittest.TestCase):
    def test_defaults_match_the_specification(self):
        t = DEFAULT_TRAINING
        self.assertEqual(t["learning_rate"], 8e-4)
        self.assertEqual(t["min_lr"], 8e-5)
        self.assertEqual(t["betas"], [0.9, 0.95])
        self.assertEqual(t["weight_decay"], 0.1)
        self.assertEqual(t["grad_clip"], 1.0)
        self.assertEqual(t["warmup_steps"], 200)
        self.assertEqual(t["eval_interval"], 250)
        self.assertEqual(t["checkpoint_interval_minutes"], 30)
        self.assertEqual(t["micro_batch_size"] * t["grad_accum_steps"] * 512, 131072)
        self.assertGreaterEqual(t["max_steps"] * 131072, 700_000_000)

    def test_learning_rate_schedule(self):
        lr = lambda step: get_learning_rate(step, 5341, 200, 8e-4, 8e-5)
        self.assertAlmostEqual(lr(100), 4e-4)
        self.assertAlmostEqual(lr(200), 8e-4)
        self.assertAlmostEqual(lr(5341), 8e-5)
        self.assertGreater(lr(1000), lr(3000))


class TestCompileFallback(unittest.TestCase):
    def test_failure_falls_back_to_the_plain_model(self):
        model = MiniGPTV3(MiniGPTConfig(**TINY_MODEL))
        training = dict(DEFAULT_TRAINING, micro_batch_size=2)
        with mock.patch("torch.compile", side_effect=RuntimeError("no compiler")):
            result, status = compile_model(model, training, 16, torch.device("cpu"))
        self.assertIs(result, model)
        self.assertEqual(status, "failed: RuntimeError")

    def test_disabled(self):
        model = MiniGPTV3(MiniGPTConfig(**TINY_MODEL))
        self.assertEqual(compile_model(model, dict(DEFAULT_TRAINING, compile=False), 16, torch.device("cpu")), (model, "disabled"))


class RunBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        rng = np.random.default_rng(0)
        # Learnable data: a repeating pattern with noise, so the loss falls
        data = (np.arange(40_000) % 32 + (rng.random(40_000) < 0.1) * rng.integers(0, 32, 40_000)).astype(np.uint16)
        data.tofile(os.path.join(self.dir, "train.bin"))
        data[:2000].tofile(os.path.join(self.dir, "val.bin"))

    def make_config(self, name, device="cpu", **training):
        config = {
            "device": device, "out_dir": os.path.join(self.dir, name), "model": TINY_MODEL,
            "data": {"train_bin": os.path.join(self.dir, "train.bin"), "val_bin": os.path.join(self.dir, "val.bin"), "val_subset_tokens": 320},
            "training": {**dict(max_steps=60, micro_batch_size=4, grad_accum_steps=2, warmup_steps=5, eval_interval=20, log_interval=10,
                                compile=False, cpu_threads=1, learning_rate=3e-3, min_lr=3e-4), **training},
        }
        path = os.path.join(self.dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return path, config["out_dir"]

    def steps(self, out_dir):
        with open(os.path.join(out_dir, "steps.jsonl"), "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def metrics(self, out_dir):
        with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def run_trainer(self, config_path, *args):
        return subprocess.run([sys.executable, TRAINER, "--config", config_path, *args], capture_output=True, text=True, cwd=project_path("."), timeout=600)

    def kill_when(self, config_path, out_dir, min_steps, *args):
        """Starts the trainer, then SIGKILLs it once it has finished `min_steps` steps."""
        process = subprocess.Popen([sys.executable, TRAINER, "--config", config_path, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=project_path("."))
        path = os.path.join(out_dir, "steps.jsonl")
        deadline = time.time() + 300
        while time.time() < deadline and process.poll() is None:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    if f.read().count(b"\n") >= min_steps:
                        break
            time.sleep(0.005)
        self.assertIsNone(process.poll(), "Trainer ended before the kill")
        process.send_signal(signal.SIGKILL)
        process.wait()
        self.assertEqual(process.returncode, -signal.SIGKILL)


class TestTraining(RunBase):
    def test_grad_norm_is_logged_every_step_and_loss_falls(self):
        path, out_dir = self.make_config("run")
        train(load_config(path))
        steps = self.steps(out_dir)
        self.assertEqual([s["step"] for s in steps], list(range(1, 61)))
        self.assertTrue(all(s["grad_norm"] > 0 for s in steps))
        self.assertLess(steps[-1]["loss"], steps[0]["loss"] - 0.5)
        metrics = self.metrics(out_dir)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["max_grad_norm"], max(s["grad_norm"] for s in steps))
        self.assertEqual(metrics["max_grad_norm_after_warmup"], max(s["grad_norm"] for s in steps if s["step"] > 5))

    def test_first_loss_is_near_uniform_entropy(self):
        path, out_dir = self.make_config("run")
        train(load_config(path, {"max_steps": 1}))
        self.assertLess(abs(self.steps(out_dir)[0]["loss"] - math.log(64)), 0.5)

    def test_val_bits_per_byte_uses_the_loss_and_bytes(self):
        path, out_dir = self.make_config("run")
        train(load_config(path))
        entry = self.metrics(out_dir)["evals"][-1]
        self.assertEqual(entry["val_tokens"], 320)
        self.assertEqual(entry["val_bytes"], 320)
        self.assertAlmostEqual(entry["val_bits_per_byte"], entry["val_loss"] / math.log(2))

    def test_evaluates_every_interval_and_at_an_early_stop(self):
        path, out_dir = self.make_config("run")
        train(load_config(path, {"stop_after_step": 50}))
        self.assertEqual([e["step"] for e in self.metrics(out_dir)["evals"]], [20, 40, 50])
        self.assertEqual(self.metrics(out_dir)["status"], "stopped_early")

    def test_non_finite_loss_stops_the_run_and_is_recorded(self):
        path, out_dir = self.make_config("run")
        with mock.patch("phase10.train_v3.accumulate_gradients", return_value=float("nan")):
            with self.assertRaises(RuntimeError):
                train(load_config(path))
        metrics = self.metrics(out_dir)
        self.assertEqual(metrics["status"], "failed_non_finite")
        self.assertEqual(metrics["non_finite_count"], 1)

    def test_changed_validation_subset_aborts(self):
        path, _ = self.make_config("run")
        config = load_config(path)
        config["data"]["val_subset_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "subset hash"):
            train(config)

    def test_changed_train_file_aborts(self):
        path, _ = self.make_config("run")
        config = load_config(path)
        config["data"]["train_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            train(config)

    def test_truncate_drops_later_steps_and_a_half_written_line(self):
        path = os.path.join(self.dir, "s.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"step": 1}\n{"step": 2}\n{"step": 3}\n{"step": 4')
        truncate_jsonl(path, 2)
        with open(path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"step": 1}\n{"step": 2}\n')


class TestKillAndResume(RunBase):
    def check_kill_and_resume(self, device, compile_model_flag, tolerance, max_steps):
        settings = dict(compile=compile_model_flag, checkpoint_every_steps=10, max_steps=max_steps)
        run_args = ["--eval-interval", "20"]
        base_path, base_dir = self.make_config("baseline", device=device, **settings)
        result = self.run_trainer(base_path, *run_args)
        self.assertEqual(result.returncode, 0, result.stderr)

        kill_path, kill_dir = self.make_config("killed", device=device, **settings)
        self.kill_when(kill_path, kill_dir, 25, *run_args)
        state = torch.load(os.path.join(kill_dir, "checkpoint.pt"), map_location="cpu", weights_only=True)
        resumed_from = state["step"]
        self.assertGreaterEqual(resumed_from, 20)
        self.assertLess(resumed_from, max_steps)

        result = self.run_trainer(kill_path, "--auto-resume", *run_args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Resumed from step {resumed_from}", result.stdout)

        # The log has each step exactly once, and the loss continues as if the kill never happened
        base, killed = self.steps(base_dir), self.steps(kill_dir)
        self.assertEqual([s["step"] for s in killed], list(range(1, max_steps + 1)))
        for a, b in zip(base, killed):
            self.assertAlmostEqual(a["loss"], b["loss"], delta=tolerance, msg=f"step {a['step']}")
            self.assertAlmostEqual(a["grad_norm"], b["grad_norm"], delta=tolerance * 50, msg=f"step {a['step']}")
        metrics = self.metrics(kill_dir)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["resumes"][0]["from_step"], resumed_from)
        self.assertEqual([e["step"] for e in metrics["evals"]], [e["step"] for e in self.metrics(base_dir)["evals"]])
        return base_dir, kill_dir

    def test_kill_then_resume_continues_exactly_on_cpu(self):
        base_dir, kill_dir = self.check_kill_and_resume("cpu", False, 0.0, 400)
        a = torch.load(os.path.join(base_dir, "checkpoint.pt"), map_location="cpu", weights_only=True)["model_state"]
        b = torch.load(os.path.join(kill_dir, "checkpoint.pt"), map_location="cpu", weights_only=True)["model_state"]
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_kill_then_resume_continues_exactly_on_gpu_without_compile(self):
        self.check_kill_and_resume("cuda", False, 0.0, 150)

    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_kill_then_resume_on_gpu_with_compile(self):
        # Compiled GPU runs are not bit-for-bit repeatable: two uninterrupted runs of this setup differed by up to
        # 0.00097 in loss. The tolerance allows that noise, while a resume that lost state would differ far more.
        self.check_kill_and_resume("cuda", True, 5e-3, 150)

    def test_auto_resume_starts_fresh_without_a_checkpoint_and_stops_when_complete(self):
        path, out_dir = self.make_config("fresh", checkpoint_every_steps=10)
        result = self.run_trainer(path, "--auto-resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Resumed", result.stdout)
        self.assertEqual(self.metrics(out_dir)["status"], "completed")
        result = self.run_trainer(path, "--auto-resume")
        self.assertIn("Already completed", result.stdout)

    def test_resume_rejects_a_changed_learning_rate(self):
        path, out_dir = self.make_config("run", checkpoint_every_steps=10)
        train(load_config(path, {"stop_after_step": 10}))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            train(load_config(path, {"learning_rate": 1e-3}), resume=True)


@unittest.skipUnless(os.path.exists(project_path(OLD_TEXT)), "Phase 5D text not present")
class TestOldTextEvaluation(unittest.TestCase):
    def test_uses_the_lf_form_whatever_the_checkout_has(self):
        from tokenizers import Tokenizer
        from phase10.tests.test_tokenizer_v2 import tiny_corpus
        from phase10.tokenizer_v2 import token_byte_lengths, train_tokenizer
        tokenizer = train_tokenizer(tiny_corpus(), vocab_size=600)
        tokens = tokenize_old_text(tokenizer)
        text = read_text_lf(project_path(OLD_TEXT))
        self.assertNotIn("\r", text)
        self.assertEqual(int(token_byte_lengths(tokenizer)[tokens].sum()), len(text.encode("utf-8")))
        self.assertEqual(len(text.encode("utf-8")), 3_517_521)

    def test_wrong_pin_aborts(self):
        from phase10.tests.test_tokenizer_v2 import tiny_corpus
        from phase10.tokenizer_v2 import train_tokenizer
        with self.assertRaisesRegex(ValueError, "Pinned hash mismatch"):
            tokenize_old_text(train_tokenizer(tiny_corpus(), vocab_size=600), expected_sha256="0" * 64)

    def test_pin_constant_matches_the_file(self):
        self.assertEqual(sha256_text(project_path(OLD_TEXT)), OLD_TEXT_SHA256)


if __name__ == "__main__":
    unittest.main()
