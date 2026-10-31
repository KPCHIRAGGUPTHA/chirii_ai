"""Phase 10A tests: the overlap filter, the stability gate and the auto-resume launcher."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from phase10.check_stability import apply_gate, summarise_run
from phase10.common import project_path
from phase10.filter_phase5d_overlap import document_overlaps, keep_mask, old_text_keys
from phase10.tests.test_train_v3 import RunBase

LONG_A = "This line is long enough to count as shared text, number one."
LONG_B = "This other line is also long enough and is not in the old text."


class TestOverlapFilter(unittest.TestCase):
    def test_flags_only_documents_sharing_a_long_line(self):
        keys = old_text_keys(f"short\n{LONG_A}\n")
        self.assertTrue(document_overlaps(f"intro\n{LONG_A}\nend", keys))
        self.assertFalse(document_overlaps(f"short\n{LONG_B}", keys))

    def test_short_lines_do_not_count(self):
        self.assertFalse(document_overlaps("short", old_text_keys("short\n")))

    def test_keep_mask_drops_the_whole_document_and_its_terminator(self):
        tokens = np.array([5, 6, 0, 7, 0, 8, 9, 9, 0], dtype=np.uint16)
        mask = keep_mask(tokens, 0, np.array([False, True, False]))
        self.assertEqual(tokens[mask].tolist(), [5, 6, 0, 8, 9, 9, 0])

    def test_keep_mask_rejects_a_misaligned_file(self):
        with self.assertRaises(ValueError):
            keep_mask(np.array([5, 0, 6], dtype=np.uint16), 0, np.array([False]))
        with self.assertRaises(ValueError):
            keep_mask(np.array([5, 0], dtype=np.uint16), 0, np.array([False, False]))


def run_steps(grads=None, loss_start=8.0, count=300):
    grads = grads or {}
    return [{"step": i, "loss": loss_start - i * 0.01, "grad_norm": grads.get(i, 0.5)} for i in range(1, count + 1)]


def run_evals(bits=1.8, values=(7.0, 6.0, 5.0)):
    return [{"step": 100 * (i + 1), "val_loss": v, "val_bits_per_byte": bits} for i, v in enumerate(values)]


class TestStabilityGate(unittest.TestCase):
    def gate(self, *pairs):
        return apply_gate([summarise_run(s, e, 200) for s, e in pairs])

    def test_good_runs_pass(self):
        self.assertTrue(self.gate((run_steps(), run_evals()), (run_steps(), run_evals(1.82)))["passed"])

    def test_difference_within_the_limit_passes_and_above_fails(self):
        self.assertTrue(self.gate((run_steps(), run_evals(1.75)), (run_steps(), run_evals(1.775)))["checks"]["final_difference_within_limit"])
        self.assertFalse(self.gate((run_steps(), run_evals(1.8)), (run_steps(), run_evals(1.84)))["passed"])

    def test_large_grad_only_fails_after_warmup(self):
        self.assertTrue(self.gate((run_steps({200: 50.0}), run_evals()), (run_steps(), run_evals()))["passed"])
        self.assertFalse(self.gate((run_steps({201: 10.5}), run_evals()), (run_steps(), run_evals()))["passed"])

    def test_non_finite_fails(self):
        steps = run_steps()
        steps[5]["loss"] = float("nan")
        self.assertFalse(self.gate((steps, run_evals()), (run_steps(), run_evals()))["checks"]["no_non_finite"])

    def test_rising_val_loss_fails(self):
        self.assertFalse(self.gate((run_steps(), run_evals(values=(7.0, 7.5, 5.0))), (run_steps(), run_evals()))["passed"])


class TestLauncher(RunBase):
    def launcher(self, config_path, log):
        env = dict(os.environ, PYTHON_BIN=sys.executable, RETRY_SECONDS="0", LOG=log, MAX_ATTEMPTS="3")
        return subprocess.Popen(["sh", project_path("phase10/run_long.sh"), config_path], cwd=project_path("."), env=env, start_new_session=True)

    def test_resumes_after_the_trainer_is_killed_and_finishes(self):
        path, out_dir = self.make_config("job", checkpoint_every_steps=10, max_steps=400)
        log = os.path.join(self.dir, "job.log")
        process = self.launcher(path, log)
        steps_path = os.path.join(out_dir, "steps.jsonl")
        deadline = time.time() + 300
        while time.time() < deadline:
            if os.path.exists(steps_path):
                with open(steps_path, "rb") as f:
                    if f.read().count(b"\n") >= 25:
                        break
            time.sleep(0.005)
        children = subprocess.run(["pgrep", "-P", str(process.pid)], capture_output=True, text=True).stdout.split()
        self.assertEqual(len(children), 1)
        os.kill(int(children[0]), signal.SIGKILL)
        self.assertEqual(process.wait(timeout=300), 0)
        self.assertEqual(self.metrics(out_dir)["status"], "completed")
        self.assertEqual([s["step"] for s in self.steps(out_dir)], list(range(1, 401)))
        with open(log, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("attempt 2", text)
        self.assertIn("Resumed from step", text)

    def test_does_not_retry_after_a_non_finite_failure(self):
        path, out_dir = self.make_config("job")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "failed_non_finite"}, f, indent=2)
        self.assertEqual(self.launcher(path, os.path.join(self.dir, "job.log")).wait(timeout=60), 3)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "steps.jsonl")))


if __name__ == "__main__":
    unittest.main()
