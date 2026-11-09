"""Phase 11A tests on a tiny CPU model: SFT masking and training, the chat runtime, and scoring through the pipeline."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model import MiniGPTConfig
from phase10.common import TOKENIZER_PATH, project_path
from phase10.model_v3 import MiniGPTV3
from phase11.chat import MAX_TOOL_CALLS, ChatReply, encode_prompt, generate_reply
from phase11.chat_format import ASSISTANT, CALC_CLOSE, CALC_OPEN, EOT, IGNORE_INDEX, RESULT_CLOSE, RESULT_OPEN, USER, build_labels, example_text, special_ids
from phase11.goal_eval_chat import GOAL_SETS, run_goal_eval, score_items
from phase11.sft_train import DEFAULT_TRAINING, collate, encode_examples, evaluate_masked, load_config, make_batches, read_rows, train
from phase9.goal_eval.scorer_v2 import load_goal_set
from phase9.train_v2 import accumulate_gradients
from tokenizers import Tokenizer

TINY = dict(vocab_size=16384, block_size=128, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)
TRAINER = project_path("phase11/sft_train.py")
DEVICE = torch.device("cpu")
CALC_REPLY = f"{CALC_OPEN}2+2{CALC_CLOSE}{RESULT_OPEN}4{RESULT_CLOSE}The answer is 4."
OVERFIT_ROWS = [
    {"prompt": "What is 2+2?", "text": CALC_REPLY},
    {"prompt": "Who are you?", "text": "I'm MiniGPT."},
    {"prompt": "Say hello.", "text": "Hello there, friend."},
    {"prompt": "Name a colour.", "text": "Blue is a colour."},
]


def tokenizer():
    return Tokenizer.from_file(project_path(TOKENIZER_PATH))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return MiniGPTV3(MiniGPTConfig(**TINY))


class TestEnvironment(unittest.TestCase):
    def test_tests_cannot_see_the_gpu(self):
        self.assertFalse(torch.cuda.is_available())


class TestMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()
        cls.ids_of = special_ids(cls.tok)

    def test_targets_are_shifted_and_only_reply_tokens_are_kept(self):
        (x, y), = encode_examples(self.tok, [{"prompt": "Say hi", "text": "Hi there."}], 128)
        ids = self.tok.encode(example_text("Say hi", "Hi there."), add_special_tokens=False).ids
        self.assertEqual(x, ids[:-1])
        self.assertEqual(len(y), len(x))
        reply_start = ids.index(self.ids_of[ASSISTANT]) + 1
        # The first trained target is the reply's first token, predicted from <|assistant|>
        self.assertTrue(all(value == IGNORE_INDEX for value in y[:reply_start - 1]))
        self.assertEqual(y[reply_start - 1:], ids[reply_start:])

    def test_calculator_call_is_trained_and_result_is_masked(self):
        (x, y), = encode_examples(self.tok, [{"prompt": "What is 2+2?", "text": CALC_REPLY}], 128)
        kept = [value for value in y if value != IGNORE_INDEX]
        self.assertEqual(self.tok.decode(kept, skip_special_tokens=False), f"{CALC_OPEN}2+2{CALC_CLOSE}The answer is 4.{EOT}")
        for token in (RESULT_OPEN, RESULT_CLOSE):
            self.assertNotIn(self.ids_of[token], kept)
        position = x.index(self.ids_of[RESULT_CLOSE])
        self.assertNotEqual(y[position], IGNORE_INDEX, "The token right after a result must be trained")
        self.assertEqual(y[x.index(self.ids_of[CALC_CLOSE])], IGNORE_INDEX, "<|result|> is supplied by the runtime")

    def test_loss_and_gradients_equal_cross_entropy_over_only_the_kept_positions(self):
        examples = encode_examples(self.tok, OVERFIT_ROWS[:2], 128)
        x, y = collate(examples, [0, 1], self.ids_of["<|pad|>"], [128])
        keep = y != IGNORE_INDEX
        self.assertTrue(keep.any() and (~keep).any())
        model = tiny_model().eval()
        logits, loss = model(x, y)
        loss.backward()
        masked_grad = model.transformer.wte.weight.grad.clone()

        model.zero_grad()
        logits, _ = model(x, y)
        by_hand = torch.nn.functional.cross_entropy(logits[keep], y[keep])
        by_hand.backward()
        self.assertAlmostEqual(loss.item(), by_hand.item(), places=5)
        self.assertLess((masked_grad - model.transformer.wte.weight.grad).abs().max().item(), 1e-5)

    def test_a_prompt_target_would_change_the_loss_if_it_were_not_masked(self):
        examples = encode_examples(self.tok, OVERFIT_ROWS[:1], 128)
        x, y = collate(examples, [0], self.ids_of["<|pad|>"], [128])
        model = tiny_model().eval()
        masked = model(x, y)[1].item()
        unmasked_targets = y.clone()
        unmasked_targets[0, 0] = x[0, 1]
        self.assertNotAlmostEqual(masked, model(x, unmasked_targets)[1].item(), places=4)

    def test_padding_does_not_change_the_loss(self):
        examples = encode_examples(self.tok, OVERFIT_ROWS[:1], 128)
        pad = self.ids_of["<|pad|>"]
        model = tiny_model().eval()
        short = collate(examples, [0], pad, [40])
        long = collate(examples, [0], pad, [128])
        self.assertAlmostEqual(model(*short)[1].item(), model(*long)[1].item(), places=4)

    def test_collate_pads_to_fixed_lengths(self):
        examples = encode_examples(self.tok, OVERFIT_ROWS, 128)
        x, y = collate(examples, [0, 1, 2], self.ids_of["<|pad|>"], [128, 256])
        self.assertEqual(x.shape, (3, 128))
        self.assertEqual(y.shape, (3, 128))

    def test_too_long_example_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "over the block size"):
            encode_examples(self.tok, [{"id": "x", "prompt": "word " * 300, "text": "hi"}], 128)


class TestBatches(unittest.TestCase):
    def test_each_epoch_uses_every_example_once(self):
        batches = make_batches(list(range(100)), 8, 2, 1)
        self.assertEqual(len(batches), 24)
        flat = [i for batch in batches[:12] for i in batch]
        self.assertEqual(len(set(flat)), len(flat))

    def test_is_deterministic_and_epochs_differ(self):
        self.assertEqual(make_batches(list(range(100)), 8, 2, 1), make_batches(list(range(100)), 8, 2, 1))
        batches = make_batches(list(range(100)), 8, 2, 1)
        self.assertNotEqual(batches[:12], batches[12:])
        self.assertNotEqual(batches, make_batches(list(range(100)), 8, 2, 2))


class TestSettings(unittest.TestCase):
    def test_defaults_match_the_specification(self):
        t = DEFAULT_TRAINING
        self.assertEqual((t["learning_rate"], t["warmup_steps"], t["epochs"], t["weight_decay"]), (2e-4, 50, 2, 0.0))

    def test_config_matches_the_specification(self):
        with open(project_path("phase11/configs/sft_v1.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        self.assertEqual(config["training"].get("weight_decay", 0.0), 0.0)
        self.assertEqual(config["training"].get("epochs", 2), 2)
        self.assertTrue(config["training"].get("compile", True))


class TestConfigPins(unittest.TestCase):
    def test_configs_pin_the_files_in_the_manifest(self):
        with open(project_path("phase11/manifests/sft_manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for name, key in (("sft_v1", "files"), ("sft_v1_noformat", "noformat_files")):
            with open(project_path(f"phase11/configs/{name}.json"), "r", encoding="utf-8") as f:
                data = json.load(f)["data"]
            for split in ("train", "val"):
                self.assertEqual(data[split], manifest[key][split]["file"])
                self.assertEqual(data[f"{split}_sha256"], manifest[key][split]["sha256"])


class RunBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.init = os.path.join(self.dir, "init.pt")
        torch.save({"format": "phase10_train_v3_weights", "step": 0, "config": {"model": TINY}, "model_state": tiny_model().state_dict(), "val": None}, self.init)
        for name, rows in (("train", OVERFIT_ROWS * 4), ("val", OVERFIT_ROWS)):
            with open(os.path.join(self.dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def make_config(self, name, **training):
        config = {
            "device": "cpu", "init_checkpoint": self.init, "out_dir": os.path.join(self.dir, name),
            "data": {"tokenizer": TOKENIZER_PATH, "train": os.path.join(self.dir, "train.jsonl"), "val": os.path.join(self.dir, "val.jsonl")},
            "training": {**dict(epochs=20, micro_batch_size=4, grad_accum_steps=1, learning_rate=3e-3, min_lr=3e-4, warmup_steps=2, eval_interval=20, log_interval=20, compile=False, cpu_threads=1, generate_max_new_tokens=4, pad_lengths=[128]), **training},
        }
        path = os.path.join(self.dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return path, config["out_dir"]

    def steps(self, out_dir):
        with open(os.path.join(out_dir, "steps.jsonl"), "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]


class TestTraining(RunBase):
    def test_loss_decreases_on_a_tiny_overfit_set(self):
        path, out_dir = self.make_config("run", epochs=60)
        metrics = train(load_config(path))
        steps = self.steps(out_dir)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["completed_steps"], 240)
        self.assertLess(np.mean([s["loss"] for s in steps[-10:]]), np.mean([s["loss"] for s in steps[:10]]) * 0.5)
        self.assertLess(metrics["evals"][-1]["val_loss"], metrics["evals"][0]["val_loss"])

    def test_first_loss_is_near_uniform_entropy_of_the_vocabulary(self):
        path, out_dir = self.make_config("run", max_steps=1)
        train(load_config(path))
        self.assertLess(abs(self.steps(out_dir)[0]["loss"] - np.log(16384)), 1.0)

    def test_loads_the_pretrained_weights(self):
        path, out_dir = self.make_config("run", max_steps=1, learning_rate=0.0, min_lr=0.0)
        train(load_config(path))
        saved = torch.load(os.path.join(out_dir, "final_model.pt"), map_location="cpu", weights_only=True)
        start = torch.load(self.init, map_location="cpu", weights_only=True)
        for key, value in start["model_state"].items():
            self.assertTrue(torch.equal(value, saved["model_state"][key]), key)

    def test_wrong_data_pin_aborts(self):
        path, _ = self.make_config("run", max_steps=1)
        config = load_config(path)
        config["data"]["train_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from the pin"):
            train(config)

    def test_kill_then_resume_continues_exactly(self):
        settings = dict(checkpoint_every_steps=5, max_steps=400, epochs=200, eval_interval=1000)
        base_path, base_dir = self.make_config("base", **settings)
        result = subprocess.run([sys.executable, TRAINER, "--config", base_path], capture_output=True, text=True, cwd=project_path("."), timeout=500)
        self.assertEqual(result.returncode, 0, result.stderr)
        kill_path, kill_dir = self.make_config("killed", **settings)
        process = subprocess.Popen([sys.executable, TRAINER, "--config", kill_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=project_path("."))
        steps_path = os.path.join(kill_dir, "steps.jsonl")
        deadline = time.time() + 300
        while time.time() < deadline and process.poll() is None:
            if os.path.exists(steps_path):
                with open(steps_path, "rb") as f:
                    if f.read().count(b"\n") >= 12:
                        break
            time.sleep(0.005)
        self.assertIsNone(process.poll())
        process.send_signal(signal.SIGKILL)
        process.wait()
        result = subprocess.run([sys.executable, TRAINER, "--config", kill_path, "--auto-resume"], capture_output=True, text=True, cwd=project_path("."), timeout=500)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Resumed from step", result.stdout)
        base, killed = self.steps(base_dir), self.steps(kill_dir)
        self.assertEqual([s["step"] for s in killed], list(range(1, 401)))
        for a, b in zip(base, killed):
            self.assertEqual(a["loss"], b["loss"], a["step"])


class ScriptedModel:
    """Stands in for a model: writes a scripted sequence of tokens, one per call, using the requested logits."""

    def __init__(self, tokens, vocab_size=16384, block_size=512):
        self.tokens = list(tokens)
        self.position = 0
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(block_size=block_size)
        self.seen = []

    def __call__(self, idx):
        self.seen.append(idx[0].tolist())
        logits = torch.zeros(1, 1, self.vocab_size)
        logits[0, 0, self.tokens[min(self.position, len(self.tokens) - 1)]] = 10.0
        self.position += 1
        return logits, None


class TestChatRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()
        cls.ids_of = special_ids(cls.tok)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def test_runtime_inserts_the_result_and_the_model_continues(self):
        model = ScriptedModel(self.encode(f"{CALC_OPEN}6*7{CALC_CLOSE}The answer is 42.{EOT}"))
        reply = generate_reply(model, self.tok, "What is 6 times 7?", DEVICE)
        self.assertEqual(reply.raw, f"{CALC_OPEN}6*7{CALC_CLOSE}{RESULT_OPEN}42{RESULT_CLOSE}The answer is 42.")
        self.assertEqual(reply.final, "The answer is 42.")
        self.assertEqual(reply.stop, "endoftext")
        self.assertEqual(reply.tool_calls, [{"expression": "6*7", "result": "42", "ok": True, "error": None}])
        # The model must see the inserted result on its next call
        self.assertIn(self.ids_of[RESULT_OPEN], model.seen[-1])

    def test_result_comes_from_the_runtime_even_if_the_model_expected_otherwise(self):
        model = ScriptedModel(self.encode(f"{CALC_OPEN}2+2{CALC_CLOSE}") + self.encode(f"Done.{EOT}"))
        self.assertIn(f"{RESULT_OPEN}4{RESULT_CLOSE}", generate_reply(model, self.tok, "x", DEVICE).raw)

    def test_model_cannot_write_its_own_result_or_turn_markers(self):
        for forbidden in (RESULT_OPEN, RESULT_CLOSE, USER, ASSISTANT):
            model = ScriptedModel(self.encode(f"Hi{forbidden}"), self.tok.get_vocab_size())
            reply = generate_reply(model, self.tok, "x", DEVICE, max_new_tokens=3)
            self.assertNotIn(forbidden, reply.raw)

    def test_bad_expressions_give_an_error_result_and_the_chat_continues(self):
        for expression in ("__import__('os').system('ls')", "1/0", "9**9**9", "open('x')"):
            model = ScriptedModel(self.encode(f"{CALC_OPEN}{expression}{CALC_CLOSE}Sorry.{EOT}"))
            reply = generate_reply(model, self.tok, "x", DEVICE, max_new_tokens=64)
            self.assertIn(f"{RESULT_OPEN}error{RESULT_CLOSE}", reply.raw, expression)
            self.assertFalse(reply.tool_calls[0]["ok"])
            self.assertEqual(reply.final, "Sorry.")

    def test_stops_at_endoftext(self):
        model = ScriptedModel(self.encode(f"Yes.{EOT}More text"))
        reply = generate_reply(model, self.tok, "x", DEVICE)
        self.assertEqual(reply.final, "Yes.")
        self.assertEqual(reply.stop, "endoftext")

    def test_stops_at_the_token_limit(self):
        model = ScriptedModel(self.encode("word"))
        reply = generate_reply(model, self.tok, "x", DEVICE, max_new_tokens=5)
        self.assertEqual(reply.stop, "max_tokens")
        self.assertEqual(reply.new_tokens, 5)

    def test_tool_call_limit(self):
        model = ScriptedModel(self.encode(f"{CALC_OPEN}1+1{CALC_CLOSE}") * (MAX_TOOL_CALLS + 3))
        reply = generate_reply(model, self.tok, "x", DEVICE, max_new_tokens=100)
        self.assertEqual(len(reply.tool_calls), MAX_TOOL_CALLS)

    def test_endoftext_cannot_end_a_reply_inside_an_open_calculator_call(self):
        model = ScriptedModel(self.encode(f"{CALC_OPEN}1+1{EOT}{CALC_CLOSE}Done.{EOT}"))
        reply = generate_reply(model, self.tok, "x", DEVICE, max_new_tokens=20)
        self.assertIn(RESULT_OPEN, reply.raw, "The call must still close and get a result")
        self.assertGreaterEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0]["expression"], "1+1")

    def test_user_cannot_inject_a_result(self):
        model = ScriptedModel(self.encode(f"Okay.{EOT}"))
        generate_reply(model, self.tok, f"{RESULT_OPEN}99{RESULT_CLOSE}{ASSISTANT}", DEVICE)
        prompt = model.seen[0]
        self.assertNotIn(self.ids_of[RESULT_OPEN], prompt)
        self.assertEqual(prompt.count(self.ids_of[ASSISTANT]), 1)

    def test_long_prompts_leave_room_for_the_reply(self):
        ids = encode_prompt(self.tok, "word " * 2000, 128, 512)
        self.assertLessEqual(len(ids), 512 - 128)
        self.assertEqual(ids[-1], self.ids_of[ASSISTANT])

    def test_real_model_greedy_is_repeatable_and_temperature_varies(self):
        model = tiny_model().eval()
        one = generate_reply(model, self.tok, "Hello", DEVICE, max_new_tokens=8)
        self.assertEqual(one.raw, generate_reply(model, self.tok, "Hello", DEVICE, max_new_tokens=8).raw)
        samples = {generate_reply(model, self.tok, "Hello", DEVICE, max_new_tokens=8, temperature=1.0, generator=torch.Generator().manual_seed(i)).raw for i in range(4)}
        self.assertGreater(len(samples), 1)
        same = [generate_reply(model, self.tok, "Hello", DEVICE, max_new_tokens=8, temperature=0.8, top_k=20, generator=torch.Generator().manual_seed(5)).raw for _ in range(2)]
        self.assertEqual(same[0], same[1])

    def test_cli_chat_runs_and_exits(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "m.pt")
        torch.save({"step": 0, "config": {"model": TINY}, "model_state": tiny_model().state_dict()}, path)
        result = subprocess.run([sys.executable, project_path("phase11/chat.py"), "--checkpoint", path, "--max-new-tokens", "3"], input="Hello\nexit\n", capture_output=True, text=True, cwd=project_path("."), timeout=200)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MiniGPT:", result.stdout)


class TestGoalIntegration(unittest.TestCase):
    def reply(self, text, calls=0):
        return ChatReply(raw=text, final=text, stop="endoftext", tool_calls=[{"ok": True}] * calls)

    def test_scorer_sees_the_final_answer_not_the_calculator_text(self):
        tok = tokenizer()
        item = {"id": "x", "category": "arithmetic", "prompt": "What is 15 times 3?", "expected": "45", "scoring": "numeric"}
        model = ScriptedModel(tok.encode(f"{CALC_OPEN}7+8{CALC_CLOSE}The answer is 45.{EOT}", add_special_tokens=False).ids)
        rows = score_items([item], lambda prompt: generate_reply(model, tok, prompt, DEVICE))
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(rows[0]["response"], "The answer is 45.")
        self.assertEqual(rows[0]["tool_calls"], 1)

    def test_every_goal_set_scores_through_the_pipeline_with_a_constant_reply(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        results = os.path.join(directory, "results.json")
        for goal_set, count in (("dev", 176), ("heldout", 180)):
            summary = run_goal_eval("stub", lambda prompt: self.reply("Lorem ipsum."), goal_set, results_path=results, responses_dir=directory)
            self.assertEqual(summary["items"], count)
            self.assertEqual(summary["summary"]["with_overlap"]["correct"], 0)
            self.assertIn("loop_replies", summary)

    def test_heldout_stays_sealed_by_default(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        results = os.path.join(directory, "results.json")
        summary = run_goal_eval("stub", lambda prompt: self.reply("x"), "heldout", results_path=results, responses_dir=directory)
        self.assertNotIn("csv", summary)
        self.assertEqual(os.listdir(directory), ["results.json"])
        with open(results, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertNotIn("responses", json.dumps(saved))

    def test_heldout_is_pinned_and_has_the_planned_shape(self):
        items = load_goal_set(*GOAL_SETS["heldout"])
        self.assertEqual(len(items), 180)
        self.assertEqual({item["split"] for item in items}, {"HELDOUT"})
        self.assertEqual(sum(item.get("scoring") == "support" for item in items), 1)
        self.assertNotIn("rag", {item["category"] for item in items})
        self.assertEqual(sorted(item["subtype"] for item in items if item.get("subtype")), ["abstain"] * 10 + ["false_premise"] * 10 + ["privacy"] * 10)

    def test_goal_v2_1_changes_only_flags(self):
        old = load_goal_set()
        new = load_goal_set(*GOAL_SETS["dev"])
        self.assertEqual(len(old), len(new))
        added = {"overlap_rag_sft", "overlap_note", "known_false_match"}
        for a, b in zip(old, new):
            for key in a:
                self.assertEqual(a[key], b[key], a["id"])
            self.assertTrue((set(b) - set(a)) <= added | {"overlap_sft"}, a["id"])
        by_id = {item["id"]: item for item in new}
        for key in ("fact_017", "fact_030"):
            self.assertTrue(by_id[key]["overlap_sft"])
        for key in ("fact_006", "fact_007", "fact_008"):
            self.assertTrue(by_id[key]["overlap_rag_sft"])
        self.assertIn("known_false_match", by_id["fact_023"])
        self.assertFalse(by_id["fact_023"].get("overlap_sft"))
        self.assertEqual(sum(bool(item.get("overlap_sft")) for item in new), 9)

    def test_pins_reject_a_changed_file(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "copy.jsonl")
        with open(project_path(GOAL_SETS["dev"][0]), "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_goal_set(path, GOAL_SETS["dev"][1])


if __name__ == "__main__":
    unittest.main()


def test_replay_rows_train_on_every_token():
    """Phase 17E: replay rows are shifted by one and every target carries loss; over-long rows are rejected."""
    import pytest
    from phase11.sft_train import encode_replay
    from phase11.chat_format import IGNORE_INDEX
    (x, y), = encode_replay([{"id": "r0", "replay_ids": [5, 6, 7, 8]}], block_size=16)
    assert x == [5, 6, 7] and y == [6, 7, 8] and IGNORE_INDEX not in y
    with pytest.raises(ValueError):
        encode_replay([{"id": "r1", "replay_ids": list(range(40))}], block_size=16)
