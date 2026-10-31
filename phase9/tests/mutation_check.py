"""
Proves the tests would have failed before each fix. Each mutation reintroduces one bug in a temporary copy of the
code, runs the tests there, and requires at least one failure. Each suite first confirms its unmodified copy passes,
so a mutation failure cannot come from the copy itself. The real files are never changed.

Usage:
    python phase9/tests/mutation_check.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "phase9", "tests", "mutation_results.json")

TRAIN_V2 = "phase9/train_v2.py"
EVAL_STANDARD = "phase9/eval_standard.py"
PHASE8E = "phase8/evaluation/evaluate_model_c_vs_d.py"
SCORER = "phase9/goal_eval/score_goal_eval.py"

SUITES = [
    {
        "name": "phase9_trainer_and_eval",
        "files": ["model.py", "bpe_tokenizer.py", "phase9/__init__.py", TRAIN_V2, EVAL_STANDARD,
                  "phase9/tests/test_train_v2.py", "phase9/tests/test_eval_standard.py"],
        "pytest_args": ["phase9/tests/test_train_v2.py", "phase9/tests/test_eval_standard.py"],
        "mutations": [
            ("Old trainer: logged loss is the sum of micro-batch losses", TRAIN_V2,
             "return total / len(micro_batches)", "return total"),
            ("Old trainer: gradients not scaled per micro-batch", TRAIN_V2,
             "(loss / len(micro_batches)).backward()", "loss.backward()"),
            ("Old trainer: sequential reads", TRAIN_V2,
             "return torch.randint(0, num_tokens - block_size, (batch_size,), generator=generator)",
             "return torch.arange(batch_size) * block_size"),
            ("Old trainer: only the first 10.24M tokens used", TRAIN_V2,
             "return torch.randint(0, num_tokens - block_size, (batch_size,), generator=generator)",
             "return torch.randint(0, min(num_tokens, 10_240_000) - block_size, (batch_size,), generator=generator)"),
            ("Old trainer: weight_decay=0.0", TRAIN_V2, "\"weight_decay\": 0.01,", "\"weight_decay\": 0.0,"),
            ("Weight decay applied to biases, LayerNorm and embeddings", TRAIN_V2,
             "(no_decay if id(param) in no_decay_ids else decay).append(param)", "decay.append(param)"),
            ("Resume does not restore the sampler state", TRAIN_V2,
             "generator.set_state(state[\"sampler_rng_state\"].cpu())", "pass"),
            ("Evaluation ignores pinned hashes", EVAL_STANDARD,
             "if actual != shard[f\"{key}_sha256\"]:", "if False:"),
            ("Evaluation chunks overlap", EVAL_STANDARD,
             "begin = start * block_size\n", "begin = start * (block_size - 1)\n"),
            ("Bits per byte computed per token (no byte normalization)", EVAL_STANDARD,
             "mean_loss * eval_tokens / math.log(2) / total_bytes", "mean_loss / math.log(2)"),
            ("Checkpoint loader rejects the Phase 6 SFT format", EVAL_STANDARD,
             "for key in (\"model_state\", \"model_state_dict\") if key in state",
             "for key in (\"model_state\",) if key in state"),
            ("Unweighted batch-mean loss misweights the final batch", EVAL_STANDARD,
             "mean_loss = float(block_losses.mean())",
             "mean_loss = float(np.mean([block_losses[i:i + batch_size].mean() for i in range(0, num_blocks, batch_size)]))"),
        ],
    },
    {
        "name": "phase8e_evaluation",
        "files": ["model.py", "phase8/__init__.py", "phase8/model_scaling_config.py", PHASE8E,
                  "phase8/evaluation/model_c_vs_d/results.json", "phase8/evaluation/model_c_vs_d/MODEL_C_VS_D_REPORT.md",
                  "tests/test_phase8e_evaluation.py"],
        # Real checkpoints are not copied; the hash-gate logic is tested on temporary files instead
        "pytest_args": ["tests/test_phase8e_evaluation.py", "-k", "not test_artifact_hashes_integrity"],
        "mutations": [
            ("Sign: loss difference is C - D", PHASE8E,
             "\"loss_diff_d_minus_c\": round(d_loss - c_loss, 4)", "\"loss_diff_d_minus_c\": round(c_loss - d_loss, 4)"),
            ("Sign: perplexity difference is C - D", PHASE8E,
             "\"ppl_diff_d_minus_c\": round(d_ppl - c_ppl, 2)", "\"ppl_diff_d_minus_c\": round(c_ppl - d_ppl, 2)"),
            ("Memory: handle return type not declared", PHASE8E,
             "    kernel32.GetCurrentProcess.restype = wintypes.HANDLE\n"
             "    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]\n",
             ""),
            ("Memory: failed lookup reported as 0.0", PHASE8E,
             "    except (AttributeError, OSError):\n        return None", "    except (AttributeError, OSError):\n        return 0.0"),
            ("Memory: missing value printed as a measurement", PHASE8E,
             "return f\"`{value} MB`\" if value is not None else \"n/a\"", "return f\"`{value or 0.0} MB`\""),
            ("Hard-coded: recommendation", PHASE8E, "**{rec['decision']}**", "**DEFER_MODEL_E**"),
            ("Hard-coded: Model C loss in the metrics table", PHASE8E,
             "**{c['validation_loss']:.4f}**", "**2.9970**"),
            ("Hard-coded: Model D loss in the metrics table", PHASE8E,
             "**{d['validation_loss']:.4f}**", "**3.9837**"),
            ("Hard-coded: conclusion text regardless of outcome", PHASE8E,
             "- It measures how these two checkpoints perform on the same validation tokens.",
             "- Model D shows degradation against Model C."),
            ("Hard-coded: model specification table", PHASE8E,
             "| {c['parameter_count']:,} | {d['parameter_count']:,} |", "| 6,613,504 | 15,164,800 |"),
            ("Decision rule: any improvement proceeds, ignoring the design gate", PHASE8E,
             "if loss_pct > 0 and d_ppl < PPL_GATE:", "if loss_pct > 0 and ppl_pct > 0:"),
            ("Decision rule: wrong perplexity gate", PHASE8E, "PPL_GATE = 15.0", "PPL_GATE = 30.0"),
            ("Validation hash: entry removed from the pinned list", PHASE8E,
             "    ),\n    \"Validation Tokens\": (\n        VAL_TOKENS_PATH,\n"
             "        \"a45af657b27115440ed995f738ebe1509d9b51f973003bbcdce55108f1f49c8a\"\n    )\n",
             "    )\n"),
            ("Validation hash: existence checked but mismatch allowed", PHASE8E,
             "passed = (actual_sha.lower() == expected_sha.lower())",
             "passed = name == \"Validation Tokens\" or (actual_sha.lower() == expected_sha.lower())"),
        ],
    },
    {
        "name": "goal_eval_scorer",
        "files": ["model.py", "bpe_tokenizer.py", "phase9/__init__.py", "phase9/train_v2.py", "phase9/eval_standard.py",
                  "phase6/preparation/prepare_sft_dataset.py", SCORER, "phase9/goal_eval/build_goal_set.py",
                  "phase9/goal_eval/goal_prompts.jsonl", "phase9/tests/test_goal_eval.py"],
        "pytest_args": ["phase9/tests/test_goal_eval.py"],
        "mutations": [
            ("RAG: copying the passage is accepted", SCORER,
             "return words_only(first_line(response)).startswith(words_only(first_sentence(passage)))", "return False"),
            ("Keywords searched in the whole response", SCORER,
             "text = first_sentence(response) if item.get(\"category\") in FIRST_SENTENCE_CATEGORIES else response", "text = response"),
            ("Repeating all options is accepted", SCORER,
             "if item.get(\"options\") and fails_options(text, item[\"options\"], item[\"expected\"][\"all\"][0]):", "if False:"),
            ("Prompt echo is accepted", SCORER,
             "if item.get(\"no_echo\") and words_only(text) in words_only(item[\"prompt\"]):", "if False:"),
            ("Minimum words ignored", SCORER,
             "if len(words_only(text).split()) < item.get(\"min_words\", 0):", "if False:"),
            ("Case ignored on literal items", SCORER,
             "return line == target if literal else line.lower() == target.lower()", "return line.lower() == target.lower()"),
            ("Literal items read number words", SCORER,
             "if not literal and re.fullmatch(", "if re.fullmatch("),
            ("Comma spacing fails exact matches", SCORER,
             "line = re.sub(r\"\\s*,\\s*\", \", \", re.sub(r\"\\s+\", \" \", line.rstrip(\".!\")))", "line = line.rstrip(\".!\")"),
            ("Number followed by a magnitude read as the small number", SCORER,
             "value *= MAGNITUDES[tokens[index]]", "pass"),
            ("Number answers take the last number mentioned", SCORER,
             "return numbers[0] if numbers else None", "return numbers[-1] if numbers else None"),
            ("min_any ignored", SCORER,
             "return sum(contains_word(response, word) for word in options) >= expected.get(\"min_any\", 1)",
             "return any(contains_word(response, word) for word in options)"),
            ("Keyword alternatives not read", SCORER,
             "alternatives = entry if isinstance(entry, list) else [entry]", "alternatives = entry[:1] if isinstance(entry, list) else [entry]"),
            ("Refusal followed by compliance is accepted", SCORER,
             "    if re.search(COMPLIANCE_PATTERNS, response.lower().replace(\"’\", \"'\")):\n        return False\n", ""),
            ("Refusal phrases accepted without a following action", SCORER,
             "(?:help|assist|provide|share|do|give|tell|explain|write|support|teach|answer|create|make)\\b", "\\w+"),
            ("Support item scored as refusal", SCORER,
             "return offers_support(response)", "return is_refusal(response)"),
            ("Premise corrections not accepted as unknown", SCORER,
             "    r\"\\bthere (?:is|are) no\\b\",\n", ""),
            ("Training overlap not separated", SCORER,
             "tally([row for row in scored if row.get(\"in_sft_training\")])", "tally([])"),
        ],
    },
]


def run_pytest(cwd, pytest_args):
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args, "-q", "-p", "no:cacheprovider", "--no-header", "--tb=no"],
        cwd=cwd, capture_output=True, text=True,
    )
    failed = [line.split(" ")[1].split("::")[-1] for line in completed.stdout.splitlines() if line.startswith("FAILED")]
    return completed.returncode, failed


def copy_suite(suite, target):
    for name in suite["files"]:
        destination = os.path.join(target, name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(os.path.join(PROJECT_ROOT, name), destination)


def run_suite(suite):
    with tempfile.TemporaryDirectory() as tmp:
        copy_suite(suite, tmp)
        returncode, failed = run_pytest(tmp, suite["pytest_args"])
        if returncode != 0:
            raise RuntimeError(f"{suite['name']}: unmodified copy does not pass (exit {returncode}): {failed}")

    results = []
    for bug, file_name, old, new in suite["mutations"]:
        with tempfile.TemporaryDirectory() as tmp:
            copy_suite(suite, tmp)
            target = os.path.join(tmp, file_name)
            with open(target, "r", encoding="utf-8") as f:
                source = f.read()
            if old not in source:
                raise ValueError(f"Mutation target not found: {bug}")
            with open(target, "w", encoding="utf-8") as f:
                f.write(source.replace(old, new))
            returncode, failed = run_pytest(tmp, suite["pytest_args"])
            results.append({"bug": bug, "caught": returncode == 1 and bool(failed), "failed_tests": failed})
    return {"baseline_passed": True, "mutations": results}


def main():
    output = {suite["name"]: run_suite(suite) for suite in SUITES}
    for name, suite_result in output.items():
        print(f"== {name}: unmodified copy passed", flush=True)
        for result in suite_result["mutations"]:
            status = "CAUGHT" if result["caught"] else "MISSED"
            print(f"{status}  {result['bug']}: {', '.join(result['failed_tests'])}", flush=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    if not all(r["caught"] for s in output.values() for r in s["mutations"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
