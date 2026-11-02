"""
Mutation check for scorer v3: each mutation breaks one rule in a temporary copy and requires the v3 tests to fail.
The rules v3 inherits from scorer_v2 are mutated in the copy of v2. Uses the runner in phase9/tests/mutation_check.py.

The phase9 check is also rerun here with its results written beside these, so its tracked results file stays as recorded.

Usage:
    CUDA_VISIBLE_DEVICES="" python phase11/tests/mutation_check_v3.py          # v3 only
    CUDA_VISIBLE_DEVICES="" python phase11/tests/mutation_check_v3.py --all    # also rerun every phase9 suite
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from phase9.tests import mutation_check as runner

V2 = "phase9/goal_eval/scorer_v2.py"
V3 = "phase11/goal_eval/scorer_v3.py"
OUTPUT_V3 = os.path.join(PROJECT_ROOT, "phase11", "logs", "mutation_results_v3.json")
OUTPUT_V2 = os.path.join(PROJECT_ROOT, "phase11", "logs", "mutation_results_phase9_rerun.json")

SCORER_MUTATIONS = [m for suite in runner.SUITES if suite["name"] == "goal_eval_v2_and_retrain" for m in suite["mutations"] if m[1] == V2]
# v3 has its own number reading, tested below. The hash, overlap and CSV code is unchanged v2 code, tested by the phase9 suite.
SKIPPED = ("Rule 2:", "Goal set hash", "Overlap not separated", "CSV verdict")
INHERITED = [m for m in SCORER_MUTATIONS if not m[0].startswith(SKIPPED)]

SIGN_MUTATIONS = [
    ("Rule 2: restated expression not skipped", V3, "if before and set(before) <= prompt_numbers:", "if False:"),
    ("Rule 2: '10 million' read as 10", V3, "value *= MAGNITUDES[parts[index]]", "pass"),
    ("Sign: minus sign dropped (the v2 bug)", V3, "numbers.append(sign * value)", "numbers.append(value)"),
    ("Sign: number words lose their sign", V3, "numbers.append(sign * (total + current))", "numbers.append(total + current)"),
    ("Sign: subtraction read as a negative", V3, r"(?<![a-z0-9)\]%])-(?=", r"-(?="),
    ("Sign: 'minus' after a number read as negative", V3, "part in SIGN_WORDS and not after_number and", "part in SIGN_WORDS and"),
    ("Sign: 'minus' word ignored", V3, "SIGN_WORDS = (\"minus\", \"negative\")", "SIGN_WORDS = ()"),
    ("Sign: 'negative' word ignored", V3, "SIGN_WORDS = (\"minus\", \"negative\")", "SIGN_WORDS = (\"minus\",)"),
    ("Sign: spaced minus after 'is' ignored", V3, "SPACED_MINUS.sub(r\"\\1 -\", clean(text).translate(MINUS_CHARACTERS))", "clean(text).translate(MINUS_CHARACTERS)"),
    ("Sign: unicode minus ignored", V3, "clean(text).translate(MINUS_CHARACTERS)", "clean(text)"),
    ("Echo: negative answer counted as an echo", V3, "return v2.is_echo(region, item) and not adds_negative_answer(region, item)", "return v2.is_echo(region, item)"),
    ("Echo: every negative answer exempt", V3,
     "return expected in extract_numbers(region) and expected not in extract_numbers(item.get(\"question\", item[\"prompt\"]))", "return True"),
    ("Restated expression compared without signs", V3, "if before and set(before) <= prompt_numbers:", "if before and {abs(n) for n in before} <= {abs(n) for n in prompt_numbers}:"),
    ("v3 scores with v2's number reading", V3,
     "return answer_number(region, item.get(\"question\", item[\"prompt\"])) == float(item[\"expected\"])",
     "return v2.answer_number(region, item.get(\"question\", item[\"prompt\"])) == float(item[\"expected\"])"),
]

SUITE = {
    "name": "scorer_v3",
    "files": ["model.py", "bpe_tokenizer.py", "phase9/__init__.py", "phase11/__init__.py", "phase11/tests/__init__.py",
              "phase9/train_v2.py", "phase9/eval_standard.py", V2, V3, "phase9/goal_eval/goal_prompts_v2.jsonl", "phase9/goal_eval/goal_set_v2_meta.json",
              "phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json", "phase9/tests/scorer_fixtures.py", "phase11/tests/test_scorer_v3.py"],
    "pytest_args": ["phase11/tests/test_scorer_v3.py"],
    "mutations": SIGN_MUTATIONS + INHERITED,
}


def main():
    results = {"scorer_v3": runner.run_suite(SUITE)}
    with open(OUTPUT_V3, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if "--all" in sys.argv:
        phase9 = {suite["name"]: runner.run_suite(suite) for suite in runner.SUITES}
        with open(OUTPUT_V2, "w", encoding="utf-8") as f:
            json.dump(phase9, f, indent=2)
        results.update(phase9)
    missed = 0
    for name, suite_result in results.items():
        caught = sum(r["caught"] for r in suite_result["mutations"])
        print(f"== {name}: unmodified copy passed; {caught}/{len(suite_result['mutations'])} mutations caught", flush=True)
        for result in suite_result["mutations"]:
            if not result["caught"]:
                missed += 1
                print(f"MISSED  {result['bug']}", flush=True)
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
