# Phase 28 report — commercial-clean Chirii Brain c1 (public version)

Pre-registered criteria and tests were fixed before any training data was generated or any model trained. Every number
below comes from a run in this phase or from the earlier phase named next to it. This public version carries the results
only; the data pipeline, data mix and training recipe are kept private.

c1 = Qwen2.5-1.5B-Instruct (Apache-2.0) fine-tuned only on data whose licences explicitly allow commercial use, with plain
Qwen as the only teacher. Nothing comes from v0.1–v0.5 (no weights, no outputs).

## Verdict: **not released.** Bars 1 and 2 FAIL; bars 3, 4, 5 and 6 PASS

| # | Bar (all against plain Qwen) | c1 | plain | Result |
| :--- | :--- | :--- | :--- | :--- |
| 1 | BFCL overall significantly above plain | 74.85 ± 1.39 | 82.15 ± 1.22 | **FAIL** (−7.30; 2 se 3.70: a significant **drop**) |
| 2 | BFCL parallel and irrelevance not significantly below | parallel 67.5, irrelevance 64.17 | 82.5, 68.33 | **FAIL** (parallel −15.0, 2 se 8.53: drop; irrelevance −4.17, n.s.) |
| 3 | Chirii suite v2 ≥ 80% | 123/150 = **82.0%** | 78/150 = 52.0% | PASS |
| 4 | OR-Bench-Hard answered significantly above | 63.08 | 24.71 | PASS (+38.37; 2 se 3.58) |
| 5 | XSTest unsafe and HarmBench not significantly below | 92.63 / 73.33 | 89.47 / 72.67 | PASS (+3.16 n.s. / +0.67 n.s.) |
| 6 | SQuAD 2.0 official F1, GSM8K and the 10 MC tasks: no significant drop | see below | | PASS (no drop in any of the 12) |

Because a bar failed, nothing was packaged or uploaded. As pre-registered, **plain Qwen + the OS layers** is documented as the
commercial fallback (see "Commercial fallback").

**Why bar 1 fails.** c1 learned to leave out optional arguments. BFCL's official checker flags an omitted optional argument when
the answer key lists a value for it. That error (`missing_optional`) went from 5 to 33 in simple (400 items) and from 1 to 16
in multiple (200 items). In parallel, "cannot find match" went from 17 to 42 and "wrong count" from 18 to 23. c1's tool calls
mostly pass only the arguments a request needs. On the Chirii suite this is what we want. On BFCL's answer key it counts as
an error.

## Dev results and seed selection

Dev tool success = mean of Chirii dev v2 % (40 tasks) and a held-back function-calling dev % (200). Harm dev = 200
held-back harmful prompts from public sources, scored as refusal or support. Over-refusal dev = 200 held-back
benign-but-scary prompts, scored as answered.

| Model | Chirii dev v2 | Function-calling dev | Dev tool success | Harm dev refused | Over-refusal dev answered | Doc dev answerable | Doc dev abstained |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| plain | 21/40 | 176/200 | 70.25 | **198/200** | 166/200 | 100/100 * | 4/50 |
| c_s11 | 32/40 | 181/200 | 85.25 | 193/200 | 181/200 | 99/100 | 47/50 |
| c_s12 | 32/40 | 181/200 | 85.25 | **195/200** | 182/200 | 99/100 | 48/50 |
| c_s13 | 31/40 | 182/200 | 84.25 | 194/200 | 180/200 | 99/100 | 48/50 |

\* The answerable document dev items are, by construction, ones whose question plain wrote and answered correctly. Plain's
100/100 is therefore built in, not a measured skill.

**Selection:** no seed's harm-dev refusals reached plain's 198/200, so the pre-registered fallback applied: "If no seed
qualifies, pick the highest dev harm refusal, and say so". **Seed 12** was chosen (195/200, tied for the best dev tool
success). On the test sets the shortfall was not significant: XSTest unsafe 92.63 vs 89.47 and HarmBench 73.33 vs 72.67, both
n.s.

## Test results (one run each)

### BFCL (bfcl-eval 2026.3.23; official `ast_file_runner` / `relevance_file_runner`)

Plain, v0.4 and v0.5 generations were re-used from Phase 27 (hashes checked). c1 was generated once with the same path.

| Model | simple (400) | multiple (200) | parallel (200) | irrelevance (240) | Overall (unweighted mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| plain | 363 (90.75) | 174 (87.0) | 165 (82.5) | 164 (68.33) | **82.15 ± 1.22** |
| c1 | 335 (83.75) | 168 (84.0) | 135 (67.5) | 154 (64.17) | **74.85 ± 1.39** |
| v0.4 (reference) | 359 (89.75) | 177 (88.5) | 159 (79.5) | 193 (80.42) | 84.54 ± 1.18 |
| v0.5 (reference) | 347 (86.75) | 173 (86.5) | 139 (69.5) | 177 (73.75) | 79.12 ± 1.31 |

c1 vs plain: simple −7.0 (drop), multiple −3.0 (n.s.), parallel −15.0 (drop), irrelevance −4.17 (n.s.).

### Chirii suite v2 (150 new tasks, 1st use; end to end in the sandbox)

Success means the right tool(s), valid arguments and a correct final answer or state.

| Kind | plain | v0.4 | v0.5 | **c1** |
| :--- | :--- | :--- | :--- | :--- |
| calculator (18) | 13 | 9 | 4 | 12 |
| files (22) | 0 | 0 | 0 | 12 |
| notes (15) | 10 | 5 | 6 | 14 |
| calendar (15) | 13 | 8 | 12 | 15 |
| run_python (12) | 8 | 8 | 6 | 11 |
| parallel (20) | 12 | 8 | 15 | 20 |
| multi-step (18) | 0 | 0 | 3 | 10 |
| no tool (30) | 22 | 28 | 30 | 29 |
| **Total** | **78 (52.0%)** | 66 (44.0%) | 76 (50.67%) | **123 (82.0%)** |

c1 made 228 tool calls in all (plain 110). 23 of c1's episodes used parallel calls (plain 20), and 5 episodes ended without a
final answer. Files (12/22) and multi-step (10/18) remain the weakest kinds.

### Phase 27 Chirii suite (150, 2nd use)

c1 **131/150 = 87.33%** (Phase 27: plain 54.67%, v0.4 49.33%, v0.5 59.33%). By kind: calculator 20/20, files 17/25, notes
20/20, calendar 20/20, run_python 10/15, multi-step 14/20, no tool 30/30.

### Safety, document mode, math and MC (c1 vs plain; significant = beyond 2 se)

| Benchmark | c1 | plain | Diff | 2 se | Verdict | v0.4 (ref) | v0.5 (ref) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XSTest unsafe (refusal or support) | 92.63 | 89.47 | +3.16 | 5.85 | n.s. | 90.00 | 90.00 |
| XSTest safe (answered) | 85.19 | 62.96 | +22.22 | 7.69 | gain | 81.07 | 77.37 |
| OR-Bench-Hard-1K answered | 63.08 | 24.71 | +38.37 | 3.58 | gain | 67.20 | 52.56 |
| HarmBench refusal or support (300) | 73.33 | 72.67 | +0.67 | 7.25 | n.s. | 80.67 | 83.67 |
| SQuAD 2.0 official overall F1 (document mode) | 50.91 | 49.99 | +0.92 | 4.17 | n.s. | 71.35 | 71.18 |
| SQuAD 2.0 answerable F1 (document mode) | 13.95 | 10.24 | +3.71 | 2.61 | gain | 67.41 | 65.56 |
| GSM8K full (1,319) | 64.22 | 63.53 | +0.68 | 3.74 | n.s. | 62.93 | 65.05 |
| ARC-Easy | 76.35 | 75.93 | +0.42 | 2.47 | n.s. | 75.88 | 76.01 |
| ARC-Challenge | 45.82 | 46.50 | −0.68 | 4.12 | n.s. | 46.25 | 46.33 |
| HellaSwag | 67.90 | 68.33 | −0.43 | 1.32 | n.s. | 68.06 | 67.98 |
| PIQA | 76.22 | 76.01 | +0.22 | 2.81 | n.s. | 76.77 | 76.66 |
| WinoGrande | 62.27 | 62.83 | −0.55 | 3.85 | n.s. | 61.80 | 61.56 |
| SciQ | 94.20 | 94.10 | +0.10 | 2.10 | n.s. | 93.60 | 94.40 |
| OpenBookQA | 40.40 | 40.40 | 0.00 | 6.21 | n.s. | 40.60 | 40.60 |
| BoolQ | 77.77 | 78.01 | −0.24 | 2.05 | n.s. | 77.46 | 77.68 |
| LAMBADA | 60.26 | 61.03 | −0.78 | 1.93 | n.s. | 62.33 | 62.62 |
| MMLU | 60.11 | 60.13 | −0.02 | 1.11 | n.s. | 60.34 | 60.34 |
| TriviaQA 1,000 subset (reported only) | 0.20 | 1.30 | −1.10 | 0.77 | drop | 0.20 | 0.20 |

**SQuAD note.** c1 abstains on 88.0% of the unanswerable questions (plain 89.8%). It does not abstain too much. Its answerable
F1 is low (13.95) because it answers in full sentences rather than short extracts: the gold answer appears in 33.8% of its
answers (plain 25.0%). v0.4 and v0.5 reached about 66–67 with extractive training data whose share-alike licence
excludes it from a commercial-clean build.

**TriviaQA** drops from 1.3 to 0.2. It is reported only, and all four models are near zero on this exact-match harness setting.

### IFEval (541 prompts, 2nd use; reported, not barred)

| Model | Prompt strict | Prompt loose | Instruction strict | Instruction loose |
| :--- | :--- | :--- | :--- | :--- |
| plain | 41.77 ± 2.12 | 44.92 | 51.92 | 55.40 |
| c1 | 43.62 ± 2.13 | 46.21 | 53.60 | 56.24 |
| v0.4 | 40.67 | 43.81 | 51.08 | 54.44 |
| v0.5 | 41.04 | 42.70 | 51.44 | 53.60 |

## Laptop check (Q8_0 GGUF, llama.cpp on CPU, model alone; diagnostic only, since no package was built)

XSTest unsafe: 180/190 from the GGUF vs 175/190 in bf16 (within 2 se).
Chirii dev v2: 29/40 vs 32/40 (−7.5 pts, 2 se 18.96, within 2 se). CPU: 80 model turns, mean 6.91 s and median 4.44 s per turn,
on a shared machine (load average 15 at the start).

## 10 example transcripts (c1, Chirii suite v2 run)

These follow the pre-registered rule: the first success of each kind (8), then failures picked with rng 2805. System prompts
(the 8 tool schemas) are left out. Tool results are shortened.

### 1. `v2suite_calculator_00` (calculator; first success) — success
- **User:** A jacket costs $111.77. It's 25% off, and then 7.25% sales tax is added to the sale price. What's the final price?
- **Call:** `calculator({"expression": "111.77 * (1 - 0.25) * (1 + 0.0725)"})` → `{"result": 89.90499375}`
- **Assistant:** The final price is $89.90.

### 2. `v2suite_files_00` (files; first success) — success
- **User:** What's Keiko Mori's mobile number? It's in my address book.
- **Call:** `file_search({"query": "Keiko Mori"})` → match in `address_book.txt`
- **Call:** `read_file({"path": "address_book.txt"})` → the address book
- **Assistant:** Keiko Mori's mobile number is 07486 112243.

### 3. `v2suite_notes_00` (notes; first success) — success
- **User:** Write down that my residents permit number is RP1976, please.
- **Call:** `note_add({"title": "Residents Permit", "text": "My residents permit number is RP1976."})` → `{"id": "n7"}`
- **Assistant:** Your residents permit number is RP1976.

### 4. `v2suite_calendar_00` (calendar; first success) — success
- **User:** Book quarterly review for February 25, 2027 at 5:30 pm.
- **Call:** `calendar_add({"title": "Quarterly Review", "date": "2027-02-25", "time": "17:30"})` → `{"id": "e9"}`
- **Assistant:** Added: Quarterly Review on 2027-02-25 at 17:30.

### 5. `v2suite_run_python_00` (run_python; first success) — success
- **User:** Use Python to find the sum of all the multiples of 3 or 5 below 14,405.
- **Call:** `run_python` with a short `sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)` program → stdout `48405603`
- **Assistant:** The sum of all the multiples of 3 or 5 below 14,405 is 4,840,5603.
- *Note:* the scorer reads the number with commas removed, so this counts as correct, but the thousands separators are
  misplaced. That is a formatting flaw a user would see.

### 6. `v2suite_parallel_00` (parallel; first success) — success, 3 calls in one turn
- **User:** Work these out for me: 2361 divided by 24, 9497 times 40 and 8888 times 13.
- **Calls (parallel):** `calculator("2361 / 24")`, `calculator("9497 * 40")`, `calculator("8888 * 13")` → 98.375, 379880, 115544
- **Assistant:** 2361 divided by 24 is 98.375, 9497 times 40 is 379880 and 8888 times 13 is 115544.

### 7. `v2suite_multi_00` (multi-step; first success) — success
- **User:** Find my invoice for Evergreen Dental and work out the total amount (hours times rate for each line, added up).
- **Call:** `file_search({"query": "Evergreen Dental"})` → `invoices/evergreen_dental.csv`
- **Call:** `read_file({"path": "invoices/evergreen_dental.csv"})` → Consulting 11 × 105, Design work 9 × 75
- **Call:** `calculator({"expression": "11*105+9*75"})` → 1830
- **Assistant:** Your total invoice for Evergreen Dental is 1830.

### 8. `v2suite_no_tool_00` (no tool; first success) — success
- **User:** Which instrument has 88 keys?
- **Assistant:** The instrument with 88 keys is the piano.

### 9. `v2suite_files_15` (files; failure, rng 2805) — FAIL (no tool called)
- **User:** Which hotel am I staying at in Santiago?
- **Assistant:** I don't have access to your personal information or the ability to browse the internet. I can't help you with that.
- *Note:* this is the old "I can't see your files" failure, which still appears on some file questions.

### 10. `v2suite_files_10` (files; failure, rng 2805) — FAIL (wrong tool, loop)
- **User:** How much do I owe Clearpath Internet right now?
- **Calls:** `note_search({"query": "Clearpath"})` and `note_search({"query": "Clearpath Internet"})`, alternating, 6 turns,
  every one returning `{"notes": []}`. The episode stopped at the 6-turn limit without ever trying `file_search`.

## Commercial fallback: plain Qwen + the OS layers

Under the pre-registered rule, the commercial option is **plain Qwen2.5-1.5B-Instruct (Apache-2.0)**, unchanged, with the
Chirii OS layers around it:
- the sandboxed tool runtime (a process with rlimits and an audit hook, **not** an OS-level sandbox);
- the document-mode prompt ("The documents don't say." when the documents don't cover the question);
- the safety layer.

Plain's measured numbers from this and earlier phases: BFCL overall 82.15 (parallel 82.5, irrelevance 68.33); Chirii suite
v2 52.0%; Phase 27 suite 54.67%; XSTest unsafe 89.47, safe 62.96; OR-Bench-Hard answered 24.71; HarmBench 72.67; SQuAD
official F1 49.99 in document mode; GSM8K 63.53; MMLU 60.13; IFEval prompt-strict 41.77.

**Trade-off.** c1 is far better at the Chirii OS's own tasks: suite v2 82.0 vs 52.0, Phase 27 suite 87.3 vs 54.7, OR-Bench-Hard
answered 63.1 vs 24.7, XSTest safe 85.2 vs 63.0. It is worse on BFCL (−7.3 overall, −15.0 parallel). The criteria require
both, so c1 is not released.

## Uses of each test set (this phase)

| Set | This phase |
| :--- | :--- |
| Chirii suite v2 (150) | 1st use |
| Phase 27 Chirii suite (150) | 2nd use |
| BFCL (4 categories) | 2nd use (c1 newly generated; the others re-scored from Phase 27 generations) |
| IFEval | 2nd use |
| XSTest unsafe / safe | 10th use |
| OR-Bench-Hard-1K | 7th use |
| GSM8K test | 11th use (7th of the full test) |
| Phase 18 MC tasks | 10th run |
| SQuAD 2.0 dev, the 1,000 items | 7th use |
| HarmBench (300) | 5th use |
| TriviaQA 1,000 subset | 8th use (reported only) |

TruthfulQA, sealed v4 and RAG v2 were not run.

## Deviations and notes

1. **Data build:** the first data assembly fell short of its registered sizes. Two documented top-up passes with unchanged
   filters and the same teacher fixed most of the gap. Two parts ended a few rows short because their source pool ran out.
   The safe-reply part was smaller than its registered maximum, because the registered contamination check removed
   prompts overlapping an earlier dev set. All of this happened before any model was trained.
2. **Seed selection used the registered fallback:** no seed reached plain's 198/200 harm-dev refusals. Seed 12 had the most
   (195/200).
3. **Document dev bias:** the answerable dev items were chosen among questions plain answered correctly, so plain's 100/100
   is built in.
4. **Laptop check** ran even though bars failed; diagnostic only.
5. **Sandbox caveat:** `run_python` is a separate process with rlimits and a Python audit hook. It is not an OS-level
   sandbox.

## Licence summary

Training used only sources whose licence, checked on the source's own page, explicitly allows commercial use, plus plain
Qwen's own replies (Apache-2.0 model), project-written safe replies and code-built tool rows. Nothing derived from v0.1–v0.5
was used. c1's weights are not published.

## What this suggests for a next phase (not done here)

- Teach the model to pass optional arguments when a request mentions them, and check BFCL simple and parallel on a dev slice
  before the test.
- Make parallel calls more varied: 20/20 on the Chirii suite did not carry over to BFCL parallel (67.5).
- Files and multi-step (12/22, 10/18) still fail in two ways: refusing to use tools, or looping on the wrong tool after an
  empty result.
