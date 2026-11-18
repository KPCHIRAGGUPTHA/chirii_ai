# Phase 29 report: Chirii Brain c2, the last training round (public version)

## How this phase was run

The criteria and tests were pre-registered before any training data was generated or any model was trained. Every number below comes from a run in this phase, or from the earlier phase named next to it.

This public version carries the results only. The data pipeline, data mix and training recipe are kept private.

**What c2 is.** c2 is Qwen2.5-1.5B-Instruct (Apache-2.0) fine-tuned only on data whose licences explicitly allow commercial use, with plain Qwen as the only teacher. Compared with c1, this round aimed at two things:
- filling every argument a request supports, optional ones included;
- making parallel calls.

Nothing comes from v0.1–v0.5 or from c1.

**What c1 is.** c1 is Phase 28's model. It is shown alongside c2 in every table.

## Verdict: **not released.** Bars 1 and 2 FAIL; bars 3, 4 and 5 PASS

| # | Bar (all against plain Qwen) | c2 | plain | Result | c1 (same bar) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | BFCL non-live overall and BFCL-live overall each not significantly below plain, and no BFCL category significantly below | non-live 78.77 ± 1.30, live 61.90 ± 3.24 | 82.15 ± 1.22, 60.90 ± 3.23 | **FAIL**. Both overalls n.s. (−3.38, 2 se 3.57; +1.00, 2 se 9.15). Two categories drop: **parallel** −12.0 (2 se 8.39) and **live_irrelevance** −5.77 (2 se 4.64) | FAIL (non-live −7.30, simple, parallel and live_irrelevance drop) |
| 2 | Chirii suite v2 ≥ 80%, with the number-format rule | 111/150 = **74.0%** | 76/150 = 50.67% | **FAIL** | 120/150 = 80.0% (re-scored; it would pass) |
| 3 | OR-Bench-Hard-1K answered significantly above | 72.92 | 24.71 | PASS (+48.21; 2 se 3.42) | 63.08, PASS |
| 4 | XSTest unsafe and HarmBench not significantly below | 91.58 / 68.00 | 89.47 / 72.67 | PASS (+2.11 n.s. / −4.67 n.s., 2 se 7.45) | 92.63 / 73.33, PASS |
| 5 | SQuAD 2.0 official F1, GSM8K full and the 10 MC tasks: no significant drop | see below | | PASS (no drop in any of the 12) | PASS |

Because a bar failed, nothing was packaged, converted to GGUF or uploaded. The **plain Qwen + OS layers** commercial fallback documented in Phase 28 stands (see "Commercial fallback").

**The BFCL bar was changed after Phase 28's result.**
- **The change:** Phase 28 required BFCL overall to be *significantly above* plain. Phase 29 requires both BFCL non-live and BFCL-live overall to be *not significantly below* plain, with no category of the 8 significantly below.
- **Who and when:** the project owner made this change after seeing c1's result, so it is a post-hoc relaxation.
- **Why:** plain already scores 82% on non-live BFCL, and the commercial-clean sources contain no parallel examples. The product needs a brain that keeps general function calling at plain's level while adding the Chirii skills. It does not need one that beats plain on a public benchmark.
- **What was tightened:** the bar now covers every category and a fresh benchmark, BFCL-live.
- **Effect:** c2 fails even the relaxed bar.

### What changed from c1, and what did not

- **Optional arguments: fixed for the most part.** BFCL's `missing_optional` errors fell from c1 to c2 in every category:

  | Category | c1 | c2 | plain |
  | :--- | ---: | ---: | ---: |
  | simple | 33 | 15 | 5 |
  | multiple | 16 | 5 | 1 |
  | live_simple | 13 | 7 | 4 |
  | live_multiple | 94 | 55 | 20 |

  Non-live overall went from a significant drop (c1 −7.30) to n.s. (c2 −3.38). simple (87.25) and multiple (89.0) are now n.s. against plain.
- **Parallel: not fixed.** BFCL parallel is 141/200 (70.5%) against plain's 165/200 (82.5%); c1 had 135.
  - The errors are mostly "cannot find match" (c2 38, c1 42, plain 17), i.e. the right number of calls but a call whose arguments do not match the answer key.
  - There are also wrong call counts (c2 21, c1 23, plain 18). In 19 of c2's 21, it made *fewer* calls than needed.
  - The parallel training did not carry over to BFCL's parallel items. live_parallel (16 items) is n.s. for both: c2 8/16, c1 9/16, plain 7/16.
- **Live irrelevance: better than c1, still below plain.** c2 answers without calling on 511/884 (57.81%), c1 on 423/884 (47.85%) and plain on 562/884 (63.57%). c2 still calls a tool when none of the offered functions fits more often than plain does. Non-live irrelevance is level with plain (164/240 for both).
- **Suite v2 fell from 80.0% (c1, re-scored) to 74.0%.** The loss is in files (8/22 vs c1's 12/22) and run_python (6/12 vs 8/12).
  - Nine of c2's 14 file failures called no tool at all (the "I don't have access to your information" reply); c1 had five such failures.
  - In run_python, 2 failures are malformed numbers caught by the new rule. In 3 others, c2 wrote code that computes a value without printing it, set `timeout_s: 5` although the request gave no timeout, and then stated a number that the tool had not returned.

## Dev results and seed selection

- **Combined dev tool score:** the mean of Chirii dev v2 % (40 tasks, number-format rule) and AST dev % (400 held-back items from a public function-calling dataset, never trained on: 300 with a call, 100 without).
- **Harm dev:** 200 held-back harmful prompts, scored as refusal or support with `scorer_v4`.

| Model | Chirii dev v2 | AST dev | Combined | Harm dev refused | z vs plain | OR-Bench-80K dev answered | Doc dev answerable | Doc dev abstained |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| plain | 20/40 | 367/400 | 70.88 | **198/200** | | 166/200 | 100/100 * | 4/50 |
| c2_s11 | 27/40 | 361/400 | 78.88 | 192/200 | −1.922 (significantly below) | 181/200 | 99/100 | 48/50 |
| c2_s12 | 27/40 | 363/400 | 79.12 | 194/200 | −1.429 | 184/200 | 99/100 | 48/50 |
| **c2_s13** | 27/40 | 364/400 | **79.25** | 194/200 | −1.429 | 184/200 | 99/100 | 48/50 |
| c1 (reference) | | 366/400 | | | | | | |

\* As in Phase 28, the answerable document dev items were chosen among questions plain wrote and answered correctly. Plain's 100/100 is therefore built in.

**Selection (registered rule).** The test is a one-sided pooled two-proportion z-test against plain's 198/200, significant at z < −1.645. Seed 11 (192/200) is significantly below plain; seeds 12 and 13 (194/200) are not. Of the two, **seed 13** has the highest combined score (79.25), so c2 = seed 13.

**Why the rule changed from Phase 28.** Phase 28's rule (≥ plain's 198/200) let no seed qualify: plain's 99% sits at the ceiling of a 200-item set, so the rule rejected differences of 1–4 prompts that are within noise. The z-test still rejects a real drop. On the test sets, c2 is not significantly below plain on XSTest unsafe (91.58 vs 89.47) or on HarmBench (68.00 vs 72.67, −4.67, 2 se 7.45).

### AST dev diagnostics (400 held-back items)

| Model | Correct | missing optional | missing required | extra args | wrong value | wrong count | wrong name | unparseable |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain | 367 | 7 | 1 | 5 | 28 | 2 | 0 | 1 |
| c2_s11 | 361 | 13 | 1 | 5 | 23 | 8 | 0 | 0 |
| c2_s12 | 363 | 13 | 1 | 4 | 20 | 9 | 0 | 0 |
| c2_s13 | 364 | 12 | 1 | 5 | 21 | 7 | 1 | 0 |
| c1 | 366 | 15 | 1 | 4 | 22 | 5 | 0 | 0 |

On this dev set, no c2 seed beats plain or c1. The combined score rose only through Chirii dev v2. On these items c2 still omits more optional arguments than plain does (12 vs 7). The source dataset has no parallel turns, so the AST dev has no parallel items, as registered.

## Test results (one run each)

### BFCL non-live (bfcl-eval 2026.3.23, official checkers; 3rd use)

Plain, v0.4 and v0.5 were re-used from Phase 27 and c1 from Phase 28, with hashes checked. c2 was generated once.

| Model | simple (400) | multiple (200) | parallel (200) | irrelevance (240) | Overall (unweighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| plain | 363 (90.75) | 174 (87.0) | 165 (82.5) | 164 (68.33) | **82.15 ± 1.22** |
| **c2** | 349 (87.25) | 178 (89.0) | 141 (70.5) | 164 (68.33) | **78.77 ± 1.30** |
| c1 | 335 (83.75) | 168 (84.0) | 135 (67.5) | 154 (64.17) | 74.85 ± 1.39 |
| v0.4 (ref) | 359 (89.75) | 177 (88.5) | 159 (79.5) | 193 (80.42) | 84.54 ± 1.18 |
| v0.5 (ref) | 347 (86.75) | 173 (86.5) | 139 (69.5) | 177 (73.75) | 79.12 ± 1.31 |

**c2 vs plain:**

| Category | Difference | 2 se | Verdict |
| :--- | ---: | ---: | :--- |
| simple | −3.5 | 4.42 | n.s. |
| multiple | +2.0 | 6.50 | n.s. |
| **parallel** | **−12.0** | **8.39** | **drop** |
| irrelevance | 0.0 | 8.49 | n.s. |

**c1 vs plain:**

| Category | Difference | 2 se | Verdict |
| :--- | ---: | ---: | :--- |
| simple | −7.0 | 4.69 | drop |
| multiple | −3.0 | 7.04 | n.s. |
| parallel | −15.0 | 8.53 | drop |
| irrelevance | −4.17 | 8.62 | n.s. |

### BFCL-live (same package and checkers; 1st use, fresh)

Plain, c1 and c2 were each generated once.

| Model | live_simple (258) | live_multiple (1,053) | live_parallel (16) | live_irrelevance (884) | Overall (unweighted) | Count-weighted |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| plain | 186 (72.09) | 676 (64.20) | 7 (43.75) | 562 (63.57) | **60.90 ± 3.23** | 64.72 |
| **c2** | 195 (75.58) | 676 (64.20) | 8 (50.0) | 511 (57.81) | **61.90 ± 3.24** | 62.87 |
| c1 | 188 (72.87) | 662 (62.87) | 9 (56.25) | 423 (47.85) | 59.96 ± 3.23 | 57.98 |

**c2 vs plain:**

| Category | Difference | 2 se | Verdict |
| :--- | ---: | ---: | :--- |
| live_simple | +3.49 | 7.73 | n.s. |
| live_multiple | 0.0 | 4.18 | n.s. |
| live_parallel | +6.25 | 35.22 | n.s. |
| **live_irrelevance** | **−5.77** | **4.64** | **drop** |

**c1 vs plain:**

| Category | Difference | 2 se | Verdict |
| :--- | ---: | ---: | :--- |
| live_simple | +0.78 | 7.86 | n.s. |
| live_multiple | −1.33 | 4.20 | n.s. |
| live_parallel | +12.5 | 35.08 | n.s. |
| live_irrelevance | −15.72 | 4.67 | drop |

live_parallel has only 16 items, so its se is wide (as registered). Replies cut at 512 tokens in live_irrelevance: plain 19, c1 21, c2 22.

### BFCL error types (official checker labels; counts)

| Error | plain | c1 | c2 |
| :--- | ---: | ---: | ---: |
| simple: missing_optional | 5 | 33 | 15 |
| multiple: missing_optional | 1 | 16 | 5 |
| parallel: cannot_find_match | 17 | 42 | 38 |
| parallel: wrong_count (too few / too many) | 18 (15 / 3) | 23 (21 / 2) | 21 (19 / 2) |
| irrelevance: called a function | 76 | 86 | 76 |
| live_simple: missing_optional | 4 | 13 | 7 |
| live_multiple: missing_optional | 20 | 94 | 55 |
| live_multiple: wrong_func_name | 61 | 52 | 52 |
| live_irrelevance: called a function | 322 | 461 | 373 |

### Chirii suite v2 (150; 2nd use; end to end in the sandbox; number-format rule)

Success means the right tool(s), valid arguments, a correct final answer or state, and **no malformed number**. Plain, v0.4, v0.5 and c1 were re-scored from their Phase 28 transcripts; nothing was generated again for them.

| Kind | plain | v0.4 | v0.5 | c1 | **c2** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| calculator (18) | 13 | 9 | 4 | 12 | 13 |
| files (22) | 0 | 0 | 0 | 12 | 8 |
| notes (15) | 10 | 5 | 6 | 14 | 14 |
| calendar (15) | 13 | 8 | 12 | 15 | 15 |
| run_python (12) | 6 | 7 | 6 | 8 | 6 |
| parallel (20) | 12 | 8 | 15 | 20 | 19 |
| multi-step (18) | 0 | 0 | 3 | 10 | 9 |
| no tool (30) | 22 | 28 | 30 | 29 | 27 |
| **Total** | 76 (50.67%) | 65 (43.33%) | 76 (50.67%) | 120 (80.0%) | **111 (74.0%)** |
| Without the number rule | 78 | 66 | 76 | 123 | 113 |
| Failed only by the number rule | 2 | 1 | 0 | 3 | 2 |

c2 made 196 tool calls (c1 228, plain 110). 20 episodes used parallel calls (c1 23), and 4 ended without a final answer (c1 5).

### Phase 27 Chirii suite (150; 3rd use; number-format rule)

| Kind | plain | v0.4 | v0.5 | c1 | **c2** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| calculator (20) | 17 | 16 | 12 | 20 | 18 |
| files (25) | 3 | 2 | 3 | 17 | 12 |
| notes (20) | 15 | 12 | 20 | 20 | 20 |
| calendar (20) | 14 | 13 | 15 | 20 | 19 |
| run_python (15) | 6 | 3 | 6 | 10 | 11 |
| multi-step (20) | 0 | 0 | 3 | 14 | 14 |
| no tool (30) | 27 | 28 | 30 | 30 | 29 |
| **Total** | 82 (54.67%) | 74 (49.33%) | 89 (59.33%) | 131 (87.33%) | **123 (82.0%)** |

No model's Phase 27 suite score was changed by the number rule.

### Safety, document mode, math and MC (vs plain; significant = beyond 2 se)

| Benchmark | c2 | plain | Diff | 2 se | Verdict | c1 | c1 verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XSTest unsafe (refusal or support) | 91.58 | 89.47 | +2.11 | 6.01 | n.s. | 92.63 | n.s. |
| XSTest safe (answered) | 84.77 | 62.96 | +21.81 | 7.72 | gain | 85.19 | gain |
| OR-Bench-Hard-1K answered | 72.92 | 24.71 | +48.21 | 3.42 | gain | 63.08 | gain |
| HarmBench refusal or support (300) | 68.00 | 72.67 | −4.67 | 7.45 | n.s. | 73.33 | n.s. |
| SQuAD 2.0 official overall F1 (document mode) | 50.97 | 49.99 | +0.98 | 4.10 | n.s. | 50.91 | n.s. |
| SQuAD 2.0 answerable F1 (document mode) | 20.19 | 10.24 | +9.95 | 2.73 | gain | 13.95 | gain |
| GSM8K full (1,319) | 65.58 | 63.53 | +2.05 | 3.73 | n.s. | 64.22 | n.s. |
| ARC-Easy | 75.59 | 75.93 | −0.34 | 2.49 | n.s. | 76.35 | n.s. |
| ARC-Challenge | 45.82 | 46.50 | −0.68 | 4.12 | n.s. | 45.82 | n.s. |
| HellaSwag | 67.86 | 68.33 | −0.48 | 1.32 | n.s. | 67.90 | n.s. |
| PIQA | 76.44 | 76.01 | +0.44 | 2.81 | n.s. | 76.22 | n.s. |
| WinoGrande | 62.19 | 62.83 | −0.63 | 3.85 | n.s. | 62.27 | n.s. |
| SciQ | 94.20 | 94.10 | +0.10 | 2.10 | n.s. | 94.20 | n.s. |
| OpenBookQA | 40.20 | 40.40 | −0.20 | 6.21 | n.s. | 40.40 | n.s. |
| BoolQ | 78.04 | 78.01 | +0.03 | 2.05 | n.s. | 77.77 | n.s. |
| LAMBADA | 60.95 | 61.03 | −0.08 | 1.92 | n.s. | 60.26 | n.s. |
| MMLU | 59.82 | 60.13 | −0.31 | 1.11 | n.s. | 60.11 | n.s. |
| TriviaQA 1,000 subset (reported only) | 0.20 | 1.30 | −1.10 | 0.77 | drop | 0.20 | drop |

**SQuAD note.** c2 abstains on 82.0% of the unanswerable questions; plain abstains on 89.8% and c1 on 88.0%. The gold answer appears in 47.0% of c2's answerable replies (c1 33.8%, plain 25.0%).

**TriviaQA** is reported only. Every model is near zero on this exact-match harness setting.

### IFEval (541 prompts; 3rd use; reported, not barred)

| Model | Prompt strict | Prompt loose | Instruction strict | Instruction loose |
| :--- | :--- | :--- | :--- | :--- |
| plain | 41.77 ± 2.12 | 44.92 | 51.92 | 55.40 |
| **c2** | 41.96 ± 2.12 | 44.73 | 52.28 | 55.52 |
| c1 | 43.62 ± 2.13 | 46.21 | 53.60 | 56.24 |

## 10 example transcripts (c2, Chirii suite v2 run)

These follow the registered rule: the first success of each kind (8), then failures picked with rng 2905. System prompts (the 8 tool schemas) are left out, and tool results are shortened. Full transcripts are kept private.

### 1. `v2suite_calculator_00` (calculator; first success): success
- **User:** A jacket costs $111.77. It's 25% off, and then 7.25% sales tax is added to the sale price. What's the final price?
- **Call:** `calculator({"expression": "111.77 * (1 - 0.25) * (1 + 0.0725)"})` → `{"result": 89.90499375}`
- **Assistant:** The final price is $89.90.

### 2. `v2suite_files_01` (files; first success): success
- **User:** When is Wanjiru Kamau's birthday? Check my address book.
- **Call:** `file_search({"query": "Wanjiru Kamau"})` → match in `address_book.txt`
- **Call:** `read_file({"path": "address_book.txt"})` → the address book
- **Assistant:** Wanjiru Kamau's birthday is January 18.

### 3. `v2suite_notes_00` (notes; first success): success
- **User:** Write down that my residents permit number is RP1976, please.
- **Call:** `note_add({"title": "Residents Permit Number", "text": "My residents permit number is RP1976."})` → `{"id": "n7"}`
- **Assistant:** I've written down your residents permit number, RP1976.

### 4. `v2suite_calendar_00` (calendar; first success): success
- **User:** Book quarterly review for February 25, 2027 at 5:30 pm.
- **Call:** `calendar_add({"title": "Quarterly Review", "date": "2027-02-25", "time": "17:30"})` → `{"id": "e9"}`
- **Assistant:** Added: Quarterly Review on 2027-02-25 at 17:30.

### 5. `v2suite_run_python_00` (run_python; first success): success
- **User:** Use Python to find the sum of all the multiples of 3 or 5 below 14,405.
- **Call:** `run_python` with a short `sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)` program → stdout `48405603`
- **Assistant:** The sum of all the multiples of 3 or 5 below 14,405 is 48,405,603.
- *Note:* c1 wrote "4,840,5603" for this same task in Phase 28. c2 formats it correctly here, but the same fault appears in two of c2's run_python failures (see below).

### 6. `v2suite_parallel_00` (parallel; first success): success, 3 calls in one turn
- **User:** Work these out for me: 2361 divided by 24, 9497 times 40 and 8888 times 13.
- **Calls (parallel):** `calculator("2361 / 24")`, `calculator("9497 * 40")`, `calculator("8888 * 13")` → 98.375, 379880, 115544
- **Assistant:** The results are: 98.375, 379880, and 115544.

### 7. `v2suite_multi_03` (multi-step; first success): success
- **User:** Find Jamal Carter's birthday in my address book and save it as a note.
- **Call:** `read_file({"path": "address_book.txt"})` → the address book
- **Call:** `note_add({"title": "Jamal Carter's birthday", "text": "April 26"})` → `{"id": "n1"}`
- **Assistant:** Saved Jamal Carter's birthday as a note.

### 8. `v2suite_no_tool_00` (no tool; first success): success
- **User:** Which instrument has 88 keys?
- **Assistant:** The instrument with 88 keys is the piano.

### 9. `v2suite_files_13` (files; failure, rng 2905): FAIL (no tool called)
- **User:** When is Toffee's parvovirus booster due?
- **Assistant:** I'm sorry, but I don't have any information about Toffee's parvovirus booster or any other specific product. I can help you with general information or tasks related to the functions I have been provided with.

### 10. `v2suite_files_07` (files; failure, rng 2905): FAIL (no tool called)
- **User:** Which hotel am I staying at in Hanoi?
- **Assistant:** I don't have access to your current location or the ability to search your device's location data. […] use a mapping service like Google Maps […]
- *Note:* this is the "I can't see your files" failure. It accounts for 9 of c2's 14 file failures (c1: 5).

**Also seen among the run_python failures (not part of the 10).**
- *Malformed numbers:* "118,449,5625" and "59,303,1645". The tool output was correct each time (1184495625 and 593031645), so both tasks fail only by the number rule.
- *Unprinted results:* three reverse-and-add tasks were run as code that computed the result without printing it, with `timeout_s: 5` although no timeout was asked for. Each got empty stdout, and c2 then stated a number that the tool had not returned.

## Commercial fallback: plain Qwen + the OS layers (stands)

As registered, the commercial option remains **plain Qwen2.5-1.5B-Instruct (Apache-2.0)**, unchanged, with the Chirii OS layers around it:
- the sandboxed tool runtime (a process with rlimits and an audit hook, **not** an OS-level sandbox);
- the document-mode prompt;
- the safety layer.

Plain's numbers in this phase: BFCL non-live 82.15, BFCL-live 60.90 (count-weighted 64.72), Chirii suite v2 50.67% with the number rule.

**Trade-off.** c2 keeps c1's gains on the Chirii OS's own work and fixes most of c1's optional-argument problem. The bars still fail on BFCL parallel and live irrelevance, and the suite falls to 74.0%.

| Measure | c2 | plain |
| :--- | :--- | :--- |
| Phase 27 suite | 82.0 | 54.67 |
| OR-Bench-Hard answered | 72.9 | 24.7 |
| XSTest safe | 84.8 | 63.0 |
| BFCL non-live overall | 78.77 (n.s.) | 82.15 |
| BFCL-live overall | 61.90 (n.s.) | 60.90 |

## Uses of each test set (this phase)

| Set | This phase |
| :--- | :--- |
| BFCL non-live (4 categories) | 3rd use (c2 newly generated; the others re-scored from earlier generations) |
| **BFCL-live (4 categories)** | **1st use** (plain, c1, c2 each generated once) |
| Chirii suite v2 (150) | 2nd use (c2 run; the others re-scored from saved transcripts) |
| Phase 27 Chirii suite (150) | 3rd use (c2 run; the others re-scored) |
| IFEval | 3rd use |
| XSTest unsafe / safe | 11th use |
| OR-Bench-Hard-1K | 8th use |
| GSM8K test | 12th use (8th of the full test) |
| Phase 18 MC tasks | 11th run |
| SQuAD 2.0 dev, the 1,000 items | 8th use |
| HarmBench (300) | 6th use |
| TriviaQA 1,000 subset | 9th use (reported only) |

TruthfulQA, sealed v4 and RAG v2 were not run.

## Deviations and notes

1. **One data-template fix before any training.** The first data build stopped at two registered data checks. The templates were fixed, and every check passed before training started.
2. **BFCL-live generation ran out of GPU memory, and batching was changed.**
   - The first live run (plain) stopped after live_simple: batches of 16 prompts with long function lists did not fit in 12 GB.
   - For live only, a batch is now capped at 16 prompts **and** at 16,384 padded prompt tokens. The rule is the same for plain, c1 and c2.
   - Plain's live_simple outputs from the first attempt, which had not been scored, were regenerated.
   - Non-live generation was not affected.
3. **The number-format rule** was clarified (small-number lists such as "2,3,5" are not judged) before the tests were committed.
4. **No GGUF and no laptop check**: they are registered to run only if every bar passes.
5. **Sandbox caveat:** `run_python` is a separate process with rlimits and a Python audit hook. It is not an OS-level sandbox.

## Licence summary

Training used only sources whose licences allow commercial use:
- glaive-function-calling-v2: Apache-2.0
- OpenAssistant oasst1: Apache-2.0
- GSM8K: MIT
- OR-Bench 80K and Toxic: CC BY 4.0
- AdvBench: MIT

The other rows are plain Qwen's own replies and project-written rows.

c2's weights are not uploaded.

## What this round shows (this was the last training round)

- **Fixed:** filling the optional arguments a request supports. BFCL `missing_optional` roughly halved, and non-live overall is no longer a significant drop.
- **Not fixed:** BFCL-style parallel calls. The parallel training did not carry over (parallel 70.5 vs plain 82.5).
- **Improved but still below plain:** tool restraint on BFCL-live irrelevance.
- **Regression on the Chirii suite:** file tasks fell back towards the "I can't see your files" reply, and suite v2 fell from 80.0% to 74.0%.
- **Result:** no c2 seed passes every bar, so the plain Qwen + OS layers fallback is the commercial configuration.
