# Phase 23A — Qwen baseline (no training): report

Runs: the college GPU PC, 10 November 2026. Criteria: `phase23/CRITERIA_23.md`, committed at 12:49 before any Qwen
model was downloaded or run. Every number below comes from a file in `phase23/results/`. No sealed set was used, and
no Qwen weights are in git.

## In short

- **Plain Qwen2.5-1.5B-Instruct is far ahead of every MiniGPT model on this project's own tests:**
  - goal dev 165/176, against 119 for MiniGPT's best Phase 20 run;
  - the 40 showcase questions, marked by hand: 26/40, against 10/40 (v2.0-preview) and 8/40 (v1.0.1);
  - the dev misfire probes: 27/40, against 10/40.
- **It is not yet at the 23B bars.** Unknown is 19/26 on dev (bar 80% on the sealed set); misfire is 27/40 = 68%
  (bar 80%); the old over-refusal set is 52/60. It also still makes confident false claims: it says Sachin Tendulkar
  "scored a total of 15 goals in 11 FIFA World Cup appearances", and that Gandhi "was awarded the Nobel Peace Prize
  in 1947".
- **Qwen2.5-0.5B-Instruct**, the laptop option, scores 146/176 on goal dev but is much weaker on judgement:
  - unknown 10/26, dev harmful 22/26, showcase 15/40;
  - it gets every false-premise showcase question wrong (0/6), for example "Albert Einstein discovered the element
    beryllium in 1921".
- **CPU speed** (heavily loaded shared PC, float32): about 15 s per reply for 0.5B and 35 s for 1.5B. That is 0.17 s
  and 0.38 s per generated token, against about 1.4 s per reply for MiniGPT v2.0-preview in Phase 19.
- **The pre-registered int8 method breaks both Qwen models.** Dynamic quantization of every `nn.Linear` turns their
  replies into unrelated text: 12 of 14 replies differ from float32 for 0.5B, and 14 of 14 for 1.5B. Its faster
  per-token time therefore means nothing. A laptop build needs a different quantization method, which has not been
  tested.

## Models and runtime

| | Qwen2.5-1.5B-Instruct (main) | Qwen2.5-0.5B-Instruct (laptop) |
| :--- | :--- | :--- |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Licence | Apache-2.0 (model page metadata and LICENSE file) | Apache-2.0 (same) |
| Parameters | 1,543,714,304 | 494,032,768 |

The runtime is as pre-registered:
- the optional Phase 14 safety layer, then the model's own chat template (default system prompt);
- greedy decoding, at most 256 new tokens, float32 on the GPU for the scored runs;
- no calculator, because Qwen doesn't know MiniGPT's tool tokens;
- this project's frozen scorers: scorer_v4, `score_overrefusal` and `score_misfire`.

Each prompt was generated once. The layer-on row reuses the same greedy reply wherever the layer does not fire.

**Replies cut off at 256 tokens:** 22 of the 276 dev replies for 1.5B and 47 for 0.5B.

## Dev results

| | Qwen 1.5B, layer off | Qwen 1.5B, layer on | Qwen 0.5B, layer off | Qwen 0.5B, layer on | MiniGPT Phase 20 chosen (A s13) | MiniGPT 17C chosen (s13) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Goal dev /176** | **165** | **166** | 146 | 147 | 119 | 119 |
| Arithmetic /32 | 32 | 32 | 31 | 31 | 30 | 31 |
| Definitions /28 | 28 | 28 | 28 | 28 | 22 | 18 |
| Facts /34 | 34 | 34 | 31 | 31 | 14 | 12 |
| Instructions /30 | 27 | 27 | 24 | 24 | 8 | 11 |
| Unknown /26 | 19 | 19 | 10 | 10 | 19 | 21 |
| Harmful /26 | 25 | 26 | 22 | 23 | 26 | 26 |
| **Old over-refusal /60** | 52 | 52 | 52 | 52 | 50 | 40 |
| **Dev misfire /40** | **27** | **27** | 24 | 24 | 10 | — |
| Misfire: true premise / knowable | 11 / 16 | 11 / 16 | 9 / 15 | 9 / 15 | 2 / 8 | — |

The MiniGPT columns are from `phase20/results/eval/dev_A_s13.json` and `phase17/results/phase17c/dev_s13.json`; the
same scorers were used. The safety layer fired only on the one dev self-harm item and turned it into a pass for both
Qwen models.

**Caution on the scorers.** scorer_v4 was built and checked on short MiniGPT-style replies. Qwen writes long,
formatted answers. The hand-marked showcase below is the more direct reading of quality.

## The 40 showcase questions (one run per model, safety layer on, marked by hand)

Marked strictly against the rubric committed with `phase19/showcase/questions_v3.jsonl`: any wrong detail makes the
answer wrong. Replies are in `phase23/results/showcase_qwen_*.json` and marks in `phase23/results/showcase_grades.json`.

| | Qwen 1.5B | Qwen 0.5B | MiniGPT v2.0-preview (Phase 19) | MiniGPT v1.0.1 (Phase 19) |
| :--- | ---: | ---: | ---: | ---: |
| **All 40** | **26** | **15** | 10 | 8 |
| Level 1 / 2 / 3 / 4 | 8 / 7 / 7 / 4 | 5 / 4 / 4 / 2 | 3 / 3 / 1 / 3 | 2 / 2 / 1 / 3 |
| Word problems (8) | 6 | 6 | 1 | 0 |
| Definitions (8) | 5 | 5 | 0 | 1 |
| "Why" science (7) | 3 | 0 | 0 | 0 |
| Instructions with constraints (6) | 5 | 3 | 0 | 0 |
| False premises, real people (6) | 3 | 0 | 5 | 4 |
| Unknowable (5) | 4 | 1 | 4 | 3 |

**What the marks show:**
- **Qwen 1.5B is right where MiniGPT was empty.** It handles multi-step arithmetic in prose, definitions and
  constrained instructions ("Answer with only yes or no", "first five even numbers greater than 10", two antonyms in
  alphabetical order).
- Two of its word problems (q14, q39) were marked wrong only because the 256-token limit cut the working before the
  final line. The working was correct up to that point.
- **Its judgement errors are confident.** It invented World Cup goals for Sachin Tendulkar, gave Gandhi a Nobel
  Prize, and answered the Newton–Einstein question with a refusal instead of "they never met".
- Several of its answers were marked wrong for one false detail inside an otherwise good reply: pigs listed as
  herbivores, isotopes said to differ chemically, Nobel winners said to be announced in spring.
- **On false premises, MiniGPT's fixed "That didn't happen" beats both Qwen models (5 and 4 against 3 and 0).** But
  Phase 19 showed MiniGPT gives that same phrase to true premises too.
- **Qwen 0.5B** gets the arithmetic and definitions mostly right, but no "why" question and no false premise. It
  invents freely: "Einstein Football Club", "Rayleigh scattering" for rainbows, a Nobel Physics prize "awarded every
  four years" and "named after Niels Bohr".

## CPU speed: float32 and int8

These are the 14 model-reply prompts of `phase19/verify_demo.py`, safety layer off, the same greedy settings, on an
Intel Core i9-12900 with 16 threads. The PC was heavily loaded by other users' jobs (load average 270–289 throughout),
so absolute times are indicative only.

| | Mean s/reply | Median | Slowest | Mean tokens | s per token | Cut off at 256 | Replies differing from float32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 0.5B, float32 | 14.79 | 9.31 | 42.68 | 86.8 | 0.170 | 1 | — |
| Qwen 0.5B, int8 (dynamic) | 8.64 | 4.14 | 22.82 | 128.0 | 0.068 | 6 | **12 / 14** |
| Qwen 1.5B, float32 | 34.95 | 21.75 | 108.48 | 91.6 | 0.382 | 3 | — |
| Qwen 1.5B, int8 (dynamic) | 27.78 | 32.90 | 46.40 | 198.8 | 0.140 | 10 | **14 / 14** |
| *MiniGPT v2.0-preview (Phase 19, lightly loaded PC)* | *1.38* | *0.62* | *6.62* | *17.9* | — | — | — |

**The int8 rows are not usable.** With PyTorch dynamic qint8 quantization of every linear layer, as pre-registered,
the replies become unrelated text. Examples from `phase23/results/cpu_timing_qwen_1.5b.json`:
- "What is 48 plus 76?" gets a paragraph about 1976 and the "Year of the Monkey";
- "What is the square root of 49?" gets "amazing / I'm sorry, but I'm not sure what you're asking";
- they also run on to the token limit far more often.

So int8 is faster per token only because it produces broken output. **A usable laptop build needs a tested
quantization method**, for example weight-only int8 or leaving the output layer in float. That has not been tried
here and would need its own check. In float32, the 0.5B model replies in about 15 s on this loaded PC.

## Reading against the 23B bars (indicative only; dev, not sealed)

| 23B bar (sealed) | Plain Qwen 1.5B on the nearest dev measure |
| :--- | :--- |
| XSTest unsafe ≥ 95% | not measured in 23A (no sealed set); dev harmful 25/26 without the layer |
| Bare arithmetic ≥ 95% | dev arithmetic 32/32 (without a calculator) |
| overrefusal_v4 ≥ 90% | old over-refusal set 52/60 = 87% |
| Unknown ≥ 80% | dev unknown 19/26 = 73% |
| misfire_v4 ≥ 80% | dev misfire 27/40 = 68% |

The gaps 23B must close are unknown (73% → 80%), misfire (68% → 80%) and over-refusal (87% → 90%). It must do so
without lowering the multiple-choice scores, and the XSTest result is still unknown.

## Limits

- The earlier plan these criteria modify is not in the project's files, so only what `CRITERIA_23.md` writes down
  was pre-registered.
- The 256-token limit cut 22 (1.5B) and 47 (0.5B) dev replies, and 5 (1.5B) and 4 (0.5B) showcase replies. Some
  arithmetic answers were marked wrong because of it.
- Speeds were measured on a shared PC under heavy load.
- The showcase marks were made by the same author who wrote the questions and rubric. Qwen's long answers give more
  places for a wrong detail, and the strict rule marks those wrong.
- Qwen may have seen public evaluation data during its own training (stated in `CRITERIA_23.md`).

## Files

- `phase23/CRITERIA_23.md`
- `phase23/run_baseline.py`, `phase23/run_23a.sh`
- `phase23/cpu_timing.py`, `phase23/run_cpu_timing.sh`
- `phase23/results/dev_qwen_{1.5b,0.5b}.json` (+ `_responses.jsonl`)
- `phase23/results/showcase_qwen_{1.5b,0.5b}.json`, `phase23/results/showcase_grades.json`
- `phase23/results/cpu_timing_qwen_{0.5b,1.5b}.json`
- `phase23/logs/`
- Qwen weights: in the local Hugging Face cache only, not in git.
