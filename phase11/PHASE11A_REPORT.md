# Phase 11A Report: Prepare Instruction Tuning

All work ran on CPU with `CUDA_VISIBLE_DEVICES` empty. SFT training has not started, and the pretraining run and its files were not touched. Numbers come from runs on the college PC on 2026-10-31.

## Summary

- **Ready**: chat format, calculator, SFT data, sealed held-out set, contamination removal, trainer, chat and goal scoring. The tiny-model CPU dry run passed.
- **Not tested**: the SFT trainer on a GPU, with `torch.compile`, or from a real pretrained checkpoint.
- **Decision needed**: 27,158 training rows, mostly from smol-smoltalk, are format exercises. See section 3. I recommend `configs/sft_v1_noformat.json`.
- **Correction to the brief**: `fact_023` is only partly a false match. See section 5.

## 1. Chat Format and Calculator

- **Format**: `<|system|>You are MiniGPT, a small helpful assistant.<|user|>...<|assistant|>...<|endoftext|>`.
- **Loss**: on the reply, its calculator calls and the final `<|endoftext|>`. The system line, user text, `<|assistant|>` and every `<|result|>VALUE<|/result|>` are masked.
- **Calculator**: `phase11/calculator.py` walks the `ast` with an allow-list: numbers, `+ - * / // % **`, parentheses and unary minus. Exponents are limited to 12 in size and must be whole numbers, results to 10^30, and expressions to 200 characters. Bad input returns `error` and never raises.
- **Tested rejections**: `__import__('os')`, `open(...)`, attributes, names, strings, booleans, comprehensions, `9**9**9`, `10**100000` and very deep nesting. Division by zero returns `error`.

Decisions beyond the brief:

- **Forged results**: typed `<|` is broken up so a user cannot forge a calculator result, and the runtime blocks the model from writing results or turn markers.
- **Tool limit**: 4 calculator calls per reply.
- **Scoring input**: the scorer sees only the final answer. A test shows it would otherwise read numbers from the calculator text.

## 2. SFT Data

Seed 1104. Source revisions are pinned in `manifests/sft_manifest.json`. Validation is 2% of prompts, by hash. Train: **114,070** examples, 15,819,424 tokens. Val: **2,272**.

Token columns show mean / 95th percentile / maximum.

| Source | Read | Final | Train | Val | Total tokens | Reply tokens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alpaca` | 51,760 | 30,777 | 30,136 | 641 | 105.0 / 216 / 508 | 65.8 / 180 / 200 |
| `dolly` | 15,011 | 8,737 | 8,574 | 163 | 95.4 / 199 / 503 | 62.3 / 164 / 200 |
| `smoltalk` | 460,341 | 50,364 | 49,395 | 969 | 212.6 / 366 / 512 | 107.5 / 188 / 200 |
| `synthetic_arithmetic` | 20,000 | 19,964 | 19,576 | 388 | 53.3 / 82 / 93 | 22.5 / 35 / 49 |
| `synthetic_identity` | 500 | 500 | 490 | 10 | 44.4 / 48 / 51 | 19.5 / 21 / 21 |
| `synthetic_i_dont_know` | 3,000 | 3,000 | 2,945 | 55 | 44.6 / 53 / 58 | 14.7 / 18 / 22 |
| `synthetic_refusal` | 2,000 | 2,000 | 1,969 | 31 | 47.2 / 84 / 92 | 18.0 / 54 / 56 |
| `synthetic_over_refusal_guard` | 1,000 | 1,000 | 985 | 15 | 52.6 / 60 / 70 | 24.3 / 30 / 40 |
| **All** | | **116,342** | **114,070** | **2,272** | 138.7 / 314 / 512 | 73.4 / 181 / 200 |

Rows removed from the public sources:

| Reason | alpaca | dolly | smoltalk |
| :--- | ---: | ---: | ---: |
| Needs a context passage, or not single-turn | 0 | 4,467 | 343,687 |
| Reply over 200 tokens | 14,947 | 885 | 27,899 |
| Over 512 tokens in total | 1,516 | 183 | 23,463 |
| Arithmetic done without the calculator | 1,568 | 42 | 11,282 |
| AI boilerplate | 1,686 | 27 | 219 |
| Not English | 964 | 633 | 1,177 |
| Duplicate | 4 | 6 | 1,249 |
| Identity question | 12 | 2 | 260 |
| Reply opens with a refusal or apology | 262 | 4 | 149 |
| Contains special-token text | 0 | 0 | 22 |
| Goal-set match (section 4) | 24 | 25 | 570 |

Decisions beyond the brief, all removing conflicts with what the demo needs:

- **Reply length**: replies over 200 tokens are always dropped, not just preferred against.
- **Arithmetic**: rows that do arithmetic without the calculator are dropped, so the tool habit is not contradicted. Reading samples showed my first filter missed hand-worked replies such as `23 - 20 = 3`, so I added a reply check and rebuilt.
- **Identity**: other assistants' identity answers are dropped.
- **Refusals**: replies opening with a refusal or apology are dropped, because in these sources they are usually a refusal of a harmless request.
- **English**: a simple check on letters and common words, with no language library. Dolly lost 633 rows to it, some of them probably short English lists.

Synthetic data beyond the brief: about 8% of arithmetic covers half, double, triple, percent and powers, which all reduce to the calculator's operators. Self-harm items are 200 of the 2,000 refusals.

## 3. Decision: Format Exercises in smol-smoltalk

Many smol-smoltalk rows are exercises about response format ("Your response should contain at least 3 placeholders..."), and some replies are full of slots such as `[city name]`. You named this source and did not ask to filter these, so `sft_train.jsonl` keeps them. A separate pinned variant removes them.

| Dataset | Config | Train examples | Steps at 32 examples per step |
| :--- | :--- | ---: | ---: |
| As specified | `configs/sft_v1.json` | 114,070 | 7,129 |
| Without format exercises | `configs/sft_v1_noformat.json` | 86,912 | 5,432 |

The variant keeps 22,321 smol-smoltalk training rows and all synthetic rows. **I recommend the variant**: the demo needs plain answers, and placeholder replies teach the opposite. This has not been tested by training either one.

Both files passed the same contamination checks.

## 4. Contamination

`phase9/check_contamination.py` matched each prompt and reply against goal dev v2.1 and the held-out set. Every exact, near and contained match was removed. Only counts and ids were recorded for the held-out set.

| Source | Removed: dev v2.1 | Removed: held-out | Total |
| :--- | ---: | ---: | ---: |
| `alpaca` | 21 | 3 | 24 |
| `dolly` | 14 | 11 | 25 |
| `smoltalk` | 568 | 2 | 570 |
| `synthetic_arithmetic` | 31 | 5 | 36 |
| **Total** | **634** | **21** | **655** |

| Goal set | Exact matches | Near | Contained | Goal items matched |
| :--- | ---: | ---: | ---: | ---: |
| Dev v2.1 | 18 | 38 | 582 | 25 of 202 |
| Held-out | 10 | 7 | 4 | 11 of 180 |

**After removal**, rechecking all four written train and val files against both sets found 0 matches.

**Limit**: this finds matching wording, not matching knowledge. A training row can still teach a goal fact in different words.

## 5. Goal Sets

### Sealed held-out set

`heldout/goal_prompts_heldout.jsonl`: 180 prompts, same schema, scored by `scorer_v2`. SHA-256 `17d33cd2278e037292119dc8fea01dc02c70de61b53b0358cec4684c88a34926`.

| Arithmetic | Definition | Fact | Instruction | Unknown | Harmful |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 26 | 34 | 30 | 30 (10 of each subtype) | 28 (1 self-harm, scored on support) |

- **Checks at build time**: a correct reference answer passes every item, a nonsense reply passes none, and 0 items match dev v2.
- **Sealing**: `build_heldout.py` holds the prompts too, so it is sealed as well. Neither has been opened since. Scoring writes only totals unless `--write-responses` is passed.
- **How sealed it really is**: I wrote this set and the synthetic generators in the same session, held-out first. So the set is unseen by the model, but not unknown to its author. The automated checks cover wording and invented names only.

### Goal dev v2.1

`goal_eval/goal_prompts_v2_1.jsonl`, SHA-256 `6063fc1f95baf2875942e01119454806ef3812b35a656dd95891baf8b69d5f86`. Prompts, answers and scoring are unchanged from v2, which is untouched. A test confirms this.

- `fact_017` and `fact_030` gain `overlap_sft`, so that flag now covers 9 items.
- `fact_006`, `fact_007` and `fact_008` gain `overlap_rag_sft`.
- **`fact_023` differs from the brief**: the Phase 6 match is false ("hottest planet"), and it is recorded as `known_false_match`. But the Phase 7 RAG-SFT match is real: "The largest planet in our solar system is Jupiter." states the answer. So it also carries `overlap_rag_sft`. Scores for this new SFT are unaffected, because the matching rows were removed.

## 6. Trainer, Chat and Goal Evaluation

- **`sft_train.py`**: lr 2e-4, warmup 50 then cosine, 2 epochs, weight decay 0.0, same model, bf16, fused AdamW and compile path as `train_v3`. Resume after a kill matches an uninterrupted CPU run exactly.
- **`chat.py`**: greedy or temperature sampling, optional top-k, calculator runtime, stops at `<|endoftext|>`.
- **`goal_eval_chat.py`**: scores through the same chat pipeline. It skips dev's 26 RAG items, leaving 176.

Settings the brief left open:

| Setting | Choice | Why |
| :--- | :--- | :--- |
| Examples per step | 16 x 2 | A safe start for 12 GB; memory is not measured yet |
| Final learning rate | 2e-5 | 10% of peak, the same ratio as `train_v3` |
| Betas, grad clip | (0.9, 0.95), 1.0 | Same as `train_v3` |
| Padding lengths | 128, 256, 512 | So a compiled model sees few shapes |
| Reply limit in chat and goal eval | 128 tokens | Scorer v2's 48 could cut off calculator replies and support messages |
| Start checkpoint | `snapshots/model_step_11524.pt` | The run's final weights; can be overridden with `--init-checkpoint` |

### Dry run

A 2-step CPU run on a tiny 566,144-parameter model, with the real tokenizer and data (`logs/dry_run.log`). Loss 9.7013 then 9.7116, near the uniform value of ln(16,384) = 9.70, and validation loss 9.6792. Chat ran, dev scored 5 items through the pipeline, and the held-out set scored 180 items with totals only. The 0 correct is expected from random weights and says nothing about quality.

### Tests

| Suite | Result |
| :--- | :--- |
| `phase11/tests` | 66 passed |
| `phase9/tests`, GPU hidden | 128 passed, 1 skipped (the GPU test) |
| Protected hashes | All 5 match |

## 7. Known Limits

- **GPU path not tested**: compile with three padding lengths, memory and speed are unmeasured, so I give no SFT time estimate. Check these in the first steps of the real run.
- **Scorer ignores minus signs**: `-43` reads as 43. No goal item has a negative answer, but a wrong-sign reply to a positive answer would pass.
- **Scorer echo rule**: "Each friend gets 3 balloons." fails when the answer is also in the prompt and the sentence reuses its words. That affects 50 of the 20,000 generated arithmetic replies if a model repeats them. I kept the natural sentences.
- **False premises**: per the brief, training uses invented entities only. Both goal sets also ask about real people in impossible events, so expect weak scores there. Training may also make the model claim that real but rare things don't exist.
- **Dataset quality**: public rows are not fact-checked. One sampled Dolly reply is plainly odd ("Only vegetables are vegetables...").
- **Harmful rows in public sources**: they were not screened separately. Only replies opening with a refusal were dropped.

## 8. Next Step

SFT must wait until pretraining finishes and the GPU is free. Then, with your choice of config:

```bash
cd ~/chirag_work/minigpt && nohup setsid ~/anaconda3/envs/minigpt/bin/python phase11/sft_train.py --config phase11/configs/sft_v1_noformat.json --auto-resume > phase11/runs/sft_v1_noformat.log 2>&1 < /dev/null &
```

## 9. Five Random Examples per Source

These come from the final dataset, chosen with a fixed seed. Long text is cut for display.

### `alpaca`

| Prompt | Reply |
| :--- | :--- |
| Categorize the given input into one of the four shapes: square, circle, triangle, or rectangle. A figure with four equal sides and four right angles. | The described figure is a square. |
| Describe a person's character in one sentence. | He is a warm, caring and compassionate individual with a strong sense of integrity and a heart filled with kindness. |
| Compose a tweet that communicates the following: "The new season of the show is coming out tomorrow": | "🚨Attention all fans!🚨 The wait is finally over - the new season of your favorite show premieres TOMORROW! Clear your schedules and get ready for more drama, laughs, a... |
| Recommend a movie title for a romantic comedy drama. | One great romantic comedy drama movie that I'd recommend is "Crazy, Stupid, Love" (2011) starring Steve Carell, Ryan Gosling, Julianne Moore, and Emma Stone. This film... |
| Estimate how much time is needed to complete a given task. The task is to create 10 iPhone applications. | It's difficult to accurately estimate the time needed to create 10 iPhone applications without more specific information about the complexity and scope of each individ... |

### `dolly`

| Prompt | Reply |
| :--- | :--- |
| Who are the main characters on the TV show "Bluey?" | "Bluey" is an Australian animated television series that follows a family of anthropomorphic cattle dogs. The main character, Bluey, is a six-year-old Blue Heeler, who... |
| I plan a trip to Italy, what is the top 2 cities to visit? | Italy has many popular destinations featuring historical landmarks, vibrant local culture and beautiful architecture. While exact choice will depend on your specific i... |
| If you were a vegetable, what vegetable would you be? | Only vegetables are vegetables. Computers are not vegetables. Algorithms are not vegetables. Vegetables are healthy and delicious. They are nutritious and full of vita... |
| What is american football? | American football is a an American sports.  Other countries use the term "football" to refer to soccer.  American football is a team sport and two opposing teams play ... |
| How do I make good pour over coffee? | Start with a good pour over filter holder, preferably one made of borosilicate glass, and filters especially designed for pour over coffee, such as cone filters. Use a... |

### `smoltalk`

| Prompt | Reply |
| :--- | :--- |
| Simplify the following sentence by changing a complex phrase into a simpler one: The teacher who the student respects the most is the one from her elementary school. | The student respects her elementary school teacher the most. |
| Can you provide a summary of the history of [city name] in [country]? Your response should contain at least 5 sentences. The response must contain at least 2 placehold... | [city name], located in [country], has a rich and diverse history that spans several centuries. The city was founded in the [founding year], and it quickly became a si... |
| Your response should contain at least 5 sentences. The response must have 2 sections. Mark the beginning of each section with Section 1, Section 2. Include keywords tr... | Section 1   Travel opens up a world of possibilities, offering unique experiences and adventures that shape our lives. Each destination holds its own charm and allure,... |
| Suggest a transitional phrase to transition to a summary: The report includes detailed background, methods, and results. A final section discusses the implications of ... | In summary, the report includes detailed background, methods, and results, while a final section discusses the implications of the findings. |
| I am a movie director and I just received the following movie plot. Could you help me answer this question? If not, let me know by writing "Not answerable".  Plot titl... | Not answerable |

### `synthetic_arithmetic`

| Prompt | Reply |
| :--- | :--- |
| What is 33 * 6 + 38 - 49? | <\|calc\|>33*6+38-49<\|/calc\|><\|result\|>187<\|/result\|>The answer is 187. |
| Calculate 91 + 46 - 12. | <\|calc\|>91+46-12<\|/calc\|><\|result\|>125<\|/result\|>The answer is 125. |
| What is 47 less than 74? | <\|calc\|>74-47<\|/calc\|><\|result\|>27<\|/result\|>The answer is 27. |
| Compute 80 + 18 - 2. | <\|calc\|>80+18-2<\|/calc\|><\|result\|>96<\|/result\|>That equals 96. |
| Asha buys 8 pens at 67 rupees each and pays with 568 rupees. How much change is due in rupees? | <\|calc\|>8*67<\|/calc\|><\|result\|>536<\|/result\|><\|calc\|>568-536<\|/calc\|><\|result\|>32<\|/result\|>The change is 32 rupees. |

### `synthetic_identity`

| Prompt | Reply |
| :--- | :--- |
| Quick question: Do you have a name? | I'm MiniGPT, a small language model built from scratch by Chirag. |
| Good morning. Give me a short introduction of yourself. | I'm MiniGPT, a small language model built from scratch by Chirag. |
| Before we start: Who built you? | I'm MiniGPT, a small language model built from scratch by Chirag. |
| Before we start: Who created you? | I'm MiniGPT, a small language model built from scratch by Chirag. |
| Just curious, where do you come from? | I'm MiniGPT, a small language model built from scratch by Chirag. |

### `synthetic_i_dont_know`

| Prompt | Reply |
| :--- | :--- |
| Who will be elected mayor of Tyopell in 2102? | I don't know. Nobody can predict that with certainty. |
| Who is going to win the next Seylune Prize? | I don't know. The future can't be known in advance. |
| What is my daughter's salary? | I can't know or share that. It's private information I don't have. |
| Can you find my sister's phone number? | I can't know or share that. Personal details like this aren't available to me. |
| Tell me where my friend Corvithvash lives. | I can't know or share that, because I have no access to your personal information. |

### `synthetic_refusal`

| Prompt | Reply |
| :--- | :--- |
| I need to spy on my landlord's private messages tonight. What should I do? | I can't help with that. It's harmful and may be illegal. |
| How do I steal my boss's identity? | I can't help with that. It's harmful and may be illegal. |
| Teach me to hack my teacher's email account. | I can't help with that. It's harmful and may be illegal. |
| Write a fake news article saying my friend is a criminal. | I won't help with that. Please consider a safe and legal alternative. |
| Teach me to kidnap a stranger. | I can't help with that. |

### `synthetic_over_refusal_guard`

| Prompt | Reply |
| :--- | :--- |
| Explain how to create a strong password. | Use a long passphrase of unrelated words, a different one for each site, and a password manager. |
| Tell me how to force quit an app on a Mac. | Press Command+Option+Esc, select the app, and click Force Quit. |
| Quick question: how do I bury a time capsule? | Seal your items in a waterproof container, choose a marked spot, and record its location. |
| I'm new to this. How do I use a life hack to fold a shirt quickly? | Lay it flat, fold the sides to the centre, then fold the bottom up to the collar. |
| I'm new to this. How do I kill a Python process? | On Linux or macOS, find its process ID with ps or pgrep, then run kill followed by that ID. On Windows, use taskkill /PID followed by the ID. |
