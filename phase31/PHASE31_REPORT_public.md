# Phase 31 report (public version: results only): the Chirii safety classifier — NOT released (bars 1, 3, 4 and 5 fail)

Criteria were committed before anything was built. Every number below comes from a run that
happened; each test was run once, after the seed and threshold were fixed on dev. Harmful items were handled only as files
and labels; none is printed here.

## Verdict

| Bar (base package + classifier vs base package alone) | With | Without | Diff | 2·SE | Result |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 1. XSTest unsafe refused or support: significantly higher | 93.68% | 89.47% | +4.21 | 5.68 | **FAIL** (n.s.) |
| 2. HarmBench refused or support: significantly higher | 81.00% | 71.33% | +9.67 | 6.91 | PASS |
| 3. XSTest safe answered: not significantly lower | 50.21% | 64.20% | −13.99 | 8.89 | **FAIL** (significant drop) |
| 4. OR-Bench-Hard-1K answered: not significantly lower | 18.76% | 25.86% | −7.10 | 3.24 | **FAIL** (significant drop) |
| 5. Median added time per check ≤ 500 ms (CPU, Q8_0) | 576.4 ms | | | | **FAIL** |

**Not all bars pass.** As registered: no release; the classifier is **not** added to either package. The Phase 30 packages
stay as they are (plain Qwen + OS layers remains the commercial package).

Summary of why:
- The prompt check blocks too many benign but sensitive-sounding prompts: 34.2% of XSTest safe and 41.3% of
  OR-Bench-Hard. Dev said 4.2%, but the dev set did not contain prompts of this kind.
- The reply check never blocked anything (0 of 852 Aegis test replies; 0 in every system run), so it added cost and no
  protection. It was trained only on replies labelled safe.
- The 0.5B check takes about 0.58 s on 4 CPU threads.

## Selection (dev only)

Three seeds (11, 12, 13) were trained. The threshold and seed were fixed on a dev set before any test:
highest dev unsafe recall with dev false-block ≤ 5%, then best dev F1. Result: **seed 11, T = 0.45** (dev F1 85.63,
unsafe recall 80.07, false-block 4.21%).

## Test (A): the classifier alone

Both models are Q8_0 GGUF. The chosen model uses T = 0.45; the baseline is plain Qwen2.5-0.5B-Instruct with the same
instruction at T = 0.5. Llama Guard 3 1B and ShieldGemma 2B were not run (gated).

| Test | n | Chosen F1 | Chosen recall | Chosen false-block | Baseline F1 | Baseline recall | Baseline false-block |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aegis 2.0 test, prompt check | 1,928 | 83.67 | 79.40 | 12.15% | 67.35 | 73.05 | 51.29% |
| Aegis 2.0 test, reply check: all | 852 | 0.00 | 0.00 | 0.00% | 50.33 | 48.48 | 37.99% |
| Aegis reply check: human-labelled (all safe) | 358 | — | — | 0.00% | — | — | 36.59% |
| Aegis reply check: jury-labelled | 494 | 0.00 | 0.00 | 0.00% | 60.83 | 48.48 | 43.00% |
| XSTest, 190 unsafe + 243 safe | 433 | 74.48 | 85.26 | 34.16% | 64.19 | 86.32 | 64.61% |
| OR-Bench-Hard-1K (all allow) | 1,311 | — | — | 41.27% | — | — | 26.39% |
| OR-Bench-Toxic (all block) | 655 | 89.92 | 81.68 | — | 81.98 | 69.47 | — |

On the prompt check, the trained classifier is much better than the untrained baseline on Aegis, XSTest and
OR-Bench-Toxic. It is worse than the baseline on OR-Bench-Hard, blocking 41.3% against 26.4%. Its reply check never says
unsafe.

Self-harm category on blocked test-A prompts:

| Test | Routed to the support message |
| :--- | ---: |
| Aegis | 24 |
| XSTest unsafe | 5 |
| XSTest safe | 1 |
| OR-Bench-Hard | 7 |
| OR-Bench-Toxic | 73 |

## Test (B): the whole system

**Settings:**
- the self-harm layer first, then the brain through llama-server chat (greedy, 512 new tokens, no tools);
- judged by `scorer_v4` / `score_overrefusal`, as in earlier phases;
- the "without" replies are the same greedy replies that the "with" system checks.

| System | XSTest unsafe refused/support | XSTest safe answered | HarmBench refused/support | OR-Bench-Hard answered |
| :--- | ---: | ---: | ---: | ---: |
| Base package alone | 89.47% (170/190) | 64.20% (156/243) | 71.33% (214/300) | 25.86% (339/1,311) |
| **Base + classifier** | 93.68% (178/190) | 50.21% (122/243) | 81.00% (243/300) | 18.76% (246/1,311) |
| c1 package alone | 94.74% (180/190) | 85.60% (208/243) | 70.67% (212/300) | 71.70% (940/1,311) |
| c1 + classifier (reported, not barred) | 96.84% (184/190) | 60.91% (148/243) | 80.67% (242/300) | 44.62% (585/1,311) |

c1 with vs without (reported only):

| Test | Diff | 2·SE | Result |
| :--- | ---: | ---: | :--- |
| XSTest unsafe | +2.10 | 4.12 | n.s. |
| XSTest safe | −24.69 | 7.71 | significant drop |
| HarmBench | +10.00 | 6.96 | significant gain |
| OR-Bench-Hard | −27.08 | 3.71 | significant drop |

On c1, the classifier takes away most of c1's over-refusal gain.

**Classifier actions, base package** (the c1 counts are within 2 of these):

| Test | Prompt blocks | Reply blocks | Routed to the self-harm message | Caught first by the self-harm layer |
| :--- | ---: | ---: | ---: | ---: |
| XSTest unsafe | 160 | 0 | 3 | 2 |
| XSTest safe | 83 | 0 | 1 | 0 |
| HarmBench | 195 | 0 | 5 | 0 |
| OR-Bench-Hard | 544 | 0 | 8 | 0 |

## Test (C): speed

The chosen classifier, run by llama-server (llama.cpp cd959ba) on CPU with 4 threads, 1 slot, Q8_0. Each check is timed
end to end. Sample: the first 200 prompt checks and the first 200 reply checks of the base-package run.

| Checks | n | Median | Mean |
| :--- | ---: | ---: | ---: |
| Prompt checks | 200 | 569.2 ms | 541.8 ms |
| Reply checks | 200 | 596.2 ms | 688.5 ms |
| **All** | 400 | **576.4 ms** | |

The machine was not idle: the load average at the start was 4.95 / 8.18 / 10.54 (1 / 5 / 15 min), so the timing may be
pessimistic. As registered, the test was run once and is not repeated. Bar 5 fails as measured.

## Optional research comparison (MiniGPT as the classifier)

**Not run.** The main classifier failed four of five bars, so a research-only 98M comparison would not change any
decision. It was not started.

## What would need to change (notes, not results)

- **Over-blocking.** The dev set did not represent sensitive-sounding benign prompts like XSTest safe and OR-Bench-Hard.
- **Reply check.** It needs unsafe reply labels. As trained,
  the reply check could be dropped to halve the cost.
- **Speed.** One prompt check alone is about 0.57 s at 4 threads. More threads, a smaller classifier, or a shorter
  instruction would be needed for ≤ 500 ms.

## Use counts in this phase

| Set | Use |
| :--- | :--- |
| Aegis 2.0 test split | 1st use |
| XSTest unsafe / safe | 12th use |
| OR-Bench-Hard-1K | 9th use |
| OR-Bench-Toxic, as a test | 1st use |
| HarmBench (300) | 7th use |


## Licences

- Classifier base: Qwen2.5-0.5B-Instruct (Apache-2.0). Training data: commercially licensed public safety datasets only
  (CC BY 4.0, Apache-2.0); gated datasets were excluded.
- Reference baselines Llama Guard 3 1B and ShieldGemma 2B: gated (manual approval); recorded and skipped.
