# Phase 31B report (public version: results only): the Chirii safety layer, last classifier round — NO WINNER, not released

Criteria were committed before any Phase 31B data was built or any model was trained. Every number below comes from a
run that happened. Each test was run once, after candidate (a)'s seed and threshold were fixed on dev sets. Data,
prompts, training recipe and model outputs are not included.

## Verdict

Each candidate in the base package (plain Qwen2.5-1.5B + OS layers) vs the base package alone. Significant means a
difference beyond 2·sqrt(se1² + se2²).

| Bar | (a) our classifier (s11, T 0.55) | (b) Llama Guard 3 1B | (c) ShieldGemma 2B |
| :--- | :--- | :--- | :--- |
| 1. Pooled unsafe refused (XSTest unsafe + HarmBench, 490): sig. higher | **PASS** 84.08 vs 77.96 (+6.12; 2·SE 5.00) | **PASS** 95.51 vs 77.96 (+17.55; 4.19) | **PASS** 87.96 vs 77.96 (+10.00; 4.76) |
| 2. XSTest safe answered: not sig. lower | **FAIL** 51.85 vs 64.20 (−12.35; 8.88) | PASS 60.91 vs 64.20 (−3.29; 8.78) | **FAIL** 48.56 vs 64.20 (−15.64; 8.88) |
| 3. OR-Bench-Hard answered: not sig. lower | PASS 23.42 vs 26.01 (−2.59; 3.37) | **FAIL** 15.94 vs 26.01 (−10.07; 3.16) | **FAIL** 17.85 vs 26.01 (−8.16; 3.22) |
| 4. Median added time per check ≤ 500 ms (CPU, Q8_0, idle) | PASS 435.9 ms | **FAIL** 1,241.5 ms | **FAIL** 7,510.9 ms |
| All bars | no | no | no |

**No candidate passes all bars, so there is no winner.** As registered: **no release**. No classifier is added to
either package. **The Chirii OS uses the brain's own safety plus the self-harm layer**, as in the Phase 30 packages.
They stay exactly as released.

Why, in short:
- **(a)** Bar 1 now passes, helped by the pooling, and so do the speed bar and OR-Bench-Hard. It still over-blocks
  XSTest's safe prompts (26.75% blocked on its own; 65 of 243 in the system run). Added sensitive-sounding but safe training
  prompts brought its sensitive dev set down to 0.95% false-blocks, but this did not carry over to XSTest's safe
  prompts.
- **(b) Llama Guard 3 1B** gives by far the strongest unsafe gain (HarmBench 70.67 → 98.67) and keeps XSTest safe.
  But it blocks 38.14% of OR-Bench-Hard on its own, and its check takes 1.24 s on 4 CPU threads.
- **(c) ShieldGemma 2B** fails three bars. Its check runs four policy requests and takes about 7.5 s per check.

**Pooling statement (required):** bar 1 pools XSTest unsafe and HarmBench. **This pooling is a change made after
Phase 31**: Phase 31 had two separate bars, and its XSTest-unsafe bar failed for lack of power. Each part is reported
separately below. Under Phase 31's separate bars, (a) would fail both XSTest unsafe (+4.74; 2·SE 5.59) and HarmBench
(+7.00; 2·SE 7.13), so pooling helped (a). It did not change the outcome, because (a) fails bar 2 either way.

## Selection (dev only)

Candidate (a) was retrained (seeds 11, 12, 13), with sensitive-sounding but safe prompts added to training and to a
separate dev set. It has no reply check: no commercially licensed dataset with unsafe-reply labels was found, so the
reply check was dropped. Threshold rule: the highest dev unsafe recall with false-block ≤ 5% on both the ordinary
and the sensitive dev set; then best ordinary-dev F1.

| Seed | T | Ordinary F1 | Ordinary recall | False-block ordinary | False-block sensitive |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **11 (chosen)** | 0.55 | **83.63** | 76.60 | 4.55% | 0.95% |
| 12 | 0.55 | 82.74 | 74.95 | 4.30% | 1.14% |
| 13 | 0.50 | 82.70 | 75.14 | 4.55% | 0.95% |

The external candidates on the same dev sets (as provided, T 0.5, reported only):
- Llama Guard 3 1B: F1 69.46, false-block 15.42% ordinary / 5.14% sensitive.
- ShieldGemma 2B: F1 76.95, false-block 6.45% / 4.19%.

## Test (A): each checker alone (prompt checks; replies where the checker has a reply check)

| Test | (a) s11 F1 / recall / false-block | (b) Llama Guard | (c) ShieldGemma | plain Qwen2.5-0.5B (ref.) |
| :--- | :--- | :--- | :--- | :--- |
| Aegis test prompts (1,928) | 82.73 / 76.52 / 9.90% | 73.35 / 67.56 / 19.46% | 77.21 / 68.82 / 11.02% | 67.14 / 72.86 / 51.63% |
| Aegis test replies (852) | (no reply check) | 61.52 / 51.52 / 13.76% | 76.03 / 70.05 / 12.23% | 49.80 / 47.72 / 37.77% |
| XSTest unsafe (190): recall | 83.16 | 74.74 | 82.11 | 85.79 |
| XSTest safe (243): false-block | 26.75% | **5.76%** | 27.98% | 64.61% |
| OR-Bench-Hard (1,311): false-block | **23.04%** | 38.14% | 26.54% | 26.39% |
| OR-Bench-Toxic (655): recall | 67.79 | **83.36** | 52.52 | 69.01 |

The Phase 31 classifier blocked 34.2% of XSTest safe and 41.3% of OR-Bench-Hard. Candidate (a) improves on both (26.75%,
23.04%), but not enough for bar 2.

## Test (B): the whole system (bars 1–3; c1 reported, not barred)

The brain was run once per package (greedy, 512 new tokens, no tools, CPU 16 threads, 4 slots). The same replies were
used for every candidate.

**Base package (barred):**

| Test | Without | + (a) | + (b) Llama Guard | + (c) ShieldGemma |
| :--- | ---: | ---: | ---: | ---: |
| XSTest unsafe refused or support | 89.47 | 94.21 (n.s.) | 90.53 (n.s.) | 97.89 (gain) |
| HarmBench refused or support | 70.67 | 77.67 (n.s.) | 98.67 (gain) | 81.67 (gain) |
| **Pooled unsafe (490)** | 77.96 | **84.08 (gain)** | **95.51 (gain)** | **87.96 (gain)** |
| XSTest safe answered | 64.20 | 51.85 (**drop**) | 60.91 (n.s.) | 48.56 (**drop**) |
| OR-Bench-Hard answered | 26.01 | 23.42 (n.s.) | 15.94 (**drop**) | 17.85 (**drop**) |

**c1 + candidate (reported, not barred):**

| Test | Without | + (a) | + (b) Llama Guard | + (c) ShieldGemma |
| :--- | ---: | ---: | ---: | ---: |
| XSTest unsafe | 94.74 | 96.84 (n.s.) | 95.79 (n.s.) | 98.42 (n.s.) |
| HarmBench | 71.33 | 78.67 (gain) | 99.00 (gain) | 84.00 (gain) |
| Pooled unsafe | 80.41 | 85.71 (gain) | 97.76 (gain) | 89.59 (gain) |
| XSTest safe answered | 85.60 | 65.02 (drop) | 80.66 (n.s.) | 65.02 (drop) |
| OR-Bench-Hard answered | 71.62 | 57.82 (drop) | 44.47 (drop) | 52.86 (drop) |

c1 answers far more OR-Bench-Hard prompts on its own (71.62%), so every classifier costs it more there.

## Test (C): speed (bar 4)

llama-server cd959ba, CPU, 4 threads, 1 slot, Q8_0, end-to-end time per check. Timing started only after the 1-minute
load average had stayed below 1.0 for 2 minutes (read every 10 s). Nothing
else of this project ran during timing, and no other user was logged in.

| Candidate | Idle reached | Prompt checks: median | Reply checks: median | All checks: median (bar) |
| :--- | :--- | ---: | ---: | ---: |
| (a) s11 | 28 Nov 10:48 (load 0.16) | 435.9 ms (n 400) | — | **435.9 ms** |
| (b) Llama Guard 3 1B | 28 Nov 10:55 (0.13) | 1,212.2 ms (200) | 1,497.8 ms (200) | **1,241.5 ms** |
| (c) ShieldGemma 2B (4 policies) | 28 Nov 11:08 (0.12) | 7,049.2 ms (200) | 10,306.5 ms (200) | **7,510.9 ms** |

Llama Guard's earlier smoke test (about 0.55 s) used 8 threads and a short prompt. It was not a measurement.

## Licences (recorded per the spec; nothing shipped)

- **(b) Llama Guard 3 1B** @ acf7aafa, Llama 3.2 Community Licence: commercial use allowed. Conditions: ship a copy of
  the licence, display "Built with Llama", include the NOTICE line, follow the Acceptable Use Policy; a separate licence
  is needed above 700M monthly active users.
- **(c) ShieldGemma 2B** @ d1dffc9c, Gemma Terms of Use (last modified 1 Apr 2026) plus the Prohibited Use Policy:
  commercial use allowed. Conditions: include the NOTICE line, pass the use restrictions on to users, the §3.2
  remote-restriction right, and the §3.3 terms on outputs.
- Access to both gated models was granted.

## Deviations (recorded; none chose anything based on a score)

- The work moved to a different lab PC after the first one crashed (out of RAM). The criteria, lost unpushed in the
  crash, were rewritten from the same spec before any 31B data or training existed. Seeds trained one at a time, and a
  RAM guard stopped any step below 4 GiB free (it never fired).
- Two contamination-reference files lost in the crash were left out. This was the owner's decision, made before any
  31B data was generated. The rebuilt base data matched the original within 0.1% (dev set identical).
- The previous round's classifier checkpoints were lost with the old PC, so there is no direct checkpoint
  comparison with Phase 31.
- One data-build step stopped on a wrong file path before writing anything, and was rerun with the same settings. No
  training, test or timing run was stopped or rerun.

## Use counts in this phase

| Set | Use |
| :--- | :--- |
| Aegis 2.0 test split | 2nd |
| XSTest unsafe / safe | 13th |
| OR-Bench-Hard-1K | 10th |
| OR-Bench-Toxic, as a test | 2nd |
| HarmBench (300) | 8th |

No other test set was run.

## What this closes

This was the last classifier round, as specified. The registered outcome is that no separate safety classifier ships.
Chirii OS keeps the brain's own safety behaviour plus the self-harm layer (Phase 30 packages, unchanged). In these runs
only Llama Guard 3 1B kept XSTest safe, but its OR-Bench-Hard over-blocking and its 1.2 s CPU cost rule it out under
these bars. These are observations, not proposals.

Score tables: `phase31b/results/public_scores_31b.json`.
