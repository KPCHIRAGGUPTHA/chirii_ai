# Chirii AI: one-page summary (K P Chirag Guptha, 18 November 2026; updated 28 November 2026)

## What was built

1. **MiniGPT, from scratch (September to 10 November).** I built my own tokeniser, pretrained models of 34M and 98M
   parameters (up to 11 billion tokens), and instruction-tuned them with a calculator and a safety layer.
   - **v1.0.1 is released** as a demo.
   - The 98M model is level with GPT-2 on 9 of 12 benchmarks but **did not beat it**, which was the stated goal.
2. **Chirii Brain, on the open Qwen2.5-1.5B model (10–18 November).** Eight fine-tuned versions aimed at safety,
   helpfulness, document answering and tool use (calculator, files, notes, calendar, Python), each tested on public
   benchmarks.
3. **Chirii OS packages (Phase 30).** These run on a normal laptop (CPU, about 1.8 GB of RAM, about 5 seconds per
   reply). Each one is a brain plus the "OS layers": a self-harm safety layer (Tele-MANAS 14416 and 112), sandboxed
   tools, a document mode, Wikipedia lookup and a router.
   - **Base (commercial):** the brain is plain Qwen.
   - **Experimental:** the brain is c1, my best commercial fine-tune.
4. **A safety classifier in front of the brain (Phases 31–31B, 18–28 November).** Two pre-registered rounds tested:
   - my own 0.5B classifier;
   - Llama Guard 3 1B and ShieldGemma 2B (31B).

   **No candidate passed all of its bars, so none ships.** The packages keep the brain's own safety plus the self-harm
   layer.

**How it was tested.** Every experiment had its pass/fail rules written down and committed before it ran, and the
results were reported whether they passed or not.

## Key results

| Brain | Safe on harmful requests (XSTest) | Answers edgy-but-harmless questions (OR-Bench-Hard) | General function calling (BFCL) | Chirii tool tasks (150) | Commercial use |
| :--- | ---: | ---: | ---: | ---: | :--- |
| plain Qwen | 89.5% | 24.7% | 82.2% | 50.7% | yes |
| v0.3 (safest research version) | 97.4% | 15.3% | — | — | no |
| v0.4 (research) | 90.0% | 67.2% | 84.5% | 43.3% | no |
| **c1** (commercial fine-tune) | 92.6% | 63.1% | 74.9% | **80.0%** | yes |
| c2 (last round) | 91.6% | 72.9% | 78.8% | 74.0% | yes |

**The packages, measured on a laptop-style setup** (Chirii tool tasks, 150):
- base package: 55.3%;
- experimental c1 package: 70.7%.

## Honest limits

- **No fine-tuned brain passed all of its own pre-registered bars.**
  - The research versions traded safety against helpfulness.
  - The commercial ones (c1, c2) are much better at Chirii tasks but worse than plain Qwen on BFCL's parallel calls.
- **I relaxed two bars after seeing a result**, and I say so openly:
  - the BFCL bar in Phase 29 (c2 still failed it);
  - pooling the two unsafe-refusal tests in Phase 31B (no candidate passed anyway).
- **v0.1–v0.5 cannot be used commercially**, because their training data included non-commercial datasets (found by
  my own licence audit).
- **The commercial package uses plain Qwen.** It often says it "can't see your files" instead of using the file tools
  (0 of 22 file tasks).
- **The safety layer is a narrow rule set, not a classifier.** Phase 31B, the last classifier round, found none
  that passed:
  - my classifier over-blocked XSTest's safe prompts (51.9% answered vs 64.2%);
  - Llama Guard 3 1B caught the most unsafe prompts (95.5% vs 78.0% refused) but over-blocked OR-Bench-Hard, and took
    1.2 s per check on CPU;
  - ShieldGemma 2B failed three bars.

- **The run_python sandbox is not an operating-system sandbox.**
- **Small numbers.** Some test sets have been used several times, and many differences are within noise. The reports
  give the standard errors.

## Plan to 31 December (lab access extended; cleanup on 30 December)

1. **Safety classifier: done, not shipped.** Phase 31 and Phase 31B (the last round) tested it with rules fixed in
   advance. No candidate passed, so the packages are unchanged. Phase 31B ran on a new lab PC after the old one
   crashed; the change is recorded as a deviation.
2. **OS core.** Make the router, tools, memory and document mode a proper service with tests, so that brains can be
   swapped behind one interface.
3. **Real use.** Use the base package daily for my own notes, calendar and study files. Record failures (with consent,
   and no private data in the repository) to drive the next fixes, especially the "can't see your files" problem.
4. **Possible brain c3, only if the data supports it:** add parallel-call examples with varied schemas, keep the Chirii
   file and multi-step skills, and use the same pre-registered bars, with no further relaxation.
5. **Before 30 December:** the PC cleanup (`CLEANUP_PLAN.md`). The pendrive backup was made from PC 30 on 28 November
   and verified (every file checked by SHA-256). The laptop-only
   items are topped up at home.

Details: `FINAL_REPORT.md` (the whole story, every bar, lessons 1–35), `phase30/PHASE30_REPORT.md` (the packages),
`LAPTOP_SETUP.md` (how to run them).
