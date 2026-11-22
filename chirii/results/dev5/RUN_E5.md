# Run (e5): S4f on development set 5, first try (the last development run)

S4f (matching by meaning, payable bill words, "by when" deadlines, ranges listed once, maths routing and the maths
check against an independent code computation), run once on development set 5. The set was written and committed
before S4f was built (e479bc2); S4f is ee86ea6; the runner was committed before the run (4dd2785). Laptop CPU,
Qwen2.5-1.5B-Instruct Q8_0, temperature 0. **A fresh first-try number.** Every reply was read by hand
(`labels_e5.json`). Intervals are 95% Wilson intervals; with 10–20 tasks per group they are wide.

As agreed, the development-set loop stops here whatever the score. The gaps below go to `docs/KNOWN_GAPS.md`; they are
not fixed now.

## Results next to run (e4)

| Group | (e4) marker | (e4) hand | **(e5) marker** | **(e5) hand** |
| :--- | ---: | ---: | ---: | ---: |
| Memory tasks | 11/15 | 10/15 | **14/20 (48–85%)** | **16/20 (58–92%)** |
| People and profile | 13/15 | 14/15 | **9/10 (60–98%)** | **10/10 (72–100%)** |
| Maths | 8/15 | 8/15 | **10/10 (72–100%)** | **10/10** |
| General knowledge | 7/10 | 7/10 | not in this set | |
| Route right | 48/55 | | **35/40 (74–95%)** | |
| Answer label right | 52/55 | | **39/40 (87–100%)** | |

The sets are different and small, so the change from (e4) is a direction, not a measured improvement.

## What the run shows

- **Maths: 10/10, all worked out by code with working shown.** Lakh/crore, weeks between dates, unit and temperature
  conversions, compound interest, EMI, GST, discount and "me and 4 friends" all routed and computed correctly.
- **Varied wording mostly works.** 8 of the 9 tasks written to test it were right: shaadi/wedding, specs/glasses,
  dawai/medicine, upanayanam/thread ceremony, "after breakfast" for morning, "hit my account", "before Navaratri
  starts", and "by when" (twice). The one miss ("When does the Jio plan run out?") was lost to the router, not to
  matching.
- **Bill words fixed:** renewals are listed to pay, never "to receive".
- **People: 10/10 by hand** (one marker false positive, the same one as in (e4)).
- **One wrong answer (D5-M12):** asked when Mr. Desai's leave starts (never saved), Qwen said "4:00 pm on Fridays",
  taken from his weekly visit. "leave" is not a key term, so the specific-item check did not stop it.

## Failures and labels

| Task | Cause | Detail |
| :--- | :--- | :--- |
| D5-M02 | model behaviour (Hinglish) | Echoed the question; gave the amount but not the day |
| D5-M07 (hand pass) | code: router | "How does Saturday look for me?" listed the week (Saturday's items all right) |
| D5-M11 | code: router | "Jio **plan**" sent to the day planner |
| D5-M12 | model behaviour + code: key terms | Made-up leave time from a weekly visit |
| D5-M15 | code: router | "Is there any bill this week?" not seen as personal; went to general |
| D5-M20 (hand pass) | scorer / answer style | Right number (700) without ₹, after an echoed question |
| D5-P10 (hand pass) | scorer | Consistency rule false positive (as D4-P15) |

Route mismatches: D5-M07 window (plan expected), D5-M11 plan (lookup), D5-M15 general (window), D5-P04 and D5-P05
lookup (people; both answers right).

## What is next

Per the plan agreed before this run: no more development sets. The gaps go to `docs/KNOWN_GAPS.md`. Next: S5
(encryption at rest, login, tester consent note), the safety layer, the 500-item scale check, the fresh extraction set,
then S6 (chat UI and the held-out set). The held-out 30 stay untouched until S6.
