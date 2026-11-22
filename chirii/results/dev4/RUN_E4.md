# Run (e4): S4d + S4e on development set 4, first try

The S4d fixes (ranges, money questions, capitalised names, "done" updates, the lead contradiction check, the maths
formula library, the glossary, the three answer labels) and S4e (people and profile memory), run once on development
set 4. The set was written and committed before either was built (75ef7ce); the runner was committed before the run
(b098a63). Laptop CPU, Qwen2.5-1.5B-Instruct Q8_0, temperature 0. **A fresh first-try number.** Every reply was read by
hand (`labels_e4.json`). Intervals are 95% Wilson intervals; with 10–15 tasks per group they are wide.

## Results next to run (e3)

| Group | (e3) marker | (e3) hand | **(e4) marker** | **(e4) hand** |
| :--- | ---: | ---: | ---: | ---: |
| Memory tasks | 23/30 (26/30 re-marked) | 25/30 | **11/15 (48–89%)** | **10/15 (42–85%)** |
| People and profile | – | – | **13/15 (62–96%)** | **14/15 (70–99%)** |
| Maths | 12/15 | 12/15 | **8/15 (30–75%)** | **8/15** |
| General knowledge | 14/15 | 14/15 | **7/10 (40–89%)** | **7/10** |
| Route right | 79/85 | | **48/55 (76–94%)** | |
| Answer label right | – | | **52/55 (85–98%)** | |

## What the run shows

- **People and profile memory works on unseen phrasings** (14/15 by hand). This covers reverse relationships ("Sunita's
  husband" → "That's you: Ramesh"; "Lalitha's children" → you and Karan), Indian terms (Didi, Bhaiya, Dadi, Appa), a
  relationship through someone else ("Kiran is your nephew (Revathi's son)"), and a gender-neutral answer for a cousin.
- **The glossary fixed the Indian-term questions:** all 5 right.
- **Maths got worse, but no wrong number was shown.** All 7 maths failures are my code: either the router didn't send
  the question to maths (weeks between dates, crore to lakh, "Write ₹45,00,000 in lakh"), or the checks rejected a
  correct choice from Qwen:
  - an extra "1" for "compounded yearly", which my own prompt asked for;
  - conversion factors added to a conversion;
  - a right expression with the wrong formula name.

  The checks are too rigid about how Qwen fills the schema.
- **General knowledge: all 3 misses are Qwen 1.5B's facts** (Missile Man, national animal, 212°F). This is a
  model-size limit; the 7B diagnostic on the GPU PC will show whether a larger model fixes it.
- **Memory tasks found new code gaps:**
  - "renews" isn't a bill word, so Qwen labelled two subscriptions as money to receive;
  - "By when…" was misread as a period;
  - range items repeat on every day of a list;
  - "wedding" doesn't match "getting married", nor "engagement" "engaged";
  - "appointment" is required as a key term even when the saved sentence just says "Dentist".

## Failures and labels

| Task | Cause | Detail |
| :--- | :--- | :--- |
| D4-M02 (hand fail) | code: bill words / hybrid kind | Subscriptions shown as "To receive" |
| D4-M04 | code: period reader | "By when…" read as today |
| D4-M05 | task ambiguity | "Anything due" read as payments and money only; the task also wanted the surgery |
| D4-M07 (hand pass) | code: range items | Exam range repeated each day; marker counts days inside a range as invented |
| D4-M09 (hand fail) | code: specific-item check | "wedding" vs "getting married" |
| D4-M13 | code: event matching | "engagement" vs "engaged" |
| D4-P13 | code: all-terms check too strict | "appointment" required; the item says "Dentist Dr. Shetty" |
| D4-P15 (hand pass) | scorer | Consistency rule false positive |
| D4-F04, F09, F14, F15 | code: maths validation | Correct choices rejected |
| D4-F08, F10, F11 | code: router | Not sent to maths |
| D4-K07, K08, K10 | model behaviour | Wrong facts from Qwen 1.5B |

Route mismatches: D4-M04 window (expected lookup), D4-M13 status (expected window), D4-F02 and D4-F03 mixed (expected
maths; the answers were right), and D4-F08, F10, F11 (not sent to maths).

## What is next (not done)

- **Fix the code gaps above as general rules:**
  - word families (engaged/engagement, married/wedding);
  - generic event words not required as key terms;
  - subscription words as bills;
  - list a range once;
  - "by when" as a question, not a period;
  - maths routing for weeks, lakh/crore and "write X in Y";
  - maths validation that ignores the factors code doesn't need and falls back to a valid plain expression.

  These are development work. A fresh set would be needed to measure them.
- **Still pending:** the safety layer, the 500-item scale check, the fresh extraction set. The held-out 30 stay
  untouched until S6.
