# S3 re-marked with the fixed marker

The two marker errors found in S3 were fixed on 22 November, with tests:
1. a month followed by a numbered list item ("15 December.⏎2.") was read as a date;
2. the empty days of an asked week ("Tuesday 15 December: no appointments") were counted as invented.

The same S3 replies are re-marked below. **The original scores stay the official S3 result** (`S3_RESULTS.md`).
Guard decisions in run (b) are as they happened: the fixed rules would have kept S1-27's correct first answer, but
that answer was replaced at the time, and the replacement is what is marked here.

| Run | Bar A original | Bar A re-marked | Bar B original → re-marked | Bar C original → re-marked |
| :--- | ---: | ---: | :--- | :--- |
| a | 28/50 | 28/50 | 5/8 → 5/8 | 2 → 1 |
| b | 28/50 | 28/50 | 5/8 → 5/8 | 0 → 0 |
| none | 10/50 | 10/50 | 8/8 → 8/8 | 1 → 1 |

## Verdicts that changed

| Run | Task | Original | Re-marked | Remaining failures |
| :--- | :--- | :--- | :--- | :--- |
| a | S1-23 | fail, invented | fail | ["contains forbidden {'amount': 7800}"] |
