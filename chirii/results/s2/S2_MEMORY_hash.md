# S2 memory store: measurements

Embedder: `hash (test stand-in)`. Run on 2026-11-21 (Windows AMD64, CPU). Development data from the S1
memories, so these are development numbers (see `chirii/eval/s2_gold.py`). No language model is involved yet.

| Check | Result |
| :--- | :--- |
| Sentences parsed fully right (development, 66) | 66/66 |
| Sentences parsed fully right (22 fresh, first try, before any fix) | 20/22 |
| Sentences parsed fully right (22 fresh, now) | 22/22 |
| Updates linked to the right item | 8/8 |
| Updates changed the item correctly | 8/8 |
| Meaning-based questions: needed sentence ranked 1st | 7/8 |
| Meaning-based questions: needed sentence in top 3 | 8/8 |
| New questions sharing no key word with the note: ranked 1st / in top 3 | 1/8 / 3/8 |
| Date windows with nothing missing and nothing extra | 6/6 |
| Delete all my data: rows, vectors removed; folder gone | 9, 9; True |
| Index matches database before delete | True |
| Time to add one sentence / one search | 9.2 ms / 9.6 ms |
| Disk space for one user with 9 items | 29.7 KB |

## Parsing, field by field (development)

| Field | Right |
| :--- | ---: |
| kind | 66/66 |
| date | 54/54 |
| end_date | 1/1 |
| time | 23/23 |
| end_time | 2/2 |
| amount | 19/19 |
| day_of_month | 4/4 |
| weekdays | 2/2 |
| other_dates | 1/1 |
| other_times | 1/1 |
| other_amounts | 1/1 |

## Updates

| Task | Linked right | Changed right | Linked to |
| :--- | :--- | :--- | :--- |
| S1-08 | True | True | My BESCOM electricity bill is ₹1,860 and it's due on 12 Dece |
| S1-09 | True | True | Dentist appointment with Dr. Kavitha on 10 December at 6 pm. |
| S1-18 | True | True | Our Metro Water bill is ₹420, due on 10 December. |
| S1-19 | True | True | Divya's school term fee of ₹24,000 must be paid by 18 Decemb |
| S1-28 | True | True | Mrs. Deshpande owes ₹3,450 on credit and promised to pay by  |
| S1-29 | True | True | Gupta Traders will deliver 50 bags of sugar on 11 December b |
| S1-37 | True | True | My next check-up is with Dr. Ramaswamy at the hospital on 18 |
| S1-48 | True | True | Hostel mess fee of ₹38,500 is due on 20 December. |

## Search by meaning

| Task | Question | Rank of needed sentence | Top result |
| :--- | :--- | ---: | :--- |
| S1-05 | What did I note about the release? | 1 | Note from today's team meeting: the release has moved to 8 January, an |
| S1-15 | What do I need to buy for Pongal? | 1 | Shopping list for Pongal: rice, jaggery, moong dal, ghee, sugarcane an |
| S1-16 | What should I carry for my mother-in-law's eye check-up? | 1 | Mother-in-law's eye check-up at the eye hospital on 16 December at 10  |
| S1-25 | What did I note about the milk price? | 1 | Note: the new milk vendor charges ₹56 a litre from January, up from ₹5 |
| S1-34 | What did the doctor tell me about food? | 2 | My next check-up is with Dr. Ramaswamy at the hospital on 18 December  |
| S1-35 | What does Ananya want for her birthday? | 1 | Granddaughter Ananya's birthday is on 9 January. She wants a story boo |
| S1-44 | Which topics should I study for DBMS? | 1 | Note from DBMS class: the exam covers normalisation, SQL joins and tra |
| S1-45 | Where did I save my PNR? | 1 | Train home to Hyderabad on 13 December at 9:40 pm from Katpadi. The PN |

## Search by meaning: new questions with no shared key words

| Question | Rank of needed sentence | Top result |
| :--- | ---: | :--- |
| What exercise was I advised? | 4 | I submitted my life certificate on 20 November. |
| What groceries for the harvest festival? | 8 | Mother-in-law's eye check-up at the eye hospital on 16 December at 10  |
| What's important for the database paper? | 2 | The post-matric scholarship form's last date is 22 December. |
| Where is my railway ticket number? | 4 | Hostel mess fee of ₹38,500 is due on 20 December. |
| Has the dairy supplier changed his rates? | 2 | I need to update the KYC for the current account at the bank by 15 Dec |
| When does the software launch happen? | 8 | The bank says I must update my KYC at the SBI Koramangala branch befor |
| What present should I buy for my grandchild? | 9 | I submitted my life certificate on 20 November. |
| What papers should I take to the ophthalmologist? | 1 | Mother-in-law's eye check-up at the eye hospital on 16 December at 10  |

## Date windows

| Task | Window | Entries | Missing | Should not be there |
| :--- | :--- | ---: | :--- | :--- |
| S1-03 | 2026-12-07 to 2026-12-14 | 3 | - | - |
| S1-04 | 2026-12-24 to 2026-12-31 | 2 | - | - |
| S1-13 | 2026-12-14 to 2026-12-20 | 6 | - | - |
| S1-14 | 2026-12-24 to 2026-12-31 | 4 | - | - |
| S1-23 | 2026-12-10 to 2026-12-15 | 4 | - | - |
| S1-33 | 2026-12-14 to 2026-12-20 | 17 | - | - |
