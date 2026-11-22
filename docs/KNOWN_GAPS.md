# Known gaps (after run (e5), 22 November 2026)

The development-set loop stopped after run (e5), as the owner decided before the run. These gaps are logged here, **not
fixed now**. Each says where it was seen. Any fix later is development work and would need a fresh set to measure.

## Wrong answers (most serious)

- **Made-up detail when one word of the question isn't saved.** "When does Mr. Desai's leave start?" (nothing about
  leave saved) → "4:00 pm on Fridays", taken from his weekly visit (D5-M12). The specific-item check only requires
  words from a fixed list of kinds of thing; "leave" is not on it. *Possible fix:* treat every content noun of a lookup
  as a key term, matched by meaning.

## Router

- **"plan" as a noun** ("the Jio plan", "my mobile plan") is read as a day plan (D5-M11).
- **"How does Saturday look for me?"** isn't a plan phrase (only "how does my X look"), so it gets a week list (D5-M07).
- **Questions without "my/I"** ("Is there any bill this week?") can go to general knowledge when the thing asked about
  ("bill") isn't on the list of specific things and "any" isn't one of the words that force memory (D5-M15).
- **"Selvi kaun hai?", "Who is Mr. Desai?"** go to lookup, not people (answers right; D5-P04, P05).

## Matching and lists

- **Generic event words are required as key terms:** "my dentist appointment" vs a saved "Dentist Dr. Shetty…"
  (D4-P13).
- **Lists are not narrowed to the kind asked for:** "which subscriptions" also lists rent; "anything to renew this
  month" lists every item (D5-M03, D5-M19). Not wrong, but over-broad.
- **"Anything due…"** is read as payments only; a surgery was left out (D4-M05).
- **Meaning matching is conservative:** the threshold (0.66, calibrated on my own pairs) catches 16 of 30 true synonyms
  and 0 of 30 near-misses. Synonyms that the embedding model scores low (flat/apartment, grocery/provisions) are not
  matched unless they are stems or in the small Indian synonym map.

## Answers in Hinglish

- **Qwen echoes Hinglish questions** ("Kiraya ₹25,000 dena hai aur kitna?", "Kabir se kitne paise lene hain? 700.")
  and can drop part of the answer (D5-M02, D5-M20).

## Model facts (no code fix; the 7B diagnostic decides)

- **General knowledge:** wrong facts from Qwen 1.5B (Missile Man, national animal, 212°F; D4-K07, K08, K10).

## Marker

- **The consistency rule** flags "You haven't told me more than that" after a right answer (D4-P15, D5-P10). Any change
  must be recorded in CRITERIA.md before the run it affects, and old scores shown next to re-marked ones.
- **Amounts without ₹** ("700") are not seen by the amount rule (D5-M20).

## Post-pilot (owner, 22 Nov: deferred on purpose)

- **Per-user password-derived keys.** Today the server holds the data key, so the server could read any user's
  memories. Wrapping each user's key with a key derived from their password (Argon2id) would mean only the person can
  decrypt their data, but a forgotten password then loses it for good, and reminders can't run while they are logged
  out. Needs a recovery design first.
- **Storage redesign for multi-user cloud.** Whole-file encryption (one encrypted SQLite file per user, held in memory
  while open, rewritten on each change) suits a laptop pilot of a few users with up to a few thousand items each. A
  cloud service with many users needs a real database with encryption (encrypted volumes or column encryption, key
  management service, backups, concurrent access from several app servers).

## Not yet done (planned, not gaps in what exists)

- S5: login and tester consent note (encryption at rest is done). **Fictional data only until S5 is done.**
- The safety layer, the 500-item scale check, the fresh extraction set (target: at most 2 of 20 wrong and not flagged).
- S6: chat UI and the held-out 30 (untouched until then).
- GPU jobs (owner): the benchmark job and the 7B diagnostic.
