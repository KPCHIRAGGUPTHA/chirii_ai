# Known gaps (after run (e5), 22 November 2026)

The development-set loop stopped after run (e5), as the owner decided before the run. These gaps are logged here, **not
fixed now**. Each says where it was seen. Any fix later is development work and would need a fresh set to measure.

## Wrong answers (most serious)

- **Made-up detail when one word of the question isn't saved.** "When does Mr. Desai's leave start?" (nothing about
  leave saved) → "4:00 pm on Fridays", taken from his weekly visit (D5-M12). The specific-item check only requires
  words from a fixed list of kinds of thing; "leave" is not on it. *Possible fix:* treat every content noun of a lookup
  as a key term, matched by meaning.

- **Fixed before s6-final (owner, 22 Nov):** "before / by / until <date or weekday>" (was read as that one day) and
  "in / during <month>" (was not read as a period). A third report, "Anything to pay before Priya's birthday?" going
  to general knowledge, was a mistake in my check (the router was called without the saved memories); in the app it
  already worked. See PROGRESS, S6 step 5a.
- **Questions with no "I/my" and no "anything/due"** ("Bills to pay before 10 Jan?") still go to general knowledge
  even though the period is now read (the router gap below, D5-M15).

## Extraction

- **Multi-day weekly repeats** written as abbreviated weekdays without commas or "and" ("Tuition for Riya every Tue
  Thu Sat 4pm") are read as the first day only (Tuesday), and not flagged for the user (extraction set C, the one
  "wrong and not flagged" of 20). Accepted with S5 (owner, 22 Nov); not fixed.

## Router

- **"plan" as a noun** ("the Jio plan", "my mobile plan") is read as a day plan (D5-M11).
- **"How does Saturday look for me?"** isn't a plan phrase (only "how does my X look"), so it gets a week list (D5-M07).
- **Questions without "my/I"** ("Is there any bill this week?") can go to general knowledge when the thing asked about
  ("bill") isn't on the list of specific things and "any" isn't one of the words that force memory (D5-M15).
- **"Selvi kaun hai?", "Who is Mr. Desai?"** go to lookup, not people (answers right; D5-P04, P05).

## Matching and lists

- **Hyphenated names never match** ("Where is the Wi-Fi router?" vs a saved "Wi-Fi router…"): memory text is split
  at the hyphen ("wi", "fi"), the question's term is not, so the specific-item check says "I don't have anything
  about Wi-Fi saved" (scale check SC-25, the one miss at both 40 and 500 items). A small fix (split both the same
  way); found in a scored run, so logged here rather than fixed silently. Accepted with S5 (owner, 22 Nov).
- **Long lists are complete but too long** (30 bills before an event; 9 badminton games on a Saturday): **addressed in
  S6 step 4** (a list over 12 lines is summarised, with "Show all").
- **Generic event words are required as key terms:** "my dentist appointment" vs a saved "Dentist Dr. Shetty…"
  (D4-P13).
- **Lists are not narrowed to the kind asked for:** "which subscriptions" also lists rent; "anything to renew this
  month" lists every item (D5-M03, D5-M19). Not wrong, but over-broad.
- **"Anything due…"** is read as payments only; a surgery was left out (D4-M05).
- **Meaning matching is conservative:** the threshold (0.66, calibrated on my own pairs) catches 16 of 30 true synonyms
  and 0 of 30 near-misses. Synonyms that the embedding model scores low (flat/apartment, grocery/provisions) are not
  matched unless they are stems or in the small Indian synonym map.

## Safety

- **A personal crisis in Hinglish that also names a relative** is suppressed as third-party wording by the S5
  personal-crisis rule: "Mujhe marna hai, mummy ko pata nahi." gets no support reply, and "Main suicide kar lunga,
  papa ko mat batana." gets the S6 helper reply (which still gives 14416 and 112) instead of the personal one. Found
  while adding the helper reply; the personal path was left unchanged on the owner's instruction. (English is fine:
  "I want to kill myself and my mom doesn't know" gets the personal reply.)

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

- S5 is done (22 Nov). Testers only with accounts made by the owner, after they accept the consent note.
- S6: chat UI and the held-out 30 (untouched until then).
- GPU jobs (owner): the benchmark job and the 7B diagnostic.
