# Chirii AI privacy notice and tester consent note

**Status:** pilot version, 22 November 2026 (S5). Applies to the pilot on the owner's laptop. Consent note version
`2026-11-22`; the app shows the same points at first login and is used only after the tester accepts them
(`chirii/app/consent.py`; a test checks that both say the same).

## Tester consent note

Before you start: what Chirii keeps, and your choices.

1. **Please use fictional data during the pilot.** Chirii is a student proof of concept, not a finished product: do not
   save real medical, financial or identity details (no account numbers, card numbers, Aadhaar or PAN, passwords).
2. **What is stored:** the sentences you save, what Chirii read from them (kind, dates, times, amounts), every change
   you make, and a search vector for each sentence. Your questions and Chirii's answers are not stored.
3. **Where:** on the pilot laptop only, in your own folder. Nothing is sent to any outside service; the language model
   runs on the same laptop.
4. **Encryption:** your memories are stored encrypted (AES-256-GCM) with a key that only the pilot laptop holds. Your
   password is stored only as an Argon2id hash.
5. **If the key is lost, the data can't be recovered.** There is no back door, for the owner either.
6. **Turn on device encryption** (Windows Device Encryption or BitLocker) on any computer that runs Chirii, so data in
   memory and the page file are protected too.
7. **How to delete:** "Delete all my data" removes every memory, search vector and change history at once, and cannot
   be undone. "Delete my account" also removes your login. You can ask the owner to do either for you.
8. **Logs** record only counts, timings and error types, never what you typed.
9. **Chirii can be wrong.** Check anything important, and for health, money or legal matters ask a qualified person.
   In a crisis, call Tele-MANAS on 14416 or the emergency number 112.
10. **You can stop at any time.**

## What Chirii stores, in detail

- **Your account:** a user name and an Argon2id hash of your password, in `accounts.enc`, encrypted with a key derived
  from the data key. The password itself is never stored or logged. Accounts are created by the owner; there is no
  open sign-up during the pilot.
- **Your memories:** each sentence you save, exactly as written, with the date you said it and what Chirii read from it
  (kind of item, dates, times, amounts, repeats, whether a bill is paid), whether you confirmed it, and a history of
  every change.
- **Search vectors:** 384 numbers made from each sentence so Chirii can find it by meaning. They come from your words,
  so they are treated as personal data and stored inside the same encrypted file.
- **Your consent:** the version of this note you accepted and when.
- **Not stored:** your questions and Chirii's answers, the text of anything in logs, your password. Sessions (which
  browser is logged in) are held only in the app's memory and end when it restarts.

## Where and how it is stored

- **One encrypted file per user,** `<data folder>/users/<code>/memory.enc`. The data folder is
  `%LOCALAPPDATA%\Chirii` on the pilot laptop: outside the code repository and outside OneDrive. The folder name is a
  keyed code made from your user name, so file names don't show who you are.
- **Encryption:** AES-256-GCM, with a separate key for each user derived from the main data key (HKDF-SHA256). A
  changed or damaged file is refused rather than half-read, and a file copied into another user's folder won't open.
  Every change is written to a temporary file and swapped in, so a crash can't leave a half-written file.
- **The data key** (`CHIRII_DATA_KEY`) is kept in the app's `.env` file on the pilot laptop and, by the owner, in a
  password manager. A wrong key stops the app before any data is touched.
- **While the app runs,** your memories are decrypted in the app's memory. Device encryption (point 6 above) protects
  that memory if it is written to the page file, and the disk if the laptop is lost.
- **Nothing is sent to an outside AI service:** the language model (Qwen2.5-1.5B through llama.cpp) and the search model
  run on the same laptop. The app answers only requests from the same computer.
- **Backups:** only if the owner makes one (`python -m chirii.app.keys backup FOLDER`). A backup holds the encrypted
  files only, never the key, and is useless without it.

## How to delete

- **One item:** delete it in the list; its row, search vector and history go with it.
- **Everything:** "Delete all my data" removes your whole folder: every memory, every search vector and all history.
  Other users' data is not touched.
- **Your account:** "Delete my account" (your password is asked again) removes all your data and your login.
- The owner can do either for you on request. Deleted data is not kept anywhere else, except inside an encrypted
  backup the owner made earlier; the owner deletes those on request too.

## Safety

Chirii checks questions and saved notes for signs of self-harm or suicide intent. If it sees them, it does not use the
language model and shows a fixed message with Tele-MANAS (14416) and the emergency number (112). This check runs on
the laptop; nothing is reported to anyone.

## Limits of this pilot

- The owner's laptop holds the data key, so the owner could technically decrypt the data. Keys that only your own
  password can unlock are planned after the pilot (`docs/KNOWN_GAPS.md`).
- This storage design suits a small pilot on one laptop, not a multi-user cloud service.

## India's DPDP Act 2023

The Digital Personal Data Protection Act, 2023 gives people in India rights over their personal data, including to
know what is held and to have it erased. Chirii is built to collect only what the service needs, use it only to help
you, let you see everything it remembers, and delete it when you ask. This is a description of design goals for a
student proof of concept, not legal advice or a compliance claim. Before any public launch, the obligations of a Data
Fiduciary under the Act and its rules must be reviewed properly.
