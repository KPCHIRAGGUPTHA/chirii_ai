# Chirii AI privacy notice (draft)

**Status:** draft, updated at S2 (21 November 2026) when the memory store was built. It will be finalised in S5 (login
and API). Sections marked *planned* describe how the system is being built, not what exists today.

## What Chirii stores

- **Your account:** a username and a password hash (Argon2). The password itself is never stored. *Planned, S5.*
- **Your memories:** each sentence you tell Chirii, exactly as you said it, with the date you said it and what Chirii
  read from it (kind of item, dates, times, amounts, repeats, and whether a bill is paid). *Built in S2.*
- **A history of changes:** when you correct something ("the dentist moved to 11 December"), Chirii keeps the old and
  new value, so it can tell you what changed. *Built in S2.*
- **Search vectors:** 384 numbers made from each sentence so Chirii can find it by meaning. They are made from your
  words and are treated as your personal data too. *Built in S2.*

## What Chirii does not store

- **Message text in logs.** Logs record only events such as "a memory was added" or "a reply took 4 seconds", with
  your user id replaced by a one-way code. The code refuses to log anything else.
- **Your data for training.** Nothing you tell Chirii is used to train any model.

## Where it is stored

On the server that runs Chirii: for the demo, the owner's laptop or a server in India. Nothing is sent to an outside
AI service: the language model and the search model both run on the same server.

Each user has one folder, `<data folder>/users/<code>/`, holding two files: `memory.db` (a SQLite database with your
memories and their history) and `index.faiss` (your search vectors). The folder name is a code made from your user id
and the server's secret key, so the file names don't show who you are, and one user's folder is never read when
another user asks a question.

**Not yet:** the files are not encrypted on disk, and there are no backups. Both are decided before any real user
data is stored (S5 and S8).

## How to delete it

- **One item:** deleting a memory removes its row, its search vector and its change history. *Built in S2.*
- **Everything:** "delete all my data" removes your whole folder: every memory, every search vector and all history.
  Other users' data is not touched. Tests check that nothing is left behind. *Built in S2; the button in the app comes
  with the API in S5, which will also delete your account.*

## India's DPDP Act 2023

The Digital Personal Data Protection Act, 2023 gives people in India rights over their personal data, including to
know what is held and to have it erased. Chirii is being built so that it collects only what the service needs, uses it
only to help you, lets you see everything it remembers, and deletes it when you ask. This is a description of design
goals for a student proof of concept, not legal advice or a compliance claim. Before any public launch, the obligations
of a Data Fiduciary under the Act and its rules must be reviewed properly.
