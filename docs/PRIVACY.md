# Chirii AI privacy notice (draft)

**Status:** draft written at S0, before any user data is stored. It will be updated in S2 (memory store) and finalised
in S5 (login and API). Sections marked *planned* describe how the system is being built, not what exists today.

## What Chirii stores

- **Your account:** a username and a password hash (Argon2). The password itself is never stored. *Planned, S5.*
- **Your memories:** the tasks, appointments, bills, notes and important dates you tell Chirii, with their dates and
  amounts, in a database file that belongs only to you. *Planned, S2.*
- **Search vectors:** a list of numbers made from each note so Chirii can find it by meaning. They are made from your
  notes and are treated as your personal data too. *Planned, S2.*

## What Chirii does not store

- **Message text in logs.** Logs record only events such as "a memory was added" or "a reply took 4 seconds", with
  your user id replaced by a one-way code. The code refuses to log anything else.
- **Your data for training.** Nothing you tell Chirii is used to train any model.

## Where it is stored

On the server that runs Chirii: for the demo, the owner's laptop or a server in India. Nothing is sent to an outside
AI service: the language model runs on the same server.

## How to delete it

A "delete all my data" option will remove your memories, your search vectors and your account. *Planned, S2 and S5.*

## India's DPDP Act 2023

The Digital Personal Data Protection Act, 2023 gives people in India rights over their personal data, including to
know what is held and to have it erased. Chirii is being built so that it collects only what the service needs, uses it
only to help you, lets you see everything it remembers, and deletes it when you ask. This is a description of design
goals for a student proof of concept, not legal advice or a compliance claim. Before any public launch, the obligations
of a Data Fiduciary under the Act and its rules must be reviewed properly.
