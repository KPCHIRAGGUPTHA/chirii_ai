# Chirii AI

A personal assistant that remembers your tasks, appointments, bills, notes and important dates, and helps you plan
your day in simple language. It runs on a small open model, Qwen2.5-1.5B-Instruct, so it can be offered cheaply.

This folder is the proof of concept. The research history behind it (Phases 0–31) is in the rest of the repository.

- **Plan:** `PLAN.md`
- **Pass/fail rules:** `CRITERIA.md`
- **Progress log:** `../docs/PROGRESS.md`
- **Privacy:** `../docs/PRIVACY.md`
- **Licences:** `data/SOURCES.md`

## Setup (Windows laptop)

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
copy .env.example .env
```

Then put a random secret in `.env`:

```
.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Run the tests from the repository root:

```
.venv\Scripts\python -m pytest chirii/tests
```

On Linux or macOS, use `.venv/bin/python` and `cp` instead.

## Run the app (pilot laptop)

The repository lives at `C:\dev\chirii_ai` (outside OneDrive, so `.env` and its keys are never synced). The model
files are not in git; download them as described in `data/SOURCES.md` (into `models/`). Run every command from
`C:\dev\chirii_ai`.

0. Once only: create the data key, save a copy in your password manager, and create an account for each tester (they
   type their own password; it is never shown):

   ```
   .venv\Scripts\python -m chirii.app.keys create
   .venv\Scripts\python -m chirii.app.keys show
   .venv\Scripts\python -m chirii.app.accounts add NAME
   ```

1. Start the language model server and leave it running:

   ```
   models\llama.cpp\b12359\llama-server.exe -m models\qwen\qwen2.5-1.5b-instruct-q8_0.gguf --host 127.0.0.1 --port 8080 -c 4096 --alias qwen2.5-1.5b-instruct-q8_0
   ```

2. In a second terminal, start the app:

   ```
   .venv\Scripts\python -m uvicorn chirii.app.api.main:build_default_app --factory --host 127.0.0.1 --port 8000
   ```

3. Open http://127.0.0.1:8000, log in, read and accept the consent note, then add memories (check each "Got it" card),
   ask questions, and try "Delete all my data".

Memories are stored encrypted in `%LOCALAPPDATA%\Chirii` (`CHIRII_DATA_DIR`). If the data key is lost, the data cannot
be recovered. The app refuses requests from other computers. Use fictional data during the pilot
(`../docs/PRIVACY.md`). On this laptop's CPU an answer takes about 1–10 seconds.

Key and backup commands: `python -m chirii.app.keys check | rotate | backup FOLDER` (stop the app before `rotate`).

## Updating dependencies

Edit `requirements.in`, then:

```
uv pip compile requirements.in --python-version 3.12 --universal -o requirements.txt
```
