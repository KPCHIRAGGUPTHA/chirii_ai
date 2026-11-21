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

## Run the demo (laptop, fictional data only)

The model files are not in git; download them as described in `data/SOURCES.md` (into `models/`).

1. Start the language model server and leave it running:

   ```
   models\llama.cpp\b12359\llama-server.exe -m models\qwen\qwen2.5-1.5b-instruct-q8_0.gguf --host 127.0.0.1 --port 8080 -c 4096 --alias qwen2.5-1.5b-instruct-q8_0
   ```

2. In a second terminal, from the repository root, start the app:

   ```
   .venv\Scripts\python -m uvicorn chirii.app.api.main:build_default_app --factory --host 127.0.0.1 --port 8000
   ```

3. Open http://127.0.0.1:8000. Add a few memories and check each "Got it" card (Yes or Edit), ask questions, and try
   "Delete all my data".

The demo has one user and no login (login comes in S5). It refuses requests from other computers. Use fictional data
only: memories are not encrypted on disk yet (`../docs/PRIVACY.md`). On this laptop's CPU an answer takes about 1–5
seconds.

## Updating dependencies

Edit `requirements.in`, then:

```
uv pip compile requirements.in --python-version 3.12 --universal -o requirements.txt
```
