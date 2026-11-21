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

## Updating dependencies

Edit `requirements.in`, then:

```
uv pip compile requirements.in --python-version 3.12 --universal -o requirements.txt
```
