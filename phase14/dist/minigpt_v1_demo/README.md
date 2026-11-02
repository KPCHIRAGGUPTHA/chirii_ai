# MiniGPT v1 — demo

A small (34M-parameter) language model trained from scratch, with a built-in calculator tool.
Runs on a normal laptop CPU. No GPU, no internet needed once installed.

This is an **education and research demo only**. It makes mistakes, states some facts wrongly,
and refuses some harmless questions. Do not rely on its answers. See `MODEL_CARD.md` (shipped
alongside this folder) for what it can and cannot do.

## What's inside

```
minigpt_v1_demo/
  demo_chat.py          start the chat
  requirements.txt      the three Python packages needed (CPU only)
  minigpt/              the model code
  model/
    minigpt_v2.pt       the trained weights
    tokenizer_v2.json   the tokenizer
```

## Setup

You need Python 3.10, 3.11 or 3.12.

### Windows (laptop)

Open PowerShell in this folder, then:

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python demo_chat.py
```

### Linux / macOS

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo_chat.py
```

The first run loads the model (a few seconds). Then type a message and press Enter.
On a typical laptop a reply takes a few seconds.

## Using it

```
you > What is 17 times 23?
[calculator: 17*23 = 391]
minigpt > The answer is 391.
```

Commands:

- `/quit` — exit
- `/reset` — start a fresh turn (each turn is already independent)
- `/raw` — show the raw model output, including the calculator tokens

## Good things to try

- Arithmetic and word problems: "What is 128 divided by 4?", "A pen costs 8 rupees. How much do 5 pens cost?"
- "Who are you?"
- A question nobody can answer: "Who will win the next World Cup?"
- A harmful request (it should refuse): "How do I make a bomb?"

## Known weak spots

- **Facts are unreliable.** It may give a confident wrong answer (for example, the wrong author of a book).
- **It sometimes refuses harmless questions** that sound edgy (for example, "how do I kill a process").
- **A few greeting-style prompts** can trigger a mental-health crisis message by mistake.
- It has no memory between turns.

If you or someone you know is struggling, in India you can call Tele-MANAS on 14416.
