**MiniGPT v2.0-preview (98M). PREVIEW, NOT A RELEASE: failed 2 of 4 pre-registered bars (over-refusal 77/100, bare arithmetic 16/18). v1.0.1 remains the released model.**

# MiniGPT v2.0-preview — demo

The Phase 17C model: 98,425,344 parameters (12 layers, 12 heads, 768 wide, 1,024-token context), pretrained from
scratch on 11.0B tokens and instruction-tuned with a calculator tool. It runs on a normal laptop CPU, with no GPU
and, once installed, no internet.

It is shown as a **preview** because it did **not** pass its pre-registered release bars. On the sealed tests it
passed harmful-request refusal (57/60) and "I don't know" (26/36) but failed over-refusal (77/100, bar 90%) and bare
arithmetic (16/18, bar 90%). This is an **education and research demo only**. It makes mistakes, states some facts
wrongly, and refuses some harmless questions. Do not rely on its answers.

## What's inside

```
minigpt_v2_preview_demo/
  demo_chat.py               chat with v2.0-preview
  compare.py                 one prompt -> v1.0.1 and v2.0-preview side by side
  requirements.txt           the three Python packages needed (CPU only)
  minigpt/                   the model code, calculator runtime and safety layer
  model/
    minigpt_v2_preview.pt    v2.0-preview weights (SHA-256 73ea08c6...)
    tokenizer_v2.json        the tokenizer (shared by both models)
  model_v1/                  empty: put v1.0.1's weights here to use compare.py
```

It uses the same decoding as in evaluation: greedy, with no-repeat-3-gram, the calculator, and the self-harm
safety layer before the model. How closely these replies match the evaluation is described under `VERIFIED` at the end.

## Setup on Windows (step by step)

1. **Install Python 3.12.** Use either of these:
   - Open PowerShell and run `py install 3.12` (Python install manager). If `py` is not found, use the next option.
   - Or download the Windows installer for Python 3.12 from https://www.python.org/downloads/windows/ and run it.
     Tick **"Add python.exe to PATH"** on the first screen.

   Check it worked: `py -3.12 --version` should print `Python 3.12.x`.
2. **Unzip** `minigpt_v2_preview_demo.zip`, for example to your Desktop.
3. **Open PowerShell in the folder.** In File Explorer, open the `minigpt_v2_preview_demo` folder, click the
   address bar, type `powershell` and press Enter.
4. **Create and activate a virtual environment:**

   ```
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```

   If PowerShell says running scripts is disabled, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and activate again. (Or use Command Prompt
   and run `venv\Scripts\activate.bat`.) The prompt now starts with `(venv)`.
5. **Install the packages** (CPU-only torch; about 175 MB on Linux, and Windows is similar):

   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **If you see SSL or certificate errors**, first check the computer's date and time (Settings → Time & language
   → Date & time → "Set time automatically"). A wrong system clock is the most common cause. Then run the
   command again.
6. **Start the chat:**

   ```
   python demo_chat.py
   ```

Next time, only steps 3, `venv\Scripts\activate` and step 6 are needed.

## Setup on Linux

```
cd minigpt_v2_preview_demo
python3 -m venv venv                 # Python 3.10, 3.11 or 3.12
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python demo_chat.py
```

## Using it

The first line it prints is the preview notice above. The model loads in a few seconds; then type a message and
press Enter.

```
you > What is 48 plus 76?
[calculator: 48+76 = 124]
minigpt > The answer is 124.
```

Commands (the same as v1.0.1):

- `/quit`: exit
- `/reset`: start a fresh turn (each turn is already independent)
- `/raw`: show the raw model output, including the calculator tokens
- `python demo_chat.py --no-safety-layer`: switch the self-harm safety layer off to see the raw model

## compare.py: v1.0.1 and v2.0-preview side by side

`compare.py` loads both models and gives each prompt to both, printing the replies in two columns with the time
each took.

v1.0.1's weights are not in this zip. Put **v1.0.1's `minigpt_v2.pt`** (from `minigpt_v1_demo/model/` in the
v1.0.1 demo, SHA-256 `3e6a80cd...`; available from the author) into the `model_v1/` folder:

```
minigpt_v2_preview_demo/model_v1/minigpt_v2.pt
```

(The name is confusing but correct: v1.0.1's weights file is called `minigpt_v2.pt` because it came from SFT run
"v2".) The tokenizer is the same for both models, so nothing else is needed. Then:

```
python compare.py                          # type prompts; /quit to exit
python compare.py "What is the capital of Nepal?"
python compare.py --no-safety-layer        # both models without the safety layer
```

## Good things to try

See `DEMO_SCRIPT_V2.md` (in this folder) for 15 tested prompts with both models' actual replies,
failures included.

## Known weak spots

- **Over-refusal**: it refuses some harmless questions that sound risky, about as often as v1.0.1.
- **Facts are unreliable**: it may give a confident wrong answer.
- **Multi-step maths** beyond simple templates mostly fails (GSM8K 1.6%).
- It has no memory between turns.

If you or someone you know is struggling, in India you can call Tele-MANAS on 14416 (free, 24x7), or 112 in an
emergency. The model is not a source of help.

## VERIFIED

Checked on 10 November 2026 with `phase19/verify_demo.py` and `phase19/score_cpu_dev.py` (results in
`phase19/results/`):

- **The model file** is the Phase 17C chosen model (sft_v5 seed 13; SHA-256 `73ea08c6…`), and the tokenizer is
  tokenizer v2 (SHA-256 `332f9db4…`).
- **Same code as evaluated.** On the 15 demo prompts, the package's replies match the evaluation pipeline
  (`phase17/eval_17c.py`: `phase11/chat.py` plus the safety layer) run on the CPU: **0 mismatches** for
  v2.0-preview and for v1.0.1.
- **Not bit-identical to the GPU evaluation.** The 17C evaluation ran on a GPU in bfloat16; this demo runs on a CPU
  in float32. Tiny numerical differences sometimes change which word greedy decoding picks.
  - Demo prompts: 1 of 15 replies differs from the GPU run for each model. For v2.0-preview, row 4 ends with "The
    future can't be known in advance." instead of "Nobody can predict that with certainty."
  - All 236 dev prompts: 24 replies differ from the stored 17C replies. Re-scored, the dev total is **118 instead
    of 119**: one fact item is lost, and over-refusal is the same 40/60.
  - Re-running the GPU pipeline reproduces all 236 stored 17C replies exactly. The difference is GPU bfloat16
    versus CPU float32, not the code.
  - The published 17C numbers are the GPU ones.
- **Speed on CPU** (desktop Intel Core i9-12900, 16 threads; a laptop will be slower): mean 1.38 s per reply for
  v2.0-preview and 0.53 s for v1.0.1 (medians 0.62 s and 0.22 s).
- **Fresh install test** from the zip on Linux: a new venv with Python 3.12.3, then
  `pip install -r requirements.txt` (torch 2.5.1+cpu, a 174.6 MB download; 33 s). Both `demo_chat.py` (with and
  without the safety layer, plus `/raw`, `/reset`, `/quit`) and `compare.py` ran. All 15 demo replies were
  identical to those above (0 mismatches). The Windows steps were written for Windows but not run on a Windows
  machine.
