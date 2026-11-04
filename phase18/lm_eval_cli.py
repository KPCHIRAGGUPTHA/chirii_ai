"""Phase 18 — lm-eval command line with the MiniGPT adapter registered (same arguments as `python -m lm_eval`).

lm-eval registers its built-in models (hf, ...) only when the registry is empty, so they are imported first;
otherwise registering "minigpt" would hide them.
"""

import lm_eval.models  # noqa: F401  built-in models first
import minigpt_lm  # noqa: F401  registers model "minigpt"
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
