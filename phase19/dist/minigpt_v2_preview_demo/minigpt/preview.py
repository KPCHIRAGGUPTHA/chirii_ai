"""Fixed text and file locations for the MiniGPT v2.0-preview demo."""

import os

PREVIEW_NOTICE = ("MiniGPT v2.0-preview (98M). PREVIEW, NOT A RELEASE: failed 2 of 4 pre-registered bars "
                  "(over-refusal 77/100, bare arithmetic 16/18). v1.0.1 remains the released model.")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_MODEL = os.path.join(HERE, "model", "minigpt_v2_preview.pt")      # 17C chosen model (sft_v5 seed 13)
V2_MODEL_SHA256 = "73ea08c6a6d7ec1b94957853bb0a79e37103a0fbbf474731c5ac4e7401e95e99"
TOKENIZER = os.path.join(HERE, "model", "tokenizer_v2.json")
TOKENIZER_SHA256 = "332f9db4368e051179e70695ff6da320a8dc16f8f17934719cd58c8a00601b98"
V1_MODEL = os.path.join(HERE, "model_v1", "minigpt_v2.pt")            # v1.0.1 weights (sft_v2), only for compare.py
V1_MODEL_SHA256 = "3e6a80cdd69e8d43f386b4381c9d9683e831d3f0726faf379036fa17f7355877"
