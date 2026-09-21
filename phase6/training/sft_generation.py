import os
import sys
import torch
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT
from bpe_tokenizer import BPETokenizer
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt

DEFAULT_PROMPTS = [
    {"instruction": "Give three tips for staying healthy.", "input": ""},
    {"instruction": "Calculate 15 multiplied by 4.", "input": ""},
    {"instruction": "Translate the phrase 'Hello world' to Spanish.", "input": ""},
    {"instruction": "Summarize the key idea of machine learning.", "input": ""},
    {"instruction": "Write a short rhyming poem about a cat.", "input": ""}
]

@torch.no_grad()
def generate_sft_responses(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompts: List[Dict[str, str]] = None,
    device: str = "cpu",
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 40
) -> List[Dict[str, Any]]:
    """
    Generate responses for a standard evaluation prompt suite.
    Used for BEFORE vs AFTER SFT qualitative generation comparison.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    model.eval()
    results = []

    for item in prompts:
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        prompt_prefix, _ = format_alpaca_prompt(inst, inp, "")

        prompt_tokens = tokenizer.encode(prompt_prefix)
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Generate tokens
        out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        generated_tokens = out_idx[0].tolist()

        full_decoded = tokenizer.decode(generated_tokens)
        response_decoded = tokenizer.decode(generated_tokens[len(prompt_tokens):])

        results.append({
            "instruction": inst,
            "input": inp,
            "prompt_prefix": prompt_prefix,
            "full_generated_text": full_decoded,
            "generated_response": response_decoded.strip()
        })

    return results
