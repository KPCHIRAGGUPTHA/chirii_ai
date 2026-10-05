import os
import sys
import time
import hashlib
import torch
from typing import Dict, Any, Tuple, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt
from ui.backend.schemas import GenerateResponse, ModelInfo

def get_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "MISSING"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

class ModelService:
    def __init__(self):
        self.device = "cpu"
        self.base_ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
        self.sft_ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt")
        self.vocab_path = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")

        self._models: Dict[str, MiniGPT] = {}
        self._tokenizer: Optional[BPETokenizer] = None

    def get_tokenizer(self) -> BPETokenizer:
        if self._tokenizer is None:
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"Tokenizer vocab missing at {self.vocab_path}")
            self._tokenizer = BPETokenizer.load(self.vocab_path)
        return self._tokenizer

    def load_model(self, model_key: str) -> MiniGPT:
        if model_key in self._models:
            return self._models[model_key]

        if model_key == "sft":
            ckpt_path = self.sft_ckpt_path
        elif model_key == "base":
            ckpt_path = self.base_ckpt_path
        else:
            raise ValueError(f"Invalid model_key: '{model_key}'. Must be 'sft' or 'base'.")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint missing at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_config = ckpt.get("config") or MiniGPTConfig(
            vocab_size=1024, block_size=128, n_layer=8, n_head=8, n_embd=256
        )
        model = MiniGPT(model_config)
        state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        self._models[model_key] = model
        return model

    def get_models_info(self) -> list[ModelInfo]:
        base_sha = get_file_sha256(self.base_ckpt_path)
        sft_sha = get_file_sha256(self.sft_ckpt_path)

        return [
            ModelInfo(
                id="sft",
                name="Phase 6E Model C (SFT Fine-Tuned)",
                parameters=6613504,
                n_layer=8,
                n_head=8,
                n_embd=256,
                vocab_size=1024,
                block_size=128,
                checkpoint_path="checkpoints/phase6/model_c_sft/best_model.pt",
                checkpoint_sha256=sft_sha,
                status="Ready (Default)" if os.path.exists(self.sft_ckpt_path) else "Missing"
            ),
            ModelInfo(
                id="base",
                name="Phase 5D Model C (Pretrained Baseline)",
                parameters=6613504,
                n_layer=8,
                n_head=8,
                n_embd=256,
                vocab_size=1024,
                block_size=128,
                checkpoint_path="checkpoints/phase5d/model_6_61m/best_model.pt",
                checkpoint_sha256=base_sha,
                status="Ready" if os.path.exists(self.base_ckpt_path) else "Missing"
            )
        ]

    def generate(
        self,
        model_key: str = "sft",
        prompt: str = "",
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> GenerateResponse:
        model = self.load_model(model_key)
        tokenizer = self.get_tokenizer()

        # Format prompt appropriately
        if model_key == "sft" and not prompt.startswith("Instruction:"):
            prompt_prefix, _ = format_alpaca_prompt(prompt.strip(), "", "")
        else:
            prompt_prefix = prompt

        prompt_tokens = tokenizer.encode(prompt_prefix)
        if not prompt_tokens:
            prompt_tokens = [0]

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        start_time = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        gen_duration = max(time.time() - start_time, 1e-4)

        gen_tokens = out_ids[0].tolist()
        resp_tokens = gen_tokens[len(prompt_tokens):]

        full_text = tokenizer.decode(gen_tokens)
        generated_response = tokenizer.decode(resp_tokens).strip()

        num_tokens = len(resp_tokens)
        tps = round(num_tokens / gen_duration, 2)

        return GenerateResponse(
            model=model_key,
            prompt=prompt,
            response=generated_response,
            full_text=full_text,
            generation_time_sec=round(gen_duration, 3),
            tokens_generated=num_tokens,
            tokens_per_second=tps
        )

# Global Singleton Instance
model_service = ModelService()
