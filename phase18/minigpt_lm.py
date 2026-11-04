"""
Phase 18 — MiniGPT adapter for EleutherAI lm-evaluation-harness (0.4.13).

Registered as model "minigpt". Model args (comma-separated, as for any lm-eval model):
    checkpoint=<path to .pt>      required
    chat_generation=True|False    generative tasks go through MiniGPT's chat runtime (chat format, calculator tool);
                                  multiple-choice / loglikelihood tasks always use the plain prompt, as for every model
    calculator=True|False         only with chat_generation: allow the calculator tool (default True)
    batch_size=<int>              loglikelihood batch size (default 32)
    truncation_log=<path.json>    where to write per-task truncation counts

Context handling. MiniGPT's context is 512 tokens. Like lm-eval's Hugging Face backend, inputs that are too long
are cut from the LEFT, so the scored continuation always stays whole (unless the continuation alone exceeds the
window). Every cut is counted per task, so the report can say how often each task was truncated.

Precision: float32 for scoring and plain generation. Decoding: greedy for every request (no repetition rules), the
same as the other models in Phase 18.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)  # appended, not prepended: the repo's evaluate.py must not shadow HF evaluate

from lm_eval.api.model import TemplateLM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window  # noqa: E402

TOKENIZER_PATH = "phase10/tokenizer/tokenizer_v2.json"
EOT = "<|endoftext|>"


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")


@register_model("minigpt")
class MiniGPTLM(TemplateLM):
    def __init__(self, checkpoint: Optional[str] = None, chat_generation: Any = False, calculator: Any = True, batch_size: Any = 32,
                 device: str = "cuda", truncation_log: Optional[str] = None, model: Any = None, tokenizer: Any = None, **kwargs):
        super().__init__()
        from tokenizers import Tokenizer

        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if model is None:
            from phase10.eval_v3 import load_model_v3
            model, _ = load_model_v3(checkpoint if os.path.isabs(checkpoint) else os.path.join(PROJECT_ROOT, checkpoint), self._device)
        self.model = model.to(self._device).eval()
        self.tokenizer = tokenizer or Tokenizer.from_file(os.path.join(PROJECT_ROOT, TOKENIZER_PATH))
        self.block_size = self.model.config.block_size
        self.batch_size = int(batch_size) if str(batch_size) != "auto" else 32
        self.chat_generation = _bool(chat_generation)
        self.calculator = _bool(calculator)
        self.truncation_log = truncation_log
        self.truncation: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._task_names: List[Optional[str]] = []

    # ---- tokenizer ---------------------------------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.token_to_id(EOT)

    @property
    def max_length(self) -> int:
        return self.block_size

    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None, **kwargs) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False).ids

    def tok_decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ---- truncation accounting ---------------------------------------------------------------------------------
    def _count(self, task: Optional[str], kind: str, truncated: bool, continuation_cut: bool = False) -> None:
        row = self.truncation[task or "unknown"]
        row[f"{kind}_requests"] += 1
        row[f"{kind}_truncated"] += int(truncated)
        if continuation_cut:
            row[f"{kind}_continuation_cut"] += 1

    def write_truncation_log(self) -> None:
        if self.truncation_log:
            os.makedirs(os.path.dirname(os.path.abspath(self.truncation_log)), exist_ok=True)
            with open(self.truncation_log, "w", encoding="utf-8") as f:
                json.dump({task: dict(row) for task, row in sorted(self.truncation.items())}, f, indent=2)

    # ---- scoring -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _logprobs(self, batch: List[List[int]]) -> torch.Tensor:
        """Log-softmax over the vocabulary at every position. Right-padded, so padding never affects real positions."""
        width = max(len(x) for x in batch)
        idx = torch.full((len(batch), width), self.eot_token_id, dtype=torch.long, device=self._device)
        for row, tokens in enumerate(batch):
            idx[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self._device)
        logits, _ = self.model(idx, torch.zeros_like(idx))  # targets make the model return logits for every position
        return F.log_softmax(logits.float(), dim=-1)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        self._task_names = [getattr(r, "task_name", None) for r in requests]
        try:
            return super().loglikelihood(requests, disable_tqdm=disable_tqdm)
        finally:
            self.write_truncation_log()

    def _prepare(self, context_enc: List[int], continuation_enc: List[int]) -> Tuple[List[int], int, bool, bool]:
        """Input tokens (at most block_size), the number of continuation tokens scored, and whether anything was cut."""
        whole = context_enc + continuation_enc
        truncated = len(whole) - 1 > self.block_size
        inp = whole[-(self.block_size + 1):][:-1]
        n_cont = min(len(continuation_enc), len(inp))
        return inp, n_cont, truncated, n_cont < len(continuation_enc)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
        tasks = self._task_names if len(self._task_names) == len(requests) else [None] * len(requests)
        prepared = []
        for i, ((_, _), context_enc, continuation_enc) in enumerate(requests):
            inp, n_cont, truncated, cont_cut = self._prepare(context_enc, continuation_enc)
            self._count(tasks[i], "loglikelihood", truncated, cont_cut)
            target = (context_enc + continuation_enc)[-(n_cont):] if n_cont else []
            prepared.append((i, inp, target))
        results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
        order = sorted(prepared, key=lambda p: -len(p[1]))  # similar lengths together: less padding
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            logprobs = self._logprobs([inp for _, inp, _ in chunk])
            for row, (i, inp, target) in enumerate(chunk):
                n = len(target)
                if n == 0:
                    results[i] = (0.0, True)
                    continue
                positions = logprobs[row, len(inp) - n:len(inp)]
                target_t = torch.tensor(target, dtype=torch.long, device=positions.device)
                total = float(positions.gather(1, target_t[:, None]).sum())
                greedy = bool((positions.argmax(dim=-1) == target_t).all())
                results[i] = (total, greedy)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        out = []
        for request in requests:
            (string,) = request.args
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self.tok_encode(string), prefix_token=self.prefix_token_id, max_seq_len=self.block_size, context_len=1)]
            self._count(getattr(request, "task_name", None), "rolling", len(windows) > 1)
            scored = self._loglikelihood_tokens([((None, None), ctx, cont) for ctx, cont in windows]) if windows else []
            out.append(sum(score for score, _ in scored))
        self.write_truncation_log()
        return out

    # ---- generation --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _forward_cached(self, new_ids: List[int], start: int, cache: Optional[list]) -> Tuple[torch.Tensor, list]:
        """Logits for the last new token, with a key/value cache, using the model's own modules in forward() order.
        Only the new tokens are computed; earlier keys/values come from the cache. Positions start at `start`."""
        m = self.model
        idx = torch.tensor([new_ids], dtype=torch.long, device=self._device)
        pos = torch.arange(start, start + len(new_ids), dtype=torch.long, device=self._device)
        x = m.transformer.drop(m.transformer.wte(idx) + m.transformer.wpe(pos))
        new_cache = []
        for layer, block in enumerate(m.transformer.h):
            attn = block.attn
            B, T, C = x.size()
            q, k, v = attn.c_attn(block.ln_1(x)).split(attn.n_embd, dim=2)
            q, k, v = (t.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2) for t in (q, k, v))
            if cache is not None:
                k = torch.cat([cache[layer][0], k], dim=2)
                v = torch.cat([cache[layer][1], v], dim=2)
            new_cache.append((k, v))
            if T == k.size(2):  # no cache: ordinary causal attention over the prompt
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif T == 1:  # one new token may attend to everything before it
                y = F.scaled_dot_product_attention(q, k, v)
            else:
                raise ValueError("cached forward supports one new token at a time")
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + attn.resid_dropout(attn.c_proj(y))
            x = x + block.mlp(block.ln_2(x))
        x = m.transformer.ln_f(x)
        return m.lm_head(x[:, -1, :])[0], new_cache

    @torch.no_grad()
    def _greedy(self, context_enc: List[int], until: List[str], max_gen_toks: int) -> Tuple[str, bool]:
        """Greedy decoding with a key/value cache. The context is cut from the left so that context + max_gen_toks fits
        the window, so the cache never has to slide and every position matches a plain forward pass."""
        room = self.block_size - max_gen_toks
        if room < 1:
            raise ValueError(f"max_gen_toks {max_gen_toks} leaves no room for context in a {self.block_size}-token window")
        truncated = len(context_enc) > room
        ids = list(context_enc[-room:]) if context_enc else [self.eot_token_id]
        generated: List[int] = []
        logits, cache = self._forward_cached(ids, 0, None)
        for step in range(max_gen_toks):
            token = int(logits.argmax())
            if token == self.eot_token_id:
                break
            ids.append(token)
            generated.append(token)
            text = self.tok_decode(generated)
            if any(stop in text for stop in until):
                break
            if step + 1 < max_gen_toks:
                logits, cache = self._forward_cached([token], len(ids) - 1, cache)
        return self.tok_decode(generated), truncated

    def _chat_reply(self, context: str, max_gen_toks: int) -> Tuple[str, bool]:
        from phase11.chat import MAX_TOOL_CALLS, Decoding, generate_reply
        from phase11.chat_format import ASSISTANT, SYSTEM, SYSTEM_LINE, USER, sanitize_user_text
        # Same rule as phase11.chat.encode_prompt, which keeps the first `limit` user tokens and cuts the rest
        fixed = len(self.tok_encode(f"{SYSTEM}{SYSTEM_LINE}{USER}")) + len(self.tok_encode(ASSISTANT))
        limit = max(1, self.block_size - max_gen_toks - fixed)
        truncated = len(self.tok_encode(sanitize_user_text(context))) > limit
        reply = generate_reply(self.model, self.tokenizer, context, self._device, max_gen_toks,
                               max_tool_calls=MAX_TOOL_CALLS if self.calculator else 0, decoding=Decoding())
        return reply.final, truncated

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        out = []
        for request in requests:
            context, gen_kwargs = request.args
            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.get("until") or []
            until = [until] if isinstance(until, str) else list(until)
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 256))
            if self.chat_generation:
                text, truncated = self._chat_reply(context, max_gen_toks)
            else:
                text, truncated = self._greedy(self.tok_encode(context), until, max_gen_toks)
            for stop in until:  # cut at the first stop sequence, as lm-eval's HF backend does
                if stop and stop in text:
                    text = text.split(stop)[0]
            self._count(getattr(request, "task_name", None), "generate", truncated)
            out.append(text)
        self.write_truncation_log()
        return out
