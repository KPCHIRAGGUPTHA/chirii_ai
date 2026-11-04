"""Tests for the Phase 18 MiniGPT lm-eval adapter, checked against hand-written computations on a tiny random model."""

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # phase18/ first (no evaluate.py here)
from minigpt_lm import PROJECT_ROOT, MiniGPTLM  # noqa: E402

from model import MiniGPTConfig  # noqa: E402
from phase10.model_v3 import MiniGPTV3  # noqa: E402

BLOCK = 16


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    model = MiniGPTV3(MiniGPTConfig(vocab_size=16384, block_size=BLOCK, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    return MiniGPTLM(model=model, device="cpu", batch_size=3)


def req(*args, task="t"):
    return types.SimpleNamespace(args=args, task_name=task)


def manual_logprob(lm, tokens, n_cont):
    """Sum of log p(token) for the last n_cont tokens, given everything before them (tokens fit in the window + 1)."""
    inp = torch.tensor([tokens[:-1]])
    logits, _ = lm.model(inp, torch.zeros_like(inp))
    logprobs = F.log_softmax(logits.float(), dim=-1)[0]
    target = torch.tensor(tokens[1:])
    per_token = logprobs.gather(1, target[:, None])[:, 0]
    return float(per_token[-n_cont:].sum()), bool((logprobs.argmax(-1)[-n_cont:] == target[-n_cont:]).all())


def test_registered():
    from lm_eval.api.registry import get_model
    assert get_model("minigpt") is MiniGPTLM


def test_loglikelihood_matches_manual(lm):
    context, continuation = "The cat sat on the", " mat"
    (score, greedy), = lm.loglikelihood([req(context, continuation)])
    whole = lm.tok_encode(context + continuation)
    n_cont = len(whole) - len(lm.tok_encode(context))
    expected, expected_greedy = manual_logprob(lm, whole, n_cont)
    assert score == pytest.approx(expected, abs=1e-4)
    assert greedy == expected_greedy


def test_batching_and_order_do_not_change_scores(lm):
    pairs = [("A short one", " yes"), ("A rather longer context sentence here", " no"), ("Mid size text", " maybe so"), ("x", " y")]
    together = lm.loglikelihood([req(c, t) for c, t in pairs])
    alone = [lm.loglikelihood([req(c, t)])[0] for c, t in pairs]
    for (a, ga), (b, gb) in zip(together, alone):
        assert a == pytest.approx(b, abs=1e-4) and ga == gb


def test_long_context_cut_from_left_and_counted(lm):
    lm.truncation.clear()
    context = " ".join(["word"] * 40)
    (score, _), = lm.loglikelihood([req(context, " end", task="long")])
    whole = lm.tok_encode(context + " end")
    n_cont = len(whole) - len(lm.tok_encode(context))
    expected, _ = manual_logprob(lm, whole[-(BLOCK + 1):], n_cont)  # the window keeps the last BLOCK+1 tokens
    assert score == pytest.approx(expected, abs=1e-4)
    row = lm.truncation["long"]
    assert row["loglikelihood_requests"] == 1 and row["loglikelihood_truncated"] == 1
    lm.loglikelihood([req("short", " one", task="short")])
    assert lm.truncation["short"]["loglikelihood_truncated"] == 0


def test_rolling_short_text_equals_scoring_after_eot(lm):
    text = "Hello there friend"
    (rolling,) = lm.loglikelihood_rolling([req(text)])
    tokens = [lm.eot_token_id] + lm.tok_encode(text)
    expected, _ = manual_logprob(lm, tokens, len(tokens) - 1)
    assert rolling == pytest.approx(expected, abs=1e-4)


def test_rolling_long_text_counts_every_token_once(lm):
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
    text = " ".join(f"token{i}" for i in range(30))
    (rolling,) = lm.loglikelihood_rolling([req(text, task="roll")])
    tokens = lm.tok_encode(text)
    windows = [make_disjoint_window(w) for w in get_rolling_token_windows(tokens, lm.eot_token_id, BLOCK, 1)]
    assert sum(len(cont) for _, cont in windows) == len(tokens)
    expected = sum(manual_logprob(lm, (ctx + cont)[-(BLOCK + 1):], len(cont))[0] for ctx, cont in windows)
    assert rolling == pytest.approx(expected, abs=1e-3)
    assert lm.truncation["roll"]["rolling_truncated"] == 1


def test_generate_matches_manual_greedy_and_respects_limits(lm):
    context = "Once upon a"
    (text,) = lm.generate_until([req(context, {"until": ["☃never"], "max_gen_toks": 5})])
    ids = lm.tok_encode(context)
    out = []
    for _ in range(5):
        logits, _ = lm.model(torch.tensor([ids[-BLOCK:]]))
        token = int(logits[0, -1].argmax())
        if token == lm.eot_token_id:
            break
        ids.append(token)
        out.append(token)
    assert text == lm.tok_decode(out)


def test_generate_stops_at_until(lm):
    (free,) = lm.generate_until([req("Once upon a", {"until": [], "max_gen_toks": 6})])
    if len(free) >= 2:
        stop = free[1]
        (cut,) = lm.generate_until([req("Once upon a", {"until": [stop], "max_gen_toks": 6})])
        assert stop not in cut and free.startswith(cut)


def test_generation_context_cut_is_counted(lm):
    lm.truncation.clear()
    lm.generate_until([req(" ".join(["word"] * 40), {"until": [], "max_gen_toks": 4}, task="gen")])
    assert lm.truncation["gen"]["generate_truncated"] == 1


REAL = os.path.join(PROJECT_ROOT, "phase11/runs/sft_v2/final_model.pt")


@pytest.mark.skipif(not os.path.exists(REAL), reason="released weights not present")
def test_real_model_prefers_the_right_answer():
    real = MiniGPTLM(checkpoint=REAL, device="cpu", batch_size=2)
    (paris, _), (banana, _) = real.loglikelihood([req("The capital of France is", " Paris"), req("The capital of France is", " banana")])
    assert paris > banana


def _uncached_greedy(lm, context_ids, max_gen_toks):
    room = lm.block_size - max_gen_toks
    ids, out = list(context_ids[-room:]), []
    for _ in range(max_gen_toks):
        logits, _ = lm.model(torch.tensor([ids[-lm.block_size:]], device=lm._device))
        token = int(logits[0, -1].argmax())
        if token == lm.eot_token_id:
            break
        ids.append(token)
        out.append(token)
    return out


def test_cached_logits_equal_full_forward_at_every_step(lm):
    ids = lm.tok_encode("The quick brown fox jumps over")
    logits, cache = lm._forward_cached(ids, 0, None)
    seq = list(ids)
    for token in lm.tok_encode(" the lazy dog and more"):
        full, _ = lm.model(torch.tensor([seq]))
        assert torch.allclose(logits, full[0, -1], atol=1e-5)
        seq.append(token)
        logits, cache = lm._forward_cached([token], len(seq) - 1, cache)
    full, _ = lm.model(torch.tensor([seq]))
    assert torch.allclose(logits, full[0, -1], atol=1e-5)


def test_cached_generation_with_cut_context_equals_uncached(lm):
    context = " ".join(f"w{i}" for i in range(30))  # far longer than the 16-token window
    text, truncated = lm._greedy(lm.tok_encode(context), [], 8)
    assert truncated
    assert text == lm.tok_decode(_uncached_greedy(lm, lm.tok_encode(context), 8))


@pytest.mark.skipif(not os.path.exists(REAL), reason="released weights not present")
def test_real_model_cached_generation_identical():
    real = MiniGPTLM(checkpoint=REAL, device="cuda" if torch.cuda.is_available() else "cpu")
    ids = real.tok_encode("Question: Janet has 3 apples and buys 5 more. How many apples does she have?\nAnswer:")
    text, _ = real._greedy(ids, [], 40)
    assert text == real.tok_decode(_uncached_greedy(real, ids, 40))
