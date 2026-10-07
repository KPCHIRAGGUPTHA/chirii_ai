import os
import sys
import time
import random
from typing import Optional, Dict, Any

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase7.retrieval.lexical_retriever import LexicalRetriever
from phase7.rag.rag_models import (
    RAGRequest,
    RAGResponse,
    ContextBudgetInfo,
)
from phase7.rag.prompt_builder import PromptBuilder
from phase7.rag.context_budget import ContextBudgetManager

DEFAULT_SFT_CKPT = os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt")
DEFAULT_TOKENIZER = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")

class RAGPipeline:
    """
    End-to-End RAG Generation Pipeline for MiniGPT.
    Connects BM25 Lexical Retriever, Context Budget Guard, and Phase 6 SFT Model C.
    """

    def __init__(
        self,
        retriever: Optional[LexicalRetriever] = None,
        model: Optional[MiniGPT] = None,
        tokenizer: Optional[BPETokenizer] = None,
        sft_ckpt_path: str = DEFAULT_SFT_CKPT,
        tokenizer_path: str = DEFAULT_TOKENIZER,
        default_score_threshold: float = 1.5,
        device: str = "cpu"
    ):
        self.device = device
        self.retriever = retriever
        self.default_score_threshold = float(default_score_threshold)

        # 1. Tokenizer initialization
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"BPE Tokenizer vocabulary file not found at: {tokenizer_path}")
            self.tokenizer = BPETokenizer.load(tokenizer_path)

        # 2. Model initialization (inference-only)
        if model is not None:
            self.model = model
        else:
            if not os.path.exists(sft_ckpt_path):
                raise FileNotFoundError(f"SFT model checkpoint file not found at: {sft_ckpt_path}")
            ckpt = torch.load(sft_ckpt_path, map_location=self.device, weights_only=False)
            config = ckpt.get("config") or MiniGPTConfig(
                vocab_size=1024, block_size=128, n_layer=8, n_head=8, n_embd=256
            )
            self.model = MiniGPT(config)
            state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
            self.model.load_state_dict(state_dict)

        # Guarantee eval mode & device placement
        self.model.eval()
        self.model.to(self.device)

        self.budget_manager = ContextBudgetManager(tokenizer=self.tokenizer)
        self.prompt_builder = PromptBuilder()

    def answer(
        self,
        question: str,
        top_k: int = 1,
        score_threshold: Optional[float] = None,
        max_new_tokens: int = 46,
        temperature: float = 0.2,
        top_k_sampling: int = 10,
        top_p_sampling: float = 0.9,
        seed: Optional[int] = None
    ) -> RAGResponse:
        """
        Execute RAG answer generation given a user question.

        Args:
            question: Input user question string.
            top_k: Number of micro-chunks to retrieve (default 1).
            score_threshold: Minimum BM25 score threshold for context injection.
            max_new_tokens: Generation token budget (default <= 46).
            temperature: Sampling temperature (default 0.2 for deterministic RAG).
            top_k_sampling: Top-k sampling candidate count (default 10).
            top_p_sampling: Top-p nucleus threshold (default 0.9).
            seed: Optional integer seed for reproducible generation.

        Returns:
            RAGResponse object.
        """
        t_start = time.perf_counter()

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        thresh = self.default_score_threshold if score_threshold is None else float(score_threshold)

        # Step 1: Retrieval
        retrieved = False
        top_item = None
        context_text = None

        if self.retriever is not None:
            envelope = self.retriever.search(query=question, top_k=top_k, score_threshold=thresh)
            if envelope.retrieved and len(envelope.results) > 0:
                retrieved = True
                top_item = envelope.results[0]
                context_text = top_item.text

        # Step 2: Context Budget Guard & Prompt Construction
        final_prompt, budget_info = self.budget_manager.allocate_budget(
            query=question,
            context_text=context_text,
            generation_reserve=max_new_tokens,
            total_budget=128
        )

        # Step 3: Inference (No Gradients)
        prompt_tokens = self.tokenizer.encode(final_prompt)
        if not prompt_tokens:
            prompt_tokens = [0]

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k_sampling,
                top_p=top_p_sampling
            )

        all_gen_tokens = out_ids[0].tolist()
        gen_tokens = all_gen_tokens[len(prompt_tokens):]
        generated_answer = self.tokenizer.decode(gen_tokens).strip()

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Step 4: Metadata Envelope Assembly
        if retrieved and top_item is not None:
            source = top_item.source
            doc_id = top_item.document_id
            chunk_id = top_item.chunk_id
            retrieval_score = top_item.score
        else:
            source = None
            doc_id = None
            chunk_id = None
            retrieval_score = 0.0

        return RAGResponse(
            answer=generated_answer,
            retrieved=retrieved,
            source=source,
            document_id=doc_id,
            chunk_id=chunk_id,
            retrieval_score=retrieval_score,
            generation_tokens=len(gen_tokens),
            latency_ms=latency_ms,
            budget=budget_info.to_dict(),
            retrieved_text=context_text,
            prompt_used=final_prompt
        )
