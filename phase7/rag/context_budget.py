from typing import Optional, Tuple
from bpe_tokenizer import BPETokenizer
from phase7.rag.rag_models import ContextBudgetInfo
from phase7.rag.prompt_builder import PromptBuilder

class ContextBudgetManager:
    """
    Enforces strict token budget guardrails for MiniGPT 128-token context window.
    Guarantees prompt_tokens + generation_reserve <= total_budget (128).
    """

    def __init__(self, tokenizer: BPETokenizer, prompt_builder: Optional[PromptBuilder] = None):
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder or PromptBuilder()

    def allocate_budget(
        self,
        query: str,
        context_text: Optional[str] = None,
        generation_reserve: int = 46,
        total_budget: int = 128
    ) -> Tuple[str, ContextBudgetInfo]:
        """
        Tokenize and measure actual prompt, truncating context or query if necessary
        to guarantee prompt_tokens + generation_reserve <= total_budget.

        Args:
            query: User input query string.
            context_text: Optional retrieved context text.
            generation_reserve: Tokens reserved for model generation output (default 46).
            total_budget: Model context limit (default 128).

        Returns:
            Tuple of (final_prompt_string, ContextBudgetInfo).
        """
        max_prompt_budget = total_budget - generation_reserve
        assert max_prompt_budget > 0, f"generation_reserve ({generation_reserve}) >= total_budget ({total_budget})"

        query_str = query.strip() if query else ""
        context_str = context_text.strip() if (context_text and context_text.strip()) else None

        raw_query_ids = self.tokenizer.encode(query_str)
        raw_context_ids = self.tokenizer.encode(context_str) if context_str else []

        truncated_query = False
        truncated_context = False

        # 1. First, check if prompt fits without any truncation
        raw_prompt = self.prompt_builder.build_prompt(query_str, context_str)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt)

        if len(raw_prompt_ids) <= max_prompt_budget:
            budget_info = ContextBudgetInfo(
                query_tokens=len(raw_query_ids),
                context_tokens=len(raw_context_ids),
                prompt_tokens=len(raw_prompt_ids),
                generation_reserve=generation_reserve,
                total_budget=total_budget,
                truncated_query=False,
                truncated_context=False,
            )
            return raw_prompt, budget_info

        # 2. Truncation required.
        # Check if query itself is excessively long (e.g., > 22 tokens or taking > 50% of prompt budget)
        max_query_target = min(22, max_prompt_budget - 15)
        curr_query_ids = list(raw_query_ids)
        if len(curr_query_ids) > max_query_target:
            curr_query_ids = curr_query_ids[:max_query_target]
            query_str = self.tokenizer.decode(curr_query_ids)
            truncated_query = True

        # 3. Fit context into remaining budget
        final_prompt = ""
        final_prompt_ids = []
        final_context_ids = []

        if context_str and raw_context_ids:
            curr_context_ids = list(raw_context_ids)
            while curr_context_ids:
                cand_context = self.tokenizer.decode(curr_context_ids)
                cand_prompt = self.prompt_builder.build_prompt(query_str, cand_context)
                cand_prompt_ids = self.tokenizer.encode(cand_prompt)

                if len(cand_prompt_ids) <= max_prompt_budget:
                    final_prompt = cand_prompt
                    final_prompt_ids = cand_prompt_ids
                    final_context_ids = curr_context_ids
                    truncated_context = True
                    break

                curr_context_ids.pop()
            else:
                # Context could not fit even with 1 token; drop context completely
                truncated_context = True

        if not final_prompt:
            # Context dropped or not provided: build query-only prompt
            cand_prompt = self.prompt_builder.build_prompt(query_str, None)
            cand_prompt_ids = self.tokenizer.encode(cand_prompt)

            # If query-only prompt still exceeds budget, truncate query tokens iteratively
            while len(cand_prompt_ids) > max_prompt_budget and len(curr_query_ids) > 1:
                curr_query_ids.pop()
                query_str = self.tokenizer.decode(curr_query_ids)
                truncated_query = True
                cand_prompt = self.prompt_builder.build_prompt(query_str, None)
                cand_prompt_ids = self.tokenizer.encode(cand_prompt)

            final_prompt = cand_prompt
            final_prompt_ids = cand_prompt_ids
            final_context_ids = []

        budget_info = ContextBudgetInfo(
            query_tokens=len(curr_query_ids),
            context_tokens=len(final_context_ids),
            prompt_tokens=len(final_prompt_ids),
            generation_reserve=generation_reserve,
            total_budget=total_budget,
            truncated_query=truncated_query,
            truncated_context=truncated_context,
        )

        return final_prompt, budget_info


def allocate_context_budget(
    query: str,
    context_text: Optional[str],
    tokenizer: BPETokenizer,
    generation_reserve: int = 46,
    total_budget: int = 128
) -> Tuple[str, ContextBudgetInfo]:
    """
    Convenience function for budget allocation.
    """
    manager = ContextBudgetManager(tokenizer=tokenizer)
    return manager.allocate_budget(
        query=query,
        context_text=context_text,
        generation_reserve=generation_reserve,
        total_budget=total_budget,
    )
