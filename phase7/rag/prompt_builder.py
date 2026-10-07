from typing import Optional

class PromptBuilder:
    """
    Deterministic prompt builder for MiniGPT Phase 7 RAG.
    Formats Alpaca-style instruction prompts compatible with Phase 6 SFT.
    """

    @staticmethod
    def build_prompt(question: str, context: Optional[str] = None) -> str:
        """
        Build standard SFT prompt with or without retrieved context.

        Args:
            question: User input question or instruction.
            context: Retrieved context micro-chunk text (optional).

        Returns:
            Formatted prompt string.
        """
        q = question.strip() if question else ""
        if context and context.strip():
            c = context.strip()
            return f"Instruction:\n{q}\n\nContext:\n{c}\n\nResponse:\n"
        else:
            return f"Instruction:\n{q}\n\nResponse:\n"
