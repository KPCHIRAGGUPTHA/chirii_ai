# MiniGPT Architecture Notes

MiniGPT Model C features 8 attention heads and an embedding dimension of 256.
It is fine-tuned using Supervised Fine-Tuning (SFT) for high response accuracy.

## Retrieval Specs
RAG micro-chunks target 35 to 45 tokens per chunk.
Overlap between consecutive chunks is approximately 10 tokens.
