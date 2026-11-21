# Data notices

This project (MiniGPT / Chirii AI) used the datasets below for training or evaluation.

- **Licences:** each licence is as listed on the dataset's Hugging Face page, checked on 14 November 2026. Where that page
  gives no licence, the authors' own repository or README was checked; the table says which source was used.
- **Copyright:** all rights in the datasets stay with their authors. Credit for each dataset goes to its creators. Please
  cite the original papers when you use them.
- **Model weights:** none are stored in this repository.

## Training data

| Dataset | Used for | Licence (source) |
| :--- | :--- | :--- |
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, sample-10BT) | MiniGPT pretraining | ODC-By (HF page) |
| Alpaca-cleaned (`yahma/alpaca-cleaned`) | instruction tuning | **CC BY-NC 4.0** (its README; the HF page tag says CC BY 4.0): non-commercial |
| Dolly 15k (`databricks/databricks-dolly-15k`) | instruction tuning | CC BY-SA 3.0 (HF page) |
| smol-smoltalk (`HuggingFaceTB/smol-smoltalk`) | instruction tuning | Apache-2.0 (HF page) |
| Simple English Wikipedia (`wikimedia/wikipedia`, 20231101.simple) | RAG corpus; documents in document-mode refusal rows | CC BY-SA 3.0 and GFDL (HF page) |
| SQuAD 2.0 train (`rajpurkar/squad_v2`) | RAG / document-mode fine-tuning | CC BY-SA 4.0 (HF page) |
| GSM8K train (`openai/gsm8k`) | self-distilled solutions for fine-tuning | MIT (HF page) |
| SciQ train (`allenai/sciq`) | inputs for self-distilled replay (Phase 26 and later) | **CC BY-NC 3.0** (HF page): non-commercial |
| BoolQ train (`aps/super_glue`, boolq) | inputs for self-distilled replay (Phase 26 and later) | CC BY-SA 3.0 (authors' README, `google-research-datasets/boolean-questions`; the HF page says "other") |
| PKU-SafeRLHF | refusal prompts in Chirii Brain v0.3.1 to v0.5 (prompts only, never responses) | **CC BY-NC 4.0**: non-commercial |
| glaive-function-calling-v2 | function-calling rows in c1 and c2 | Apache-2.0 |
| oasst1 | conversation rows in c1 and c2 | Apache-2.0 |
| OR-Bench 80K / Toxic | over-refusal and refusal rows in c1 and c2 | CC BY 4.0 (attribution) |
| AdvBench | refusal rows in c1 and c2 | MIT |

These five rows follow the licence summary in `FINAL_REPORT.md`; check each source's own page before reuse.

Alpaca-cleaned is treated as **non-commercial**, matching the licence summary in `FINAL_REPORT.md`: its README gives
CC BY-NC 4.0, and the stricter of the two licences applies. So every model trained on it is research-only: the MiniGPT
SFT models (v1.0, v1.0.1, v2.0-preview, Phases 17C, 17E and 20) and Chirii Brain v0.1 to v0.5.

Because SciQ is CC BY-NC 3.0, the Chirii Brain v0.2 and later fine-tunes, which were trained on plain Qwen's answers to
SciQ train inputs, are also **non-commercial**.

The base model for Chirii Brain is Qwen2.5-1.5B-Instruct (Apache-2.0, Alibaba Cloud). Its own training data is not
covered here.

## Evaluation data

| Benchmark | Licence (source) | What this repository contains |
| :--- | :--- | :--- |
| XSTest (`Paul/XSTest`) | CC BY 4.0 (HF page) | totals only |
| OR-Bench-Hard-1K (`bench-llm/or-bench`) | CC BY 4.0 (HF page) | totals and item ids; no prompts |
| HarmBench (`centerforaisafety/HarmBench`, text behaviours) | MIT (GitHub repository) | totals only; no prompts or replies |
| TruthfulQA (`truthfulqa/truthful_qa`) | Apache-2.0 (HF page) | scores |
| GSM8K test (`openai/gsm8k`) | MIT (HF page) | scores; Phase 18 subset sample logs (model replies with questions) |
| SQuAD 2.0 dev (`rajpurkar/squad_v2`) | CC BY-SA 4.0 (HF page) | scores; model replies on the evaluated items (CC BY-SA 4.0) |
| TriviaQA (`mandarjoshi/trivia_qa`) | see below | scores only |
| ARC-Easy / ARC-Challenge (`allenai/ai2_arc`) | CC BY-SA 4.0 (HF page) | scores |
| HellaSwag (`Rowan/hellaswag`) | MIT (HF card, pointing to the authors' GitHub licence) | scores |
| PIQA (`baber/piqa`; original `ybisk/piqa`) | not stated (HF pages: none / "unknown") | scores |
| WinoGrande (`allenai/winogrande`) | Apache-2.0 (authors' GitHub repository; the HF page states none) | scores |
| SciQ test (`allenai/sciq`) | CC BY-NC 3.0 (HF page) | scores |
| OpenBookQA (`allenai/openbookqa`) | Apache-2.0 (authors' GitHub repository; the HF page says "unknown") | scores |
| BoolQ validation (`aps/super_glue`) | CC BY-SA 3.0 (authors' README) | scores |
| LAMBADA (`EleutherAI/lambada_openai`) | MIT (HF page) | scores |
| MMLU (`cais/mmlu`) | MIT (HF page) | scores |

Notes:
- **HellaSwag:** the authors' GitHub repository (`rowanz/hellaswag`) was not reachable when checked. GitHub reports
  it as blocked under a DMCA notice dated 14 September 2026, which concerns WikiHow content. This repository holds
  HellaSwag scores only.
- **Author-written sets:** the dev, sealed, over-refusal, misfire, showcase and RAG test sets were written for this
  project. Their items stay private where the reports say so.

## TriviaQA

- **Hugging Face page (`mandarjoshi/trivia_qa`):** the licence is listed as **"unknown"**. The card says: "The
  University of Washington does not own the copyright of the questions and documents included in TriviaQA."
- **The authors' repository (`mandarjoshi90/triviaqa`):** licensed **Apache-2.0**, and its README says "The Apache
  2.0 License applies to both the code and the data."
- **Our reading:** TriviaQA is available under Apache-2.0 per its authors. **With a caveat:** the questions were
  collected from trivia websites, and the authors state they do not own their copyright.
- **Credit:** TriviaQA was created by Mandar Joshi, Eunsol Choi, Daniel S. Weld and Luke Zettlemoyer (University of
  Washington): "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension", ACL 2017.
- **What changed here:** item-level TriviaQA files (the Phase 18 `subset_samples_triviaqa.jsonl` sample logs, which
  contain questions and gold answers) were removed from this repository's main branch on 14 November 2026. They
  remain in its earlier git history, because history is not rewritten. Only TriviaQA scores are kept.
