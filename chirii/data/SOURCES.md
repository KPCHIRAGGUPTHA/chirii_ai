# Chirii AI: sources and licences

Every model, dataset and library that ships with or trains Chirii AI is listed here with its licence. Anything marked
**non-commercial** must not be used in the product or its training data. Checked 21 November 2026; not legal advice.

## Models

| Model | Licence | Use |
| :--- | :--- | :--- |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | Default brain; ship its LICENSE with the weights |
| Qwen2.5-7B-Instruct | Apache-2.0 | Possible later upgrade |
| Qwen2.5-3B-Instruct | Qwen Research License: **non-commercial** | **Excluded** |
| sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | Embeddings for note search |
| MiniGPT (this repository) | Project's own | Research benchmark only, not in the product |

## Software

| Library | Licence |
| :--- | :--- |
| llama.cpp | MIT |
| FAISS (`faiss-cpu`) | MIT |
| fastembed | Apache-2.0 |
| FastAPI, Starlette, Pydantic | MIT |
| argon2-cffi | MIT |

## Datasets

None are used yet. Synthetic examples written for this project are the project's own. Every dataset added later
gets a row here before it is used.

**Never used for Chirii training (non-commercial):** alpaca-cleaned (CC BY-NC 4.0), SciQ (CC BY-NC 3.0) and
PKU-SafeRLHF (CC BY-NC 4.0). See the repository's `DATA_NOTICES.md`.
