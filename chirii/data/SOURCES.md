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

## Downloaded files (S1, 21 November 2026)

All three come from official sources and sit under `models/`, which git ignores. To recreate them, download the same
files and check the SHA-256.

| What | Source | Version | Licence | File and SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| llama.cpp, Windows CPU build | ggml-org/llama.cpp GitHub release `b12359` (13 Nov 2026), asset `llama-b12359-bin-win-cpu-x64.zip` | build 12359, commit b1b770d90. The newest stable tag, v1.0.0, has no Windows assets; its `nightly-tag.txt` points to b12359 | MIT | zip, 25,097,509 bytes, `837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69` |
| Qwen2.5-1.5B-Instruct Q8_0 GGUF | Hugging Face `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q8_0.gguf` | repo revision `91cad51170dc346986eccefdc2dd33a9da36ead9` | Apache-2.0 (the repo's LICENSE is saved next to the file) | 1,894,532,128 bytes, `d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8` |
| all-MiniLM-L6-v2 embeddings (ONNX) | fetched by fastembed 0.9.1, which downloads Qdrant's ONNX export `Qdrant/all-MiniLM-L6-v2-onnx` of `sentence-transformers/all-MiniLM-L6-v2` | revision `d13954661f83248295ba75c1ed411eef3b7b936e`, 384 dimensions | Apache-2.0 | `model.onnx`, `bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5` (cache in `models/fastembed`) |

**Note on the embedding file:** fastembed doesn't load sentence-transformers' own files. It loads Qdrant's ONNX
conversion of the same model, published under the same Apache-2.0 licence.

**Smoke test on this laptop (CPU, 8 threads), not an eval run:** llama-server loaded the Q8_0 file and answered one
neutral question. It read the prompt at about 64 tokens/s and wrote about 16 tokens/s, using about 1.75 GB RAM with
a 4,096-token context. The embedding model loaded and turned 3 sentences into 384-number vectors. A related pair
scored 0.32 and an unrelated pair 0.02.

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
