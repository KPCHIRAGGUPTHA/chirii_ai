# MiniGPT Playground & Phase 6 Dashboard

A modern, full-stack web interface and REST API for interacting with MiniGPT Model C (6.61M parameters) and inspecting Phase 6 Supervised Fine-Tuning (SFT) evaluation benchmarks.

---

## Architecture Overview

```text
ui/
  backend/
    main.py           # FastAPI application & REST routing
    model_service.py  # Lazy CPU model loading & inference engine
    schemas.py        # Pydantic request/response data models
  frontend/
    src/
      components/     # Sidebar navigation, StatCard components
      pages/          # Chat, Compare, Dashboard, Evaluation, Model, Training
      services/       # API client service layer
      App.jsx         # Main layout & routing container
      main.jsx        # React entry point
    vite.config.js    # Vite configuration & backend API proxy
    index.html        # HTML entry point
    package.json      # Dependencies (React, Vite, Lucide-React)
```

---

## Setup & Running

### 1. Backend Server (FastAPI)
The backend loads MiniGPT checkpoints lazily on CPU and exposes the REST API on port `8000`.

```bash
# Navigate to repository root
cd c:\Users\dell9\OneDrive\Desktop\minigpt

# Launch FastAPI server via Uvicorn
python -m uvicorn ui.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Development Server (React + Vite)
The frontend runs on Vite dev server at port `3000` with automatic API proxying to backend `http://127.0.0.1:8000`.

```bash
# Navigate to frontend directory
cd ui/frontend

# Install dependencies (first time only)
npm install

# Start Vite dev server
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## Available API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/health` | `GET` | Health check endpoint returning backend CPU status |
| `GET /api/models` | `GET` | Returns architectural details & SHA hashes for Base and SFT models |
| `GET /api/evaluation/phase6` | `GET` | Returns stored Phase 6 evaluation benchmarks & JSON results |
| `POST /api/generate` | `POST` | Generates response for given model (`'sft'` or `'base'`) & prompt |
| `POST /api/compare` | `POST` | Generates side-by-side responses from both Base and SFT models |

---

## Model Checkpoint Locations

- **Base Model (Pretrained Baseline)**: `checkpoints/phase5d/model_6_61m/best_model.pt`
- **SFT Model (Instruction Aligned)**: `checkpoints/phase6/model_c_sft/best_model.pt`
- **Subword Tokenizer**: `tokenizers/phase5d/bpe_vocab_1024.json`

---

## Features

1. **💬 Chat Playground**: Chat directly with SFT or Base model with adjustable temperature, top-k sampling, and context max token parameters.
2. **⚖ Model Compare**: Execute side-by-side comparison of Base Model vs SFT Model under identical prompt and sampling conditions.
3. **📊 Dashboard**: View summary metrics cards (Test PPL `13.5777` ↓ 31.81%, Test Loss `2.6084` ↓ 12.80%), progress tables, and empirical plot charts.
4. **🧪 Evaluation**: Inspect validation set metrics, held-out test set benchmarks, and interactive 30 fixed prompt qualitative outputs.
5. **🧠 Model Architecture**: Inspect 6.61M parameter architecture specs and SHA-256 byte integrity status.
6. **⚙ Training & SFT**: Review SFT hyperparameters, hardware execution stats (AMD Ryzen 7 CPU), and context-length truncation audit.
