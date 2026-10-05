import os
import sys
import json
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ui.backend.schemas import (
    GenerateRequest, GenerateResponse,
    CompareRequest, CompareResponse,
    HealthResponse, ModelsResponse
)
from ui.backend.model_service import model_service

app = FastAPI(
    title="MiniGPT API Server",
    description="REST API for MiniGPT Model C (Pretrained Baseline & SFT Instruction Model)",
    version="1.0.0"
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount plots static directory first under /api/plots
plots_dir = os.path.join(REPO_ROOT, "phase6", "evaluation", "plots")
if os.path.exists(plots_dir):
    app.mount("/api/plots", StaticFiles(directory=plots_dir), name="plots")

@app.get("/api/health", response_model=HealthResponse)
def get_health():
    models_ok = os.path.exists(model_service.sft_ckpt_path) and os.path.exists(model_service.base_ckpt_path)
    return HealthResponse(
        status="ok",
        device="cpu",
        models_available=models_ok
    )

@app.get("/api/models", response_model=ModelsResponse)
def get_models():
    models = model_service.get_models_info()
    return ModelsResponse(models=models)

@app.get("/api/evaluation/phase6")
def get_phase6_evaluation():
    eval_dir = os.path.join(REPO_ROOT, "phase6", "evaluation")
    results_dir = os.path.join(REPO_ROOT, "results", "phase6")

    phase6g_file = os.path.join(eval_dir, "phase6g_results.json")
    phase6h_file = os.path.join(eval_dir, "phase6h_comparison.json")
    validation_file = os.path.join(eval_dir, "checkpoint_validation.json")
    summary_file = os.path.join(results_dir, "sft_summary.json")

    phase6g_data = {}
    if os.path.exists(phase6g_file):
        with open(phase6g_file, "r", encoding="utf-8") as f:
            phase6g_data = json.load(f)

    phase6h_data = {}
    if os.path.exists(phase6h_file):
        with open(phase6h_file, "r", encoding="utf-8") as f:
            phase6h_data = json.load(f)

    validation_data = {}
    if os.path.exists(validation_file):
        with open(validation_file, "r", encoding="utf-8") as f:
            validation_data = json.load(f)

    sft_summary = {}
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            sft_summary = json.load(f)

    return {
        "phase6g_results": phase6g_data,
        "phase6h_comparison": phase6h_data,
        "checkpoint_validation": validation_data,
        "sft_summary": sft_summary
    }

@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if req.model not in ["sft", "base"]:
        raise HTTPException(status_code=400, detail="Model must be 'sft' or 'base'.")

    try:
        response = model_service.generate(
            model_key=req.model,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        base_res = model_service.generate(
            model_key="base",
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )
        sft_res = model_service.generate(
            model_key="sft",
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )
        return CompareResponse(
            prompt=req.prompt,
            base_response=base_res,
            sft_response=sft_res
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

# ALWAYS MOUNT SPA FRONTEND LAST SO API ROUTES TAKE PRECEDENCE
dist_dir = os.path.join(REPO_ROOT, "ui", "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
