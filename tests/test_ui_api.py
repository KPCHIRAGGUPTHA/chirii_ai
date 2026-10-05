import pytest
from fastapi.testclient import TestClient
from ui.backend.main import app

client = TestClient(app)

def test_ui_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["device"] == "cpu"
    assert "models_available" in data

def test_ui_api_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 2
    model_ids = [m["id"] for m in data["models"]]
    assert "sft" in model_ids and "base" in model_ids

def test_ui_api_evaluation():
    response = client.get("/api/evaluation/phase6")
    assert response.status_code == 200
    data = response.json()
    assert "phase6g_results" in data
    assert "phase6h_comparison" in data
    assert "checkpoint_validation" in data

def test_ui_api_generate_validation():
    # Empty prompt validation
    response = client.post("/api/generate", json={"prompt": "", "model": "sft"})
    assert response.status_code == 400

    # Invalid model selection validation
    response = client.post("/api/generate", json={"prompt": "Hello", "model": "invalid_name"})
    assert response.status_code == 400

def test_ui_api_generate_sft_success():
    response = client.post("/api/generate", json={
        "prompt": "What is 2 + 2?",
        "model": "sft",
        "max_new_tokens": 10,
        "temperature": 0.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "sft"
    assert "response" in data
    assert "generation_time_sec" in data
    assert "tokens_generated" in data
