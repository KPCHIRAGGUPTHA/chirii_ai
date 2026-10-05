from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GenerateRequest(BaseModel):
    model: str = Field(default="sft", description="Model variant to generate from: 'sft' or 'base'")
    prompt: str = Field(..., description="Prompt text or instruction")
    max_new_tokens: int = Field(default=40, ge=1, le=128, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature (0.0 = greedy)")
    top_k: Optional[int] = Field(default=50, ge=0, description="Top-k filtering threshold")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling threshold")

class GenerateResponse(BaseModel):
    model: str
    prompt: str
    response: str
    full_text: str
    generation_time_sec: float
    tokens_generated: int
    tokens_per_second: float

class CompareRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to send to both Base and SFT models")
    max_new_tokens: int = Field(default=40, ge=1, le=128)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)

class CompareResponse(BaseModel):
    prompt: str
    base_response: GenerateResponse
    sft_response: GenerateResponse

class HealthResponse(BaseModel):
    status: str
    device: str
    models_available: bool

class ModelInfo(BaseModel):
    id: str
    name: str
    parameters: int
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int
    checkpoint_path: str
    checkpoint_sha256: str
    status: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
