"""
Demo slice web app: add a memory, see your memories, ask a question, delete everything.

There is no login until S5, so this demo has one user and only answers requests from this computer (127.0.0.1).
Fictional data only until encryption at rest and the tester consent note are done (PLAN.md).

Run from the repository root, with llama-server already running:
    .venv/Scripts/python.exe -m uvicorn chirii.app.api.main:build_default_app --factory --host 127.0.0.1 --port 8000
"""

from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from chirii.app.assistant import Assistant
from chirii.app.config import REPO_ROOT, get_settings
from chirii.app.formatting import understood
from chirii.app.llm.client import ChatModel, LLMClient, LLMError
from chirii.app.logging_setup import configure_logging, log_event
from chirii.app.memory.embeddings import Embedder, FastEmbedder
from chirii.app.memory.hybrid import HybridExtractor
from chirii.app.memory.store import MemoryStore

DEMO_USER = "demo"
LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
PAGE = REPO_ROOT / "chirii" / "web" / "index.html"


class NewMemory(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class Correction(BaseModel):
    kind: str
    title: str
    date: Optional[str] = None
    end_date: Optional[str] = None
    time: Optional[str] = None
    amount: Optional[int] = None
    day_of_month: Optional[int] = None
    weekdays: List[str] = []


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def local_only(request: Request) -> None:
    if request.client is None or request.client.host not in LOCAL_CLIENTS:
        raise HTTPException(status_code=403, detail="This demo only accepts requests from this computer.")


def create_app(data_dir: Path, secret: str, embedder: Embedder, model: ChatModel,
               today: Callable[[], date] = date.today) -> FastAPI:
    app = FastAPI(title="Chirii AI demo", docs_url=None, redoc_url=None, dependencies=[Depends(local_only)])
    state = {"store": None}

    def store() -> MemoryStore:
        if state["store"] is None or not state["store"].dir.exists():
            state["store"] = MemoryStore(data_dir, DEMO_USER, secret, embedder, extractor=HybridExtractor(model))
        return state["store"]

    @app.get("/")
    def page():
        return FileResponse(PAGE)

    def card(item: Dict[str, Any]) -> Dict[str, Any]:
        return {**item, "understood": understood(item)}

    @app.get("/api/memories")
    def list_memories():
        items = [card(i) for i in store().items() if not (i["kind"] == "update" and i["status"] == "applied")]
        return {"today": today().isoformat(), "items": items}

    @app.post("/api/memories")
    def add_memory(body: NewMemory):
        result = store().add(body.text.strip(), today())
        return {"item": result["item"], "understood": result["understood"], "linked_to": result["linked_to"], "changes": result["changes"]}

    @app.post("/api/memories/{item_id}/confirm")
    def confirm_memory(item_id: int):
        item = store().confirm(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="No such memory.")
        return card(item)

    @app.put("/api/memories/{item_id}")
    def correct_memory(item_id: int, body: Correction):
        try:
            item = store().edit(item_id, body.model_dump(), today())
        except KeyError:
            raise HTTPException(status_code=404, detail="No such memory.")
        except ValueError as bad:
            raise HTTPException(status_code=422, detail=f"Please check the {bad.args[0].replace('_', ' ')}.")
        return card(item)

    @app.delete("/api/memories/{item_id}")
    def delete_memory(item_id: int):
        if not store().delete_item(item_id):
            raise HTTPException(status_code=404, detail="No such memory.")
        return {"deleted": item_id}

    @app.post("/api/ask")
    def ask(body: Question):
        try:
            answer = Assistant(store(), model).ask(body.question.strip(), today())
        except LLMError:
            raise HTTPException(status_code=503, detail="The language model is not running. Start llama-server and try again.")
        return {"answer": answer.text, "facts": answer.facts, "duration_ms": answer.duration_ms, "checked": answer.guard,
                "unconfirmed": answer.unconfirmed}

    @app.post("/api/delete-all")
    def delete_all():
        counts = store().delete_all()
        state["store"] = None
        return {"deleted": counts}

    return app


def build_default_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    log_event("app_start", stage="slice")
    return create_app(
        data_dir=settings.data_dir, secret=settings.secret_key,
        embedder=FastEmbedder(settings.embed_model, settings.embed_cache_dir),
        model=LLMClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout_seconds),
    )

