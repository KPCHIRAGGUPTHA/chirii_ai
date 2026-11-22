"""
Chirii web app: log in, accept the consent note, add a memory, see your memories, ask a question, delete everything.

S5: every memory request needs a logged-in user who has accepted the current consent note, and is answered from that
user's own encrypted store only. The app still answers only requests from this computer (127.0.0.1), and requests
that change something must come from this app's own page (the Origin check), on top of SameSite=Strict cookies.

Run from the repository root, with llama-server already running:
    .venv/Scripts/python.exe -m uvicorn chirii.app.api.main:build_default_app --factory --host 127.0.0.1 --port 8000
"""

import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from chirii.app.accounts import COOKIE, IDLE_SECONDS, Accounts, Locked, Sessions
from chirii.app.assistant import Assistant
from chirii.app.consent import CONSENT_POINTS, CONSENT_TITLE, CONSENT_VERSION
from chirii.app.config import ENV_FILE, REPO_ROOT, get_settings
from chirii.app.keys import read_env
from chirii.app.memory.vault import check_key
from chirii.app import people as people_module
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


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class Consent(BaseModel):
    version: str


class Confirmation(BaseModel):
    password: str = Field(min_length=1, max_length=256)


def people_mentioned(item: Dict[str, Any], saved: List[Dict[str, Any]]) -> List[str]:
    """S4e: the known people a new memory is about ("Amma's eye check-up" → "Amma: Lalitha, your mother")."""
    ppl = people_module.build(saved)
    found = []
    for p in ppl.people.values():
        names = ([p.name.split()[0]] if p.name else []) + sorted(p.aliases)
        if p.relation and any(re.search(r"\b" + re.escape(n) + r"\b", item["text"]) for n in names) and item["kind"] not in ("person", "profile"):
            found.append(f"{' / '.join(names)}: {people_module.describe(p, ppl)}")
    return found


def local_only(request: Request) -> None:
    if request.client is None or request.client.host not in LOCAL_CLIENTS:
        raise HTTPException(status_code=403, detail="This app only accepts requests from this computer.")


def same_origin(request: Request) -> None:
    """A request that changes something must come from this app's own page, not from another site."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="Requests must come from the Chirii page.")


def create_app(data_dir: Path, secret: str, embedder: Embedder, model: ChatModel,
               today: Callable[[], date] = date.today, data_key: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Chirii AI", docs_url=None, redoc_url=None, openapi_url=None,
                  dependencies=[Depends(local_only), Depends(same_origin)])
    key = data_key or secret                          # tests may leave data_key out; the app never does
    accounts, sessions = Accounts(data_dir, key), Sessions()
    stores: Dict[str, MemoryStore] = {}
    stores_lock = threading.RLock()                   # re-entrant: delete_all holds it while calling store_for
    app.state.accounts, app.state.sessions = accounts, sessions

    def logged_in(request: Request) -> str:
        user = sessions.user(request.cookies.get(COOKIE))
        if user is None or accounts.get(user) is None:
            raise HTTPException(status_code=401, detail="Please log in.")
        return user

    def consented(user: str = Depends(logged_in)) -> str:
        consent = accounts.get(user).get("consent")
        if not consent or consent.get("version") != CONSENT_VERSION:
            raise HTTPException(status_code=403, detail="Please read and accept the consent note first.")
        return user

    def store_for(user: str) -> MemoryStore:
        """The logged-in user's own store; the user name comes only from the session, never from the request."""
        with stores_lock:
            s = stores.get(user)
            if s is None or not s.dir.exists():
                s = stores[user] = MemoryStore(data_dir, user, secret, embedder, extractor=HybridExtractor(model), data_key=key)
            return s

    @app.get("/")
    def page():
        return FileResponse(PAGE, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                                           "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                                                      "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"})

    # ---------------------------------------------------------------- login and consent

    @app.post("/api/login")
    def login(body: Login, request: Request, response: Response):
        try:
            user = accounts.verify(body.username, body.password)
        except Locked:
            raise HTTPException(status_code=429, detail="Too many wrong passwords. Try again in 15 minutes.")
        if user is None:
            raise HTTPException(status_code=401, detail="Wrong user name or password.")
        sessions.end(request.cookies.get(COOKIE))
        response.set_cookie(COOKIE, sessions.create(user), max_age=IDLE_SECONDS, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", path="/")
        return me_for(user)

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        sessions.end(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")
        return {"logged_out": True}

    def me_for(user: str) -> Dict[str, Any]:
        consent = accounts.get(user).get("consent") or {}
        return {"username": user, "consented": consent.get("version") == CONSENT_VERSION}

    @app.get("/api/me")
    def me(user: str = Depends(logged_in)):
        return me_for(user)

    @app.get("/api/consent-note")
    def consent_note():
        return {"version": CONSENT_VERSION, "title": CONSENT_TITLE, "points": CONSENT_POINTS}

    @app.post("/api/consent")
    def accept_consent(body: Consent, user: str = Depends(logged_in)):
        if body.version != CONSENT_VERSION:
            raise HTTPException(status_code=409, detail="The consent note has changed; please read it again.")
        accounts.set_consent(user, CONSENT_VERSION)
        log_event("consent_accepted", stage="S5")
        return me_for(user)

    # ---------------------------------------------------------------- memories (the logged-in user's only)

    def card(item: Dict[str, Any]) -> Dict[str, Any]:
        return {**item, "understood": understood(item)}

    @app.get("/api/memories")
    def list_memories(user: str = Depends(consented)):
        items = [card(i) for i in store_for(user).items() if not (i["kind"] == "update" and i["status"] == "applied")]
        return {"today": today().isoformat(), "items": items}

    @app.post("/api/memories")
    def add_memory(body: NewMemory, user: str = Depends(consented)):
        store = store_for(user)
        result = store.add(body.text.strip(), today())
        card = dict(result["understood"])
        card["people"] = people_mentioned(result["item"], store.items(include_updates=False))
        return {"item": result["item"], "understood": card, "linked_to": result["linked_to"], "changes": result["changes"]}

    @app.post("/api/memories/{item_id}/confirm")
    def confirm_memory(item_id: int, user: str = Depends(consented)):
        item = store_for(user).confirm(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="No such memory.")
        return card(item)

    @app.put("/api/memories/{item_id}")
    def correct_memory(item_id: int, body: Correction, user: str = Depends(consented)):
        try:
            item = store_for(user).edit(item_id, body.model_dump(), today())
        except KeyError:
            raise HTTPException(status_code=404, detail="No such memory.")
        except ValueError as bad:
            raise HTTPException(status_code=422, detail=f"Please check the {bad.args[0].replace('_', ' ')}.")
        return card(item)

    @app.delete("/api/memories/{item_id}")
    def delete_memory(item_id: int, user: str = Depends(consented)):
        if not store_for(user).delete_item(item_id):
            raise HTTPException(status_code=404, detail="No such memory.")
        return {"deleted": item_id}

    @app.post("/api/ask")
    def ask(body: Question, user: str = Depends(consented)):
        try:
            answer = Assistant(store_for(user), model).ask(body.question.strip(), today())
        except LLMError:
            raise HTTPException(status_code=503, detail="The language model is not running. Start llama-server and try again.")
        return {"answer": answer.text, "facts": answer.facts, "duration_ms": answer.duration_ms, "checked": answer.guard,
                "unconfirmed": answer.unconfirmed, "label": answer.label, "route": answer.route}

    @app.post("/api/delete-all")
    def delete_all(user: str = Depends(consented)):
        with stores_lock:
            counts = store_for(user).delete_all()
            stores.pop(user, None)
        return {"deleted": counts}

    @app.post("/api/delete-account")
    def delete_account(body: Confirmation, request: Request, response: Response, user: str = Depends(logged_in)):
        """Removes all of the user's data and their login. Needs the password again."""
        try:
            if accounts.verify(user, body.password) != user:
                raise HTTPException(status_code=401, detail="Wrong password; nothing was deleted.")
        except Locked:
            raise HTTPException(status_code=429, detail="Too many wrong passwords. Try again in 15 minutes.")
        with stores_lock:
            counts = store_for(user).delete_all()
            stores.pop(user, None)
        accounts.remove(user)
        sessions.end_all(user)
        response.delete_cookie(COOKIE, path="/")
        return {"deleted": counts, "account_removed": True}

    return app


def build_default_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    # S5: no start without the data key, with a wrong key, or in the middle of an interrupted key rotation
    if not settings.data_key:
        raise SystemExit("CHIRII_DATA_KEY is not set. Create it with: python -m chirii.app.keys create")
    if read_env(ENV_FILE).get("CHIRII_DATA_KEY_PREVIOUS"):
        raise SystemExit("A key rotation was interrupted. Finish it with: python -m chirii.app.keys rotate")
    check_key(settings.data_dir, settings.data_key)
    log_event("app_start", stage="slice")
    return create_app(
        data_dir=settings.data_dir, secret=settings.secret_key,
        embedder=FastEmbedder(settings.embed_model, settings.embed_cache_dir),
        model=LLMClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout_seconds),
        data_key=settings.data_key,
    )

