"""The local-AI proxy's parts that don't need Ollama: what it forwards, and who it refuses."""
import importlib

import pytest


@pytest.fixture
def llm(tmp_path, monkeypatch):
    monkeypatch.setenv("BOSWELL_SERVER_DATA", str(tmp_path))
    import boswell_server.config, boswell_server.auth, boswell_server.llm
    importlib.reload(boswell_server.config)
    importlib.reload(boswell_server.auth)
    return importlib.reload(boswell_server.llm)


def test_the_phones_request_is_made_ollamas(llm):
    phone = {"model": "home", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}],
             "max_tokens": 400, "temperature": 0.2, "usage": {"include": True},
             "reasoning": {"effort": "low", "exclude": True}, "plugins": [{"id": "web"}]}
    out = llm.request_for_ollama(phone)
    assert out["model"] == llm.MODEL and out["stream"] is False
    assert out["reasoning_effort"] == llm.THINK
    assert out["tools"] == phone["tools"] and out["max_tokens"] == 400
    assert not {"usage", "reasoning", "plugins"} & set(out)


def test_only_paired_phones_may_ask(llm):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(llm.router)
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    assert c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    token = llm.auth.pair(llm.auth.new_code(), "Pixel")
    assert c.post("/v1/chat/completions", content=b"nope", headers={"Authorization": f"Bearer {token}"}).status_code == 400
