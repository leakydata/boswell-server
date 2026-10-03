"""The parts that don't need a GPU: word grouping (the phone's Words.fromTokens), pairing, the API's refusals, and the service."""
import importlib

import pytest


def test_words_follow_the_phone():
    from boswell_server.pipeline import words_from_tokens
    w = words_from_tokens([" So", " I", "'", "m", " g", "onna"], [0.5, 0.7, 0.8, 0.9, 1.0, 1.1], 3.0)
    assert [x["text"] for x in w] == ["So", "I'm", "gonna"]
    assert w[0]["start"] == 0.5 and w[0]["end"] == 0.7          # up to the next word
    assert w[-1]["end"] == 2.0                                    # capped at a second before the end


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("BOSWELL_SERVER_DATA", str(tmp_path))
    import boswell_server.config, boswell_server.auth
    importlib.reload(boswell_server.config)
    return importlib.reload(boswell_server.auth)


def test_a_code_works_once(fresh):
    code = fresh.new_code()
    token = fresh.pair(code, "Pixel")
    assert token and fresh.check(token)["device"] == "Pixel"
    assert fresh.pair(code, "Pixel") is None                     # used up
    assert fresh.check("not-a-key") is None


def test_keys_are_stored_hashed(fresh):
    token = fresh.pair(fresh.new_code(), "Pixel")
    assert token not in fresh.TOKENS.read_text()


def test_hot_words_include_the_spellings_parakeet_uses():
    from boswell_server.pipeline import boost_phrases
    p = boost_phrases(["OpenRouter", "Lindsey", "  Omi  ", ""])
    assert {"OpenRouter", "open router", "Open Router", "openrouter", "Lindsey", "lindsey", "Omi", "omi"} <= set(p)
    assert "" not in p
    assert boost_phrases(["b", "a"]) == boost_phrases(["a", "b"])   # stable: the boosting tree is rebuilt only on change



def test_local_status_is_for_this_computer_only():
    from fastapi.testclient import TestClient
    from boswell_server import api
    api.note("hello")
    here = TestClient(api.app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 5000))
    assert TestClient(api.app, base_url="http://127.0.0.1:8765", client=("100.64.0.7", 5000)).get("/v1/local/status").status_code == 403
    # `tailscale serve` connects from 127.0.0.1 too: its Host and its headers give it away.
    proxied = TestClient(api.app, base_url="https://kubuntu.tail6f2378.ts.net", client=("127.0.0.1", 5000))
    assert proxied.get("/v1/local/status").status_code == 403
    for h in ({"X-Forwarded-For": "100.64.0.7"}, {"Forwarded": "for=100.64.0.7"}, {"Tailscale-User-Login": "someone@example.com"}):
        assert here.get("/v1/local/status", headers=h).status_code == 403, h
    assert TestClient(api.app, base_url="http://localhost:8765", client=("::1", 5000)).get("/v1/local/status").status_code == 200
    r = here.get("/v1/local/status").json()
    assert r["name"] == "Boswell Server" and r["ready"] and r["gpu"]
    assert r["log"][-1][1] == "hello" and isinstance(r["recent"], list)


def test_running_tells_a_free_port_from_a_busy_one():
    import socket
    from boswell_server.local import running
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    assert running(port) is None                                  # bound, not listening: free
    s.listen(); s.settimeout(0)
    assert running(port) == "other"                               # listening, but not Boswell Server
    s.close()


def test_the_service_runs_serve_from_this_venv(monkeypatch):
    from boswell_server import service
    monkeypatch.setenv("BOSWELL_PORT", "8799")
    unit = service.unit_text()
    assert "/.venv/bin/boswell-server serve --host 0.0.0.0" in unit and "Restart=on-failure" in unit
    assert '"TORCHINDUCTOR_COMPILE_THREADS=1"' in unit and '"BOSWELL_PORT=8799"' in unit
    assert "WantedBy=default.target" in unit


def test_no_pool_of_compile_workers():
    import os, subprocess, sys
    env = {k: v for k, v in os.environ.items() if k != "TORCHINDUCTOR_COMPILE_THREADS"}
    out = subprocess.run([sys.executable, "-c", "import boswell_server, os; print(os.environ['TORCHINDUCTOR_COMPILE_THREADS'])"],
                         env=env, capture_output=True, text=True).stdout
    assert out.strip() == "1"
