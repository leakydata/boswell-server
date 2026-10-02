"""The parts that don't need a GPU: word grouping (the phone's Words.fromTokens), pairing, and the API's refusals."""
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
