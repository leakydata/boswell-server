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


def test_snr_is_the_voice_over_the_recordings_noise_floor():
    import numpy as np
    from boswell_server.pipeline import snr_db
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 10) * 0.001).astype(np.float32)         # a quiet room
    audio[16000 * 2:16000 * 4] += (0.1 * np.sin(np.arange(32000) * 0.2)).astype(np.float32)   # someone close, 2-4 s
    near = snr_db(audio, [(2.0, 4.0)])
    assert 30 < near < 40                     # 0.1 sine (power 0.005) over 1e-6 noise: 37 dB
    assert abs(snr_db(audio, [(6.0, 8.0)])) < 2           # the room itself: about 0 dB
    assert snr_db(audio, [(2.0, 2.01)]) > 20              # shorter than a frame still counts
    # Snr.kt's numbers for this recording, so the phone and the server agree.
    assert round(near, 2) == round(snr_db(audio.astype(np.float64), [(2.0, 4.0)]), 2)


def test_backups_are_kept_per_phone_and_only_seen_by_it(fresh, monkeypatch):
    from fastapi.testclient import TestClient
    from boswell_server import api
    import itertools
    names = (f"boswell-backup-2026-10-{d:02d}-120000.zip" for d in itertools.count(1))
    monkeypatch.setattr(api.time, "strftime", lambda fmt: next(names))
    pixel, other = (fresh.pair(fresh.new_code(), d) for d in ("Google Pixel 8", "../Other phone"))
    web = TestClient(api.app)
    key = lambda t: {"Authorization": f"Bearer {t}"}
    zip_ = b"PK\x03\x04" + b"x" * 100_000
    assert web.post("/v1/backup", content=zip_).status_code == 401
    assert web.post("/v1/backup", content=b"", headers=key(pixel)).status_code == 400
    assert web.post("/v1/backup", content=b"not a zip", headers=key(pixel)).status_code == 400
    for _ in range(9):
        r = web.post("/v1/backup", content=iter([zip_[:10], zip_[10:]]), headers=key(pixel))   # streamed, in pieces
        assert r.status_code == 200 and r.json()["bytes"] == len(zip_)
    folder = fresh.TOKENS.parent / "backups" / "Google-Pixel-8"
    # The refused two used up a name each: 3 to 11 arrived, and the newest 7 are kept, with no .part left.
    assert sorted(f.name for f in folder.iterdir())[0] == "boswell-backup-2026-10-05-120000.zip" and len(list(folder.iterdir())) == 7
    listed = web.get("/v1/backups", headers=key(pixel)).json()["backups"]
    assert [b["name"] for b in listed][:2] == ["boswell-backup-2026-10-11-120000.zip", "boswell-backup-2026-10-10-120000.zip"]
    assert len(listed) == 7 and listed[0]["bytes"] == len(zip_) and listed[0]["created"] > 0
    got = web.get(f"/v1/backups/{listed[0]['name']}", headers=key(pixel))
    assert got.status_code == 200 and got.content == zip_
    # Another phone sees only its own, and no name reaches outside the folder.
    assert web.get("/v1/backups", headers=key(other)).json()["backups"] == []
    assert web.get(f"/v1/backups/{listed[0]['name']}", headers=key(other)).status_code == 404
    for bad in ("..%2F..%2Fphones.json", "boswell-backup-2026-10-11-120000.zip.part", "phones.json"):
        assert web.get(f"/v1/backups/{bad}", headers=key(pixel)).status_code == 404, bad
    assert web.post("/v1/backup", content=zip_, headers=key(other)).status_code == 200
    assert (folder.parent / "Other-phone").is_dir()                        # its name can't climb out either


# --- incremental backups ---------------------------------------------------------------------

def _phone_files(n=6, seed=0):
    """What a phone's backup holds, in the order Backup.export writes it."""
    import random
    rng = random.Random(seed)
    files = {"boswell-backup.json": b'{"format":1,"app":"1.0","created":1790000000,"recordings":%d,"keys":false}' % n,
             "settings.json": b'{"name":{"t":"s","v":"Dan"}}',
             "databases/speakers.db": rng.randbytes(50_000), "databases/todo.db": rng.randbytes(8_000)}
    for i in range(n):
        files[f"files/clips/omi_{1790000000 + i}.ogg"] = rng.randbytes(20_000 + i)
        files[f"files/clips/omi_{1790000000 + i}.json"] = b'{"clip":%d}' % i
    for i in range(n):
        files[f"files/transcripts/omi_{1790000000 + i}.json"] = b'{"words":["hello %d"]}' % i
    return files


def _full_zip(files: dict) -> bytes:
    """A zip as the phone's Backup.export makes it: what POST /v1/backup receives."""
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, data in files.items():
            z.writestr(path, data, zipfile.ZIP_STORED if path.endswith(".ogg") else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def _frames(blobs, corrupt=False) -> bytes:
    import hashlib, struct
    out = b""
    for data in blobs:
        sha = hashlib.sha256(data).digest()
        if corrupt:
            data = data[:-1] + bytes([data[-1] ^ 1])
        out += struct.pack(">Q", len(data)) + data + sha
    return out


def _incremental(web, headers, files: dict, chunks=1):
    """The phone's side: list, send what's missing, finish. Returns (start answer, blobs answer, finish answer)."""
    import hashlib
    listed = [{"path": p, "size": len(d), "mtime": 0, "sha256": hashlib.sha256(d).hexdigest()} for p, d in files.items()]
    r = web.post("/v1/backup/start", json={"files": listed}, headers=headers)
    assert r.status_code == 200, r.text
    started = r.json()
    by_sha = {hashlib.sha256(d).hexdigest(): d for d in files.values()}
    body = _frames([by_sha[s] for s in started["missing"]])
    pieces = [body[i:i + max(1, len(body) // chunks + 1)] for i in range(0, len(body), max(1, len(body) // chunks + 1))]
    sent = web.post("/v1/backup/blobs", content=iter(pieces or [b""]), headers=headers)
    assert sent.status_code == 200, sent.text
    done = web.post("/v1/backup/finish", json={"session": started["session"]}, headers=headers)
    assert done.status_code == 200, done.text
    return started, sent.json(), done.json()


@pytest.fixture
def phone(fresh, monkeypatch):
    from fastapi.testclient import TestClient
    from boswell_server import api, backups
    import itertools
    names = (f"boswell-backup-2026-10-{d:02d}-120000.zip" for d in itertools.count(1))
    monkeypatch.setattr(api.time, "strftime", lambda fmt: next(names))
    monkeypatch.setattr(backups, "GRACE", 0)
    token = fresh.pair(fresh.new_code(), "Google Pixel 10")
    return TestClient(api.app), {"Authorization": f"Bearer {token}"}, fresh.TOKENS.parent / "backups" / "Google-Pixel-10"


def _entries(data: bytes) -> list:
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        return [(i.filename, z.read(i)) for i in z.infolist()]


def test_an_incremental_backup_downloads_as_the_full_zip(phone):
    web, key, folder = phone
    files = _phone_files()
    started, sent, done = _incremental(web, key, files, chunks=7)     # streamed, cut anywhere
    assert len(started["missing"]) == len(set(files.values())) and sent["stored"] == len(started["missing"])
    assert done["name"] == "boswell-backup-2026-10-01-120000.zip" and done["files"] == len(files)
    listed = web.get("/v1/backups", headers=key).json()["backups"]
    assert [b["name"] for b in listed] == [done["name"]] and listed[0]["bytes"] == done["bytes"] and listed[0]["created"] > 0
    got = web.get(f"/v1/backups/{done['name']}", headers=key)
    assert got.status_code == 200 and int(got.headers["content-length"]) == len(got.content) == done["bytes"]
    # The same entries, in the same order, with the same bytes as the phone's own zip: the manifest first.
    assert _entries(got.content) == _entries(_full_zip(files))
    # Every entry is stored with its sizes and CRC up front (no data descriptor), as Java's ZipInputStream needs.
    import struct
    assert got.content[:4] == b"PK\x03\x04" and struct.unpack("<H", got.content[6:8])[0] & 0x08 == 0
    assert (folder / "files").is_dir() and not list((folder / "pending").iterdir())


def test_the_next_backup_sends_only_what_changed(phone):
    import hashlib
    web, key, folder = phone
    files = _phone_files()
    _incremental(web, key, files)
    files["files/transcripts/omi_1790000003.json"] = b'{"words":["rechecked"]}'
    started, sent, done = _incremental(web, key, files)
    assert started["missing"] == [hashlib.sha256(b'{"words":["rechecked"]}').hexdigest()] and sent["stored"] == 1
    assert started["missing_bytes"] == len(b'{"words":["rechecked"]}')
    names = [b["name"] for b in web.get("/v1/backups", headers=key).json()["backups"]]
    assert names == ["boswell-backup-2026-10-02-120000.zip", "boswell-backup-2026-10-01-120000.zip"]
    assert _entries(web.get(f"/v1/backups/{names[0]}", headers=key).content) == _entries(_full_zip(files))
    # The older one still restores as it was.
    old = dict(_entries(web.get(f"/v1/backups/{names[1]}", headers=key).content))
    assert old["files/transcripts/omi_1790000003.json"] == b'{"words":["hello 3"]}'


def test_a_corrupt_blob_is_refused(phone):
    import hashlib
    web, key, folder = phone
    files = _phone_files()
    listed = [{"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in files.items()]
    started = web.post("/v1/backup/start", json={"files": listed}, headers=key).json()
    ogg = files["files/clips/omi_1790000000.ogg"]
    r = web.post("/v1/backup/blobs", content=_frames([ogg], corrupt=True), headers=key)
    assert r.status_code == 400 and "corrupt" in r.text
    assert not [f for f in (folder / "files").iterdir()]                  # nothing kept, no .part left
    # Cut off halfway: refused, and nothing kept either.
    assert web.post("/v1/backup/blobs", content=_frames([ogg])[:5000], headers=key).status_code == 400
    assert not list((folder / "files").iterdir())
    # A manifest whose files haven't all come is refused.
    r = web.post("/v1/backup/finish", json={"session": started["session"]}, headers=key)
    assert r.status_code == 409 and "haven't been sent" in r.text
    assert web.get("/v1/backups", headers=key).json()["backups"] == []
    assert web.post("/v1/backup/finish", json={"session": "../../phones"}, headers=key).status_code == 404


def test_files_that_changed_or_went_while_sending(phone):
    """Recording goes on during a backup: a file can change or vanish between the list and its upload."""
    import hashlib
    web, key, folder = phone
    files = _phone_files()
    listed = [{"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in files.items()]
    started = web.post("/v1/backup/start", json={"files": listed}, headers=key).json()
    gone, edited = "files/clips/omi_1790000001.ogg", "files/transcripts/omi_1790000002.json"
    new = b'{"words":["snr filled in"]}'
    sha = {hashlib.sha256(d).hexdigest(): d for p, d in files.items() if p not in (gone, edited)}
    assert web.post("/v1/backup/blobs", content=_frames(list(sha.values()) + [new]), headers=key).status_code == 200
    r = web.post("/v1/backup/finish", headers=key, json={
        "session": started["session"], "drop": [gone],
        "changed": [{"path": edited, "size": len(new), "sha256": hashlib.sha256(new).hexdigest()}]})
    assert r.status_code == 200, r.text
    del files[gone]; files[edited] = new
    assert _entries(web.get(f"/v1/backups/{r.json()['name']}", headers=key).content) == _entries(_full_zip(files))


def test_paths_are_checked(phone):
    web, key, folder = phone
    ok = {"path": "boswell-backup.json", "size": 1, "sha256": "a" * 64}
    for bad in ("files/clips/../../phones.json", "../phones.json", "/etc/passwd", "files/clips/a/b.ogg", "files/clips/..",
                "files/clips/", "files/other/x.json", "databases/../x.db", "databases/evil.db", "keys.json",
                "files/clips/x\\..\\y", "files/clips/.hidden"):
        r = web.post("/v1/backup/start", json={"files": [ok, {"path": bad, "size": 1, "sha256": "b" * 64}]}, headers=key)
        assert r.status_code == 400, bad
    for bad in ({"path": "files/clips/a.ogg", "size": 1, "sha256": "../../x"}, {"path": "files/clips/a.ogg", "size": -1, "sha256": "b" * 64}):
        assert web.post("/v1/backup/start", json={"files": [ok, bad]}, headers=key).status_code == 400
    # The phone's manifest comes first, and nothing is listed twice.
    assert web.post("/v1/backup/start", json={"files": [{**ok, "path": "settings.json"}, ok]}, headers=key).status_code == 400
    assert web.post("/v1/backup/start", json={"files": [ok, ok]}, headers=key).status_code == 400
    assert web.post("/v1/backup/start", json={"files": [ok]}).status_code == 401
    assert not (folder / "pending").exists() or not list((folder / "pending").iterdir())


def test_the_newest_7_are_kept_and_unused_files_let_go(phone):
    import hashlib
    web, key, folder = phone
    files = _phone_files(n=2)
    shas = []
    for day in range(9):
        files[f"files/transcripts/omi_{1790000100 + day}.json"] = b'{"day":%d}' % day     # a new recording each day
        files["databases/todo.db"] = b"todo on day %d" % day                              # and the database changes
        shas.append(hashlib.sha256(b"todo on day %d" % day).hexdigest())
        _incremental(web, key, files)
    listed = [b["name"] for b in web.get("/v1/backups", headers=key).json()["backups"]]
    assert len(listed) == 7 and listed[0] == "boswell-backup-2026-10-09-120000.zip" and listed[-1] == "boswell-backup-2026-10-03-120000.zip"
    assert len(list(folder.glob("boswell-backup-*.json"))) == 7
    blobs = {f.name for f in (folder / "files").iterdir()}
    assert shas[0] not in blobs and shas[1] not in blobs and set(shas[2:]) <= blobs        # the first two days' databases are gone
    assert hashlib.sha256(b'{"day":0}').hexdigest() in blobs                                # a recording still listed stays
    # The oldest kept day still downloads whole.
    oldest = dict(_entries(web.get(f"/v1/backups/{listed[-1]}", headers=key).content))
    assert oldest["databases/todo.db"] == b"todo on day 2" and "files/transcripts/omi_1790000108.json" not in oldest


def test_old_full_zips_and_incremental_backups_live_together(phone):
    web, key, folder = phone
    files = _phone_files()
    whole = _full_zip(files)
    r = web.post("/v1/backup", content=whole, headers=key)                 # an older phone, or before updating
    assert r.status_code == 200 and r.json()["name"] == "boswell-backup-2026-10-01-120000.zip"
    _incremental(web, key, files)
    listed = web.get("/v1/backups", headers=key).json()["backups"]
    assert [b["name"] for b in listed] == ["boswell-backup-2026-10-02-120000.zip", "boswell-backup-2026-10-01-120000.zip"]
    assert web.get("/v1/backups/boswell-backup-2026-10-01-120000.zip", headers=key).content == whole
    assert _entries(web.get("/v1/backups/boswell-backup-2026-10-02-120000.zip", headers=key).content) == _entries(whole)
    # Full zips age out with the rest.
    for _ in range(7):
        _incremental(web, key, files)
    assert not list(folder.glob("*.zip")) and len(list(folder.glob("boswell-backup-*.json"))) == 7
    for bad in ("boswell-backup-2026-10-01-120000.json", "..%2Fphones.json", "files"):
        assert web.get(f"/v1/backups/{bad}", headers=key).status_code == 404, bad


def test_zip64_when_the_backup_outgrows_plain_zip(phone, monkeypatch):
    """Past 65,535 entries or 4 GB, a zip needs zip64 records: tried here with tiny limits."""
    from boswell_server import backups
    monkeypatch.setattr(backups, "ZIP64_COUNT", 5)
    monkeypatch.setattr(backups, "ZIP64_OFFSET", 30_000)
    web, key, folder = phone
    files = _phone_files()
    _, _, done = _incremental(web, key, files)
    got = web.get(f"/v1/backups/{done['name']}", headers=key).content
    assert b"PK\x06\x06" in got and b"PK\x06\x07" in got and len(got) == done["bytes"]
    assert _entries(got) == _entries(_full_zip(files))


def test_health_says_incremental_backups_work():
    import inspect
    from boswell_server import api
    assert '"backup": 2' in inspect.getsource(api.health)
