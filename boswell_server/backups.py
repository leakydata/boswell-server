"""Incremental backups: each phone's files kept once, and a small manifest per backup.

    DATA/backups/<phone>/
      files/<sha256>                      every file of every kept backup, stored once
      boswell-backup-<date>.json          one backup: its entries in zip order (path, sha256, size, crc)
      boswell-backup-<date>.zip           a full zip, as an older phone sends it (POST /v1/backup)
      pending/<session>.json              a backup still being sent

A backup goes up in three steps: the phone sends its whole file list (start), the
server answers with the hashes it lacks, the phone sends those (blobs), and the
server writes the manifest (finish). A download is a standard backup zip put together
from the manifest and the blobs as it's sent, in the order the phone wrote it (the
phone's manifest, boswell-backup.json, first), so the phone restores it as any other.

The zip stores every entry (no compression): that way its exact size, and each
entry's size and CRC, are known before the first byte goes out, so the download has
a Content-Length and Java's ZipInputStream can read every entry. Audio is already
compressed; only the JSON is larger than in a phone-made zip.
"""
import hashlib
import json
import os
import re
import secrets
import struct
import time
import zlib
from pathlib import Path

KEEP = 7
SHA = re.compile(r"[0-9a-f]{64}")
ZIP_NAME = re.compile(r"boswell-backup-\d{4}-\d{2}-\d{2}-\d{6}\.zip")
MANIFEST_NAME = re.compile(r"boswell-backup-\d{4}-\d{2}-\d{2}-\d{6}\.json")
SESSION = re.compile(r"[0-9a-f]{32}")
DATABASES = ("speakers.db", "todo.db", "life.db", "assistant.db")
FOLDERS = ("clips", "transcripts")
FILE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._ -]{0,199}")
MAX_BLOB = 0xFFFFFFFF - 1           # each entry stays below zip64 sizes
PENDING_HOURS = 48
GRACE = 3600                        # seconds a fresh blob is safe from the garbage collector


class Refused(Exception):
    """A request that can't be done; its message says why (an HTTP 400 or 409)."""
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def valid_path(p: str) -> bool:
    """Only what a phone's backup holds (Backup.kt), and nothing that climbs out of its folder.
    keys.json is refused too: a home backup never holds API keys."""
    if p in ("boswell-backup.json", "settings.json"):
        return True
    if p.startswith("databases/"):
        return p[len("databases/"):] in DATABASES
    for d in FOLDERS:
        pre = f"files/{d}/"
        if p.startswith(pre):
            name = p[len(pre):]
            return bool(FILE_NAME.fullmatch(name)) and ".." not in name
    return False


def _entries(files) -> list:
    """The phone's file list, checked: a known path for each, a sha256, a size, no path twice,
    and the phone's manifest first."""
    if not isinstance(files, list) or not files:
        raise Refused("no files")
    out, seen = [], set()
    for f in files:
        if not isinstance(f, dict):
            raise Refused("a file isn't an object")
        path, sha, size = f.get("path"), f.get("sha256"), f.get("size")
        if not isinstance(path, str) or not valid_path(path):
            raise Refused(f"not a path a backup holds: {str(path)[:100]!r}")
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise Refused(f"bad sha256 for {path}")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_BLOB:
            raise Refused(f"bad size for {path}")
        if path in seen:
            raise Refused(f"{path} is listed twice")
        seen.add(path)
        out.append({"path": path, "sha256": sha, "size": size})
    if out[0]["path"] != "boswell-backup.json":
        raise Refused("the backup's manifest (boswell-backup.json) must come first")
    return out


def _write_json(path: Path, obj):
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(obj, separators=(",", ":")))
    part.replace(path)


def blobs_dir(d: Path) -> Path:
    return d / "files"


def start(d: Path, files) -> dict:
    """A backup begins: remember its list, and say which files the server doesn't have yet."""
    entries = _entries(files)
    store = blobs_dir(d)
    store.mkdir(parents=True, exist_ok=True)
    pending = d / "pending"
    pending.mkdir(exist_ok=True)
    for old in pending.glob("*.json"):
        if old.stat().st_mtime < time.time() - PENDING_HOURS * 3600:
            old.unlink(missing_ok=True)
    have = set(os.listdir(store))
    missing = sorted({e["sha256"] for e in entries if e["sha256"] not in have})
    session = secrets.token_hex(16)
    _write_json(pending / f"{session}.json", {"created": time.time(), "entries": entries})
    sizes = {e["sha256"]: e["size"] for e in entries}
    return {"session": session, "missing": missing, "missing_bytes": sum(sizes[s] for s in missing),
            "files": len(entries), "bytes": sum(e["size"] for e in entries)}


class BlobReader:
    """The body of POST /v1/backup/blobs, as it arrives: a run of blobs, each
        8 bytes   its length (big-endian)
        n bytes   the file
        32 bytes  its sha256, as the phone read it
    Each is written to a .part file while it's hashed, and moved into place only if the
    hash matches; a blob that doesn't is refused (and the request with it)."""

    def __init__(self, store: Path):
        self.store = store
        self.buf = bytearray()
        self.state = "length"
        self.need = 8
        self.remaining = 0
        self.file = None
        self.part = None
        self.hash = None
        self.stored = 0
        self.received = 0
        self.bytes = 0

    def feed(self, chunk: bytes):
        view = memoryview(chunk)
        while view:
            if self.state == "data":
                n = min(len(view), self.remaining)
                self.file.write(view[:n])
                self.hash.update(view[:n])
                self.remaining -= n
                self.bytes += n
                view = view[n:]
                if self.remaining == 0:
                    self.file.close()
                    self.file = None
                    self.state, self.need = "sha", 32
                continue
            take = min(len(view), self.need - len(self.buf))
            self.buf += view[:take]
            view = view[take:]
            if len(self.buf) < self.need:
                continue
            if self.state == "length":
                (self.remaining,) = struct.unpack(">Q", bytes(self.buf))
                self.buf.clear()
                if self.remaining > MAX_BLOB:
                    raise Refused("a blob is too big")
                self.part = self.store / f"upload-{secrets.token_hex(8)}.part"
                self.file = open(self.part, "wb")
                self.hash = hashlib.sha256()
                if self.remaining:
                    self.state = "data"
                else:
                    self.file.close()
                    self.file = None
                    self.state, self.need = "sha", 32
            else:   # the sha256 after the blob
                said = bytes(self.buf)
                self.buf.clear()
                got = self.hash.digest()
                self.received += 1
                if got != said:
                    self.part.unlink(missing_ok=True)
                    self.part = None
                    raise Refused(f"blob {self.received} is corrupt: its sha256 doesn't match")
                target = self.store / got.hex()
                if target.exists():
                    self.part.unlink(missing_ok=True)
                else:
                    self.part.replace(target)
                    self.stored += 1
                self.part = None
                self.state, self.need = "length", 8

    def close_quietly(self):
        """Drop anything half-sent."""
        if self.file:
            self.file.close()
            self.file = None
        if self.part:
            self.part.unlink(missing_ok=True)
            self.part = None

    def close(self):
        """The end of the body: it must end between blobs. Anything half-sent is dropped."""
        self.close_quietly()
        if self.state != "length" or self.buf:
            raise Refused("the upload ended in the middle of a blob")


def _known_crcs(d: Path) -> dict:
    """sha256 -> CRC-32, from the manifests already here, so a file's CRC is worked out once."""
    crcs = {}
    for m in d.glob("boswell-backup-*.json"):
        try:
            for e in json.loads(m.read_text())["entries"]:
                crcs[e["sha256"]] = e["crc"]
        except Exception:
            pass
    return crcs


def _crc(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            crc = zlib.crc32(b, crc)
    return crc


def finish(d: Path, session: str, changed=None, drop=None, name: str | None = None) -> dict:
    """Write the backup's manifest, once every file it lists is here. [changed] are files that
    changed between the list and their upload (their new sha256 and size), [drop] files that
    were gone by then."""
    if not isinstance(session, str) or not SESSION.fullmatch(session):
        raise Refused("no such backup session", 404)
    pending = d / "pending" / f"{session}.json"
    if not pending.is_file():
        raise Refused("no such backup session", 404)
    entries = json.loads(pending.read_text())["entries"]
    by_path = {e["path"]: e for e in entries}
    for p in drop or []:
        if p == "boswell-backup.json" or p not in by_path:
            raise Refused(f"can't drop {str(p)[:100]!r}")
        del by_path[p]
    for c in _entries([{"path": "boswell-backup.json", "sha256": "0" * 64, "size": 0}] + list(changed or []))[1:]:
        if c["path"] not in by_path:
            raise Refused(f"{c['path']} isn't in this backup")
        by_path[c["path"]].update(sha256=c["sha256"], size=c["size"])
    entries = [e for e in entries if e["path"] in by_path]
    store = blobs_dir(d)
    absent = [e["path"] for e in entries
              if not (f := store / e["sha256"]).is_file() or f.stat().st_size != e["size"]]
    if absent:
        raise Refused(f"{len(absent)} files haven't been sent, such as {absent[0]}", 409)
    crcs = _known_crcs(d)
    for e in entries:
        e["crc"] = crcs.get(e["sha256"])
        if e["crc"] is None:
            e["crc"] = crcs[e["sha256"]] = _crc(store / e["sha256"])
    created = time.time()
    name = name or time.strftime("boswell-backup-%Y-%m-%d-%H%M%S.json")
    total = zip_size(entries, created)
    _write_json(d / name, {"format": 1, "created": created, "bytes": total, "entries": entries})
    pending.unlink(missing_ok=True)
    keep(d)
    return {"name": name.removesuffix(".json") + ".zip", "bytes": total, "files": len(entries)}


def listing(d: Path) -> list:
    """This phone's backups, newest first: full zips and manifests alike, each named as the zip
    a download gives. Each item: (name, file, created, bytes)."""
    if not d.is_dir():
        return []
    found = {}
    for f in d.glob("boswell-backup-*.json"):
        if MANIFEST_NAME.fullmatch(f.name):
            try:
                m = json.loads(f.read_text())
                found[f.stem + ".zip"] = (f, m["created"], m["bytes"])
            except Exception:
                continue
    for f in d.glob("boswell-backup-*.zip"):    # a full zip, if both have the same name
        if ZIP_NAME.fullmatch(f.name):
            st = f.stat()
            found[f.name] = (f, st.st_mtime, st.st_size)
    return [(n, *found[n]) for n in sorted(found, reverse=True)]


def keep(d: Path, n: int = KEEP):
    """The newest [n] backups stay; then any blob no kept backup (or one being sent) lists goes."""
    for _, f, _, _ in listing(d)[n:]:
        f.unlink(missing_ok=True)
    gc(d)


def gc(d: Path):
    store = blobs_dir(d)
    if not store.is_dir():
        return
    used = set()
    sources = [f for f in d.glob("boswell-backup-*.json") if MANIFEST_NAME.fullmatch(f.name)] + list((d / "pending").glob("*.json"))
    for m in sources:
        try:
            used.update(e["sha256"] for e in json.loads(m.read_text())["entries"])
        except FileNotFoundError:
            continue
        except Exception:
            return      # a manifest that can't be read: delete nothing rather than something it needs
    # Nothing from the last GRACE seconds: a blob just sent may be for a backup still going
    # (a file that changed while it was listed isn't in that backup's list yet).
    recent = time.time() - GRACE
    for f in store.iterdir():
        if f.name not in used and f.stat().st_mtime < recent:
            f.unlink(missing_ok=True)     # listed only by backups now gone, or an upload cut off


# --- the zip, put together as it's sent ------------------------------------------------------

ZIP64_COUNT = 0xFFFF            # more entries than this, or
ZIP64_OFFSET = 0xFFFFFFFF       # an offset this far, needs the zip64 records


def _dos_time(t: float) -> tuple[int, int]:
    lt = time.localtime(t)
    return (lt.tm_hour << 11 | lt.tm_min << 5 | lt.tm_sec // 2,
            max(lt.tm_year - 1980, 0) << 9 | lt.tm_mon << 5 | lt.tm_mday)


def _zip_parts(entries: list, created: float):
    """The zip, in order: bytes to send as they are, or (sha256, size) for a blob's content."""
    tm, dt = _dos_time(created)
    central = []
    offset = 0
    for e in entries:
        name = e["path"].encode()
        local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0x800, 0, tm, dt,
                            e["crc"], e["size"], e["size"], len(name), 0) + name
        yield local
        yield (e["sha256"], e["size"])
        if offset >= ZIP64_OFFSET:
            extra, field, need = struct.pack("<HHQ", 1, 8, offset), 0xFFFFFFFF, 45
        else:
            extra, field, need = b"", offset, 20
        central.append(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 45, need, 0x800, 0, tm, dt,
                                   e["crc"], e["size"], e["size"], len(name), len(extra), 0, 0, 0, 0, field)
                       + name + extra)
        offset += len(local) + e["size"]
    cd_start = offset
    cd = b"".join(central)
    yield cd
    count, cd_size, end = len(entries), len(cd), cd_start + len(cd)
    if count >= ZIP64_COUNT or cd_start >= ZIP64_OFFSET or cd_size >= ZIP64_OFFSET:
        yield struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, count, count, cd_size, cd_start)
        yield struct.pack("<IIQI", 0x07064B50, 0, end, 1)
    yield struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, min(count, 0xFFFF), min(count, 0xFFFF),
                      min(cd_size, 0xFFFFFFFF), min(cd_start, 0xFFFFFFFF), 0)


def zip_size(entries: list, created: float) -> int:
    return sum(p[1] if isinstance(p, tuple) else len(p) for p in _zip_parts(entries, created))


def manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def zip_stream(d: Path, m: dict, chunk: int = 1 << 20):
    """The backup zip for manifest [m], a piece at a time (never all in memory)."""
    store = blobs_dir(d)
    for p in _zip_parts(m["entries"], m["created"]):
        if isinstance(p, bytes):
            yield p
            continue
        sha, size = p
        sent = 0
        with open(store / sha, "rb") as f:
            while b := f.read(min(chunk, size - sent)):
                sent += len(b)
                yield b
        if sent != size:
            raise OSError(f"blob {sha} is shorter than its manifest says")
