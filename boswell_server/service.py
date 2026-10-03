"""Running `boswell-server serve` in the background: a systemd user service that starts at boot."""
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

UNIT = "boswell-server"
# Settings that change where the server listens or keeps things, carried into the unit if set here.
PASS_ON = ("BOSWELL_PORT", "BOSWELL_SERVER_DATA", "BOSWELL_SERVER_MODELS", "HF_HOME")


def unit_path(name: str = UNIT) -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user" / f"{name}.service"


def unit_text(host: str = "0.0.0.0", name: str = UNIT) -> str:
    """The unit: this virtual environment's `boswell-server serve`, from this checkout."""
    exe = Path(sys.prefix) / "bin" / "boswell-server"
    repo = Path(__file__).resolve().parents[1]
    workdir = repo if (repo / "pyproject.toml").exists() else Path.home()
    # CUDA and cuDNN need nothing here: PyTorch and onnxruntime find their nvidia-* wheels
    # themselves, and the sound-tagging helper sets its own LD_LIBRARY_PATH (pipeline.SpeechHelper).
    env = {"PYTHONUNBUFFERED": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1",
           "PATH": f"{exe.parent}:/usr/local/bin:/usr/bin:/bin"}   # tailscale, for the address
    env |= {k: os.environ[k] for k in PASS_ON if os.environ.get(k)}
    lines = "\n".join(f'Environment="{k}={v}"' for k, v in env.items())
    return f"""[Unit]
Description=Boswell Server (home processing for Boswell Phone)

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={exe} serve --host {host}
{lines}
Restart=on-failure
RestartSec=10
# The screen (`boswell-server`) shows what it's doing; this is for `journalctl --user -u {name}`.
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


def _run(*cmd) -> bool:
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).strip().splitlines():
        print("   ", line)
    return r.returncode == 0


def state(name: str = UNIT) -> str:
    """How the service is: "not installed", or enabled/disabled and active/inactive/failed."""
    if not unit_path(name).exists():
        return "not installed"
    q = lambda what: subprocess.run(["systemctl", "--user", what, name], capture_output=True, text=True).stdout.strip() or "unknown"
    return f"installed, {q('is-enabled')}, {q('is-active')}"


def install(host: str = "0.0.0.0", name: str = UNIT):
    from .config import PORT
    from .local import running
    path = unit_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit_text(host, name))
    print(f"wrote {path}")
    _run("systemctl", "--user", "daemon-reload")
    busy = running(PORT)
    if busy:
        # Started now, it couldn't have the port and would only keep retrying.
        _run("systemctl", "--user", "enable", name)
        who = "another Boswell Server (a terminal?)" if busy == "boswell" else "another program"
        print(f"\nPort {PORT} is in use by {who}, so the service is enabled but not started.\n"
              f"Stop that one, then:  systemctl --user start {name}")
    else:
        _run("systemctl", "--user", "enable", "--now", name)
    # Without lingering, user services start at login and stop at logout; with it, at boot.
    user = getpass.getuser()
    if not _run("loginctl", "enable-linger", user) and shutil.which("sudo"):
        _run("sudo", "-n", "loginctl", "enable-linger", user)
    print(f"\nService: {state(name)}. It starts at boot, without anyone logging in.\n"
          f"Logs: journalctl --user -u {name} -f    Screen: boswell-server (it attaches to the service)")


def uninstall(name: str = UNIT):
    path = unit_path(name)
    if not path.exists():
        print(f"{path} isn't there; nothing to do")
        return
    _run("systemctl", "--user", "disable", "--now", name)
    path.unlink()
    print(f"removed {path}")
    _run("systemctl", "--user", "daemon-reload")
    print("Lingering (user services at boot) is left on; `loginctl disable-linger` turns it off.")
