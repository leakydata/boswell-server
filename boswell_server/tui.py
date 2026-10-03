"""The terminal screen: is it up, what's it doing, who's paired, and the pairing code."""
import threading
import time

import qrcode
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from . import __version__, api, auth
from .config import PORT, base_url, tailscale_name


def qr_text(payload: str) -> str:
    """A QR code in half-block characters: two rows of modules per line of text."""
    q = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(payload)
    q.make(fit=True)
    m = q.get_matrix()
    lines = []
    for y in range(0, len(m), 2):
        top, bottom = m[y], m[y + 1] if y + 1 < len(m) else [False] * len(m[y])
        # Dark modules print as spaces on a light background, for phone cameras.
        lines.append("".join({(True, True): " ", (True, False): "▄", (False, True): "▀", (False, False): "█"}[(t, b)]
                             for t, b in zip(top, bottom)))
    return "\n".join(lines)


class PairScreen(ModalScreen):
    BINDINGS = [Binding("escape,p,q", "dismiss", "Close")]

    def __init__(self):
        super().__init__()
        self.code = auth.new_code()
        self.url = base_url()

    def compose(self) -> ComposeResult:
        import json
        payload = json.dumps({"boswell": 1, "server": self.url, "code": self.code})
        with Vertical(id="pair"):
            yield Static("[b]Pair a phone[/b]\n\nIn Boswell Phone: Device → Home server → Pair, and scan this.\n"
                         f"Or type the address and code by hand.\n\n[b]{self.url}[/b]    code [b]{self.code}[/b]\n"
                         "(works once, for 10 minutes)\n")
            yield Static(qr_text(payload), id="qr")


class ServerApp(App):
    TITLE = "Boswell Server"
    CSS = """
    #top { height: 9; }
    #status { width: 1fr; border: round $accent; padding: 0 1; }
    #phones { width: 1fr; border: round $accent; }
    #jobs { height: 1fr; border: round $accent; }
    #log { height: 12; border: round $accent; }
    PairScreen { align: center middle; }
    #pair { width: auto; height: auto; background: $surface; border: thick $accent; padding: 1 2; }
    #qr { color: black; background: white; width: auto; }
    """
    BINDINGS = [Binding("p", "pair", "Pair a phone"), Binding("f", "forget", "Forget selected phone"), Binding("q", "quit", "Quit")]

    def __init__(self, host: str = "0.0.0.0"):
        super().__init__()
        self.host = host
        self.ready = "loading models…"
        self.seen_log = 0
        self.seen_jobs = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top"):
            yield Static(id="status")
            yield DataTable(id="phones", cursor_type="row")
        yield DataTable(id="jobs")
        yield RichLog(id="log", wrap=True, markup=True)
        yield Footer()

    def on_mount(self):
        self.sub_title = f"v{__version__} · {base_url()}"
        self.query_one("#phones", DataTable).add_columns("Phone", "Paired", "Last seen")
        self.query_one("#jobs", DataTable).add_columns("Time", "Phone", "Recording", "Audio", "Took", "Words", "Speakers")
        threading.Thread(target=self._serve, daemon=True).start()
        threading.Thread(target=self._warm, daemon=True).start()
        self.set_interval(2, self.refresh_view)
        self.refresh_view()

    def _serve(self):
        import uvicorn
        uvicorn.Server(uvicorn.Config(api.app, host=self.host, port=PORT, log_level="warning")).run()

    def _warm(self):
        t = time.time()
        try:
            api.engine.warm(("wespeaker-resnet34-lm", "redimnet2-b6-vb2vox2-lm"))
            self.ready = f"ready (models loaded in {time.time() - t:.0f} s)"
        except Exception as e:
            self.ready = f"[red]models failed: {e}[/red]"
        api.note(self.ready)

    def _gpu(self) -> str:
        try:
            import pynvml
            pynvml.nvmlInit()
            best = None
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                if best is None or mem.total > best[1].total:
                    best = (h, mem)
            h, mem = best
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            name = pynvml.nvmlDeviceGetName(h)
            return f"{name} · {util}% busy · {mem.used / 2**30:.1f} of {mem.total / 2**30:.0f} GB"
        except Exception:
            return "no NVIDIA GPU found"

    def refresh_view(self):
        jobs = list(api.recent)
        last = jobs[-20:]
        avg = sum(j["ms"] for j in last) / len(last) if last else 0
        ts = tailscale_name()
        self.query_one("#status", Static).update(
            f"[b]Models[/b]   {self.ready}\n[b]GPU[/b]      {self._gpu()}\n"
            f"[b]Address[/b]  {base_url()}" + ("" if ts else "  [yellow](Tailscale is off: only this network)[/yellow]") + "\n"
            f"[b]Today[/b]    {sum(1 for j in jobs if time.time() - j['at'] < 86400)} recordings"
            + (f", {avg:.0f} ms each lately" if last else "") + "\n\n[dim]p pair a phone · f forget one · q quit[/dim]")
        phones = self.query_one("#phones", DataTable)
        phones.clear()
        for p in auth.phones():
            fmt = lambda t: time.strftime("%b %d %H:%M", time.localtime(t)) if t else "never"
            phones.add_row(p["device"], fmt(p["paired"]), fmt(p["last"]), key=p["hash"])
        table = self.query_one("#jobs", DataTable)
        for j in jobs[self.seen_jobs:]:
            table.add_row(time.strftime("%H:%M:%S", time.localtime(j["at"])), j["phone"], j["clip"] or "-",
                          f"{j['seconds']} s", f"{j['ms']} ms", str(j["words"]), str(j["speakers"]))
        self.seen_jobs = len(jobs)
        log = self.query_one("#log", RichLog)
        entries = list(api.log)
        for t, msg in entries[self.seen_log:]:
            log.write(f"[dim]{time.strftime('%H:%M:%S', time.localtime(t))}[/dim] {msg}")
        self.seen_log = len(entries)

    def action_pair(self):
        self.push_screen(PairScreen())

    def action_forget(self):
        phones = self.query_one("#phones", DataTable)
        if phones.row_count == 0:
            return
        key = phones.coordinate_to_cell_key(phones.cursor_coordinate).row_key.value
        auth.forget(key)
        api.note(f"forgot {key}")
        self.refresh_view()
