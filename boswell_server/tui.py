"""The terminal screen: is it up, what's it doing, who's paired, and the pairing code."""
import threading
import time

import qrcode
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from . import __version__, api, auth, local
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
    BINDINGS = [Binding("p", "pair", "Pair a phone"), Binding("f", "forget", "Forget selected phone"), Binding("c", "clear", "Clear the list"), Binding("q", "quit", "Quit")]

    def __init__(self, host: str = "0.0.0.0", attach: bool = False):
        super().__init__()
        self.host = host
        # Attached: the server runs elsewhere (the service); this only shows it, through
        # /v1/local/status. Pairing and forgetting work as usual: they're files.
        self.attach = attach
        self.remote: dict | None = None
        # By time, not count: the server keeps only the latest jobs and log lines, so a
        # count stops moving once it's full and new ones would never show.
        self.shown_job = 0.0
        self.shown_log = 0.0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top"):
            yield Static(id="status")
            yield DataTable(id="phones", cursor_type="row")
        yield DataTable(id="jobs")
        yield RichLog(id="log", wrap=True, markup=True)
        yield Footer()

    def on_mount(self):
        self.sub_title = f"v{__version__} · {base_url()}" + (" · showing the running server" if self.attach else "")
        self.query_one("#phones", DataTable).add_columns("Phone", "Paired", "Last seen")
        self.query_one("#jobs", DataTable).add_columns("Time", "Phone", "Recording", "Audio", "Took", "Words", "Speakers")
        if self.attach:
            self.remote = local.status()
            threading.Thread(target=self._follow, daemon=True).start()
        else:
            threading.Thread(target=self._serve, daemon=True).start()
            threading.Thread(target=api.warm, daemon=True).start()
        self.set_interval(2, self.refresh_view)
        self.refresh_view()

    def _serve(self):
        import uvicorn
        uvicorn.Server(uvicorn.Config(api.app, host=self.host, port=PORT, log_level="warning")).run()

    def _follow(self):
        """Attached: the running server's status, every two seconds (off the screen's thread)."""
        while True:
            time.sleep(2)
            self.remote = local.status()

    def _source(self) -> tuple[str, str, list, list]:
        """Models, GPU, jobs and log lines: this process's own, or the running server's."""
        if not self.attach:
            return api.state["ready"], api.gpu(), list(api.recent), list(api.log)
        r = self.remote
        if r is None:
            return f"[red]the server on port {PORT} isn't answering[/red]", api.gpu(), [], []
        if r.get("old"):
            return "running (restart it to see its recordings and log here)", api.gpu(), [], []
        return r["ready"], r["gpu"], r["recent"], [tuple(e) for e in r["log"]]

    def refresh_view(self):
        ready, gpu, jobs, entries = self._source()
        last = jobs[-20:]
        avg = sum(j["ms"] for j in last) / len(last) if last else 0
        ts = tailscale_name()
        quit_note = "q quit (the server keeps running)" if self.attach else "q quit"
        self.query_one("#status", Static).update(
            f"[b]Models[/b]   {ready}\n[b]GPU[/b]      {gpu}\n"
            f"[b]Address[/b]  {base_url()}" + ("" if ts else "  [yellow](Tailscale is off: only this network)[/yellow]") + "\n"
            f"[b]Today[/b]    {sum(1 for j in jobs if time.time() - j['at'] < 86400)} recordings"
            + (f", {avg:.0f} ms each lately" if last else "")
            + ("\n[b]Server[/b]   in the background (the service); this screen only shows it" if self.attach else "\n")
            + f"\n[dim]p pair a phone · f forget one · c clear the list · {quit_note}[/dim]")
        phones = self.query_one("#phones", DataTable)
        phones.clear()
        for p in auth.phones():
            fmt = lambda t: time.strftime("%b %d %H:%M", time.localtime(t)) if t else "never"
            phones.add_row(p["device"], fmt(p["paired"]), fmt(p["last"]), key=p["hash"])
        table = self.query_one("#jobs", DataTable)
        for j in jobs:
            if j["at"] > self.shown_job:
                table.add_row(time.strftime("%H:%M:%S", time.localtime(j["at"])), j["phone"], j["clip"] or "-",
                              f"{j['seconds']} s", f"{j['ms']} ms", str(j["words"]), str(j["speakers"]))
                self.shown_job = j["at"]
        log = self.query_one("#log", RichLog)
        for t, msg in entries:
            if t > self.shown_log:
                log.write(f"[dim]{time.strftime('%H:%M:%S', time.localtime(t))}[/dim] {msg}")
                self.shown_log = t

    def action_pair(self):
        self.push_screen(PairScreen())

    def action_clear(self):
        """Empty the recordings list and the log on screen; today's count and timings stay."""
        self.query_one("#jobs", DataTable).clear()
        self.query_one("#log", RichLog).clear()

    def action_forget(self):
        phones = self.query_one("#phones", DataTable)
        if phones.row_count == 0:
            return
        key = phones.coordinate_to_cell_key(phones.cursor_coordinate).row_key.value
        name = next((p["device"] for p in auth.phones() if p["hash"] == key), "a phone")
        auth.forget(key)
        if self.attach:   # the running server's log is its own; say it here
            self.query_one("#log", RichLog).write(f"[dim]{time.strftime('%H:%M:%S')}[/dim] forgot {name}")
        else:
            api.note(f"forgot {name}")
        self.refresh_view()
