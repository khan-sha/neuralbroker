"""
neuralfit_tui.py — NeuralFit Advanced full-screen TUI
======================================================
Fetches model data from the compiled hardware-scoring engine (via --json),
then renders a fully interactive Pink-Matrix terminal UI.

Features:
  • Keyboard navigation (↑↓ PgUp PgDn Home End)
  • Tab-cycle sort (score, tok/s, params, vram, context, name)
  • Live incremental search  (press /)
  • Detail pane with score breakdown (press Enter)
  • Background Ollama pull  (press o)
  • ANSI-aware column alignment — columns never drift
  • All 900+ models from the engine, no cap by default
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

# ─── colour palette ──────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
PINK    = "\033[95m"
MAGENTA = "\033[35m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
RED     = "\033[91m"
AMBER   = "\033[93m"
MATRIX  = "\033[92m"
CLEAR   = "\033[2J\033[H"
# row backgrounds
BG_SEL  = "\033[48;5;53m"    # deep purple
BG_HDR  = "\033[48;5;17m"    # dark navy

# ─── ANSI-aware string utilities ─────────────────────────────────────────────
_ANSI = re.compile(r"\033\[[0-9;]*m")

def _vlen(s: str) -> int:
    """Visible length of a string (ANSI codes have zero width)."""
    return len(_ANSI.sub("", s))

def _pad(s: str, width: int, align: str = "<") -> str:
    """Pad s to *visible* width, so columns don't drift with colour codes."""
    vis = _vlen(s)
    gap = max(0, width - vis)
    return (s + " " * gap) if align == "<" else (" " * gap + s)

def _trunc(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

# ─── score bar (pure ASCII fill) ─────────────────────────────────────────────
def _bar(pct: float, width: int = 10, color: str = PINK) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET

# ─── fit-level badge ─────────────────────────────────────────────────────────
# Maps every value llmfit actually returns → coloured label (fixed visual width=10)
_FIT = {
    "perfect":   MATRIX + BOLD + "✅ perfect " + RESET,
    "good":      MATRIX + BOLD + "✅ good    " + RESET,
    "too tight": AMBER  + BOLD + "⚠ too tight" + RESET,
    "tight":     AMBER  + BOLD + "⚠  tight  " + RESET,
    "marginal":  RED    +        "⚡ marginal " + RESET,
    "partial":   RED    +        "⚡ partial  " + RESET,
    "too_large": DIM    +        "✗ too large" + RESET,
}

def _fit_badge(fit_level: str) -> str:
    key = fit_level.strip().lower()
    return _FIT.get(key, DIM + f"{'?' + fit_level[:8]:<10}" + RESET)

# ─── column spec ─────────────────────────────────────────────────────────────
# (header_label, visible_width, data_key_or_fn, align)
COLS = [
    ("#",        4,  None,                 ">"),
    ("Model",   30,  "name",               "<"),
    ("Params",   7,  "params_b",           ">"),
    ("VRAM",     5,  "memory_required_gb", ">"),
    ("Tok/s",    5,  "estimated_tps",      ">"),
    ("Ctx",      5,  "context_length",     ">"),
    ("Score",    5,  "score",              ">"),
    ("Fit",     12,  "fit_level",          "<"),
    ("Quality", 10,  "quality_bar",        "<"),
    ("Use-Case",22,  "use_case",           "<"),
]

# ─── main TUI class ───────────────────────────────────────────────────────────
class NeuralFitTUI:
    SORTS      = ["score", "tps", "params", "mem", "ctx", "name"]
    SORT_LABEL = {
        "score":  "Score ↓",
        "tps":    "Tok/s ↓",
        "params": "Params ↓",
        "mem":    "VRAM ↓",
        "ctx":    "Context ↓",
        "name":   "Name A→Z",
    }

    def __init__(self, models: list, system: dict):
        self.all_models  = models
        self.system      = system
        self.cursor      = 0
        self.scroll      = 0
        self.sort_idx    = 0
        self.query       = ""
        self.detail_open = False
        self.pull_status: dict[str, str] = {}
        self._filtered: list = []
        self._apply()

    # ── data ──────────────────────────────────────────────────────────────────

    def _sort_key(self, m: dict):
        s = self.SORTS[self.sort_idx]
        if s == "score":  return -(m.get("score") or 0)
        if s == "tps":    return -(m.get("estimated_tps") or 0)
        if s == "params": return -(m.get("params_b") or 0)
        if s == "mem":    return -(m.get("memory_required_gb") or 0)
        if s == "ctx":    return -(m.get("context_length") or 0)
        if s == "name":   return (m.get("name") or "").lower()
        return 0

    def _apply(self):
        q = self.query.lower()
        self._filtered = [
            m for m in self.all_models
            if not q
            or q in (m.get("name") or "").lower()
            or q in (m.get("use_case") or "").lower()
            or q in (m.get("category") or "").lower()
            or q in (m.get("fit_level") or "").lower()
        ]
        self._filtered.sort(key=self._sort_key)
        n = len(self._filtered)
        self.cursor = max(0, min(self.cursor, n - 1))

    # ── terminal size ─────────────────────────────────────────────────────────

    @staticmethod
    def _sz() -> tuple[int, int]:
        cols, rows = shutil.get_terminal_size((160, 40))
        return cols, rows

    # ── header (5 lines) ─────────────────────────────────────────────────────

    def _header(self, cols: int) -> list[str]:
        sys = self.system
        gpu   = sys.get("gpu_name", "Unknown GPU")
        vram  = sys.get("gpu_vram_gb", 0)
        ram   = sys.get("total_ram_gb", 0)
        cpu   = _trunc(sys.get("cpu_name", "?"), 38)
        bknd  = sys.get("backend", "CPU")
        sort  = self.SORT_LABEL[self.SORTS[self.sort_idx]]
        n_all = len(self.all_models)
        n_sh  = len(self._filtered)
        srch  = (f"  {CYAN}🔍 {self.query}{RESET}" if self.query
                 else f"  {DIM}[/] search{RESET}")

        title = f"{PINK}{BOLD}⌬ NEURALFIT ADVANCED{RESET}"
        hw    = (f"{CYAN}{BOLD}{gpu}{RESET}"
                 f"  {DIM}{vram:.0f}GB VRAM{RESET}"
                 f"  {DIM}{ram:.0f}GB RAM{RESET}"
                 f"  {DIM}{bknd}{RESET}")
        ctrl  = (f"{DIM}[↑↓] nav  [PgUp/Dn] page  [Enter] detail  "
                 f"[o] pull  [Tab] sort:{RESET}{AMBER}{sort}{RESET}"
                 f"{DIM}  [q] quit{RESET}")
        return [
            "─" * cols,
            f"  {title}   {hw}",
            f"  {DIM}CPU:{RESET} {cpu}  {DIM}|  {n_sh}/{n_all} models{RESET}{srch}",
            f"  {ctrl}",
            "─" * cols,
        ]

    # ── column header row ────────────────────────────────────────────────────

    def _col_hdr(self) -> str:
        cells = []
        for label, width, *_ in COLS:
            cells.append(_pad(label, width))
        return BG_HDR + DIM + "  " + "  ".join(cells) + RESET

    # ── single data row ──────────────────────────────────────────────────────

    def _row(self, idx: int, m: dict, selected: bool) -> str:
        bg   = BG_SEL if selected else ""
        name = m.get("name") or "?"
        inst = m.get("installed", False)
        pull = self.pull_status.get(name)

        if   pull == "pulling": dot = AMBER  + "⟳" + RESET
        elif pull == "done":    dot = MATRIX + "✓" + RESET
        elif pull == "error":   dot = RED    + "✗" + RESET
        elif inst:              dot = MATRIX + "●" + RESET
        else:                   dot = DIM    + "○" + RESET

        sc   = m.get("score_components") or {}
        qual = sc.get("quality", 0)

        def _cell(col):
            label, width, key, align = col
            if label == "#":
                raw = f"{idx+1}"
                return _pad(DIM + raw + RESET, width + len(DIM+RESET), align)
            if label == "Model":
                short = _trunc(name.split("/")[-1], width - 2)
                colored = ((PINK + BOLD if selected else CYAN) + short + RESET)
                return dot + " " + _pad(colored, width - 2 + len((PINK+BOLD if selected else CYAN)+RESET))
            if label == "Params":
                v = f"{m.get('params_b', 0):.1f}B"
                return _pad(DIM + v + RESET, width + len(DIM+RESET), align)
            if label == "VRAM":
                v = f"{m.get('memory_required_gb', 0):.1f}"
                return _pad(PINK + v + RESET, width + len(PINK+RESET), align)
            if label == "Tok/s":
                v = f"{m.get('estimated_tps', 0):.0f}"
                return _pad(DIM + v + RESET, width + len(DIM+RESET), align)
            if label == "Ctx":
                ctx = m.get("context_length") or 0
                v = f"{ctx//1000}k" if ctx >= 1000 else str(ctx)
                return _pad(DIM + v + RESET, width + len(DIM+RESET), align)
            if label == "Score":
                s = m.get("score") or 0
                c = MATRIX if s >= 85 else CYAN if s >= 70 else PINK if s >= 50 else DIM
                v = f"{s:.0f}"
                return _pad(c + BOLD + v + RESET, width + len(c+BOLD+RESET), align)
            if label == "Fit":
                badge = _fit_badge(m.get("fit_level") or "")
                # badge has fixed 10-char visible width + colour codes
                return badge
            if label == "Quality":
                return _bar(qual, width, PINK)
            if label == "Use-Case":
                v = _trunc(m.get("use_case") or m.get("category") or "", width)
                return _pad(DIM + v + RESET, width + len(DIM+RESET), align)
            return " " * width

        cells = [_cell(c) for c in COLS]
        return bg + "  " + "  ".join(cells) + RESET

    # ── detail pane ───────────────────────────────────────────────────────────

    def _detail(self, m: dict, cols: int) -> list[str]:
        sc    = m.get("score_components") or {}
        name  = m.get("name", "?")
        pull  = self.pull_status.get(name, "")
        lines = [
            "═" * cols,
            f"  {PINK}{BOLD}⌬  MODEL DETAIL{RESET}   {CYAN}{BOLD}{name}{RESET}",
            "─" * cols,
            f"  {DIM}Provider:{RESET}      {m.get('provider', '?')}",
            f"  {DIM}Parameters:{RESET}    {m.get('parameter_count', '?')}   "
            f"{DIM}Quant:{RESET} {m.get('best_quant', '?')}   "
            f"{DIM}MoE:{RESET} {'Yes' if m.get('is_moe') else 'No'}",
            f"  {DIM}Context:{RESET}       {(m.get('context_length') or 0):,} tokens  "
            f"({DIM}effective: {(m.get('effective_context_length') or 'N/A'):,}{RESET})",
            f"  {DIM}VRAM Required:{RESET} {PINK}{m.get('memory_required_gb', 0):.2f} GB{RESET}   "
            f"{DIM}Disk:{RESET} {m.get('disk_size_gb', 0):.2f} GB   "
            f"{DIM}Utilization:{RESET} {m.get('utilization_pct', 0):.1f}%",
            f"  {DIM}Throughput:{RESET}    {CYAN}{m.get('estimated_tps', 0):.1f} tok/s{RESET}   "
            f"{DIM}Runtime:{RESET} {m.get('runtime_label', '?')}   "
            f"{DIM}Mode:{RESET} {m.get('run_mode', '?')}",
            f"  {DIM}License:{RESET}       {m.get('license') or 'Unknown'}",
            f"  {DIM}Fit level:{RESET}     {_fit_badge(m.get('fit_level') or '')}",
            "",
            f"  {BOLD}Score Breakdown{RESET}",
            f"  {DIM}Quality{RESET}   {_bar(sc.get('quality', 0),  16, PINK)}  "
            f"{PINK}{sc.get('quality', 0):.0f}/100{RESET}",
            f"  {DIM}Speed  {RESET}   {_bar(sc.get('speed', 0),    16, GREEN)}  "
            f"{GREEN}{sc.get('speed', 0):.0f}/100{RESET}",
            f"  {DIM}Fit    {RESET}   {_bar(sc.get('fit', 0),      16, CYAN)}  "
            f"{CYAN}{sc.get('fit', 0):.0f}/100{RESET}",
            f"  {DIM}Context{RESET}   {_bar(sc.get('context', 0),  16, AMBER)}  "
            f"{AMBER}{sc.get('context', 0):.0f}/100{RESET}",
            f"  {DIM}Composite:{RESET}     {PINK}{BOLD}{m.get('score', 0):.1f} / 100{RESET}",
            "",
        ]
        caps  = m.get("capabilities") or []
        notes = m.get("notes") or []
        gguf  = m.get("gguf_sources") or []
        if caps:
            lines.append(f"  {DIM}Capabilities:{RESET}  {', '.join(caps)}")
        for note in notes[:3]:
            lines.append(f"  {DIM}⚑  {_trunc(note, cols - 8)}{RESET}")
        if gguf:
            lines.append(f"  {DIM}GGUF source:{RESET}   {gguf[0]}")
        if pull:
            ps = {
                "pulling": AMBER  + "Downloading…" + RESET,
                "done":    MATRIX + "Downloaded ✓" + RESET,
                "error":   RED    + "Pull failed ✗" + RESET,
            }.get(pull, pull)
            lines.append(f"  {DIM}Pull status:{RESET}   {ps}")
        lines += [
            "",
            f"  {DIM}[o] Pull with Ollama   [Enter / Esc] close{RESET}",
            "═" * cols,
        ]
        return lines

    # ── Ollama pull ───────────────────────────────────────────────────────────

    def _pull(self, name: str):
        short = name.split("/")[-1]
        self.pull_status[name] = "pulling"
        try:
            subprocess.run(["ollama", "pull", short], check=True,
                           capture_output=True, timeout=600)
            self.pull_status[name] = "done"
        except Exception:
            self.pull_status[name] = "error"

    # ── full render ───────────────────────────────────────────────────────────

    def _render(self):
        cols, rows = self._sz()
        hdr   = self._header(cols)
        n_hdr = len(hdr) + 1        # +1 for col-header row
        list_h = rows - n_hdr - 1   # -1 for footer

        # scroll tracking
        if self.cursor >= self.scroll + list_h:
            self.scroll = self.cursor - list_h + 1
        if self.cursor < self.scroll:
            self.scroll = self.cursor

        out = [CLEAR]
        out.extend(hdr)
        out.append(self._col_hdr())

        visible = self._filtered[self.scroll: self.scroll + list_h]
        for i, m in enumerate(visible):
            abs_i = self.scroll + i
            out.append(self._row(abs_i, m, abs_i == self.cursor))

        # blank padding
        for _ in range(list_h - len(visible)):
            out.append("")

        # footer
        out.append(
            f"  {DIM}NeuralFit Advanced  ·  "
            f"{len(self._filtered)}/{len(self.all_models)} models  ·  "
            f"row {self.cursor + 1}  ·  "
            f"sort: {self.SORT_LABEL[self.SORTS[self.sort_idx]]}{RESET}"
        )

        # detail overlay
        if self.detail_open and self._filtered:
            detail_lines = self._detail(self._filtered[self.cursor], cols)
            # splice in from below the header
            out = out[:n_hdr] + detail_lines

        sys.stdout.write("\n".join(out))
        sys.stdout.flush()

    # ── keyboard ─────────────────────────────────────────────────────────────

    def _getch(self) -> str:
        if sys.platform == "win32":
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\xe0", "\x00"):
                ch2 = msvcrt.getwch()
                return {"H": "UP", "P": "DOWN", "G": "HOME", "O": "END",
                        "I": "PGUP", "Q": "PGDN"}.get(ch2, "")
            return {"\\r": "ENTER", "\r": "ENTER", "\x1b": "ESC",
                    "\x03": "QUIT", "\t": "TAB",
                    "\x08": "BS", "\x7f": "BS"}.get(ch, ch)
        else:
            import tty, termios
            fd   = sys.stdin.fileno()
            old  = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    rest = sys.stdin.read(2)
                    return {"[A": "UP", "[B": "DOWN", "[H": "HOME",
                            "[F": "END", "[5": "PGUP", "[6": "PGDN"
                            }.get(rest, "ESC")
                return {"\r": "ENTER", "\n": "ENTER", "\x03": "QUIT",
                        "\t": "TAB", "\x08": "BS", "\x7f": "BS"}.get(ch, ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ── search ────────────────────────────────────────────────────────────────

    def _search(self):
        self.query = ""
        _, rows = self._sz()
        while True:
            self._render()
            sys.stdout.write(
                f"\033[{rows};0H"
                f"  {PINK}🔍 Search:{RESET} {self.query}{CYAN}▌{RESET}   "
            )
            sys.stdout.flush()
            key = self._getch()
            if key in ("ENTER", "ESC", "QUIT"):
                break
            elif key == "BS":
                self.query = self.query[:-1]
            elif len(key) == 1 and key.isprintable():
                self.query += key
            self._apply()
            self.cursor = 0

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleMode(
                    ctypes.windll.kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass

        while True:
            self._render()
            key = self._getch()

            n = len(self._filtered)
            _, rows = self._sz()
            page    = max(1, rows - 7)

            if   key in ("q", "QUIT"):                break
            elif key == "UP":    self.cursor = max(0, self.cursor - 1)
            elif key == "DOWN":  self.cursor = min(n - 1, self.cursor + 1)
            elif key == "PGUP":  self.cursor = max(0, self.cursor - page)
            elif key == "PGDN":  self.cursor = min(n - 1, self.cursor + page)
            elif key == "HOME":  self.cursor = 0
            elif key == "END":   self.cursor = max(0, n - 1)
            elif key == "TAB":
                self.sort_idx = (self.sort_idx + 1) % len(self.SORTS)
                self._apply()
            elif key == "ENTER":
                self.detail_open = not self.detail_open
            elif key == "ESC":
                if   self.detail_open: self.detail_open = False
                elif self.query:
                    self.query = ""
                    self._apply()
                else:
                    break
            elif key == "/":
                self._search()
            elif key.lower() == "o":
                if self._filtered:
                    m = self._filtered[self.cursor]
                    threading.Thread(
                        target=self._pull, args=(m["name"],), daemon=True
                    ).start()

        sys.stdout.write(CLEAR)
        sys.stdout.flush()
        print(f"\n  {DIM}NeuralFit Advanced closed.{RESET}\n")


# ─── entry point ─────────────────────────────────────────────────────────────

def run_advanced(limit: int = 500, use_case: Optional[str] = None):
    """Fetch all models from the compiled engine and launch the TUI."""
    # ── splash ─────────────────────────────────────────────────────────────
    sys.stdout.write(CLEAR)
    sys.stdout.flush()
    W = "═" * 58
    print(f"\n  {DIM}╔{W}╗{RESET}")
    print(f"  {DIM}║{RESET}  {DIM}10011110 11100101 10101111 01001001 01010100{RESET}          {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {PINK}{BOLD}NEURALFIT ADVANCED{RESET}  {DIM}hardware intelligence engine{RESET}      {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {DIM}Scanning · scoring · loading model catalogue…{RESET}        {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}╚{W}╝{RESET}\n")

    # ── fetch ──────────────────────────────────────────────────────────────
    result_box: dict = {}
    err_box:    dict = {}
    spinners = ["◐", "◓", "◑", "◒"]

    def _fetch():
        try:
            cmd = ["llmfit", "fit", "--json", "--no-dashboard"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            result_box["data"] = json.loads(r.stdout)
        except Exception as e:
            err_box["err"] = str(e)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    idx = 0
    while t.is_alive():
        sys.stdout.write(
            f"\r  {PINK}{spinners[idx % 4]}{RESET}  "
            f"{DIM}Running hardware scan & scoring entire model catalogue…{RESET}   "
        )
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1

    sys.stdout.write(
        f"\r  {MATRIX}✓{RESET}  {DIM}Scan complete.{RESET}"
        f"                                              \n\n"
    )
    sys.stdout.flush()

    if err_box:
        print(f"  {RED}Error: {err_box['err']}{RESET}")
        print(f"  {DIM}Make sure the scoring engine is installed.{RESET}")
        return

    data    = result_box.get("data", {})
    models  = data.get("models", [])
    system  = data.get("system", {})

    if not models:
        print(f"  {RED}No model data returned.{RESET}")
        return

    time.sleep(0.3)
    NeuralFitTUI(models, system).run()
