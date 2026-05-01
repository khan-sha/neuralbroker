"""
NeuralFit Advanced TUI
─────────────────────
Full-screen interactive terminal UI powered by the compiled hardware-scoring
engine. Features: keyboard navigation, live search, sort cycling, detail pane,
Ollama pull shortcut, and a NeuralBroker Pink-Matrix aesthetic.

Zero llmfit branding visible to the end user.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import threading
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour palette (matches NeuralBroker Pink-Matrix theme)
# ──────────────────────────────────────────────────────────────────────────────
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
BG_SEL  = "\033[48;5;53m"   # deep purple row highlight
BG_HDR  = "\033[48;5;16m"   # near-black header bg
CLEAR   = "\033[2J\033[H"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


def _trunc(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ──────────────────────────────────────────────────────────────────────────────
# Score bar renderer  ████████░░  (Pink=quality, Green=speed, Cyan=fit)
# ──────────────────────────────────────────────────────────────────────────────
def _bar(pct: float, width: int = 10, color: str = PINK) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET


# ──────────────────────────────────────────────────────────────────────────────
# Fit-level badge
# ──────────────────────────────────────────────────────────────────────────────
FIT_BADGE = {
    "good":        (_c("✅ optimal  ", MATRIX, BOLD),   "optimal"),
    "comfortable": (_c("✅ optimal  ", MATRIX, BOLD),   "optimal"),
    "tight":       (_c("⚠️  tight    ", AMBER, BOLD),    "tight"),
    "partial":     (_c("⚡ partial  ", RED),             "partial"),
    "too_large":   (_c("❌ too large", DIM),             "large"),
}

def _fit_badge(fit_level: str) -> str:
    key = fit_level.lower().replace(" ", "_")
    return FIT_BADGE.get(key, (_c("? unknown ", DIM), "?"))[0]


# ──────────────────────────────────────────────────────────────────────────────
# Main TUI class
# ──────────────────────────────────────────────────────────────────────────────
class NeuralFitTUI:
    SORTS = ["score", "tps", "params", "mem", "ctx", "name"]
    SORT_LABELS = {
        "score": "Score↓", "tps": "Tok/s↓", "params": "Params↓",
        "mem": "VRAM↓", "ctx": "Context↓", "name": "Name A-Z",
    }

    def __init__(self, models: list, system: dict, limit: int = 40):
        self.all_models = models[:limit]
        self.system = system
        self.cursor = 0
        self.scroll = 0
        self.sort_idx = 0
        self.query = ""
        self.detail_open = False
        self.pull_status: dict[str, str] = {}   # name -> "pulling"|"done"|"error"
        self._filtered: list = []
        self._apply_sort_filter()

    # ── data helpers ──────────────────────────────────────────────────────────

    def _sort_key(self, m: dict):
        s = self.SORTS[self.sort_idx]
        if s == "score":  return -(m.get("score") or 0)
        if s == "tps":    return -(m.get("estimated_tps") or 0)
        if s == "params": return -(m.get("params_b") or 0)
        if s == "mem":    return -(m.get("memory_required_gb") or 0)
        if s == "ctx":    return -(m.get("context_length") or 0)
        if s == "name":   return (m.get("name") or "").lower()
        return 0

    def _apply_sort_filter(self):
        q = self.query.lower()
        self._filtered = [
            m for m in self.all_models
            if q in (m.get("name") or "").lower()
            or q in (m.get("use_case") or "").lower()
            or q in (m.get("category") or "").lower()
        ]
        self._filtered.sort(key=self._sort_key)
        self.cursor = min(self.cursor, max(0, len(self._filtered) - 1))

    # ── terminal sizing ───────────────────────────────────────────────────────

    @staticmethod
    def _size() -> tuple[int, int]:
        cols, rows = shutil.get_terminal_size((120, 40))
        return cols, rows

    # ── header ────────────────────────────────────────────────────────────────

    def _render_header(self, cols: int) -> list[str]:
        sys_gpu   = self.system.get("gpu_name", "Unknown GPU")
        sys_vram  = self.system.get("gpu_vram_gb", 0)
        sys_ram   = self.system.get("total_ram_gb", 0)
        sys_cpu   = self.system.get("cpu_name", "Unknown CPU")
        backend   = self.system.get("backend", "CPU")
        n_results = len(self._filtered)

        sort_label = self.SORT_LABELS[self.SORTS[self.sort_idx]]
        search_str = f"  🔍 {CYAN}{self.query}{RESET}" if self.query else f"  {DIM}/ to search{RESET}"

        brand = f"{PINK}{BOLD}⌬ NEURALFIT ADVANCED{RESET}"
        hw    = (
            f"{CYAN}{BOLD}{sys_gpu}{RESET}"
            f"  {DIM}{sys_vram:.0f}GB VRAM{RESET}"
            f"  {DIM}{sys_ram:.0f}GB RAM{RESET}"
            f"  {DIM}{backend}{RESET}"
        )
        controls = (
            f"{DIM}[↑↓] scroll  [Enter] detail  [o] pull  "
            f"[Tab] sort:{RESET}{AMBER}{sort_label}{RESET}"
            f"{DIM}  [q] quit{RESET}"
        )

        lines = [
            "─" * cols,
            f"  {brand}   {hw}",
            f"  {DIM}CPU:{RESET} {_trunc(sys_cpu, 40)}  "
            f"{DIM}|  {n_results} models matched{RESET}{search_str}",
            f"  {controls}",
            "─" * cols,
        ]
        return lines

    # ── column header ─────────────────────────────────────────────────────────

    def _render_col_header(self, cols: int) -> str:
        return (
            BG_HDR
            + f"  {'#':>3}  {'Model':<28}  {'Params':>6}  "
              f"{'VRAM':>5}  {'Tok/s':>5}  {'Ctx':>6}  "
              f"{'Score':>5}  {'Fit':<11}  {'Quality':<12}  {'Use-Case':<20}"
            + RESET
        )

    # ── single model row ──────────────────────────────────────────────────────

    def _render_row(self, idx: int, m: dict, is_selected: bool, cols: int) -> str:
        bg   = BG_SEL if is_selected else ""
        num  = _c(f"{idx+1:>3}", DIM)
        name = m.get("name") or "?"
        inst = f"{MATRIX}●{RESET}" if m.get("installed") else f"{DIM}○{RESET}"

        pull_s = self.pull_status.get(name)
        if pull_s == "pulling":
            inst = f"{AMBER}⟳{RESET}"
        elif pull_s == "done":
            inst = f"{MATRIX}✓{RESET}"
        elif pull_s == "error":
            inst = f"{RED}✗{RESET}"

        params  = f"{m.get('params_b', 0):.1f}B"
        vram    = f"{m.get('memory_required_gb', 0):.1f}"
        tps     = f"{m.get('estimated_tps', 0):.0f}"
        ctx     = f"{(m.get('context_length') or 0)//1000}k"
        score   = m.get("score") or 0
        sc_col  = MATRIX if score >= 85 else CYAN if score >= 70 else PINK if score >= 50 else DIM
        score_s = _c(f"{score:.0f}", sc_col, BOLD)

        fit_s   = _fit_badge(m.get("fit_level", ""))
        sc_comp = m.get("score_components") or {}
        quality = sc_comp.get("quality", 0)
        q_bar   = _bar(quality, 8, PINK)
        use     = _trunc(m.get("use_case") or m.get("category") or "", 20)

        name_display = _trunc(name.split("/")[-1], 26)
        if is_selected:
            name_display = _c(name_display, PINK, BOLD)
        else:
            name_display = _c(name_display, CYAN)

        row = (
            bg
            + f"  {num}  {inst} {name_display:<26}  {_c(params, DIM):>9}  "
              f"{_c(vram, DIM):>8}  {_c(tps, DIM):>8}  {_c(ctx, DIM):>9}  "
              f"{score_s:>5}  {fit_s}  {q_bar}  {_c(use, DIM)}"
            + RESET
        )
        return row

    # ── detail pane ───────────────────────────────────────────────────────────

    def _render_detail(self, m: dict, cols: int) -> list[str]:
        sc   = m.get("score_components") or {}
        name = m.get("name", "?")
        lines = [
            "═" * cols,
            f"  {PINK}{BOLD}⌬ MODEL DETAIL{RESET}  {CYAN}{BOLD}{name}{RESET}",
            "─" * cols,
            f"  {DIM}Provider:{RESET}    {m.get('provider','?')}",
            f"  {DIM}Parameters:{RESET}  {m.get('parameter_count','?')}   "
            f"{DIM}Quant:{RESET} {m.get('best_quant','?')}   "
            f"{DIM}MoE:{RESET} {'Yes' if m.get('is_moe') else 'No'}",
            f"  {DIM}Context:{RESET}     {(m.get('context_length') or 0):,} tokens  "
            f"({DIM}effective: {m.get('effective_context_length') or 'N/A'}{RESET})",
            f"  {DIM}VRAM Required:{RESET} {m.get('memory_required_gb',0):.2f} GB   "
            f"{DIM}Disk:{RESET} {m.get('disk_size_gb',0):.2f} GB   "
            f"{DIM}Utilization:{RESET} {m.get('utilization_pct',0):.1f}%",
            f"  {DIM}Throughput:{RESET}  {CYAN}{m.get('estimated_tps',0):.1f} tok/s{RESET}   "
            f"{DIM}Runtime:{RESET} {m.get('runtime_label','?')}   "
            f"{DIM}Mode:{RESET} {m.get('run_mode','?')}",
            f"  {DIM}License:{RESET}     {m.get('license') or 'Unknown'}",
            "",
            f"  {BOLD}Score Breakdown:{RESET}",
            f"  Quality  {_bar(sc.get('quality',0),  14, PINK)}  {sc.get('quality',0):.0f}/100",
            f"  Speed    {_bar(sc.get('speed',0),    14, GREEN)}  {sc.get('speed',0):.0f}/100",
            f"  Fit      {_bar(sc.get('fit',0),      14, CYAN)}  {sc.get('fit',0):.0f}/100",
            f"  Context  {_bar(sc.get('context',0),  14, AMBER)}  {sc.get('context',0):.0f}/100",
            f"  {DIM}Composite:{RESET}   {PINK}{BOLD}{m.get('score',0):.1f}/100{RESET}",
            "",
        ]
        caps = m.get("capabilities") or []
        if caps:
            lines.append(f"  {DIM}Capabilities:{RESET}  {', '.join(caps)}")
        notes = m.get("notes") or []
        for note in notes[:3]:
            lines.append(f"  {DIM}⚑  {_trunc(note, cols-8)}{RESET}")
        gguf  = m.get("gguf_sources") or []
        if gguf:
            lines.append(f"  {DIM}GGUF source:{RESET} {gguf[0]}")
        pull_s = self.pull_status.get(name)
        if pull_s:
            status_str = {
                "pulling": f"{AMBER}Downloading…{RESET}",
                "done":    f"{MATRIX}Downloaded ✓{RESET}",
                "error":   f"{RED}Pull failed ✗{RESET}",
            }.get(pull_s, pull_s)
            lines.append(f"  {DIM}Pull status:{RESET}  {status_str}")
        lines += [
            "",
            f"  {DIM}[o] Pull with Ollama   [Esc/Enter] close{RESET}",
            "═" * cols,
        ]
        return lines

    # ── pull via ollama ───────────────────────────────────────────────────────

    def _pull_model(self, name: str):
        short = name.split("/")[-1]
        self.pull_status[name] = "pulling"
        try:
            subprocess.run(
                ["ollama", "pull", short],
                check=True, capture_output=True, timeout=600,
            )
            self.pull_status[name] = "done"
        except Exception:
            self.pull_status[name] = "error"

    # ── full render ───────────────────────────────────────────────────────────

    def _render(self):
        cols, rows = self._size()
        buf: list[str] = []

        header  = self._render_header(cols)
        col_hdr = self._render_col_header(cols)
        n_head  = len(header) + 1           # +1 for col header
        list_h  = rows - n_head - 2         # -2 for footer

        # Scroll logic
        if self.cursor >= self.scroll + list_h:
            self.scroll = self.cursor - list_h + 1
        if self.cursor < self.scroll:
            self.scroll = self.cursor

        buf.append(CLEAR)
        buf.extend(header)
        buf.append(col_hdr)

        visible = self._filtered[self.scroll: self.scroll + list_h]
        for i, m in enumerate(visible):
            abs_idx = self.scroll + i
            buf.append(self._render_row(abs_idx, m, abs_idx == self.cursor, cols))

        # Pad empty rows
        for _ in range(list_h - len(visible)):
            buf.append("")

        # Footer
        n_total = len(self.all_models)
        n_shown = len(self._filtered)
        buf.append(
            f"  {DIM}NeuralFit Advanced  ·  {n_shown}/{n_total} models  "
            f"·  row {self.cursor+1}/{n_shown}  ·  "
            f"sort: {self.SORT_LABELS[self.SORTS[self.sort_idx]]}{RESET}"
        )

        # Detail overlay
        if self.detail_open and self._filtered:
            m = self._filtered[self.cursor]
            detail_lines = self._render_detail(m, cols)
            # Print from bottom up, overlaid
            start_row = max(0, rows - len(detail_lines) - 1)
            # Clear and re-render from start_row
            buf_prefix = buf[:n_head + start_row]
            overlay = detail_lines
            buf = buf_prefix + overlay

        sys.stdout.write("\n".join(buf))
        sys.stdout.flush()

    # ── input (Windows getch + Unix) ─────────────────────────────────────────

    def _getch(self) -> str:
        """Read a single keystroke, returns a symbolic string."""
        if sys.platform == "win32":
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\xe0", "\x00"):
                ch2 = msvcrt.getwch()
                return {"H": "UP", "P": "DOWN", "G": "HOME", "O": "END",
                        "I": "PGUP", "Q": "PGDN"}.get(ch2, "")
            if ch == "\r":  return "ENTER"
            if ch == "\x1b": return "ESC"
            if ch == "\x03": return "QUIT"
            if ch == "\t":  return "TAB"
            return ch
        else:
            import tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    rest = sys.stdin.read(2)
                    return {"[A": "UP", "[B": "DOWN", "[H": "HOME", "[F": "END",
                            "[5": "PGUP", "[6": "PGDN"}.get(rest, "ESC")
                if ch == "\r" or ch == "\n": return "ENTER"
                if ch == "\x03": return "QUIT"
                if ch == "\t":  return "TAB"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            os.system("" if sys.platform != "win32" else "cls")   # init
            # Enable ANSI on Windows
            if sys.platform == "win32":
                os.system("color")
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

        while True:
            self._render()
            key = self._getch()

            if key in ("q", "QUIT"):
                break
            elif key == "UP":
                self.cursor = max(0, self.cursor - 1)
            elif key == "DOWN":
                self.cursor = min(len(self._filtered) - 1, self.cursor + 1)
            elif key == "PGUP":
                _, rows = self._size()
                self.cursor = max(0, self.cursor - (rows // 2))
            elif key == "PGDN":
                _, rows = self._size()
                self.cursor = min(len(self._filtered) - 1, self.cursor + (rows // 2))
            elif key == "HOME":
                self.cursor = 0
            elif key == "END":
                self.cursor = max(0, len(self._filtered) - 1)
            elif key == "TAB":
                self.sort_idx = (self.sort_idx + 1) % len(self.SORTS)
                self._apply_sort_filter()
            elif key == "ENTER":
                self.detail_open = not self.detail_open
            elif key == "ESC":
                if self.detail_open:
                    self.detail_open = False
                elif self.query:
                    self.query = ""
                    self._apply_sort_filter()
                else:
                    break
            elif key == "/":
                self._search_mode()
            elif key.lower() == "o":
                if self._filtered:
                    m = self._filtered[self.cursor]
                    t = threading.Thread(target=self._pull_model, args=(m["name"],), daemon=True)
                    t.start()

        sys.stdout.write(CLEAR)
        sys.stdout.flush()
        print(f"\n  {DIM}NeuralFit Advanced closed.{RESET}\n")

    # ── incremental search ────────────────────────────────────────────────────

    def _search_mode(self):
        """Simple in-place search input at bottom of screen."""
        _, rows = self._size()
        self.query = ""
        while True:
            self._render()
            # overwrite last footer line with prompt
            sys.stdout.write(
                f"\033[{rows};0H"  # move to last row
                f"  {PINK}🔍 Search:{RESET} {self.query}{CYAN}▌{RESET}   "
            )
            sys.stdout.flush()

            key = self._getch()
            if key in ("ENTER", "ESC"):
                break
            elif key == "QUIT":
                self.query = ""
                break
            elif key in ("\x08", "\x7f", "BACKSPACE"):
                self.query = self.query[:-1]
            elif len(key) == 1 and key.isprintable():
                self.query += key
            self._apply_sort_filter()
            self.cursor = 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — fetches data and launches TUI
# ──────────────────────────────────────────────────────────────────────────────

def run_advanced(limit: int = 40, use_case: Optional[str] = None):
    """Fetch data from the compiled scoring engine and launch the NeuralFit TUI."""
    # ── loading screen ────────────────────────────────────────────────────────
    sys.stdout.write(CLEAR)
    sys.stdout.flush()
    W = "═" * 58
    print(f"\n  {DIM}╔{W}╗{RESET}")
    print(f"  {DIM}║{RESET}  {DIM}10011110 11100101 10101111 01001001 01010100{RESET}          {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {PINK}{BOLD}NEURALFIT ADVANCED{RESET}  {DIM}hardware intelligence engine{RESET}      {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {DIM}Scanning hardware · scoring models · loading TUI{RESET}     {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}╚{W}╝{RESET}\n")

    spinners = ["◐", "◓", "◑", "◒"]
    result_box: dict = {}
    err_box: dict = {}

    def _fetch():
        try:
            cmd = ["llmfit", "fit", "--json", "--no-dashboard", "-n", str(limit)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            result_box["data"] = json.loads(r.stdout)
        except Exception as e:
            err_box["err"] = str(e)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    idx = 0
    while t.is_alive():
        sys.stdout.write(
            f"\r  {PINK}{spinners[idx % 4]}{RESET}  {DIM}Running hardware scan & model scoring…{RESET}   "
        )
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r  {MATRIX}✓{RESET}  {DIM}Scan complete.{RESET}                                   \n\n")
    sys.stdout.flush()

    if err_box:
        print(f"  {RED}Error: {err_box['err']}{RESET}")
        print(f"  {DIM}Make sure the hardware scoring engine is installed.{RESET}")
        return

    data    = result_box.get("data", {})
    models  = data.get("models", [])
    system  = data.get("system", {})

    if not models:
        print(f"  {RED}No model data returned.{RESET}")
        return

    time.sleep(0.4)
    tui = NeuralFitTUI(models, system, limit=limit)
    tui.run()
