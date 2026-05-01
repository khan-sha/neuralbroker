"""
neuralfit_tui.py — NeuralFit Advanced  (fixed full-screen TUI)
================================================================
• Writes EXACTLY `rows` lines using \033[row;1H cursor positioning
• Each line is clipped to `cols` visible chars — no wrapping ever
• Scrollable viewport stays fixed on screen (no content pouring below)
• Adapts to any terminal size on each keypress
• All 900+ models, no cap
"""
from __future__ import annotations

import json, os, re, shutil, subprocess, sys, threading, time
from typing import Optional

# ─── colours ──────────────────────────────────────────────────────────────────
RST  = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
PNK  = "\033[95m"
CYN  = "\033[96m"
GRN  = "\033[92m"
RED  = "\033[91m"
AMB  = "\033[93m"
MAT  = "\033[92m"
BG_S = "\033[48;5;53m"   # purple  — selected row
BG_H = "\033[48;5;17m"   # navy    — column header

# ─── ANSI-strip helpers ───────────────────────────────────────────────────────
_ESC = re.compile(r"\033\[[0-9;]*[mABCDHJKSTfnrlupsu]")

def vlen(s: str) -> int:
    """Visible character count (strips ANSI codes)."""
    return len(_ESC.sub("", s))

def clip(s: str, width: int) -> str:
    """
    Return a string whose VISIBLE length is exactly `width` chars.
    Truncates at the visible boundary, then pads with spaces, then
    appends RST to close any open colour sequences.
    """
    vis = 0
    out = []
    i   = 0
    while i < len(s):
        if s[i] == "\033":
            # consume the full escape sequence (ends at a letter)
            j = i + 1
            while j < len(s) and not s[j].isalpha():
                j += 1
            out.append(s[i:j + 1])
            i = j + 1
        else:
            if vis < width:
                out.append(s[i])
                vis += 1
            i += 1
    # pad to width
    out.append(" " * max(0, width - vis))
    out.append(RST)
    return "".join(out)

def trunc(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"

# ─── score bar ────────────────────────────────────────────────────────────────
def bar(pct: float, w: int, color: str = PNK) -> str:
    f = max(0, min(w, round(pct / 100 * w)))
    return color + "█" * f + DIM + "░" * (w - f) + RST

# ─── fit badge (all real llmfit values, fixed 11 visible chars) ──────────────
_FIT: dict[str, str] = {
    "perfect":   MAT + BOLD + "● perfect  " + RST,
    "good":      MAT + BOLD + "● good     " + RST,
    "too tight": AMB + BOLD + "▲ too tight" + RST,
    "tight":     AMB + BOLD + "▲ tight    " + RST,
    "marginal":  RED +        "⚡ marginal " + RST,
    "partial":   RED +        "⚡ partial  " + RST,
    "too_large": DIM +        "✗ too large" + RST,
}
FIT_W = 11  # visible width of every badge above

def fit_badge(level: str) -> str:
    return _FIT.get(level.strip().lower(), DIM + trunc("? " + level, FIT_W).ljust(FIT_W) + RST)

# ─── write one line to an exact terminal row ──────────────────────────────────
def put(row: int, line: str, cols: int):
    """Position cursor at (row, 1) and write a clipped, padded line."""
    sys.stdout.write(f"\033[{row};1H{clip(line, cols)}")

# ─── NeuralFitTUI ─────────────────────────────────────────────────────────────
class NeuralFitTUI:
    SORTS      = ["score", "tps", "params", "mem", "ctx", "name"]
    SORT_LABEL = {
        "score":  "Score↓", "tps": "Tok/s↓", "params": "Params↓",
        "mem":    "VRAM↓",  "ctx": "Ctx↓",   "name":   "Name A-Z",
    }

    def __init__(self, models: list, system: dict):
        self.models      = models        # all models, never mutated
        self.system      = system
        self.cursor      = 0
        self.scroll      = 0
        self.sort_idx    = 0
        self.query       = ""
        self.detail      = False
        self.pull_st:    dict[str, str] = {}
        self._filt:      list = []
        self._rebuild()

    # ── data ─────────────────────────────────────────────────────────────────
    def _rebuild(self):
        q = self.query.lower()
        self._filt = [
            m for m in self.models
            if not q
            or q in (m.get("name") or "").lower()
            or q in (m.get("use_case") or "").lower()
            or q in (m.get("category") or "").lower()
            or q in (m.get("fit_level") or "").lower()
        ]
        sk = self.SORTS[self.sort_idx]
        if   sk == "score":  self._filt.sort(key=lambda m: -(m.get("score") or 0))
        elif sk == "tps":    self._filt.sort(key=lambda m: -(m.get("estimated_tps") or 0))
        elif sk == "params": self._filt.sort(key=lambda m: -(m.get("params_b") or 0))
        elif sk == "mem":    self._filt.sort(key=lambda m: -(m.get("memory_required_gb") or 0))
        elif sk == "ctx":    self._filt.sort(key=lambda m: -(m.get("context_length") or 0))
        elif sk == "name":   self._filt.sort(key=lambda m: (m.get("name") or "").lower())
        n = len(self._filt)
        self.cursor = max(0, min(self.cursor, n - 1))

    # ── terminal size ────────────────────────────────────────────────────────
    @staticmethod
    def _sz() -> tuple[int, int]:
        c, r = shutil.get_terminal_size((120, 40))
        return c, r

    # ── build a single model row string (no clipping yet) ────────────────────
    def _row_str(self, abs_idx: int, m: dict, sel: bool) -> str:
        bg   = BG_S if sel else ""
        name = (m.get("name") or "?").split("/")[-1]
        inst = self.pull_st.get(m.get("name", ""))
        if   inst == "pulling": dot = AMB + "⟳" + RST
        elif inst == "done":    dot = MAT + "●" + RST
        elif inst == "error":   dot = RED + "✗" + RST
        elif m.get("installed"): dot = MAT + "●" + RST
        else:                   dot = DIM + "○" + RST

        sc    = m.get("score_components") or {}
        qual  = sc.get("quality", 0)
        score = m.get("score") or 0
        sc_c  = MAT if score >= 85 else CYN if score >= 70 else PNK if score >= 50 else DIM
        ctx   = m.get("context_length") or 0
        ctx_s = f"{ctx//1000}k" if ctx >= 1000 else str(ctx)
        vram  = m.get("memory_required_gb") or 0
        tps   = m.get("estimated_tps") or 0
        prm   = m.get("params_b") or 0
        badge = fit_badge(m.get("fit_level") or "")
        qbar  = bar(qual, 6, PNK)
        nm_c  = PNK + BOLD if sel else CYN
        use   = trunc((m.get("use_case") or m.get("category") or ""), 18)

        return (
            bg
            + f" {DIM}{abs_idx+1:>3}{RST} "
            + dot + " "
            + nm_c + f"{trunc(name,24):<24}" + RST + " "
            + DIM + f"{prm:>5.1f}B " + RST
            + PNK + f"{vram:>4.1f} " + RST
            + DIM + f"{tps:>4.0f} " + RST
            + DIM + f"{ctx_s:>5} " + RST
            + sc_c + BOLD + f"{score:>3.0f} " + RST
            + badge + " "
            + qbar + " "
            + DIM + use + RST
        )

    # ── detail pane lines ────────────────────────────────────────────────────
    def _detail_lines(self, m: dict) -> list[str]:
        sc   = m.get("score_components") or {}
        name = m.get("name", "?")
        pull = self.pull_st.get(name, "")
        caps = ", ".join(m.get("capabilities") or []) or "—"
        notes = (m.get("notes") or [])
        ps   = {"pulling": AMB + "Downloading…" + RST,
                "done":    MAT + "Downloaded ✓" + RST,
                "error":   RED + "Pull failed ✗" + RST}.get(pull, "")
        lines = [
            PNK + BOLD + " ⌬  MODEL DETAIL" + RST + "  " + CYN + BOLD + name + RST,
            DIM + "─" * 60 + RST,
            f" {DIM}Provider  {RST}{m.get('provider','?')}   "
            f"{DIM}Params{RST} {m.get('parameter_count','?')}   "
            f"{DIM}Quant{RST} {m.get('best_quant','?')}",
            f" {DIM}VRAM{RST} {PNK}{vram:.2f}GB{RST}  "
            f"{DIM}Disk{RST} {m.get('disk_size_gb',0):.1f}GB  "
            f"{DIM}Context{RST} {(m.get('context_length') or 0):,} tokens  "
            f"{DIM}Tok/s{RST} {CYN}{m.get('estimated_tps',0):.1f}{RST}",
            f" {DIM}License{RST} {m.get('license') or 'Unknown'}   "
            f"{DIM}Fit{RST} {fit_badge(m.get('fit_level') or '')}",
            f" {DIM}Capabilities{RST} {caps}",
            "",
            f" {BOLD}Score Breakdown{RST}",
            f"  Quality  {bar(sc.get('quality',0),14,PNK)}  {PNK}{sc.get('quality',0):.0f}{RST}",
            f"  Speed    {bar(sc.get('speed',0),14,GRN)}  {GRN}{sc.get('speed',0):.0f}{RST}",
            f"  Fit      {bar(sc.get('fit',0),14,CYN)}  {CYN}{sc.get('fit',0):.0f}{RST}",
            f"  Context  {bar(sc.get('context',0),14,AMB)}  {AMB}{sc.get('context',0):.0f}{RST}",
            f"  {DIM}Composite{RST}  {PNK}{BOLD}{m.get('score',0):.1f}/100{RST}",
        ]
        for note in notes[:2]:
            lines.append(f"  {DIM}⚑ {trunc(note, 70)}{RST}")
        if ps:
            lines.append(f"  {DIM}Pull status{RST}  {ps}")
        lines += ["", f"  {DIM}[o] Ollama pull   [Enter/Esc] close{RST}",
                  DIM + "═" * 60 + RST]
        return lines

    # ── render — writes to exact terminal rows, never scrolls ─────────────────
    def _render(self):
        cols, rows = self._sz()

        # ── layout constants ──
        HDR_ROWS = 3   # brand + hw line + controls line
        COL_ROW  = HDR_ROWS + 1
        LIST_TOP = COL_ROWS = COL_ROW + 1
        FOOT_ROW = rows          # last row (1-indexed)
        LIST_H   = rows - LIST_TOP  # available list rows (footer occupies last)

        # ── scroll clamping ──
        n = len(self._filt)
        if n > 0:
            if self.cursor >= self.scroll + LIST_H:
                self.scroll = self.cursor - LIST_H + 1
            if self.cursor < self.scroll:
                self.scroll = self.cursor
            self.scroll = max(0, min(self.scroll, max(0, n - LIST_H)))

        sys.stdout.write("\033[?25l")   # hide cursor while drawing

        # ── row 1: brand ──
        srt   = self.SORT_LABEL[self.SORTS[self.sort_idx]]
        n_sh  = len(self._filt)
        srch  = (f" {CYN}🔍{self.query}{RST}" if self.query else "")
        gpu   = self.system.get("gpu_name", "GPU")
        vram_s = self.system.get("gpu_vram_gb", 0)
        ram_s  = self.system.get("total_ram_gb", 0)
        bk    = self.system.get("backend", "CPU")
        put(1, (PNK + BOLD + " ⌬ NEURALFIT ADVANCED" + RST + "  "
                + CYN + BOLD + trunc(gpu, 30) + RST
                + f"  {DIM}{vram_s:.0f}GB VRAM  {ram_s:.0f}GB RAM  {bk}{RST}"
                + srch), cols)

        # ── row 2: model count + sort ──
        put(2, (f"  {DIM}{n_sh}/{len(self.models)} models  "
                f"row {self.cursor+1}  sort: {AMB}{srt}{DIM}"
                f"  [↑↓] nav  [PgUp/Dn] page  [Enter] detail"
                f"  [o] pull  [/] search  [Tab] sort  [q] quit{RST}"), cols)

        # ── row 3: divider ──
        put(3, DIM + "─" * cols + RST, cols)

        # ── row 4: column header ──
        hdr = (BG_H + DIM
               + "  #    Model                      Params VRAM  TPS   Ctx  Sc "
               + f"{'Fit':<{FIT_W+1}} Quality  Use-Case"
               + RST)
        put(4, hdr, cols)

        # ── rows 5…FOOT_ROW-1: model list ──
        visible = self._filt[self.scroll: self.scroll + LIST_H]
        for i, m in enumerate(visible):
            abs_i = self.scroll + i
            put(LIST_TOP + i, self._row_str(abs_i, m, abs_i == self.cursor), cols)

        # blank remaining rows
        for i in range(len(visible), LIST_H):
            put(LIST_TOP + i, "", cols)

        # ── footer (last row) ──
        put(FOOT_ROW, DIM + "─" * cols + RST, cols)

        # ── detail overlay ──
        if self.detail and self._filt:
            dlines = self._detail_lines(self._filt[self.cursor])
            for i, dl in enumerate(dlines):
                r = LIST_TOP + i
                if r >= FOOT_ROW:
                    break
                put(r, dl, cols)

        sys.stdout.write("\033[?25h")   # restore cursor
        sys.stdout.flush()

    # ── keyboard ─────────────────────────────────────────────────────────────
    def _getch(self) -> str:
        if sys.platform == "win32":
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\xe0", "\x00"):
                ch2 = msvcrt.getwch()
                return {"H":"UP","P":"DOWN","G":"HOME","O":"END","I":"PGUP","Q":"PGDN"}.get(ch2,"")
            return {"\r":"ENTER","\x1b":"ESC","\x03":"QUIT","\t":"TAB",
                    "\x08":"BS","\x7f":"BS"}.get(ch, ch)
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                rest = sys.stdin.read(2)
                return {"[A":"UP","[B":"DOWN","[H":"HOME","[F":"END",
                        "[5":"PGUP","[6":"PGDN"}.get(rest,"ESC")
            return {"\r":"ENTER","\n":"ENTER","\x03":"QUIT",
                    "\t":"TAB","\x08":"BS","\x7f":"BS"}.get(ch, ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ── search mode ───────────────────────────────────────────────────────────
    def _search_mode(self):
        self.query = ""
        while True:
            self._rebuild()
            self.cursor = 0
            self._render()
            cols, rows = self._sz()
            sys.stdout.write(f"\033[{rows};1H"
                             f"  {PNK}Search:{RST} {self.query}{CYN}▌{RST}   ")
            sys.stdout.flush()
            k = self._getch()
            if k in ("ENTER","ESC","QUIT"):
                break
            elif k == "BS":
                self.query = self.query[:-1]
            elif len(k) == 1 and k.isprintable():
                self.query += k

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        # enable VT on Windows
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleMode(
                    ctypes.windll.kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass
        sys.stdout.write("\033[2J")   # clear once at start
        sys.stdout.flush()

        try:
            while True:
                self._render()
                k = self._getch()
                n = len(self._filt)
                _, rows = self._sz()
                page = max(1, rows - 5)

                if   k in ("q","QUIT"):  break
                elif k == "UP":    self.cursor = max(0, self.cursor - 1)
                elif k == "DOWN":  self.cursor = min(n - 1, self.cursor + 1)
                elif k == "PGUP":  self.cursor = max(0, self.cursor - page)
                elif k == "PGDN":  self.cursor = min(n - 1, self.cursor + page)
                elif k == "HOME":  self.cursor = 0
                elif k == "END":   self.cursor = max(0, n - 1)
                elif k == "TAB":
                    self.sort_idx = (self.sort_idx + 1) % len(self.SORTS)
                    self._rebuild()
                elif k == "ENTER": self.detail = not self.detail
                elif k == "ESC":
                    if   self.detail: self.detail = False
                    elif self.query:  self.query = ""; self._rebuild()
                    else:             break
                elif k == "/":     self._search_mode()
                elif k.lower() == "o" and self._filt:
                    m = self._filt[self.cursor]
                    threading.Thread(target=self._pull,
                                     args=(m["name"],), daemon=True).start()
        finally:
            sys.stdout.write("\033[2J\033[H\033[?25h")
            sys.stdout.flush()
        print(f"\n  {DIM}NeuralFit Advanced closed.{RST}\n")

    def _pull(self, name: str):
        short = name.split("/")[-1]
        self.pull_st[name] = "pulling"
        try:
            subprocess.run(["ollama","pull",short], check=True,
                           capture_output=True, timeout=600)
            self.pull_st[name] = "done"
        except Exception:
            self.pull_st[name] = "error"


# ─── entry point ─────────────────────────────────────────────────────────────
def run_advanced(limit: int = 9999, use_case: Optional[str] = None):
    """Fetch all models from the compiled engine and launch the TUI."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    W = "═" * 56
    print(f"\n  {DIM}╔{W}╗{RST}")
    print(f"  {DIM}║{RST}  {PNK}{BOLD}NEURALFIT ADVANCED{RST}  {DIM}hardware intelligence engine{RST}  {DIM}║{RST}")
    print(f"  {DIM}║{RST}  {DIM}scanning · scoring · loading model catalogue…{RST}    {DIM}║{RST}")
    print(f"  {DIM}╚{W}╝{RST}\n")

    result: dict = {}
    err:    dict = {}
    spin = ["◐","◓","◑","◒"]

    def _fetch():
        try:
            r = subprocess.run(
                ["llmfit","fit","--json","--no-dashboard"],
                capture_output=True, text=True, timeout=60)
            result["data"] = json.loads(r.stdout)
        except Exception as e:
            err["msg"] = str(e)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    i = 0
    while t.is_alive():
        sys.stdout.write(f"\r  {PNK}{spin[i%4]}{RST}  {DIM}Running engine…{RST}   ")
        sys.stdout.flush()
        time.sleep(0.1); i += 1

    sys.stdout.write(f"\r  {MAT}✓{RST}  {DIM}Done.{RST}                \n\n")
    sys.stdout.flush()

    if err:
        print(f"  {RED}Error: {err['msg']}{RST}")
        return

    data    = result.get("data", {})
    models  = data.get("models", [])
    system  = data.get("system", {})
    if not models:
        print(f"  {RED}No models returned.{RST}"); return

    time.sleep(0.2)
    NeuralFitTUI(models, system).run()
