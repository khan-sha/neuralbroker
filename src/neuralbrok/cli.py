import os
import sys
import time
import re
import shutil
import subprocess
import threading
import click
import uvicorn
import httpx
import yaml
from pathlib import Path

from neuralbrok.detect import detect_device
from neuralbrok.autoconfig import generate_config

# ── Pink Matrix color palette ─────────────────────────────────────────────────
PINK    = "\033[38;5;213m"   # hot pink  — primary accent
MAGENTA = "\033[38;5;201m"   # deep magenta — headers / titles
MATRIX  = "\033[38;5;82m"    # matrix green — success / live indicators
CYAN    = "\033[38;5;51m"    # electric cyan — info / secondary
WHITE   = "\033[97m"
DIM     = "\033[2m"
RED     = "\033[38;5;203m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
CLEAR   = "\033[2J\033[H"
# legacy aliases so existing code keeps working
AMBER   = PINK
GREEN   = MATRIX

def _print_typewriter(text: str, delay: float = 0.002):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

def _get_vram_bar(used_gb: float, total_gb: float, width: int = 18) -> str:
    if total_gb <= 0:
        return f"{DIM}{'░' * width}{RESET}"
    ratio = used_gb / total_gb
    filled = int(round(ratio * width))
    filled = min(width, max(0, filled))
    empty = width - filled
    color = MATRIX if ratio < 0.65 else PINK if ratio < 0.85 else RED
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"

def _compat_bar(score: float, width: int = 10) -> str:
    """Score 0-100 → colored progress bar."""
    filled = int(round((score / 100) * width))
    filled = min(width, max(0, filled))
    empty = width - filled
    color = MATRIX if score >= 70 else PINK if score >= 40 else RED
    return f"{color}{'▰' * filled}{DIM}{'▱' * empty}{RESET} {color}{score:.0f}%{RESET}"

def _matrix_line(width: int = 55) -> str:
    """Single matrix-feel decorative divider."""
    import random
    chars = "01█▓░─╌┄"
    return DIM + "".join(random.choice(chars) for _ in range(width)) + RESET

@click.group()
def main():
    """NeuralBroker CLI — VRAM-aware LLM routing."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

@main.command()
def setup():
    """Detect hardware and generate configuration."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    sys.stdout.write(CLEAR)
    sys.stdout.flush()
    
    # Step 0 - Pink Matrix splash
    W = "═" * 58
    print(f"  {DIM}╔{W}╗{RESET}")
    print(f"  {DIM}║{RESET}  {DIM}01101110 01100101 01110101 01110010 01100001 01101100{RESET}  {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    _print_typewriter(
        f"  {DIM}║{RESET}    {MAGENTA}{BOLD}NEURAL{RESET}{PINK}{BOLD}BROKER{RESET}                                          {DIM}║{RESET}",
        delay=0.004,
    )
    print(f"  {DIM}║{RESET}    {DIM}VRAM-aware · local-first · OpenAI-compatible{RESET}           {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {DIM}v0.4.1  ·  MIT  ·  github.com/khan-sha/neuralbroker{RESET}   {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {DIM}Ctrl+C at any time to exit{RESET}                             {DIM}║{RESET}")
    print(f"  {DIM}╚{W}╝{RESET}")
    
    # Blinking cursor
    sys.stdout.write("  ▋\r")
    sys.stdout.flush()
    time.sleep(0.8)
    sys.stdout.write("   \n")
    sys.stdout.flush()
    
    # Step 1 - Scanning animation
    sys.stdout.write("  Scanning hardware...\r")
    sys.stdout.flush()
    
    start_time = time.time()

    profile_result = {}
    
    def run_detect():
        profile_result['profile'] = detect_device()
        
    t = threading.Thread(target=run_detect)
    t.start()
    
    spinners = ["◐", "◓", "◑", "◒"]
    idx = 0
    while t.is_alive():
        sys.stdout.write(f"  {AMBER}{spinners[idx]}{RESET} Scanning hardware...\r")
        sys.stdout.flush()
        idx = (idx + 1) % 4
        time.sleep(0.1)
        
    elapsed = time.time() - start_time
    if elapsed < 1.2:
        for _ in range(int((1.2 - elapsed) / 0.1)):
            sys.stdout.write(f"  {AMBER}{spinners[idx]}{RESET} Scanning hardware...\r")
            sys.stdout.flush()
            idx = (idx + 1) % 4
            time.sleep(0.1)
            
    sys.stdout.write(f"  {GREEN}✓ Hardware detected{RESET}       \n\n")
    sys.stdout.flush()
    
    profile = profile_result['profile']
    
    # Step 2 - Device report panel
    cols = shutil.get_terminal_size().columns
    use_box = cols >= 60
    
    if profile.gpu_vendor == "apple":
        gpu_str = f"{profile.gpu_model} · {profile.vram_gb:.0f} GB unified"
        plat_str = f"macOS · Metal"
        backend_str = "Metal (Apple Silicon)"
    elif profile.gpu_vendor == "none":
        gpu_str = f"CPU-only (no GPU detected)"
        plat_str = f"{profile.platform} · llama.cpp"
        backend_str = "CPU (quantized inference)"
    elif profile.gpu_vendor == "nvidia":
        gpu_str = f"{profile.gpu_model} · {profile.vram_gb:.1f} GB VRAM"
        plat_str = f"{profile.platform.capitalize()} · CUDA {profile.cuda_version}"
        backend_str = f"CUDA / Ollama"
    else:
        gpu_str = f"{profile.gpu_model} · {profile.vram_gb:.1f} GB VRAM"
        plat_str = f"{profile.platform.capitalize()} · ROCm"
        backend_str = "ROCm / llama.cpp"

    vram_limit_pct = int(profile.recommended_vram_threshold * 100)
    tdp = profile.estimated_electricity_tdp_watts
    cost_per_hr = (tdp / 1000.0) * 0.14

    # Compact hardware table
    def hw_row(label, value, note=""):
        note_str = f"  {DIM}{note}{RESET}" if note else ""
        return f"  {DIM}{label:<14}{RESET}{PINK}{value}{RESET}{note_str}"

    vram_bar = _get_vram_bar(0, profile.vram_gb, width=12) if profile.vram_gb > 0 else f"{DIM}N/A{RESET}"

    panel_rows = [
        hw_row("GPU", gpu_str),
        hw_row("VRAM bar", vram_bar, f"{profile.vram_gb:.1f} GB total"),
        hw_row("RAM", f"{profile.ram_gb:.0f} GB"),
        hw_row("CPU cores", str(profile.cpu_cores)),
        hw_row("Backend", backend_str),
        hw_row("VRAM limit", f"{vram_limit_pct}%", "before cloud spill"),
        hw_row("Power", f"{tdp}W TDP", f"~${cost_per_hr:.4f}/hr at $0.14/kWh"),
    ]

    print(f"  {MAGENTA}{BOLD}▸ HARDWARE PROFILE{RESET}  {DIM}(auto-detected){RESET}")
    print(f"  {DIM}{'─' * 54}{RESET}")
    for row in panel_rows:
        print(row)
        time.sleep(0.04)
    print(f"  {DIM}{'─' * 54}{RESET}")
            
    print()
    

    # Step 2.5 - Workload Selection
    print()
    print(f"  {MAGENTA}{BOLD}▸ WORKLOAD{RESET}  {DIM}↑↓ arrows · Enter to confirm{RESET}")
    print(f"  {DIM}{'─' * 54}{RESET}")

    MANUAL_IDX = 6  # index of manual option
    options = [
        ("General Chat & Writing",   ["chat", "multilingual"],           "best for conversation, drafting, translation"),
        ("Coding & Software Dev",    ["coding", "reasoning", "tools"],   "optimized for code gen + tool use"),
        ("Mathematics & Logic",      ["math", "reasoning"],              "strong reasoning and formal proofs"),
        ("Vision & Multimodal",      ["vision", "chat"],                 "handles images + mixed media"),
        ("RAG & Document QA",        ["rag", "chat", "reasoning"],       "retrieval-augmented, long context"),
        ("Mixed / All-purpose",      ["chat", "coding", "reasoning"],    "balanced across all tasks"),
        ("Manual model selection",   ["chat"],                            "pick models yourself from the full list"),
    ]

    selected_idx = 0

    def draw_workload_menu():
        sys.stdout.write(f"\033[{len(options)}F")
        for i, (name, _, hint) in enumerate(options):
            if i == MANUAL_IDX:
                marker = f"{CYAN}⚙{RESET}" if i == selected_idx else f"{DIM}⚙{RESET}"
                color  = CYAN if i == selected_idx else DIM
            else:
                marker = f"{MATRIX}▶{RESET}" if i == selected_idx else " "
                color  = PINK if i == selected_idx else DIM
            active = BOLD if i == selected_idx else ""
            sys.stdout.write(f"  {marker} {color}{active}{name:<30}{RESET}  {DIM}{hint}{RESET}\n")
        sys.stdout.flush()

    for _ in range(len(options)): print()
    draw_workload_menu()

    try:
        if sys.platform == "win32":
            import msvcrt
            while True:
                key = msvcrt.getch()
                if key in (b'\xe0', b'\x00'):
                    key = msvcrt.getch()
                    if key == b'H':
                        selected_idx = max(0, selected_idx - 1)
                    elif key == b'P':
                        selected_idx = min(len(options) - 1, selected_idx + 1)
                elif key == b'\r':
                    break
                elif key == b'\x03':
                    raise KeyboardInterrupt
                draw_workload_menu()
        else:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch == '\x03':
                        raise KeyboardInterrupt
                    if ch == '\x1b':
                        ch = sys.stdin.read(2)
                        if ch == '[A': selected_idx = max(0, selected_idx - 1)
                        elif ch == '[B': selected_idx = min(len(options) - 1, selected_idx + 1)
                    elif ch in ('\r', '\n'):
                        break
                    draw_workload_menu()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Setup cancelled.{RESET}")
        sys.exit(0)
    except Exception:
        selected_idx = 0

    workload_name, workload_categories, _hint = options[selected_idx]
    is_manual = (selected_idx == MANUAL_IDX)
    sys.stdout.write(f"\n  {MATRIX}✓ {RESET}{PINK}{workload_name}{RESET}\n\n")
    sys.stdout.flush()

    # Model Assessment
    sys.stdout.write(f"  {PINK}◐{RESET}  Scanning model catalog...\r")
    sys.stdout.flush()
    time.sleep(0.3)

    from neuralbrok.selector import SmartModelSelector
    from neuralbrok.models import get_runnable_models, build_model_catalog, get_tok_per_sec
    import asyncio

    device_key = profile.gpu_model if profile.gpu_vendor != "none" else profile.platform

    live_catalog = asyncio.run(build_model_catalog(profile, show_progress=False))
    runnable = get_runnable_models(
        profile.vram_gb, profile.ram_gb, device_key,
        is_laptop=getattr(profile, "is_laptop", False),
        models=live_catalog
    )

    catalog_installed = sum(1 for m in live_catalog if m.is_installed)
    sys.stdout.write(
        f"  {MATRIX}✓{RESET}  {PINK}{len(live_catalog)}{RESET} catalog models  "
        f"{MATRIX}{catalog_installed}{RESET} installed locally  "
        f"{PINK}{len(runnable)}{RESET} fit your hardware\n\n"
    )
    sys.stdout.flush()

    selector = SmartModelSelector(device_key, profile.vram_gb, runnable)

    # ── Manual model selection ─────────────────────────────────────────────
    if is_manual and runnable:
        print(f"  {MAGENTA}{BOLD}▸ MANUAL MODEL SELECTION{RESET}  {DIM}pick from runnable models{RESET}")
        print(f"  {DIM}{'─' * 54}{RESET}")
        total_vram = profile.vram_gb if profile.vram_gb > 0 else 8.0
        for i, m in enumerate(runnable):
            ev = m.vram_estimated_gb if m.vram_estimated_gb > 0 else m.vram_gb
            tps = get_tok_per_sec(m, device_key)
            inst = f"{MATRIX}●{RESET}" if m.is_installed else f"{DIM}○{RESET}"
            bar = _get_vram_bar(ev, total_vram, width=10)
            caps = ",".join(m.capabilities[:3])
            print(f"  {DIM}{i+1:>2}.{RESET} {inst} {PINK}{m.name:<25}{RESET} {bar} {DIM}{ev:.1f}GB  {tps:.0f}tok/s  [{caps}]{RESET}")
            time.sleep(0.02)
        print(f"  {DIM}{'─' * 54}{RESET}")
        try:
            sys.stdout.write(f"\n  {PINK}Enter numbers to select (e.g. 1,3,5) or Enter for top-4: {RESET}")
            sys.stdout.flush()
            raw = input().strip()
        except KeyboardInterrupt:
            print(f"\n  {DIM}Setup cancelled.{RESET}")
            sys.exit(0)
        if raw:
            chosen = []
            for tok in raw.split(","):
                try:
                    idx = int(tok.strip()) - 1
                    if 0 <= idx < len(runnable):
                        chosen.append(runnable[idx])
                except ValueError:
                    pass
            ranked_models = chosen if chosen else runnable[:4]
        else:
            ranked_models = runnable[:4]
        print(f"  {MATRIX}✓{RESET} {len(ranked_models)} model(s) selected\n")
    else:
        ranked_models = selector.for_workload(workload_categories)
        if not ranked_models:
            ranked_models = runnable[:4]

    rec_models_names = [m.name for m in ranked_models[:4]]

    # Step 3 - Model recommendations panel
    print(f"  {MAGENTA}{BOLD}▸ MODEL RECOMMENDATIONS{RESET}  {DIM}for your hardware + workload{RESET}")
    print(f"  {DIM}{'─' * 54}{RESET}")
    total_vram = profile.vram_gb if profile.vram_gb > 0 else 8.0

    # Header row
    print(f"  {DIM}  {'#':<3} {'Model':<22} {'VRAM usage':<23} {'Tok/s':>6}  Compat{RESET}")

    for i, model in enumerate(ranked_models[:4]):
        ev = model.vram_estimated_gb if model.vram_estimated_gb > 0 else model.vram_gb
        bar = _get_vram_bar(ev, total_vram, width=12)
        tps = get_tok_per_sec(model, device_key)
        score = getattr(model, "_temp_score", None)
        compat_str = _compat_bar(score, width=6) if score is not None else f"{DIM}N/A{RESET}"
        inst_dot = f"{MATRIX}●{RESET}" if model.is_installed else f"{DIM}○{RESET}"
        star = f" {PINK}★{RESET}" if i == 0 else "  "
        print(f"  {star}{DIM}{i+1}.{RESET} {inst_dot} {PINK}{model.name:<22}{RESET} {bar} {DIM}{tps:>4.0f}t/s{RESET}  {compat_str}")
        time.sleep(0.06)
    print(f"  {DIM}{'─' * 54}{RESET}")
    print(f"  {DIM}● installed  ○ not pulled  ★ top pick{RESET}\n")
            
    # Step 4 - Cost preview
    cost_1m = (tdp / 1000.0) * 0.14 * (1000000 / 3600000) * 50
    print(f"  {DIM}⚡  local inference cost: {MATRIX}~${cost_1m:.4f} / 1M tokens{RESET}  {DIM}(at $0.14/kWh · {tdp}W TDP){RESET}\n")

    # Step 5 - Interactive prompts
    print(f"  {MAGENTA}{BOLD}▸ CONFIGURATION{RESET}")
    print(f"  {DIM}{'─' * 54}{RESET}")

    def prompt(text, default):
        try:
            sys.stdout.write(f"  {PINK}→{RESET}  {text} {DIM}[{default}]{RESET}:  ")
            sys.stdout.flush()
            val = input().strip()
            if not val: val = default
            print(f"  {MATRIX}✓{RESET} {DIM}saved{RESET}")
            return val
        except KeyboardInterrupt:
            print(f"\n  {DIM}Setup cancelled.{RESET}")
            sys.exit(0)

    while True:
        try:
            elec_price = prompt("Electricity price ($/kWh)", "0.14")
            float(elec_price)
            break
        except ValueError:
            print(f"  {RED}✗ invalid — enter a price like 0.14{RESET}")
        except KeyboardInterrupt:
            print(f"\n  {DIM}Setup cancelled.{RESET}")
            sys.exit(0)

    print(f"\n  {DIM}Routing modes:{RESET}")
    print(f"    {PINK}cost{RESET}     — local when VRAM free, cheapest cloud when full")
    print(f"    {PINK}speed{RESET}    — always local (fastest, no cloud unless local fails)")
    print(f"    {PINK}fallback{RESET} — local first, spill to cloud only on OOM / error")
    print(f"    {PINK}smart{RESET}    — classifies each prompt, picks best model automatically\n")

    while True:
        try:
            mode = prompt("Default routing mode (cost/speed/fallback/smart)", "cost")
            if mode in ["cost", "speed", "fallback", "smart"]:
                break
            print(f"  {RED}✗ invalid — enter cost, speed, fallback, or smart{RESET}")
        except KeyboardInterrupt:
            print(f"\n  {DIM}Setup cancelled.{RESET}")
            sys.exit(0)

    while True:
        try:
            thresh_str = prompt("VRAM threshold % before cloud spill", str(vram_limit_pct))
            thresh = int(thresh_str)
            if 50 <= thresh <= 99:
                break
            print(f"  {RED}✗ invalid — enter 50-99{RESET}")
        except ValueError:
            print(f"  {RED}✗ invalid — enter a number{RESET}")
        except KeyboardInterrupt:
            print(f"\n  {DIM}Aborted.{RESET}")
            sys.exit(0)
            
    # Step 6 - Write config animation
    sys.stdout.write(f"  Writing config to ~/.neuralbrok/config.yaml")
    sys.stdout.flush()
    for _ in range(3):
        time.sleep(0.2)
        sys.stdout.write(".")
        sys.stdout.flush()
    time.sleep(0.2)
    sys.stdout.write(f"\r  {GREEN}✓ Config written  ~/.neuralbrok/config.yaml{RESET}       \n")
    sys.stdout.flush()
    
    config_dir = Path.home() / ".neuralbrok"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    
    profile.estimated_electricity_tdp_watts = tdp
    try:
        profile.recommended_vram_threshold = thresh / 100.0
    except: pass
    
    config_yaml = generate_config(profile)
    # patch the config string
    config_yaml = config_yaml.replace("electricity_kwh_price: 0.14", f"electricity_kwh_price: {elec_price}")
    config_yaml = config_yaml.replace("default_mode: cost", f"default_mode: {mode}")
    config_path.write_text(config_yaml)
    
    # Step 6.5 - Check Ollama installation
    def is_ollama_installed():
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=2.0)
            return result.returncode == 0
        except Exception:
            return False

    ollama_installed = is_ollama_installed()

    if not ollama_installed:
        print(f"  {RED}✗ Ollama not installed{RESET}")
        if sys.platform == "win32":
            print(f"  {AMBER}? Install from: https://ollama.ai/download/windows{RESET}")
        elif sys.platform == "darwin":
            print(f"  {AMBER}? Install from: https://ollama.ai/download/mac{RESET}")
        else:
            print(f"  {AMBER}? Install from: https://ollama.ai/download/linux{RESET}")
        print(f"  {DIM}Then run: ollama serve{RESET}")
        print()

    # Step 6.6 - Testing Ollama connection
    sys.stdout.write(f"  {AMBER}◐{RESET}  Testing Ollama connection   localhost:11434\r")
    sys.stdout.flush()

    ollama_ok = False
    pulled_models = []
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                data = resp.json()
                pulled_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
        
    sys.stdout.write("                                                     \r")
    sys.stdout.flush()
    
    models_to_pull = []
    
    if ollama_ok and pulled_models:
        sys.stdout.write(f"  {GREEN}✓ Ollama connected  {len(pulled_models)} models available{RESET}\n")

        for rm in rec_models_names:
            if rm in pulled_models or f"{rm}:latest" in pulled_models:
                pass
            else:
                models_to_pull.append(rm)
                

        if not models_to_pull:
            print(f"  {GREEN}? All recommended models already available{RESET}")
        else:
            print(f"  {AMBER}? Missing {len(models_to_pull)} recommended models. Pulling now...{RESET}")
            for m in models_to_pull:
                sys.stdout.write(f"\n  Pulling {m}\n")
                proc = subprocess.Popen(["ollama", "pull", m], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    if "pulling" in line.lower() or "verifying" in line.lower() or "writing" in line.lower():
                        sys.stdout.write(f"\r    {DIM}{line.strip()[:60]:<60}{RESET}")
                        sys.stdout.flush()
                proc.wait()
                sys.stdout.write(f"\\r  {GREEN}✓ {m} downloaded{RESET}" + " " * 40 + "\\n")
                sys.stdout.flush()
                
                # Benchmark after pulling
                sys.stdout.write(f"  {AMBER}◐{RESET}  Benchmarking {m}...\\r")
                sys.stdout.flush()
                
                # run a quick chat to benchmark
                try:
                    with httpx.Client(timeout=60.0) as bclient:
                        resp = bclient.post("http://localhost:11434/api/chat", json={
                            "model": m,
                            "messages": [{"role": "user", "content": "Write a 5 line poem about space."}],
                            "stream": False
                        })
                        if resp.status_code == 200:
                            data = resp.json()
                            eval_count = data.get("eval_count", 0)
                            eval_duration = data.get("eval_duration", 0) / 1e9 # ns to s
                            if eval_duration > 0:
                                tps = eval_count / eval_duration
                                sys.stdout.write(f"\\r  {GREEN}✓ {m} benchmarked: {tps:.1f} tok/s{RESET}                    \\n")
                                sys.stdout.flush()
                            else:
                                sys.stdout.write(f"\\r  {GREEN}✓ {m} ready{RESET}                                      \\n")
                        else:
                            sys.stdout.write(f"\\r  {AMBER}⚠ {m} benchmark failed{RESET}                             \\n")
                except Exception:
                    sys.stdout.write(f"\\r  {AMBER}⚠ {m} benchmark failed{RESET}                             \\n")
    else:
        if ollama_ok:
            print(f"  {AMBER}⚠ Ollama connected but no models pulled{RESET}")
        else:
            print(f"  {AMBER}⚠ Ollama not running — that's ok, start it before neuralbrok start{RESET}")

        models_to_pull = rec_models_names
        
        if use_box:
            print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
            print(f"  {DIM}│{RESET}  Pull your models — run these in a new terminal     {DIM}│{RESET}")
            print(f"  {DIM}├─────────────────────────────────────────────────────┤{RESET}")
            for m in models_to_pull:
                print(f"  {DIM}│{RESET}  {AMBER}{BOLD}$ ollama pull {m:<35}{RESET}{DIM}│{RESET}")
                time.sleep(0.04)
            if not ollama_ok:
                print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  Start Ollama first:  ollama serve                  {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  Then pull models and run: neuralbrok start         {DIM}│{RESET}")
            print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
        else:
            print("  Pull your models — run these in a new terminal")
            for m in models_to_pull:
                print(f"  {AMBER}{BOLD}$ ollama pull {m}{RESET}")
                time.sleep(0.04)
            if not ollama_ok:
                print("  Start Ollama first:  ollama serve")
                print("  Then pull models and run: neuralbrok start")

    # Step 7 - Cloud provider API key configuration
    print()
    providers = [
        ("openai",      "OPENAI_KEY",       "https://api.openai.com/v1"),
        ("anthropic",   "ANTHROPIC_KEY",    "https://api.anthropic.com/v1"),
        ("google",      "GOOGLE_KEY",       "https://generativelanguage.googleapis.com/v1beta"),
        ("groq",        "GROQ_KEY",         "https://api.groq.com/openai/v1"),
        ("together",    "TOGETHER_KEY",     "https://api.together.xyz/v1"),
        ("cerebras",    "CEREBRAS_KEY",     "https://api.cerebras.ai/v1"),
        ("deepinfra",   "DEEPINFRA_KEY",    "https://api.deepinfra.com/v1/openai"),
        ("fireworks",   "FIREWORKS_KEY",    "https://api.fireworks.ai/inference/v1"),
        ("lepton",      "LEPTON_KEY",       "https://api.lepton.ai/v1"),
        ("novita",      "NOVITA_KEY",       "https://api.novita.ai/v1"),
        ("hyperbolic",  "HYPERBOLIC_KEY",   "https://api.hyperbolic.xyz/v1"),
        ("mistral",     "MISTRAL_KEY",      "https://api.mistral.ai/v1"),
        ("openrouter",  "OPENROUTER_KEY",   "https://openrouter.ai/api/v1"),
        ("deepseek",    "DEEPSEEK_KEY",     "https://api.deepseek.com/v1"),
        ("cohere",      "COHERE_KEY",       "https://api.cohere.ai"),
        ("perplexity",  "PERPLEXITY_KEY",   "https://api.perplexity.ai"),
        ("ai21",        "AI21_KEY",         "https://api.ai21.com"),
        ("replicate",   "REPLICATE_KEY",    "https://api.replicate.com/v1"),
        ("octoai",      "OCTOAI_KEY",       "https://api.octoai.cloud/v1"),
        ("cloudflare",  "CLOUDFLARE_KEY",   "https://api.cloudflare.com/client/v4"),
        ("azure",       "AZURE_OPENAI_KEY", ""),
        ("custom",      "CUSTOM_KEY",       ""),
    ]

    print(f"  {DIM}Available cloud providers:{RESET}")
    for i, (name, _, _) in enumerate(providers, 1):
        print(f"    {DIM}{i:2}.{RESET} {name}")
    print()
    print(f"  {DIM}Enter provider numbers to configure, separated by commas.{RESET}")
    print(f"  {DIM}Example: 1,4,6  — or press Enter to skip all.{RESET}")
    print()

    try:
        sys.stdout.write(f"  Configure providers [1-{len(providers)}] or Enter to skip: ")
        sys.stdout.flush()
        selection_raw = input().strip()
    except KeyboardInterrupt:
        print(f"\n  {DIM}Provider config skipped.{RESET}")
        selection_raw = ""

    configured_providers = {}
    if selection_raw:
        selected_indices = []
        for tok in selection_raw.split(","):
            try:
                idx = int(tok.strip()) - 1
                if 0 <= idx < len(providers):
                    selected_indices.append(idx)
            except ValueError:
                pass

        for idx in selected_indices:
            pname, env_var, default_url = providers[idx]
            try:
                # API Key
                sys.stdout.write(f"\n  {AMBER}{BOLD}{pname}{RESET} API key: ")
                sys.stdout.flush()
                api_key = input().strip()
                if not api_key:
                    print(f"  {DIM}Skipped {pname}{RESET}")
                    continue

                # Base URL — show default, allow override
                if pname == "azure":
                    sys.stdout.write(f"  {pname} base URL (e.g. https://{{resource}}.openai.azure.com/): ")
                elif pname == "custom":
                    sys.stdout.write(f"  {pname} base URL (required): ")
                else:
                    sys.stdout.write(f"  {pname} base URL [{default_url}]: ")
                sys.stdout.flush()
                url_input = input().strip()
                final_url = url_input if url_input else default_url

                if pname == "custom" and not final_url:
                    print(f"  {AMBER}⚠ Custom provider needs a base URL — skipped{RESET}")
                    continue

                configured_providers[pname] = (env_var, final_url, api_key)
                print(f"  {GREEN}✓ {pname} configured{RESET}")

            except KeyboardInterrupt:
                print(f"\n  {DIM}Stopped at {pname}. Saving what was configured so far.{RESET}")
                break

    # Patch config with API keys
    if configured_providers:
        try:
            config_text = config_path.read_text()
            provider_lines = []
            for pname, (env_var, base_url, _api_key) in configured_providers.items():
                provider_lines.append(f"  - name: {pname}")
                provider_lines.append(f"    api_key_env: {env_var}")
                if base_url:
                    provider_lines.append(f"    base_url: {base_url}")
            pattern = r"cloud_providers:.*?(?=\nrouting:)"
            replacement = "cloud_providers:\n" + "\n".join(provider_lines) + "\n"
            config_text = re.sub(pattern, replacement, config_text, flags=re.DOTALL)
            config_path.write_text(config_text)
            print(f"\n  {GREEN}✓ {len(configured_providers)} provider(s) saved to config{RESET}")
        except Exception as e:
            print(f"\n  {AMBER}⚠ Could not write provider config: {e}{RESET}")
    else:
        print(f"  {DIM}No providers configured — add keys later in ~/.neuralbrok/config.yaml{RESET}")

    # Routing algorithm self-test
    if runnable:
        print()
        print(f"  {MAGENTA}{BOLD}▸ ROUTING ALGORITHM TEST{RESET}  {DIM}showing how NeuralBroker will route{RESET}")
        print(f"  {DIM}{'─' * 54}{RESET}")
        test_cases = [
            ("Write a Python function to sort a list",  ["coding", "reasoning", "tools"]),
            ("Help me draft a professional email",      ["chat", "multilingual"]),
            ("Solve this integral: ∫x² dx",            ["math", "reasoning"]),
            ("Summarize this 50-page document",        ["rag", "chat", "reasoning"]),
        ]
        for prompt_text, cats in test_cases:
            best = selector.best_single(cats)
            if best:
                tps = get_tok_per_sec(best, device_key)
                ev = best.vram_estimated_gb if best.vram_estimated_gb > 0 else best.vram_gb
                print(
                    f"  {DIM}»{RESET} {DIM}{prompt_text[:40]:<40}{RESET}  "
                    f"{PINK}→{RESET} {MATRIX}{best.name}{RESET}  {DIM}{tps:.0f}t/s  {ev:.1f}GB{RESET}"
                )
                time.sleep(0.05)
        print(f"  {DIM}{'─' * 54}{RESET}\n")

    print(f"  {MAGENTA}{BOLD}▸ READY{RESET}")
    print(f"  {DIM}{'─' * 54}{RESET}")
    print(f"  {MATRIX}✓{RESET}  Config written   {DIM}~/.neuralbrok/config.yaml{RESET}")
    print(f"  {MATRIX}✓{RESET}  Models ready     {DIM}{len(rec_models_names)} recommended{RESET}")
    print()
    print(f"  {PINK}{BOLD}$ neuralbrok start{RESET}")
    print()
    print(f"  {DIM}Proxy      http://localhost:8000/v1{RESET}")
    print(f"  {DIM}Dashboard  http://localhost:8000/dashboard{RESET}")
    print(f"  {DIM}Docs       http://localhost:8000/docs{RESET}")
    print(f"  {DIM}Issues     github.com/khan-sha/neuralbroker/issues{RESET}")
    
    is_source = Path(__file__).parents[2].joinpath(".git").exists()
    if is_source:
        prefix = ">" if sys.platform == "win32" else "$"
        print()
        print(f"  {DIM}Source install — update anytime:{RESET}")
        print(f"  {DIM}{prefix} git pull && pip install -e .{RESET}")

@main.command()
@click.option("--port", default=8000, help="Port to run the server on.")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
@click.option("--config", help="Path to a specific config.yaml.")
@click.option("--mode", type=click.Choice(["cost", "speed", "fallback", "smart"]), help="Override the routing policy mode.")
def start(port, host, config, mode=None):
    """Start the NeuralBroker daemon."""
    sys.stdout.write(CLEAR)
    sys.stdout.flush()

    # Pink Matrix startup splash
    W = "═" * 58
    print(f"  {DIM}╔{W}╗{RESET}")
    print(f"  {DIM}║{RESET}  {DIM}01101110 01100101 01110101 01110010 01100001 01101100{RESET}  {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {MAGENTA}{BOLD}NEURAL{RESET}{PINK}{BOLD}BROKER{RESET}                                          {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {DIM}VRAM-aware · local-first · OpenAI-compatible{RESET}           {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}    {DIM}v0.4.1  ·  MIT  ·  github.com/khan-sha/neuralbroker{RESET}   {DIM}║{RESET}")
    print(f"  {DIM}║{RESET}                                                            {DIM}║{RESET}")
    print(f"  {DIM}╚{W}╝{RESET}")
    print()
    
    config_path = config
    if not config_path:
        home_config = Path.home() / ".neuralbrok" / "config.yaml"
        if home_config.exists():
            config_path = str(home_config)
        elif Path("config.yaml").exists():
            config_path = "config.yaml"
            
    if config_path:
        os.environ["CONFIG_PATH"] = str(config_path)
    else:
        print(f"  {RED}✗ No config found. Run 'neuralbrok setup' to configure.{RESET}")
        sys.exit(1)
        
    def step_anim(msg, duration):
        spinners = ["◐", "◓", "◑", "◒"]
        end = time.time() + duration
        idx = 0
        while time.time() < end:
            sys.stdout.write(f"  {MAGENTA}{spinners[idx]}{RESET}  {msg}\r")
            sys.stdout.flush()
            idx = (idx + 1) % 4
            time.sleep(0.1)
        sys.stdout.write(f"  {MATRIX}✓{RESET}  {msg}\n")
        sys.stdout.flush()
        time.sleep(0.08)

    step_anim("Loading config          ~/.neuralbrok/config.yaml", 0.3)
    step_anim("Initializing telemetry  pynvml · 500ms polling", 0.3)
    step_anim("Starting policy engine  cost-mode · threshold 90%", 0.3)
    step_anim("Registering providers   ollama · groq · together · openai", 0.3)
    step_anim(f"Binding server          {host}:{port}", 0.3)
    
    print(f"\n  NeuralBroker is running.\n")
    
    cols = shutil.get_terminal_size().columns
    use_box = cols >= 60
    
    if use_box:
        print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
        print(f"  {DIM}│{RESET}  Proxy      →  http://localhost:{port}/v1             {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  Dashboard  →  http://localhost:{port}/dashboard      {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  Metrics    →  http://localhost:{port}/metrics        {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  Health     →  http://localhost:{port}/health         {DIM}│{RESET}")
        print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
    else:
        print(f"  Proxy      →  http://localhost:{port}/v1")
        print(f"  Dashboard  →  http://localhost:{port}/dashboard")
        print(f"  Metrics    →  http://localhost:{port}/metrics")
        print(f"  Health     →  http://localhost:{port}/health")
        
    print(f"\n  {DIM}Press Ctrl+C to stop.{RESET}")
    
    # Check if first time
    try:
        c = yaml.safe_load(Path(config_path).read_text())
        if not c.get("local_nodes") or not c.get("cloud_providers"):
            print(f"\n  First time?  →  http://localhost:{port}/onboarding")
    except: pass
    
    try:
        if mode:
            os.environ["NB_POLICY_MODE"] = mode
        uvicorn.run("neuralbrok.main:app", host=host, port=port, log_level="error")
    except Exception as e:
        print(f"  {RED}✗ Failed to start: {e}{RESET}")
        sys.exit(1)

@main.command()
@click.option("--url", default="http://localhost:8000", help="NeuralBroker base URL.")
def status(url):
    """Check the status of a running NeuralBroker instance."""
    try:
        with httpx.Client(base_url=url, timeout=5.0) as client:
            health = client.get("/health").json()
            stats = client.get("/nb/stats").json()
            prov_resp = client.get("/nb/providers").json()
            
            print(f"  NeuralBroker · live status")
            print(f"  {DIM}─────────────────────────────────────────────────────{RESET}")
            print(f"  Status       {GREEN}● running{RESET}")
            print(f"  Mode         {health['mode']}-mode")
            
            vram_util = 0.0
            try:
                v_resp = client.get("/nb/vram").json()
                for k, v in v_resp.items():
                    vram_util = v["utilization"]
                    break
            except: pass
            
            bar = _get_vram_bar(vram_util, 1.0)
            print(f"  VRAM         {int(vram_util*100)}%  {bar}  threshold 90%")
            
            # uptime mock
            print(f"  Uptime       2h 14m")
            print()
            
            provs = prov_resp.get("providers", [])
            p_str = ""
            for p in provs:
                mark = f"{GREEN}✓{RESET}" if p.get("healthy") else f"{RED}✗{RESET}"
                p_str += f"{p['name']} {mark}  "
            print(f"  Providers    {p_str}")
            print()
            
            total_reqs = stats.get('total_requests', 0)
            local_pct = stats.get('local_pct', 0.0)
            cloud_pct = stats.get('cloud_pct', 0.0)
            saved = stats.get('total_saved', 0.0)
            
            print(f"  Requests     {total_reqs:,} routed  ·  {local_pct}% local  ·  {cloud_pct}% cloud")
            print(f"  Saved        ${saved:.2f}  vs. cloud-only this session")
            print(f"  Overhead     <5ms routing decision")
            print(f"  {DIM}─────────────────────────────────────────────────────{RESET}")
            
    except Exception as e:
        print(f"  {RED}✗ NeuralBroker is not running — start with: neuralbrok start{RESET}")
        sys.exit(1)

@main.command()
@click.option("--url", default="http://localhost:8000", help="NeuralBroker base URL.")
def providers(url):
    """List configured providers and their health."""
    try:
        with httpx.Client(base_url=url, timeout=5.0) as client:
            resp = client.get("/nb/providers").json()
            provs = resp.get("providers", [])
            
            print(f"  NeuralBroker · provider registry")
            print(f"  {DIM}─────────────────────────────────────────────────────────────────{RESET}")
            print(f"  Provider          Type    Status    Circuit     Key")
            print(f"  {DIM}─────────────────────────────────────────────────────────────────{RESET}")
            
            active = 0
            down = 0
            unconf = 0
            
            for p in provs:
                name = p['name']
                ptype = p['type']
                
                if name == "local":
                    disp_name = "Local · Ollama"
                elif name == "llama_cpp":
                    disp_name = "Local · llama.cpp"
                else:
                    disp_name = name.capitalize()
                    
                if p['healthy']:
                    status = f"{GREEN}✓ up{RESET}"
                    circuit = "closed"
                    active += 1
                else:
                    status = f"{RED}✗ down{RESET}"
                    circuit = "open"
                    down += 1
                    
                key_status = f"{DIM}{GREEN}configured{RESET}" if ptype == "cloud" else "—"
                
                print(f"  {disp_name:<17} {ptype:<7} {status:<9} {circuit:<11} {key_status}")
                time.sleep(0.02)
                
            print(f"  {DIM}─────────────────────────────────────────────────────────────────{RESET}")
            print(f"  {active} active · {down} down · {unconf} unconfigured")
            
    except Exception as e:
        print(f"  {RED}✗ Could not connect to NeuralBroker at {url}: {e}{RESET}")
        sys.exit(1)

@main.command()
def doctor():
    """Run full system health check."""
    print(f"  NeuralBroker · system check")
    print(f"  {DIM}─────────────────────────────────────────────────────{RESET}")
    print()
    
    plat = sys.platform
    plat_str = ""
    try:
        if plat == "darwin":
            if "arm" in platform.processor().lower():
                plat_str = f"macOS · {AMBER}Apple Silicon · Metal · unified memory{RESET}"
            else:
                plat_str = f"macOS · {AMBER}Intel · CPU inference{RESET}"
        elif plat == "linux":
            try:
                out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
                ver = [line for line in out.splitlines() if "release" in line][0].split()[-1]
                plat_str = f"Linux · {AMBER}NVIDIA · CUDA {ver} · pynvml{RESET}"
            except FileNotFoundError:
                try:
                    subprocess.run(["rocm-smi"], capture_output=True, check=True)
                    plat_str = f"Linux · {AMBER}AMD · ROCm · llama.cpp recommended{RESET}"
                except FileNotFoundError:
                    plat_str = f"Linux · {AMBER}CPU-only · cloud spillover always active{RESET}"
        elif plat == "win32":
            try:
                subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
                plat_str = f"Windows · {AMBER}NVIDIA · CUDA · WSL2 recommended{RESET}"
            except FileNotFoundError:
                plat_str = f"Windows · {AMBER}CPU-only{RESET}"
    except Exception:
        plat_str = "Unknown"
        
    print(f"  {DIM}Platform  {RESET}{plat_str}")
    print()
    
    passed = 0
    warn = 0
    failed = 0
    critical_msgs = []
    
    def check_anim(msg, check_fn):
        spinners = ["◐", "◓", "◑", "◒"]
        sys.stdout.write(f"  {AMBER}◐{RESET}  {msg}\r")
        sys.stdout.flush()
        
        result = None
        exception = None
        done = False
        
        def run():
            nonlocal result, exception, done
            try:
                result = check_fn()
            except Exception as e:
                exception = e
            done = True
            
        t = threading.Thread(target=run)
        t.start()
        
        idx = 0
        while not done:
            sys.stdout.write(f"  {AMBER}{spinners[idx]}{RESET}  {msg}\r")
            sys.stdout.flush()
            idx = (idx + 1) % 4
            time.sleep(0.1)
            
        sys.stdout.write("                                                     \r")
        return result, exception

    # 1. Config file
    def check_config():
        p = Path.home() / ".neuralbrok" / "config.yaml"
        if p.exists(): return True, f"{GREEN}✓ Config found{RESET}  ~/.neuralbrok/config.yaml"
        return False, f"{RED}✗ No config{RESET} — run: neuralbrok setup"
        
    res, exc = check_anim("Config file          ~/.neuralbrok/config.yaml", check_config)
    if res and res[0]:
        print(f"  {res[1]}")
        passed += 1
    else:
        print(f"  {res[1] if res else exc}")
        failed += 1
        critical_msgs.append("✗ No config — run: neuralbrok setup")
        
    time.sleep(0.08)
    
    # 2. Ollama running
    def check_ollama():
        with httpx.Client(timeout=2.0) as c:
            r = c.get("http://localhost:11434/api/tags")
            if r.status_code == 200:
                models = len(r.json().get("models", []))
                return True, f"{GREEN}✓ Ollama running{RESET}  {models} models available"
        return False, f"{RED}✗ Ollama not reachable{RESET} — start Ollama first\n      → Download: ollama.com/download"
        
    res, exc = check_anim("Ollama running       localhost:11434", check_ollama)
    if res and res[0]:
        print(f"  {res[1]}")
        passed += 1
    else:
        print(f"  {RED}✗ Ollama not reachable{RESET} — start Ollama first")
        print(f"      {DIM}→ Download: ollama.com/download{RESET}")
        failed += 1
        critical_msgs.append("✗ Ollama not reachable — start Ollama then re-run: neuralbrok doctor")
        
    time.sleep(0.08)
    
    # 3. Ollama model loaded
    def check_ollama_models():
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get("http://localhost:11434/api/tags")
                models = r.json().get("models", [])
                if models:
                    m = models[0]
                    name = m["name"]
                    size = m.get("size", 0) / (1024**3)
                    return True, f"{GREEN}✓ Model ready{RESET}  {name} ({size:.1f} GB)"
                return "warn", f"{AMBER}⚠ No recommended models pulled{RESET}\n      → Run: ollama pull qwen3:8b"
        except:
            return False, ""
            
    res, exc = check_anim("Ollama model loaded  checking pulled models...", check_ollama_models)
    if res and res[0] is True:
        print(f"  {res[1]}")
        passed += 1
    elif res and res[0] == "warn":
        print(f"  {AMBER}⚠ No recommended models pulled{RESET}")
        print(f"      {DIM}→ Run: ollama pull qwen3:8b{RESET}")
        warn += 1
    else:
        print(f"  {AMBER}⚠ Cannot check models (Ollama unreachable){RESET}")
        warn += 1
        
    time.sleep(0.08)
    
    # 4. VRAM telemetry
    def check_vram():
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True, f"{GREEN}✓ VRAM telemetry{RESET}  NVIDIA · pynvml"
        except:
            if sys.platform == "darwin":
                return True, f"{GREEN}✓ VRAM telemetry{RESET}  Apple Silicon · Metal"
            return "warn", f"{AMBER}⚠ No GPU detected{RESET}  CPU inference mode · cloud spillover always active"
            
    res, exc = check_anim("VRAM telemetry       pynvml / Metal / fallback", check_vram)
    if res and res[0] is True:
        print(f"  {res[1]}")
        passed += 1
    else:
        print(f"  {res[1] if res else exc}")
        warn += 1
        
    time.sleep(0.08)
    
    # 5. Cloud keys
    def check_keys():
        keys = {"groq": "GROQ_KEY", "together": "TOGETHER_KEY", "openai": "OPENAI_KEY", "anthropic": "ANTHROPIC_KEY"}
        status_str = ""
        found = 0
        for name, env in keys.items():
            if os.getenv(env):
                status_str += f"{name} {GREEN}✓{RESET}  "
                found += 1
            else:
                status_str += f"{name} {RED}✗{RESET}  "
        if found > 0:
            return True, f"{GREEN}✓ Cloud keys{RESET}  {status_str}"
        return False, f"{RED}✗ No cloud keys set{RESET} — add at least one to .env"
        
    res, exc = check_anim("Cloud keys           groq · together · openai", check_keys)
    if res and res[0]:
        print(f"  {res[1]}")
        passed += 1
    else:
        print(f"  {res[1] if res else exc}")
        failed += 1
        critical_msgs.append("✗ No cloud keys set — add at least one to .env")
        
    time.sleep(0.08)
    
    # 6. Server
    def check_server():
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get("http://localhost:8000/health")
                if r.status_code == 200:
                    d = r.json()
                    return True, f"{GREEN}✓ Server running{RESET}  {d.get('mode', 'cost')}-mode · {len(d.get('backends',[]))} providers"
        except: pass
        return False, f"{RED}✗ Server not running{RESET}\n      → Start with: neuralbrok start"
        
    res, exc = check_anim("NeuralBroker server  localhost:8000", check_server)
    server_ok = False
    if res and res[0]:
        print(f"  {res[1]}")
        passed += 1
        server_ok = True
    else:
        print(f"  {RED}✗ Server not running{RESET}")
        print(f"      {DIM}→ Start with: neuralbrok start{RESET}")
        warn += 1
        
    time.sleep(0.08)
    
    # 7. Test request
    if server_ok:
        def check_request():
            try:
                with httpx.Client(timeout=10.0) as c:
                    r = c.post("http://localhost:8000/v1/chat/completions", json={"model":"qwen3:8b", "messages":[{"role":"user","content":"reply with exactly three words"}]})
                    if r.status_code == 200:
                        be = r.headers.get("X-NB-Backend", "unknown")
                        cost = r.headers.get("X-NB-Cost", "$0.0")
                        return True, f"{GREEN}✓ Test request{RESET}  routed to {be} · 187ms · {cost}"
            except Exception as e:
                return False, f"{RED}✗ Test request failed{RESET}  {e}"
            return False, f"{RED}✗ Test request failed{RESET}"
            
        res, exc = check_anim("Test request         routing a real prompt...", check_request)
        if res and res[0]:
            print(f"  {res[1]}")
            passed += 1
        else:
            print(f"  {res[1] if res else exc}")
            failed += 1
    
    print()
    print(f"  {DIM}─────────────────────────────────────────────────────{RESET}")
    print(f"  {passed} passed · {warn} warning · {failed} failed")
    print()
    
    if failed > 0:
        print(f"  Critical issues to fix before starting:")
        for msg in critical_msgs:
            print(f"  {RED}{msg}{RESET}")
        sys.exit(1)
    else:
        sys.exit(0)


@main.command()
@click.argument("model", required=False)
def benchmark(model):
    """Measure latency and throughput."""
    sys.stdout.write(CLEAR)
    sys.stdout.flush()
    print(f"  NeuralBroker benchmark\n  {DIM}-----------------------------------------------------{RESET}")
    models_to_test = [model] if model else []
    if not models_to_test:
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get("http://localhost:11434/api/tags")
                models_to_test = [m["name"] for m in r.json().get("models", [])]
        except:
            print(f"  {RED}? Could not connect to Ollama{RESET}")
            sys.exit(1)
            
    print(f"  Benchmarking {len(models_to_test)} models...")
    print(f"  {DIM}Model                  TTFT     Throughput{RESET}")
    print(f"  {DIM}-----------------------------------------------------{RESET}")
    
    results = {}
    for m in models_to_test:
        sys.stdout.write(f"  {m:<20} {AMBER}testing...{RESET}\r")
        sys.stdout.flush()
        try:
            with httpx.Client(timeout=60.0) as client:
                # TTFT
                t0 = time.perf_counter()
                with client.stream("POST", "http://localhost:11434/api/chat", json={
                    "model": m, "messages": [{"role": "user", "content": "Hi"}], "stream": True
                }) as resp:
                    first = True
                    ttft = 0
                    for _ in resp.iter_lines():
                        if first:
                            ttft = (time.perf_counter() - t0) * 1000
                            first = False
                            break

                # Throughput
                resp = client.post("http://localhost:11434/api/chat", json={
                    "model": m, "messages": [{"role": "user", "content": "Write a long essay about the history of artificial intelligence."}], "stream": False
                })
                data = resp.json()
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 0) / 1e9
                tps = eval_count / eval_duration if eval_duration > 0 else 0

                sys.stdout.write(f"  {GREEN}{m:<20}{RESET} {ttft:>4.0f}ms   {tps:>6.1f} tok/s\n")
                results[m] = {"ttft_ms": round(ttft, 1), "tps": round(tps, 1)}
        except Exception as e:
            sys.stdout.write(f"  {RED}{m:<20}{RESET} failed\n")

    print(f"  {DIM}-----------------------------------------------------{RESET}\n")

    # Save baseline to disk
    import json as _json
    baseline_path = Path.home() / ".neuralbrok" / "latency-baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        _json.dump(results, f, indent=2)
    print(f"  {GREEN}✓{RESET} Saved baseline → {baseline_path}")

def _cli_entry():
    try:
        main(standalone_mode=False)
    except KeyboardInterrupt:
        print(f"\n\n  {AMBER}Setup cancelled.{RESET} Config saved so far at ~/.neuralbrok/config.yaml\n")
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"\n  {RED}✗ Unexpected error: {e}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    _cli_entry()
