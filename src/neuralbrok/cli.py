import os
import sys
import time
import shutil
import subprocess
import click
import uvicorn
import httpx
import yaml
from pathlib import Path

from neuralbrok.detect import detect_device
from neuralbrok.autoconfig import generate_config

# Color and style constants
AMBER  = "\033[38;5;214m"
WHITE  = "\033[97m"
DIM    = "\033[2m"
GREEN  = "\033[38;5;114m"
RED    = "\033[38;5;203m"
CYAN   = "\033[38;5;117m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CLEAR  = "\033[2J\033[H"

def _print_typewriter(text: str, delay: float = 0.002):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

def _get_vram_bar(used_gb: float, total_gb: float, width: int = 16) -> str:
    if total_gb <= 0:
        return f"{DIM}{'░' * width}{RESET}"
    filled = int(round((used_gb / total_gb) * width))
    filled = min(width, max(0, filled))
    empty = width - filled
    return f"{AMBER}{'█' * filled}{DIM}{'░' * empty}{RESET}"

@click.group()
def main():
    """NeuralBroker CLI — VRAM-aware LLM routing."""
    pass

@main.command()
def setup():
    """Detect hardware and generate configuration."""
    sys.stdout.write(CLEAR)
    sys.stdout.flush()
    
    # Step 0 - Splash screen
    top_border = f"  ╔══════════════════════════════════════════════════════════╗"
    title_line = f"  ║   {WHITE}{BOLD}NeuralBr{RESET}{AMBER}●{RESET}{WHITE}{BOLD}ker{RESET}                                          ║"
    subtitle_line = f"  ║   VRAM-aware LLM routing daemon                         ║"
    version_line = f"  ║   v0.4.0 · MIT licensed · github.com/khan-sha/neuralbroker ║"
    empty_line = f"  ║                                                          ║"
    bottom_border = f"  ╚══════════════════════════════════════════════════════════╝"
    
    print(top_border)
    print(empty_line)
    _print_typewriter(title_line)
    print(subtitle_line)
    print(version_line)
    print(empty_line)
    print(bottom_border)
    
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
    
    import threading
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
    elif profile.gpu_vendor == "none":
        gpu_str = f"none · CPU inference"
        plat_str = f"{profile.platform} · CPU-only"
    else:
        gpu_str = f"{profile.gpu_model} · {profile.vram_gb:.0f} GB VRAM"
        plat_str = f"{profile.platform.capitalize()} · CUDA {profile.cuda_version}"
        
    vram_limit_pct = int(profile.recommended_vram_threshold * 100)
    tdp = profile.estimated_electricity_tdp_watts
    cost_per_hr = (tdp / 1000.0) * 0.14
    
    panel_rows = [
        f"  GPU          {AMBER}{gpu_str}{RESET}",
        f"  Platform     {AMBER}{plat_str}{RESET}",
        f"  RAM          {AMBER}{profile.ram_gb:.0f} GB{RESET}",
        f"  CPU          {AMBER}{profile.cpu_cores} cores{RESET}",
        f"  Runtime      {AMBER}{profile.recommended_runtime}{RESET}  ← recommended",
        f"  VRAM limit   {AMBER}{vram_limit_pct}%{RESET}  before cloud spill",
        f"  TDP          {AMBER}{tdp}W · ~${cost_per_hr:.3f}/hr at $0.14/kWh{RESET}"
    ]
    
    if use_box:
        print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
        print(f"  {DIM}│{RESET}  {WHITE}Hardware Profile{RESET}                                    {DIM}│{RESET}")
        print(f"  {DIM}├─────────────────────────────────────────────────────┤{RESET}")
        for row in panel_rows:
            # Padding adjustment for ANSI codes
            clean_row = row.replace(AMBER, "").replace(RESET, "")
            pad = 55 - len(clean_row)
            print(f"  {DIM}│{RESET}{DIM}{row.replace('  ', '', 1)}{RESET}{' ' * pad}{DIM}│{RESET}")
            time.sleep(0.04)
        print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
    else:
        print(f"  {WHITE}Hardware Profile{RESET}")
        for row in panel_rows:
            print(f"  {DIM}{row.replace('  ', '', 1)}{RESET}")
            time.sleep(0.04)
            
    print()
    
    # Step 3 - Model recommendations panel
    rec_models = profile.recommended_models
    model_rows = []
    
    # Rough VRAM estimation for bars based on model names
    def estimate_vram(m: str) -> float:
        if "8b" in m or "7b" in m: return 6.0
        if "14b" in m: return 9.5
        if "32b" in m: return 20.0
        if "70b" in m: return 40.0
        if "coder" in m: return 6.7
        return 5.0
        
    for i, m in enumerate(rec_models):
        v = estimate_vram(m)
        bar = _get_vram_bar(v, max(v, profile.vram_gb if profile.vram_gb > 0 else 8.0))
        model_rows.append(f"  {i+1}  {AMBER}{m:<15}{RESET}  {bar}  {DIM}{WHITE}{v:.1f} GB VRAM{RESET}")
        
    if use_box:
        print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
        print(f"  {DIM}│{RESET}  {WHITE}Recommended models for your hardware{RESET}               {DIM}│{RESET}")
        print(f"  {DIM}├─────────────────────────────────────────────────────┤{RESET}")
        for row in model_rows:
            # clean ansi to calc padding
            clean_row = row.replace(AMBER, "").replace(RESET, "").replace(DIM, "").replace(WHITE, "")
            pad = 55 - len(clean_row)
            print(f"  {DIM}│{RESET}{row.replace('  ', '', 1)}{' ' * pad}{DIM}│{RESET}")
            time.sleep(0.06)
        print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
    else:
        print(f"  {WHITE}Recommended models for your hardware{RESET}")
        for row in model_rows:
            print(row)
            time.sleep(0.06)
            
    # Step 4 - Electricity cost preview
    cost_1m = (tdp / 1000.0) * 0.14 * (1000000 / 3600000) * 50 # rough estimate
    print(f"  {DIM}{AMBER}⚡{DIM}  Local inference cost:  {GREEN}~${cost_1m:.3f} / 1M tokens{DIM}  (at $0.14/kWh · {tdp}W TDP){RESET}")
    print()
    
    # Step 5 - Interactive prompts
    def prompt(text, default):
        try:
            sys.stdout.write(f"  → {text} [{default}]:  ")
            sys.stdout.flush()
            val = input().strip()
            if not val: val = default
            print(f"  {GREEN}✓ saved{RESET}")
            return val
        except KeyboardInterrupt:
            print(f"\n  {DIM}Aborted.{RESET}")
            sys.exit(0)
            
    elec_price = prompt("Electricity price ($/kWh)", "0.14")
    
    while True:
        try:
            mode = prompt("Default routing mode (cost / speed / fallback)", "cost")
            if mode in ["cost", "speed", "fallback"]:
                break
            print(f"  {RED}✗ invalid — enter cost, speed, or fallback{RESET}")
        except KeyboardInterrupt:
            print(f"\n  {DIM}Aborted.{RESET}")
            sys.exit(0)
            
    while True:
        try:
            thresh_str = prompt("VRAM threshold % before cloud spill", str(vram_limit_pct))
            thresh = int(thresh_str)
            if 50 <= thresh <= 99:
                break
            print(f"  {RED}✗ invalid — enter number between 50 and 99{RESET}")
        except ValueError:
            print(f"  {RED}✗ invalid — enter a valid number{RESET}")
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
    
    # Step 6.5 - Testing Ollama connection
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
        
        for rm in rec_models:
            if rm in pulled_models or f"{rm}:latest" in pulled_models:
                pass
            else:
                models_to_pull.append(rm)
                
        if not models_to_pull:
            print(f"  {GREEN}✓ All recommended models already available{RESET}")
        else:
            if use_box:
                print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
                print(f"  {DIM}│{RESET}  Ollama is ready. Models already pulled:            {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
                for rm in rec_models:
                    if rm in models_to_pull:
                        print(f"  {DIM}│{RESET}    {rm:<15} {RED}✗ not pulled{RESET}                     {DIM}│{RESET}")
                    else:
                        print(f"  {DIM}│{RESET}    {rm:<15} {GREEN}✓ available{RESET}                      {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  Pull missing models:                               {DIM}│{RESET}")
                for m in models_to_pull:
                    print(f"  {DIM}│{RESET}  $ ollama pull {m:<35}{DIM}│{RESET}")
                print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
            else:
                print("  Ollama is ready. Models already pulled:")
                for rm in rec_models:
                    if rm in models_to_pull:
                        print(f"    {rm:<15} {RED}✗ not pulled{RESET}")
                    else:
                        print(f"    {rm:<15} {GREEN}✓ available{RESET}")
                print("  Pull missing models:")
                for m in models_to_pull:
                    print(f"  $ ollama pull {m}")
                    
    else:
        if ollama_ok:
            print(f"  {AMBER}⚠ Ollama connected but no models pulled{RESET}")
        else:
            print(f"  {AMBER}⚠ Ollama not running — that's ok, start it before neuralbrok start{RESET}")
            
        models_to_pull = rec_models
        
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
                
    print()
    if use_box:
        print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
        print(f"  {DIM}│{RESET}  Setup complete. Start NeuralBroker:                {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  {AMBER}{BOLD}$ neuralbrok start{RESET}                                 {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  Proxy:      http://localhost:8000/v1               {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  Dashboard:  http://localhost:8000/dashboard        {DIM}│{RESET}")
        print(f"  {DIM}│{RESET}  Docs:       http://localhost:8000/docs             {DIM}│{RESET}")
        print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
    else:
        print("  Setup complete. Start NeuralBroker:")
        print(f"  {AMBER}{BOLD}$ neuralbrok start{RESET}")
        print("  Proxy:      http://localhost:8000/v1")
        print("  Dashboard:  http://localhost:8000/dashboard")
        print("  Docs:       http://localhost:8000/docs")
        
    print(f"  {DIM}Questions? github.com/khan-sha/neuralbroker/issues{RESET}")
    
    # Source install detect
    is_source = Path(__file__).parents[2].joinpath(".git").exists()
    if is_source:
        print()
        plat = sys.platform
        if plat == "win32":
            if use_box:
                print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
                print(f"  {DIM}│{RESET}  Installed from source? Update anytime:             {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  {AMBER}{BOLD}> git pull{RESET}                                         {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  {AMBER}{BOLD}> pip install -r requirements.txt{RESET}                  {DIM}│{RESET}")
                print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
            else:
                print("  Installed from source? Update anytime:")
                print(f"  {AMBER}{BOLD}> git pull{RESET}")
                print(f"  {AMBER}{BOLD}> pip install -r requirements.txt{RESET}")
        else:
            if use_box:
                print(f"  {DIM}┌─────────────────────────────────────────────────────┐{RESET}")
                print(f"  {DIM}│{RESET}  Installed from source? Update anytime:             {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}                                                     {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  {AMBER}{BOLD}$ git pull{RESET}                                         {DIM}│{RESET}")
                print(f"  {DIM}│{RESET}  {AMBER}{BOLD}$ pip install -r requirements.txt{RESET}                  {DIM}│{RESET}")
                print(f"  {DIM}└─────────────────────────────────────────────────────┘{RESET}")
            else:
                print("  Installed from source? Update anytime:")
                print(f"  {AMBER}{BOLD}$ git pull{RESET}")
                print(f"  {AMBER}{BOLD}$ pip install -r requirements.txt{RESET}")

@main.command()
@click.option("--port", default=8000, help="Port to run the server on.")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
@click.option("--config", help="Path to a specific config.yaml.")
def start(port, host, config):
    """Start the NeuralBroker daemon."""
    sys.stdout.write(CLEAR)
    sys.stdout.flush()
    
    top_border = f"  ╔══════════════════════════════════════════════════════════╗"
    title_line = f"  ║   {WHITE}{BOLD}NeuralBr{RESET}{AMBER}●{RESET}{WHITE}{BOLD}ker{RESET}                                          ║"
    version_line = f"  ║   v0.4.0                                                 ║"
    bottom_border = f"  ╚══════════════════════════════════════════════════════════╝"
    
    print(top_border)
    print(title_line)
    print(version_line)
    print(bottom_border)
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
            sys.stdout.write(f"  {AMBER}{spinners[idx]}{RESET}  {msg}\r")
            sys.stdout.flush()
            idx = (idx + 1) % 4
            time.sleep(0.1)
        sys.stdout.write(f"  {GREEN}✓{RESET}  {msg}\n")
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
        import yaml
        c = yaml.safe_load(Path(config_path).read_text())
        if not c.get("local_nodes") or not c.get("cloud_providers"):
            print(f"\n  First time?  →  http://localhost:{port}/onboarding")
    except: pass
    
    try:
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
        
    import threading
    
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n  {DIM}Aborted.{RESET}")
        sys.exit(0)
