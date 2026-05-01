"""
MCP (Model Context Protocol) server for NeuralBroker.

Exposes NeuralBroker's capabilities as MCP tools for Claude Code,
Cursor, and other MCP-compatible clients. Supports stdio transport.

Usage:
    neuralbrok mcp          # stdio mode (for Claude Code)
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TOOLS = [
    {"name": "nb_chat", "description": "Send a chat completion through NeuralBroker routing.",
     "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}, "model": {"type": "string"}, "temperature": {"type": "number"}}, "required": ["message"]}},
    {"name": "nb_route", "description": "Preview routing decision without executing.",
     "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    {"name": "nb_recommend", "description": "Get llmfit-scored model recommendations for your hardware.",
     "inputSchema": {"type": "object", "properties": {"use_case": {"type": "string"}, "max_results": {"type": "integer"}}}},
    {"name": "nb_hardware", "description": "Get detected hardware profile.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "nb_providers", "description": "List configured providers with health.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "nb_set_mode", "description": "Change routing mode (cost/speed/fallback/smart).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["cost","speed","fallback","smart"]}}, "required": ["mode"]}},
    {"name": "nb_stats", "description": "Get routing statistics.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "nb_vram", "description": "Get current VRAM utilization.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "nb_agent_list", "description": "List available agents.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "nb_agent_run", "description": "Execute a task through a NeuralBroker agent.",
     "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}, "agent": {"type": "string"}}, "required": ["task"]}},
    {"name": "nb_swarm_create", "description": "Create multi-agent swarm for complex objectives.",
     "inputSchema": {"type": "object", "properties": {"objective": {"type": "string"}}, "required": ["objective"]}},
    {"name": "nb_model_fit", "description": "Run llmfit scoring on a specific model.",
     "inputSchema": {"type": "object", "properties": {"model": {"type": "string"}, "use_case": {"type": "string"}}, "required": ["model"]}},
    {"name": "nb_model_download", "description": "Download a model via Ollama.",
     "inputSchema": {"type": "object", "properties": {"model": {"type": "string"}}, "required": ["model"]}},
]

SERVER_INFO = {"name": "neuralbroker", "version": "2.0.0"}
CAPABILITIES = {"tools": {}}

async def _call_nb(method: str, path: str, body: dict = None) -> dict:
    nb_url = os.environ.get("NB_URL", "http://localhost:8000")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("NB_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                r = await client.get(f"{nb_url}{path}", headers=headers)
            else:
                r = await client.post(f"{nb_url}{path}", headers=headers, json=body or {})
            return r.json()
    except Exception as e:
        return {"error": str(e)}

async def handle_tool(name: str, args: dict) -> list[dict]:
    if name == "nb_chat":
        r = await _call_nb("POST", "/v1/chat/completions", {
            "model": args.get("model", "auto"), "messages": [{"role": "user", "content": args["message"]}],
            "temperature": args.get("temperature", 0.7), "stream": False})
        try: text = r["choices"][0]["message"]["content"]
        except: text = json.dumps(r, indent=2)
        return [{"type": "text", "text": text}]
    elif name == "nb_route":
        from neuralbrok.orchestrator import AgentRouter, agent_decision_to_dict
        d = await AgentRouter().route(args["message"])
        return [{"type": "text", "text": json.dumps(agent_decision_to_dict(d), indent=2)}]
    elif name == "nb_recommend":
        from neuralbrok.llmfit_scorer import rank_models, detect_system_specs, model_fit_to_dict
        fits = rank_models(detect_system_specs(), use_case=args.get("use_case","general"), max_results=args.get("max_results",10))
        return [{"type": "text", "text": json.dumps([model_fit_to_dict(f) for f in fits], indent=2)}]
    elif name == "nb_hardware":
        from neuralbrok.llmfit_scorer import detect_system_specs
        s = detect_system_specs()
        return [{"type": "text", "text": json.dumps({"gpu": s.gpu_name, "vram_gb": s.vram_gb, "bandwidth_gbps": s.bandwidth_gbps, "ram_gb": s.ram_gb, "cpu_cores": s.cpu_cores, "runtimes": {"ollama": s.ollama_available, "llama_cpp": s.llamacpp_available, "lm_studio": s.lmstudio_available}}, indent=2)}]
    elif name in ("nb_providers", "nb_stats", "nb_vram"):
        path_map = {"nb_providers": "/nb/providers", "nb_stats": "/nb/stats", "nb_vram": "/nb/vram"}
        return [{"type": "text", "text": json.dumps(await _call_nb("GET", path_map[name]), indent=2)}]
    elif name == "nb_set_mode":
        return [{"type": "text", "text": json.dumps(await _call_nb("POST", "/nb/mode", {"mode": args["mode"]}), indent=2)}]
    elif name == "nb_agent_list":
        from neuralbrok.agents import list_agents
        return [{"type": "text", "text": json.dumps([{"slug": a.slug, "name": a.name, "role": a.role, "icon": a.icon, "capabilities": a.capabilities} for a in list_agents()], indent=2)}]
    elif name == "nb_agent_run":
        from neuralbrok.orchestrator import AgentRouter; from neuralbrok.agents import get_agent
        agent = get_agent(args["agent"]) if args.get("agent") else (await AgentRouter().route(args["task"])).agent
        if not agent: return [{"type": "text", "text": f"Agent not found"}]
        r = await _call_nb("POST", "/v1/chat/completions", {"model": "auto", "messages": [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": args["task"]}], "temperature": agent.temperature, "stream": False})
        try: text = r["choices"][0]["message"]["content"]
        except: text = json.dumps(r, indent=2)
        return [{"type": "text", "text": f"[{agent.icon} {agent.name}]\n\n{text}"}]
    elif name == "nb_swarm_create":
        from neuralbrok.orchestrator import AgentRouter, SwarmCoordinator, swarm_to_dict
        router = AgentRouter(); coord = SwarmCoordinator(router)
        swarm = coord.create_swarm(args["objective"]); await coord.decompose(swarm)
        asyncio.create_task(coord.execute_swarm(swarm))
        return [{"type": "text", "text": json.dumps(swarm_to_dict(swarm), indent=2)}]
    elif name == "nb_model_fit":
        from neuralbrok.llmfit_scorer import detect_system_specs, score_model, model_fit_to_dict
        from neuralbrok.models import FALLBACK_MODELS
        hw = detect_system_specs(); matching = [m for m in FALLBACK_MODELS if args["model"].lower() in m.name.lower()]
        if not matching: return [{"type": "text", "text": f"Model not found"}]
        return [{"type": "text", "text": json.dumps([model_fit_to_dict(score_model(m, hw, use_case=args.get("use_case","general"))) for m in matching[:5]], indent=2)}]
    elif name == "nb_model_download":
        try:
            async with httpx.AsyncClient(timeout=300.0) as c:
                r = await c.post("http://localhost:11434/api/pull", json={"name": args["model"], "stream": False})
                return [{"type": "text", "text": f"✓ Downloaded '{args['model']}'" if r.status_code == 200 else f"Failed: {r.text}"}]
        except Exception as e: return [{"type": "text", "text": f"Error: {e}"}]
    return [{"type": "text", "text": f"Unknown tool: {name}"}]

async def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", ""); msg_id = msg.get("id"); params = msg.get("params", {})
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": "2024-11-05", "serverInfo": SERVER_INFO, "capabilities": CAPABILITIES}}
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        try:
            content = await handle_tool(params.get("name",""), params.get("arguments",{}))
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": content}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}}
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None

async def run_stdio():
    logger.info("NeuralBroker MCP server starting (stdio)")
    
    # Fallback for environments where connect_read_pipe/connect_write_pipe fails (e.g. some Win environments)
    try:
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
        wt, wp = await asyncio.get_event_loop().connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout.buffer)
        writer = asyncio.StreamWriter(wt, wp, None, asyncio.get_event_loop())
        
        while True:
            line = await reader.readline()
            if not line: break
            try:
                msg = json.loads(line.decode("utf-8").strip())
                resp = await handle_message(msg)
                if resp:
                    writer.write(json.dumps(resp).encode("utf-8") + b"\n")
                    await writer.drain()
            except json.JSONDecodeError: continue
            except Exception as e: logger.error(f"MCP error: {e}")
            
    except (OSError, AttributeError, NotImplementedError):
        # Synchronous fallback for restricted pipes
        logger.info("Using synchronous fallback for stdio transport")
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            if not line: break
            try:
                msg = json.loads(line.decode("utf-8").strip())
                resp = await handle_message(msg)
                if resp:
                    out = (json.dumps(resp) + "\n").encode("utf-8")
                    await loop.run_in_executor(None, sys.stdout.buffer.write, out)
                    await loop.run_in_executor(None, sys.stdout.buffer.flush)
            except json.JSONDecodeError: continue
            except Exception as e: logger.error(f"MCP fallback error: {e}")

def main():
    asyncio.run(run_stdio())

if __name__ == "__main__":
    main()
