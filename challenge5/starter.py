# ============================================================
# Challenge 5 — MCP Chatbot using Strands SDK + Ollama
# Model  : llama3.2:3b  (runs 100% locally, no API keys needed)
# MCP    : Model Context Protocol — local stdio MCP server
# ============================================================
#
# WHAT IS MCP?
# ────────────
# The Model Context Protocol (MCP) is an open standard that lets
# AI agents connect to external tools through a well-defined
# client–server interface.  Instead of defining tools with Python
# @tool decorators (Challenge 2–4), MCP tools live in a separate
# server process.  The agent speaks to that server over stdio (or
# HTTP/SSE) to discover and call tools dynamically.
#
# HOW THIS CHALLENGE WORKS
# ────────────────────────
#  1. We launch a local MCP server (mcp_server.py) as a subprocess.
#     It exposes three tools: add, get_weather, get_system_info.
#  2. Strands connects to that server via MCPClient (stdio transport).
#  3. The agent uses MCPClient.get_tools() to discover all tools
#     the server offers — no hardcoding required.
#  4. Tool calls flow:  agent → MCP client → subprocess stdin/stdout
#     → tool runs → result returned to agent → LLM formats reply.
#
# TWO FILES IN THIS CHALLENGE
# ────────────────────────────
#  • mcp_server.py   — the MCP server (run automatically)
#  • starter.py      — this file, the MCP chatbot (run manually)
#
# SETUP — run these commands once before the first start:
# ────────────────────────────────────────────────────────────
#   pip install strands-agents strands-agents-tools
#   pip install mcp                      ← MCP Python SDK
#
# Ollama must already be running with llama3.2:3b pulled:
#   ollama serve                         (starts the Ollama server)
#   ollama pull llama3.2:3b
#
# Then run this chatbot with:
#   python challenge5/starter.py
# ============================================================


# ── Standard library ─────────────────────────────────────────
import sys          # used to detect Python executable path
import os           # used to build the path to mcp_server.py

# ── Strands SDK ───────────────────────────────────────────────
# Agent      → core class managing the conversation + tool loop
# OllamaModel → connects the Agent to a local Ollama instance
from strands import Agent
from strands.models.ollama import OllamaModel

# ── Strands MCP integration ───────────────────────────────────
# MCPClient    → connects Strands to any MCP server
# stdio_client → async context manager that opens subprocess pipes;
#                this is the actual transport object MCPClient needs
# StdioServerParameters → dataclass that describes HOW to launch
#                         the server (command, args, env)
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client


# ════════════════════════════════════════════════════════════
# SECTION 1 — MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════
# Same OllamaModel pattern used across all challenges.
# Temperature 0.3 keeps tool-calling deterministic while still
# allowing natural, conversational prose in answers.

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,
)


# ════════════════════════════════════════════════════════════
# SECTION 2 — MCP SERVER PARAMETERS + TRANSPORT FACTORY
# ════════════════════════════════════════════════════════════
#
# Two things are needed:
#
#  a) StdioServerParameters — a dataclass describing HOW to
#     launch the server subprocess (command, args, env).
#
#  b) A transport factory — a zero-argument callable that
#     returns an ASYNC CONTEXT MANAGER yielding the read/write
#     streams MCPClient needs.  That callable is:
#
#       lambda: stdio_client(server_params)
#
#     stdio_client() is the async context manager from the mcp
#     library.  It spawns the subprocess and wires up its
#     stdin/stdout as anyio memory streams.
#
# Common mistake: passing StdioServerParameters directly to
# MCPClient.  That fails with:
#   TypeError: 'StdioServerParameters' object does not support
#   the asynchronous context manager protocol
# because MCPClient expects the async CM, not the params object.

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")

server_params = StdioServerParameters(
    command=sys.executable,   # same Python interpreter → same venv
    args=[SERVER_SCRIPT],     # the MCP server script to launch
    env=None,                 # inherit current environment
)

# transport_factory is the callable MCPClient accepts.
# It returns stdio_client(server_params) which IS an async CM.
transport_factory = lambda: stdio_client(server_params)  # noqa: E731


# ════════════════════════════════════════════════════════════
# SECTION 3 — SYSTEM PROMPT
# ════════════════════════════════════════════════════════════
# Tells the agent about its role and the MCP tools it has access
# to.  We describe the tools here so the LLM knows when to call
# them — even before it reads their formal JSON schemas.

SYSTEM_PROMPT = """You are a helpful AI assistant connected to a local MCP (Model Context Protocol) server.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MCP TOOLS AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• add              — add two numbers together
• get_weather      — get the current weather for a city
• get_system_info  — retrieve information about this machine

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIOUR RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always prefer calling an MCP tool when the user's question matches one.
- When you call a tool, explain what you are doing before showing the result.
- Be concise, friendly, and helpful.
- If a tool call fails, explain the error clearly and suggest a fix.
"""


# ════════════════════════════════════════════════════════════
# SECTION 4 — MCP CLIENT + AGENT SETUP
# ════════════════════════════════════════════════════════════
#
# MCPClient is a context manager that:
#   - Starts the MCP server subprocess on __enter__.
#   - Performs the MCP handshake (initialize → list_tools).
#   - Exposes get_tools() which returns Strands-compatible tool
#     wrappers around every tool the server declares.
#   - Terminates the subprocess on __exit__.
#
# We wrap the entire chat session inside the MCPClient context
# so the server stays alive for the full duration of the session.

print("=" * 62)
print("  MCP Chatbot — Strands SDK + Ollama  (llama3.2:3b)")
print("  MCP Transport : stdio (local subprocess)")
print("=" * 62)
print()
print("  EXAMPLE QUERIES")
print("    ➕  'What is 42 plus 58?'")
print("    🌦  'What is the weather in London?'")
print("    💻  'Tell me about this system'")
print("    🔍  'What MCP tools do you have?'")
print()
print("  Type 'tools' to list connected MCP tools.")
print("  Type 'quit'  to exit.")
print("=" * 62)
print()

# ── Start the MCP client (launches the server subprocess) ────
print("⏳  Connecting to MCP server …", flush=True)

# Pass transport_factory — the lambda that returns stdio_client(...).
# MCPClient calls this factory internally to open the async transport.
# Do NOT pass server_params directly — it is not an async context manager.
mcp_client = MCPClient(transport_factory)

# Enter the context manager — this starts the server subprocess
# and performs the MCP protocol handshake.
with mcp_client:

    # ── Discover tools from the MCP server ───────────────────
    # list_tools_sync() is the synchronous method that queries the
    # server's tool registry over the MCP protocol and returns a
    # PaginatedList of MCPAgentTool objects.
    #
    # MCPAgentTool is a subclass of AgentTool, so the list can be
    # passed directly into Agent(tools=...) without any conversion.
    #
    # Method reference (strands-agents 1.42):
    #   load_tools()      → async coroutine, must be awaited
    #   list_tools_sync() → synchronous, safe to call here  ✅
    mcp_tools = mcp_client.list_tools_sync()

    print(f"✅  Connected. {len(mcp_tools)} MCP tool(s) discovered:")
    for t in mcp_tools:
        # MCPAgentTool exposes .tool_name
        print(f"      • {t.tool_name}")
    print()

    # ── Create the Strands Agent with MCP tools ───────────────
    # Passing mcp_tools into the tools list is all that's needed.
    # The agent will include each tool's JSON schema in the LLM
    # context automatically so the model knows how to call them.
    agent = Agent(
        model=ollama_model,
        tools=mcp_tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # ════════════════════════════════════════════════════════
    # SECTION 5 — INTERACTIVE CHAT LOOP
    # ════════════════════════════════════════════════════════
    # Classic REPL pattern consistent with all previous challenges.
    # The agent object keeps in-session history automatically, so
    # follow-up questions work naturally within a session.

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C / Ctrl-D — exit cleanly without a traceback
            print("\nGoodbye!")
            break

        # ── Exit commands ─────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # ── Skip blank lines ──────────────────────────────────
        if not user_input:
            continue

        # ── Special command: list MCP tools ───────────────────
        # Let the user see which tools are connected at any time.
        if user_input.lower() == "tools":
            print(f"\n🔌  Connected MCP tools ({len(mcp_tools)}):")
            for t in mcp_tools:
                print(f"    • {t.tool_name}")
            print()
            continue

        # ── Normal chat turn ──────────────────────────────────
        # The agent decides whether to:
        #   a) Call an MCP tool (routes the call to the server)
        #   b) Answer directly from the LLM's own knowledge
        #
        # MCP tool flow:
        #   agent → JSON-RPC call_tool → server subprocess
        #   → tool function runs → result returned via stdout
        #   → Strands feeds result back to LLM → final answer
        print("\nAgent: ", end="", flush=True)

        try:
            agent(user_input)
        except Exception as agent_err:
            # Surface errors clearly so the user knows what went wrong.
            print(f"\n⚠  Agent error: {agent_err}")

        print()  # blank line between turns for readability

# ── MCP server subprocess is automatically terminated here ───
# (the `with mcp_client:` context manager calls __exit__)
print("MCP server disconnected. Session ended.")
