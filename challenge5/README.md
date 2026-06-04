# Challenge 5 — MCP Chatbot

Connects the Strands agent to a **local MCP server** using the **Model Context Protocol**. Instead of defining tools with Python decorators inside the agent script, tools live in a separate server process. The agent discovers and calls them dynamically over a standard protocol.

---

## What this challenge covers

- What MCP (Model Context Protocol) is and why it matters
- Running a local MCP server using `FastMCP`
- Connecting Strands to an MCP server with `MCPClient` + stdio transport
- Tool discovery via `get_tools()`
- How tool calls flow through the MCP protocol
- Two-file architecture: server + chatbot client

---

## What is MCP?

MCP is an open standard (created by Anthropic) that defines a clean interface between AI agents and external tools. Instead of writing tools as Python functions inside your agent, you write them in a **server** that exposes them over a protocol. Any MCP-compatible client — Strands, Claude Desktop, etc. — can connect and use those tools without modification.

```
Without MCP (Challenges 2–4)       With MCP (Challenge 5)
─────────────────────────────       ──────────────────────────────
Agent                               Agent
  @tool def calculator(...)           MCPClient  ←→  MCP Server
  @tool def get_weather(...)                          @mcp.tool() add()
  @tool def calculate_age(...)                        @mcp.tool() get_weather()
  (all in one file)                                   @mcp.tool() get_system_info()
                                                      (separate process)
```

**Benefits of MCP:**
- Tools can be written in any language
- The same server can serve multiple different agents
- Tools are discovered dynamically — no hardcoding in the agent
- The protocol is standardised — swapping servers requires no agent code changes

---

## How this challenge works

```
python challenge5/starter.py
         ↓
  MCPClient spawns mcp_server.py as a subprocess
         ↓
  MCP handshake: initialize → tools/list
  (agent discovers: add, get_weather, get_system_info)
         ↓
  Agent created with mcp_tools list
         ↓
  Chat loop starts

When you send a message:
  You → Agent → LLM decides to call a tool
                    ↓
              MCPClient sends JSON-RPC "tools/call" to server stdin
                    ↓
              mcp_server.py runs the tool function
                    ↓
              Result written to server stdout as JSON-RPC response
                    ↓
              MCPClient reads result → feeds to LLM → final answer → You
```

---

## MCP tools available

| Tool | Trigger phrase examples | What it does |
|------|------------------------|-------------|
| `add` | "What is 42 plus 58?", "Add 3.14 and 2" | Adds two numbers |
| `get_weather` | "Weather in London?", "Is it hot in Dubai?" | Returns simulated weather for 10 cities |
| `get_system_info` | "Tell me about this system", "What OS am I on?" | Returns OS, CPU arch, Python version, hostname |

---

## Special commands

| Command | What it does |
|---------|-------------|
| `tools` | Lists all MCP tools currently connected |
| `quit` / `exit` / `q` | Disconnects MCP server and exits |

---

## File structure

```
challenge5/
├── starter.py      ← MCP chatbot client  (run this)
└── mcp_server.py   ← MCP server          (launched automatically)
```

> You only need to run `starter.py`. It launches `mcp_server.py` automatically as a subprocess.

---

## Setup

**1. Install Python dependencies**

```bash
pip install strands-agents strands-agents-tools
pip install mcp
```

**2. Start Ollama**

```bash
ollama serve
ollama pull llama3.2:3b
```

**3. Run the chatbot**

```bash
python challenge5/starter.py
```

---

## Example session

```
==============================================================
  MCP Chatbot — Strands SDK + Ollama  (llama3.2:3b)
  MCP Transport : stdio (local subprocess)
==============================================================

  EXAMPLE QUERIES
    ➕  'What is 42 plus 58?'
    🌦  'What is the weather in London?'
    💻  'Tell me about this system'
    🔍  'What MCP tools do you have?'

  Type 'tools' to list connected MCP tools.
  Type 'quit'  to exit.
==============================================================

⏳  Connecting to MCP server …
✅  Connected. 3 MCP tool(s) discovered:
      • add
      • get_weather
      • get_system_info

You: What is 42 plus 58?

Agent: I'll use the add tool for that.
42.0 + 58.0 = 100.0

You: What is the weather in London?

Agent: Weather in London:
  Temperature : 15°C
  Conditions  : Cloudy
  Humidity    : 72%

You: Tell me about this system

Agent: System Information:
  OS          : Windows 10.0.22631
  Architecture: AMD64
  Python      : 3.11.5
  Hostname    : MY-PC

You: tools

🔌  Connected MCP tools (3):
    • add
    • get_weather
    • get_system_info

You: quit
Goodbye!
MCP server disconnected. Session ended.
```

---

## Code walkthrough

### `starter.py`

#### Section 1 — Model configuration

Same `OllamaModel` setup used in all challenges. `temperature=0.3` for reliable tool-calling.

#### Section 2 — MCP server parameters + transport factory

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command=sys.executable,   # same Python interpreter as the chatbot
    args=[SERVER_SCRIPT],     # path to mcp_server.py
    env=None,                 # inherit current environment (same venv)
)

transport_factory = lambda: stdio_client(server_params)
```

`StdioServerParameters` describes **how** to launch the subprocess. But `MCPClient` does not accept the params object directly — it needs a callable that returns an **async context manager**. That is `stdio_client()` from `mcp.client.stdio`.

Passing `server_params` directly causes:
```
TypeError: 'StdioServerParameters' object does not support the asynchronous context manager protocol
```

The fix is always: `lambda: stdio_client(server_params)`.

#### Section 3 — System prompt

Describes the agent's role and lists the available MCP tools so the LLM knows when to reach for them.

#### Section 4 — MCPClient + Agent setup

```python
mcp_client = MCPClient(transport_factory)   # pass the factory, not server_params

with mcp_client:
    mcp_tools = mcp_client.list_tools_sync()  # sync → PaginatedList[MCPAgentTool]
    agent = Agent(model=ollama_model, tools=mcp_tools, system_prompt=SYSTEM_PROMPT)
```

`MCPClient` is a context manager:
- `__enter__` — spawns the server subprocess, performs the MCP handshake
- `list_tools_sync()` — synchronously queries `tools/list`; returns `MCPAgentTool` objects (subclass of `AgentTool`) ready for `Agent(tools=...)`
- `__exit__` — terminates the subprocess cleanly

> Method reference for strands-agents 1.42:
> - `list_tools_sync()` → synchronous ✅ use this
> - `load_tools()` → async coroutine, must be awaited — cannot be called directly in sync code
> - `get_tools()` → does not exist in this version

`mcp_tools` is just a list that gets passed to `Agent(tools=...)` exactly like `@tool`-decorated functions in previous challenges.

#### Section 5 — Chat loop

Standard REPL. The `tools` command lets you inspect what's connected at any time. Errors from the agent are caught and displayed without crashing the loop.

---

### `mcp_server.py`

#### Section 1 — Create the server

```python
mcp = FastMCP("LocalDemoServer")
```

`FastMCP` handles all JSON-RPC plumbing. The string `"LocalDemoServer"` is the server name sent during the handshake.

#### Section 2 — Tool definitions

```python
@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers ..."""
    result = a + b
    print(f"[MCP Server] add({a}, {b}) = {result}", file=sys.stderr)
    return f"{a} + {b} = {result}"
```

`@mcp.tool()` works like Strands' `@tool` — the docstring is the description, type annotations build the schema. **Important:** debug prints must go to `stderr` only. `stdout` is reserved for MCP JSON-RPC messages.

#### Section 3 — Start the server

```python
mcp.run(transport="stdio")
```

Starts the blocking event loop that reads JSON-RPC from stdin and writes responses to stdout. Runs until the parent process (starter.py) closes the pipe.

---

## Key concepts

**stdio transport** — The simplest MCP transport. The client and server communicate over subprocess stdin/stdout pipes using newline-delimited JSON-RPC 2.0 messages. No network port is opened.

**JSON-RPC** — A lightweight remote procedure call protocol. MCP uses it to send requests like `{"method": "tools/call", "params": {"name": "add", "arguments": {"a": 42, "b": 58}}}` and receive structured responses.

**Tool discovery** — `get_tools()` calls the server's `tools/list` endpoint and gets back every tool's name, description, and parameter schema. The agent never has hardcoded tool names — it learns them at runtime from the server.

**Separation of concerns** — The server knows nothing about Strands or Ollama. The client knows nothing about how tools are implemented. This is the value of MCP: either side can be replaced independently.

---

## Extending this challenge

**Add more tools to the server** — just add more `@mcp.tool()` functions in `mcp_server.py`. The client picks them up automatically on the next run.

**Connect to a real MCP server** — replace `server_params` with parameters pointing to any MCP-compatible server (filesystem, database, web search, etc.). The rest of the code stays the same.

**Use HTTP/SSE transport** — for remote servers, swap `StdioServerParameters` for `SSEServerParameters` with a URL. No other changes needed.
