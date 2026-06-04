# Challenge 4 — Full Agent

The complete agent — **tools** from Challenge 2 combined with **persistent memory** from Challenge 3 in a single cohesive program. The agent can do math, look up weather, calculate ages, and remember who you are across sessions.

---

## What this challenge covers

- Combining `@tool` functions with Mem0 memory in one agent
- Memory-aware system prompt that also lists available tools
- `build_agent()` factory that injects both tool context and relevant memories
- Extended REPL with `memory`, `clear`, and `help` commands

---

## How it works

```
You  →  input
          ↓
  1. Retrieve relevant memories from FAISS (vector search)
          ↓
  2. Build Agent:  system_prompt = tool list + memory block
          ↓
  3. Agent decides: call a tool  OR  answer from knowledge + memory
          ↓  (if tool called)
  Python function runs → result fed back to LLM
          ↓
  4. LLM composes final answer  →  printed to terminal
          ↓
  5. Store turn in Mem0 → FAISS index (disk)
```

---

## Tools available

| Tool | Trigger phrase examples | What it does |
|------|------------------------|-------------|
| `calculator` | "What is sqrt(144)?", "15% of 240" | Evaluates any Python math expression safely |
| `get_weather` | "Weather in Dubai", "Is it sunny in Sydney?" | Returns simulated weather for 10 cities |
| `calculate_age` | "How old is someone born in 1985?", "Age from July 4, 1999" | Computes exact age in years and months |

---

## Special commands

| Command | What it does |
|---------|-------------|
| `memory` | Lists all facts stored in Mem0 |
| `clear` | Wipes all stored memories |
| `help` | Reprints the usage guide |
| `quit` / `exit` / `q` | Exits cleanly |

---

## File structure

```
challenge4/
├── starter.py          ← tools + memory + agent + REPL
└── memory_store/       ← created automatically on first run
    ├── mem0.faiss      ← FAISS vector index
    └── mem0.json       ← memory metadata
```

---

## Setup

**1. Install Python dependencies**

```bash
pip install strands-agents strands-agents-tools
pip install mem0ai faiss-cpu sentence-transformers
```

**2. Start Ollama**

```bash
ollama serve
ollama pull llama3.2:3b
```

**3. Run the agent**

```bash
python challenge4/starter.py
```

---

## Example session

```
==============================================================
  Full Agent — Strands SDK + Ollama  (llama3.2:3b)
  Tools : calculator · weather · age calculator
  Memory: Mem0 + FAISS  (persisted locally)
==============================================================

You: My name is Alice and I love hiking

Agent: Great to meet you, Alice! I'll remember that you love hiking.

You: What is 17 * 48 + sqrt(256)?

Agent: 17 × 48 + √256 = 816 + 16 = 832

You: What's the weather in London?

Agent: Weather in London:
  🌡  Temperature : 15°C
  🌤  Conditions  : Cloudy
  💧 Humidity    : 72%

You: How old is someone born on June 15, 1995?

Agent: Someone born on June 15, 1995 is 30 years and 11 month(s) old.

--- restart the program ---

You: What is my name?

Agent: Your name is Alice — you told me that in a previous session!

You: What do I enjoy?

Agent: You enjoy hiking, Alice.
```

---

## Code walkthrough

### Section 1 — Tool definitions

Three `@tool`-decorated functions: `calculator`, `get_weather`, `calculate_age`. Identical to Challenge 2. Each has a detailed docstring that acts as its specification for the LLM.

### Section 2 — Mem0 + FAISS configuration

Identical to Challenge 3. Three components:
- `faiss` vector store writing to `memory_store/`
- `huggingface` embedder using `all-MiniLM-L6-v2` (384-dim vectors)
- `ollama` LLM for internal fact extraction

### Section 3 — Memory helper functions

`store_memory(user_msg, reply)` — Saves the conversation turn to FAISS after each response.

`retrieve_memories(query)` — Returns the top-5 most relevant memories as a formatted string ready for the system prompt.

### Section 4 — ANSI strip utility

Removes terminal escape codes from agent output before storing in Mem0.

### Section 5 — Model configuration

`temperature=0.3` — lower than Challenge 1 and 3. Tool-calling is more reliable with a deterministic model.

### Section 6 — Agent factory

```python
def build_agent(query: str) -> Agent:
    recalled = retrieve_memories(query)
    system_prompt = BASE_SYSTEM_PROMPT.format(memory_block=...)
    return Agent(
        model=ollama_model,
        tools=[calculator, get_weather, calculate_age],
        system_prompt=system_prompt,
    )
```

The key insight: the agent is rebuilt on every turn so the system prompt always contains:
1. The tool list (static)
2. The most relevant memories for this specific question (dynamic)

### Section 7 — REPL

Extended with `memory`, `clear`, and `help` special commands on top of the standard `quit`/`exit`.

---

## Key concepts

**Why rebuild the agent every turn?** — Strands Agents have a fixed system prompt set at construction time. By rebuilding, we can inject a different memory block for each question — showing only the memories most relevant to what the user just asked, rather than dumping every stored fact into context.

**Tools vs memory** — Tools answer questions that require *computation or real-time data* (math, weather). Memory answers questions that require *recall* (name, preferences). The agent uses both in the same turn if needed.

**temperature=0.3 for tools** — When the LLM needs to decide whether to call `calculator` and what expression to pass, a lower temperature produces more reliable, parseable tool calls.

---

## Next challenge

Challenge 5 introduces **MCP (Model Context Protocol)** — a standard way to connect agents to tools living in separate server processes.
