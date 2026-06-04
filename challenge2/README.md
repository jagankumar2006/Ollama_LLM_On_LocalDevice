# Challenge 2 — Tools Agent

Extends the simple agent from Challenge 1 by giving it **callable tools**. The LLM can now decide on its own when to call a Python function instead of answering from memory alone.

---

## What this challenge covers

- Using the `@tool` decorator to expose Python functions to the LLM
- How Strands converts type annotations into JSON schemas
- How the agent decides which tool to call (and when not to)
- Passing a `tools` list to `Agent`

---

## How it works

```
You  →  Agent  →  LLM decides: answer directly OR call a tool
                        ↓ tool call
                  Python function runs locally
                        ↓ result
                  LLM formats the final answer  →  You
```

When you ask *"What is 17 × 48?"* the LLM does **not** do the math itself — it emits a structured tool call, Strands intercepts it, runs `calculator("17 * 48")`, and feeds the result back to the LLM to compose a natural-language reply.

---

## Tools available

| Tool | Trigger phrase examples | What it does |
|------|------------------------|-------------|
| `calculator` | "What is 17 * 48?", "sqrt(256)", "2 ** 10" | Evaluates a Python math expression safely |
| `get_weather` | "Weather in Tokyo", "Is it raining in Paris?" | Returns simulated weather for 10 cities |
| `calculate_age` | "How old is someone born in 1990?", "Age from June 15, 1995" | Computes exact age in years and months |

---

## File structure

```
challenge2/
└── starter.py      ← tools + agent + REPL (single file)
```

---

## Setup

**1. Install Python dependencies**

```bash
pip install strands-agents strands-agents-tools
```

**2. Start Ollama**

```bash
ollama serve
ollama pull llama3.2:3b
```

**3. Run the agent**

```bash
python challenge2/starter.py
```

---

## Example session

```
==========================================================
  Tools Agent — Strands SDK + Ollama  (llama3.2:3b)
==========================================================
  Available tools:
    🧮  Calculator   e.g. 'What is 17 * 48?'
    🌦  Weather      e.g. 'What's the weather in Tokyo?'
    🎂  Age Calc     e.g. 'How old is someone born in 1990?'

You: What is 17 * 48 + sqrt(256)?

Agent: 17 × 48 + √256 = 816 + 16 = 832

You: What's the weather in Tokyo?

Agent: Weather in Tokyo:
  🌡  Temperature : 28°C
  🌤  Conditions  : Partly Cloudy
  💧 Humidity    : 68%

You: How old is someone born on March 3, 1992?

Agent: Someone born on March 03, 1992 is 34 years and 3 month(s) old.
```

---

## Code walkthrough

### Section 1 — Tool definitions

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression ..."""
    ...
```

The `@tool` decorator registers the function. The **docstring** is sent verbatim to the LLM as the tool's description — this is how the model knows when to call it. The **type annotations** (`str`, `int`) are converted to a JSON schema so the model knows exactly what arguments to pass.

### Section 2 — Model configuration

```python
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,   # lower = more reliable tool-calling
)
```

Temperature is lowered to `0.3` compared to Challenge 1. Tool-calling works better when the model is more deterministic.

### Section 3 — Agent creation

```python
agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, calculate_age],
    system_prompt="...",
)
```

The only difference from Challenge 1: the `tools` list. Strands automatically sends each tool's schema to the LLM in every conversation turn.

### Section 4 — REPL

Same pattern as Challenge 1. `Ctrl-C` and `Ctrl-D` are handled cleanly.

---

## Key concepts

**@tool decorator** — Marks a regular Python function as a tool the LLM can invoke. The function runs on your machine — the LLM just decides *when* to call it and *what arguments* to pass.

**JSON schema** — Strands reads the type annotations and builds a JSON schema describing each tool's parameters. This schema is included in the LLM's context so it knows the exact parameter names and types to use.

**Tool safety** — The `calculator` tool uses `eval()` with a restricted namespace (`{"__builtins__": {}}` plus `math` functions only), so arbitrary code cannot be injected through a math expression.

---

## Next challenge

Challenge 3 adds **persistent memory** — the agent remembers things you told it even after you restart the program.
