# Challenge 1 — Simple AI Agent

A minimal conversational agent built with the **Strands SDK** and **Ollama**, running entirely on your local machine. No API keys. No internet required.

---

## What this challenge covers

- Importing and configuring `OllamaModel` from Strands
- Creating an `Agent` with a system prompt
- Building an interactive REPL (Read-Eval-Print Loop)
- Understanding how the agent maintains conversation history automatically

---

## How it works

```
You  →  input()  →  Agent  →  OllamaModel  →  llama3.2:3b (Ollama)
                                                      ↓
You  ←  stdout   ←  Agent  ←────────────────── response
```

1. `OllamaModel` connects to Ollama running at `localhost:11434`.
2. `Agent` wraps the model and manages conversation history.
3. Every message you send is automatically kept in the agent's context so follow-up questions work naturally.
4. The chat loop runs until you type `quit`, `exit`, or `q`.

---

## File structure

```
challenge1/
└── starter.py      ← the complete agent (single file)
```

---

## Setup

Run these commands once before the first start.

**1. Install Python dependencies**

```bash
pip install strands-agents strands-agents-tools
```

**2. Install and start Ollama**

Download Ollama from https://ollama.com, then:

```bash
ollama serve              # start the Ollama server
ollama pull llama3.2:3b   # download the model (~2 GB, one-time)
```

**3. Run the agent**

```bash
python challenge1/starter.py
```

---

## Example session

```
==================================================
  Simple AI Agent — Strands SDK + Ollama
  Model : llama3.2:3b  (running locally)
  Type  : 'quit' or 'exit' to stop
==================================================

You: What is the capital of France?

Agent: The capital of France is Paris.

You: Tell me more about it.

Agent: Paris is a major European city and the capital of France ...
```

---

## Code walkthrough

| Section | What it does |
|---------|-------------|
| `OllamaModel(...)` | Points Strands at your local Ollama server and selects the model |
| `Agent(model=..., system_prompt=...)` | Creates the agent and gives it a persona |
| `agent(user_input)` | Sends a message, streams the response to stdout, updates history |
| `while True` loop | Keeps the conversation going until the user exits |

---

## Key concepts

**temperature** — Controls how creative/random the model's responses are.
- `0.0` = fully deterministic, same answer every time
- `0.7` = balanced (used here)
- `1.0` = very creative, can be unpredictable

**system_prompt** — A fixed instruction that is prepended to every conversation. The model always "sees" it, shaping its personality and behaviour.

**conversation history** — Strands keeps track of every message in the session automatically. You do not need to manage a message list yourself.

---

## Next challenge

Challenge 2 adds **tools** — Python functions the LLM can call to perform real tasks like calculations and weather lookups.
