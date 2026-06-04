# Challenge 3 — Memory Agent

Adds **persistent long-term memory** to the agent from Challenge 1. The agent remembers facts you share across sessions using **Mem0** and a local **FAISS** vector store. No cloud services. Everything stays on your machine.

---

## What this challenge covers

- What Mem0 is and how it stores memories
- Using FAISS as a local vector store
- Embedding text with `sentence-transformers` (runs locally)
- Injecting retrieved memories into the system prompt
- Storing and retrieving conversation turns across restarts
- Special commands: `memory`, `clear`

---

## How it works

```
You  →  input
         ↓
  1. Search Mem0 for relevant past memories (vector similarity)
         ↓
  2. Build Agent with fresh system prompt containing those memories
         ↓
  3. Agent replies  →  printed to terminal
         ↓
  4. Store this turn (user + reply) in Mem0 → FAISS index on disk
```

On the next run, step 1 loads memories from disk — so the agent remembers things you said in previous sessions.

---

## Memory architecture

```
Conversation turn
      ↓
  Mem0.add()
      ↓
  LLM (llama3.2:3b via Ollama)
  extracts key facts:
  "user's name is Thamarai"
  "user enjoys hiking"
      ↓
  HuggingFace embedder
  converts facts → 384-dim vectors
      ↓
  FAISS index  ←→  disk (memory_store/)
```

When you ask a question, the question is embedded and compared against all stored vectors using cosine similarity. The top 5 matches are returned and injected into the system prompt.

---

## File structure

```
challenge3/
├── starter.py          ← agent + memory logic
└── memory_store/       ← created automatically on first run
    ├── mem0.faiss      ← FAISS vector index (binary)
    └── mem0.json       ← metadata for each stored memory
```

---

## Setup

**1. Install Python dependencies**

```bash
pip install strands-agents strands-agents-tools
pip install mem0ai faiss-cpu sentence-transformers
```

> On Apple Silicon Macs, use `faiss-cpu` not `faiss-gpu`.

**2. Start Ollama**

```bash
ollama serve
ollama pull llama3.2:3b
```

> The first run also downloads `sentence-transformers/all-MiniLM-L6-v2` (~22 MB) automatically.

**3. Run the agent**

```bash
python challenge3/starter.py
```

---

## Example session

```
==========================================================
  Memory Agent — Strands SDK + Ollama  (llama3.2:3b)
  Memory: Mem0 + FAISS  (persisted locally)
==========================================================

You: My name is Thamarai

Agent: Nice to meet you, Thamarai! I'll remember that.

You: I love hiking and coffee

Agent: Great choices! I'll keep that in mind.

You: quit

--- restart the program ---

You: What is my name?

Agent: Your name is Thamarai — you told me that earlier!

You: What do I enjoy?

Agent: You enjoy hiking and coffee, Thamarai.
```

---

## Special commands

| Command | What it does |
|---------|-------------|
| `memory` | Lists all facts stored in Mem0 for this user |
| `clear` | Deletes all stored memories (cannot be undone) |
| `quit` / `exit` / `q` | Exits the program |

---

## Code walkthrough

### Section 1 — Mem0 configuration

```python
MEM0_CONFIG = {
    "vector_store": {"provider": "faiss", "config": {"path": MEMORY_DIR, ...}},
    "embedder":     {"provider": "huggingface", "config": {"model": "...MiniLM..."}},
    "llm":          {"provider": "ollama", "config": {"model": "llama3.2:3b", ...}},
}
memory = Memory.from_config(MEM0_CONFIG)
```

Three components are configured:
- **vector_store** — FAISS writes two files to `memory_store/` on disk
- **embedder** — `all-MiniLM-L6-v2` converts text to 384-dim vectors locally
- **llm** — Ollama is used by Mem0 internally to extract structured facts from conversations

### Section 2 — Helper functions

`store_memory(user_msg, reply)` — Calls `memory.add()` which runs fact extraction and saves the embedding to FAISS.

`retrieve_memories(query)` — Calls `memory.search()` which embeds the query and returns the top-5 most similar stored memories.

### Section 3 — Model configuration

Same as Challenge 1 (`temperature=0.7`).

### Section 4 — Agent factory

```python
def build_agent(query: str) -> Agent:
    recalled = retrieve_memories(query)
    system_prompt = BASE_SYSTEM_PROMPT.format(memory_block=recalled)
    return Agent(model=ollama_model, system_prompt=system_prompt)
```

The agent is **recreated on every turn** so the system prompt can be updated with the memories most relevant to the current question. Without this, the system prompt would be fixed and memories couldn't be injected dynamically.

### Section 5 — ANSI strip utility

Strands may include terminal colour codes in its output string. These are stripped before storing in Mem0 so saved memories contain clean plain text.

### Section 6 — REPL

Same as previous challenges, extended with `memory` and `clear` commands.

---

## Key concepts

**Vector similarity search** — Each memory is stored as a dense vector (a list of 384 numbers). When you ask a question, it is also converted to a vector. FAISS finds stored vectors with the highest cosine similarity — meaning the most semantically related memories, not just keyword matches.

**Fact extraction** — Mem0 runs a quick LLM pass over each conversation turn to pull out discrete facts. "My name is Alice" becomes a single stored fact: `"user's name is Alice"`. This keeps the memory store clean and concise.

**Persistence** — The FAISS index is written to disk after every `memory.add()` call. The files in `memory_store/` survive program restarts and accumulate across sessions.

---

## Next challenge

Challenge 4 combines everything: tools from Challenge 2 + persistent memory from Challenge 3.
