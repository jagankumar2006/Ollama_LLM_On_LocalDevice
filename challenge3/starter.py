# ============================================================
# Challenge 3 — Memory Agent using Strands SDK + Ollama
# Model  : llama3.2:3b  (runs 100% locally, no API keys needed)
# Memory : Mem0 with FAISS vector store (persistent, local)
# ============================================================
#
# HOW IT WORKS
# ────────────
#  1. Every message you send is stored in Mem0.
#  2. Mem0 embeds the text and saves it to a local FAISS index
#     (a file on disk) so memories survive restarts.
#  3. Before each reply, the agent searches that index for the
#     most relevant past memories and injects them into the
#     system prompt — giving the LLM "long-term recall".
#
# SETUP — run these commands once before the first run:
# ────────────────────────────────────────────────────────────
#   pip install strands-agents strands-agents-tools
#   pip install mem0ai faiss-cpu sentence-transformers
#
# Ollama must already be running with llama3.2:3b pulled:
#   ollama serve          (starts the server)
#   ollama pull llama3.2:3b
# ============================================================


# ── Standard library ─────────────────────────────────────────
import os               # used to create the directory that holds FAISS files
import re               # regex — strips ANSI escape codes from agent output

# ── Strands SDK ───────────────────────────────────────────────
from strands import Agent
from strands.models.ollama import OllamaModel

# ── Mem0 ──────────────────────────────────────────────────────
# Mem0 is a memory layer for AI apps. It supports many storage
# back-ends; here we use FAISS (a local vector store) so
# everything stays on your machine with no external services.
from mem0 import Memory


# ════════════════════════════════════════════════════════════
# SECTION 1 — MEM0 CONFIGURATION
# ════════════════════════════════════════════════════════════

# Directory where FAISS will write its index files.
# Using a relative path keeps it tidy inside this challenge folder.
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory_store")
os.makedirs(MEMORY_DIR, exist_ok=True)   # create the folder if it doesn't exist

# Mem0 config dictionary — tells Mem0 which embedding model and
# which vector store to use.
MEM0_CONFIG = {
    # ── Vector store ──────────────────────────────────────────
    # FAISS stores dense vectors locally; no server required.
    # 'path' is the folder where FAISS files are written.
    "vector_store": {
        "provider": "faiss",
        "config": {
            "embedding_model_dims": 384,   # must match the embedding model below
            "path": MEMORY_DIR,            # persist the index here
        },
    },

    # ── Embedder ──────────────────────────────────────────────
    # sentence-transformers/all-MiniLM-L6-v2 is a small, fast
    # model that converts text → 384-dimensional vectors.
    # It runs locally — no API key, no internet required.
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },

    # ── LLM for Mem0 internal reasoning ───────────────────────
    # Mem0 uses an LLM to extract key facts from conversations
    # (e.g. it reads "My name is Thamarai" and saves "name: Thamarai").
    # We reuse Ollama so everything stays local.
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0,   # 0 = deterministic for fact extraction
            "max_tokens": 2000,
        },
    },
}

# Initialise the Mem0 Memory object with the config above.
# This loads (or creates) the FAISS index from disk.
print("⏳  Loading memory store …", flush=True)
memory = Memory.from_config(MEM0_CONFIG)
print("✅  Memory store ready.\n")

# A fixed user identifier.  Mem0 can store memories for many
# users; we use a single constant ID for this single-user demo.
USER_ID = "user_001"


# ════════════════════════════════════════════════════════════
# SECTION 2 — HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def store_memory(user_message: str, agent_reply: str) -> None:
    """
    Persist a conversation turn in Mem0.

    Mem0's add() method:
      - embeds the text using the HuggingFace model
      - runs the Mem0 LLM to extract structured facts
      - saves both the raw text and the embedding to FAISS

    Parameters
    ----------
    user_message : what the user said
    agent_reply  : what the agent replied (strip any ANSI colour codes first)
    """
    # Build the conversation in Mem0's expected format:
    # a list of {"role": ..., "content": ...} dicts.
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": agent_reply},
    ]
    # add() returns metadata about what was stored; we ignore it here.
    memory.add(messages, user_id=USER_ID)


def retrieve_memories(query: str, top_k: int = 5) -> str:
    """
    Search Mem0 for the most relevant past memories.

    Uses cosine similarity between the query embedding and all
    stored embeddings. Returns the top-k matches formatted as a
    numbered list, or an empty string if nothing is found.

    Parameters
    ----------
    query : the current user message (used as the search query)
    top_k : maximum number of memories to return (default 5)
    """
    results = memory.search(query=query, filters={"user_id": USER_ID}, limit=top_k)

    # Mem0 returns a dict with a "results" key containing a list of
    # memory objects, each having a "memory" text field.
    memories_list = results.get("results", [])

    if not memories_list:
        return ""   # no memories yet — agent replies without context

    # Format each memory as a numbered line for easy reading in
    # the system prompt.
    formatted = "\n".join(
        f"{i+1}. {m['memory']}"
        for i, m in enumerate(memories_list)
        if m.get("memory")
    )
    return formatted


# ════════════════════════════════════════════════════════════
# SECTION 3 — MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,    # balanced creativity for a conversational agent
)


# ════════════════════════════════════════════════════════════
# SECTION 4 — AGENT FACTORY
# ════════════════════════════════════════════════════════════
# We recreate the Agent on every turn so we can inject a fresh,
# query-specific system prompt that contains the most relevant
# memories.  The Strands Agent stores in-session history
# internally; the cross-session history comes from Mem0.

BASE_SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with a long-term memory.

When relevant memories are provided below, USE them to personalise your
answers. Always address the user by name if you know it.

{memory_block}

Rules:
- If the user tells you something personal (name, preference, fact), acknowledge it.
- When asked about something you remember, recall it naturally.
- Never say you cannot remember things that appear in your memories above.
"""

def build_agent(query: str) -> Agent:
    """
    Create a Strands Agent whose system prompt is enriched with
    the memories most relevant to the current query.

    Steps
    -----
    1. Search Mem0 for memories related to `query`.
    2. Format them into the system prompt.
    3. Return a fresh Agent ready to handle this turn.
    """
    recalled = retrieve_memories(query)

    if recalled:
        memory_block = f"Relevant memories about the user:\n{recalled}"
    else:
        memory_block = "(No memories stored yet.)"

    system_prompt = BASE_SYSTEM_PROMPT.format(memory_block=memory_block)

    return Agent(
        model=ollama_model,
        system_prompt=system_prompt,
    )


# ════════════════════════════════════════════════════════════
# SECTION 5 — ANSI STRIP UTILITY
# ════════════════════════════════════════════════════════════
# Strands sometimes emits ANSI colour/cursor codes in its output
# string.  We strip them before storing in Mem0 so the saved
# memories contain clean plain text, not escape sequences.

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return ANSI_ESCAPE.sub("", text)


# ════════════════════════════════════════════════════════════
# SECTION 6 — INTERACTIVE REPL
# ════════════════════════════════════════════════════════════

print("=" * 58)
print("  Memory Agent — Strands SDK + Ollama  (llama3.2:3b)")
print("  Memory: Mem0 + FAISS  (persisted locally)")
print("=" * 58)
print("  Try:")
print("    'My name is Thamarai'")
print("    'I love hiking and coffee'")
print("    'What is my name?'          ← tests recall")
print("    'What do I enjoy?'          ← tests recall")
print()
print("  Type 'memory' to list all stored memories.")
print("  Type 'clear'  to wipe all memories.")
print("  Type 'quit'   to exit.")
print("=" * 58)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    # ── Exit ─────────────────────────────────────────────────
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye! Your memories are saved for next time.")
        break

    # ── Empty input ──────────────────────────────────────────
    if not user_input:
        continue

    # ── Special command: list all memories ───────────────────
    if user_input.lower() == "memory":
        all_mem = memory.get_all(filters={"user_id": USER_ID})
        items = all_mem.get("results", [])
        if items:
            print("\n📦  Stored memories:")
            for i, m in enumerate(items, 1):
                print(f"  {i}. {m.get('memory', '—')}")
        else:
            print("\n  (no memories stored yet)")
        print()
        continue

    # ── Special command: clear all memories ──────────────────
    if user_input.lower() == "clear":
        memory.delete_all(filters={"user_id": USER_ID})
        print("🗑  All memories cleared.\n")
        continue

    # ── Normal chat turn ──────────────────────────────────────
    # 1. Build an agent whose system prompt includes relevant memories.
    agent = build_agent(user_input)

    # 2. Capture the agent's response as a string so we can store it.
    #    Strands returns the result object; printing it streams to stdout.
    print("\nAgent: ", end="", flush=True)
    response = agent(user_input)   # streams output to terminal automatically

    # 3. Extract clean text from the response for storage.
    #    response.message contains the full assistant text.
    reply_text = strip_ansi(str(response.message.get("content", "") if hasattr(response, "message") else str(response)))

    print()  # blank line between turns

    # 4. Store this turn in Mem0 so it can be recalled later.
    try:
        store_memory(user_input, reply_text)
    except Exception as mem_err:
        # Memory storage failure should never crash the chat loop.
        print(f"  ⚠  Memory save warning: {mem_err}\n")
