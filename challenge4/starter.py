# ============================================================
# Challenge 4 — Full Agent using Strands SDK + Ollama
# Model  : llama3.2:3b  (runs 100% locally, no API keys needed)
# Tools  : calculator · weather · age calculator
# Memory : Mem0 with FAISS vector store (persistent, local)
# ============================================================
#
# This is the "full agent" — it combines everything from the
# previous challenges:
#
#   Challenge 2 → three callable tools (calculator, weather, age)
#   Challenge 3 → persistent memory via Mem0 + FAISS
#
# HOW IT WORKS
# ────────────
#  1. The agent exposes three @tool functions the LLM can call.
#  2. Every conversation turn is saved to Mem0 / FAISS on disk.
#  3. Before each reply the agent retrieves the most relevant
#     past memories and injects them into the system prompt so
#     the LLM can recall personal details across sessions.
#  4. The REPL provides special commands: 'memory', 'clear', 'help'.
#
# SETUP — run once before the first start:
# ────────────────────────────────────────────────────────────
#   pip install strands-agents strands-agents-tools
#   pip install mem0ai faiss-cpu sentence-transformers
#
# Ollama must be running with llama3.2:3b pulled:
#   ollama serve
#   ollama pull llama3.2:3b
# ============================================================


# ── Standard library ─────────────────────────────────────────
import os               # directory creation for FAISS store
import re               # ANSI escape-code stripping
from datetime import date  # used by the age calculator tool

# ── Strands SDK ───────────────────────────────────────────────
# Agent  → the core class that manages the conversation loop
# tool   → decorator that registers a Python function as an LLM tool
from strands import Agent, tool
from strands.models.ollama import OllamaModel

# ── Mem0 ──────────────────────────────────────────────────────
# Mem0 is a memory layer for AI apps.  We use the FAISS backend
# so every memory is stored locally — no external services needed.
from mem0 import Memory


# ════════════════════════════════════════════════════════════
# SECTION 1 — TOOL DEFINITIONS
# ════════════════════════════════════════════════════════════
#
# The @tool decorator does three things:
#   1. Registers the function with the Strands runtime.
#   2. Uses the function name as the tool name.
#   3. Sends the docstring to the LLM so it knows when / how to
#      call the tool (the docstring IS the tool's specification).
#
# Type annotations build the JSON schema the model receives,
# so always annotate every parameter and the return type.


# ── Tool 1 : Calculator ──────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Use this tool whenever the user asks you to perform any
    arithmetic or mathematical calculation such as addition,
    subtraction, multiplication, division, powers, percentages,
    or trigonometry.

    Args:
        expression: A valid Python math expression as a string.
                    Examples: "2 + 2", "100 * 0.15", "2 ** 10",
                              "math.sqrt(144)", "math.sin(math.pi/2)"

    Returns:
        The result as a string, or an error message if the
        expression cannot be evaluated.
    """
    try:
        import math

        # Restrict eval to math-safe names so arbitrary code
        # cannot be injected through the expression string.
        safe_globals = {"__builtins__": {}}
        safe_globals.update(vars(math))   # allow sin, cos, sqrt, pi …
        result = eval(expression, safe_globals)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Calculator error: {exc}"


# ── Tool 2 : Weather ─────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """
    Return the current weather conditions for a given city.

    Use this tool whenever the user asks about the weather,
    temperature, forecast, or climate for a specific location.

    Args:
        city: The name of the city to look up.
              Examples: "London", "New York", "Tokyo"

    Returns:
        A weather summary with temperature, conditions, and humidity.

    Note:
        This uses a simulated dataset for demonstration.  Replace
        the weather_data dict with a real API call (e.g. OpenWeatherMap)
        without changing any other part of this agent.
    """
    # Simulated weather data.
    # To connect a real API, replace this dict lookup with:
    #   import requests
    #   resp = requests.get(f"https://api.openweathermap.org/data/2.5/weather"
    #                       f"?q={city}&appid=YOUR_KEY&units=metric")
    #   data = resp.json()
    weather_data = {
        "london":      {"temp": 15, "condition": "Cloudy",        "humidity": 72},
        "new york":    {"temp": 22, "condition": "Sunny",         "humidity": 55},
        "tokyo":       {"temp": 28, "condition": "Partly Cloudy", "humidity": 68},
        "paris":       {"temp": 18, "condition": "Rainy",         "humidity": 80},
        "sydney":      {"temp": 20, "condition": "Clear",         "humidity": 60},
        "dubai":       {"temp": 38, "condition": "Hot & Sunny",   "humidity": 40},
        "toronto":     {"temp": 12, "condition": "Windy",         "humidity": 65},
        "berlin":      {"temp": 14, "condition": "Overcast",      "humidity": 75},
        "mumbai":      {"temp": 32, "condition": "Humid",         "humidity": 85},
        "los angeles": {"temp": 26, "condition": "Sunny",         "humidity": 45},
    }

    key = city.lower().strip()
    if key in weather_data:
        w = weather_data[key]
        return (
            f"Weather in {city.title()}:\n"
            f"  🌡  Temperature : {w['temp']}°C\n"
            f"  🌤  Conditions  : {w['condition']}\n"
            f"  💧 Humidity    : {w['humidity']}%"
        )
    # Graceful fallback for unlisted cities
    available = ", ".join(c.title() for c in weather_data)
    return (
        f"No weather data for '{city}'. "
        f"Available cities: {available}"
    )


# ── Tool 3 : Age Calculator ──────────────────────────────────
@tool
def calculate_age(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> str:
    """
    Calculate a person's exact age from their date of birth.

    Use this tool whenever the user asks how old someone is,
    what their age is, or anything related to calculating age
    from a birth date.

    Args:
        birth_year:  Four-digit year of birth.  Example: 1990
        birth_month: Month of birth as a number 1–12.
                     Defaults to 1 (January) if not provided.
        birth_day:   Day of birth 1–31.
                     Defaults to 1 if not provided.

    Returns:
        Age in years and months, plus the weekday they were born on.
    """
    try:
        today     = date.today()
        birthdate = date(birth_year, birth_month, birth_day)

        if birthdate > today:
            return "That birth date is in the future — please check the year."

        # Full completed years
        years = today.year - birthdate.year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            years -= 1  # birthday hasn't occurred yet this calendar year

        # Remaining months since last birthday
        months = today.month - birthdate.month
        if today.day < birthdate.day:
            months -= 1
        if months < 0:
            months += 12

        day_name = birthdate.strftime("%A")

        return (
            f"Someone born on {birthdate.strftime('%B %d, %Y')} is "
            f"{years} years and {months} month(s) old.\n"
            f"They were born on a {day_name}.\n"
            f"(Calculated relative to today: {today.strftime('%B %d, %Y')})"
        )
    except ValueError as exc:
        return f"Age calculator error: {exc} — please check the date values."


# ════════════════════════════════════════════════════════════
# SECTION 2 — MEM0 + FAISS MEMORY CONFIGURATION
# ════════════════════════════════════════════════════════════
#
# Mem0 is the memory manager.  It:
#   a) Accepts raw conversation turns (user + assistant messages).
#   b) Uses a small LLM to extract key facts ("name: Alice").
#   c) Embeds those facts into 384-dim vectors.
#   d) Stores them in a FAISS index on disk — survives restarts.
#
# On the next session, relevant facts are retrieved via vector
# similarity search and injected into the agent's system prompt.

# Folder where FAISS writes its index files.
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory_store")
os.makedirs(MEMORY_DIR, exist_ok=True)   # create if it doesn't exist

MEM0_CONFIG = {
    # ── Vector store ──────────────────────────────────────────
    # FAISS runs entirely locally.  'path' is the directory
    # where mem0.faiss and mem0.json are written.
    "vector_store": {
        "provider": "faiss",
        "config": {
            "embedding_model_dims": 384,   # must match the embedder below
            "path": MEMORY_DIR,
        },
    },

    # ── Embedder ──────────────────────────────────────────────
    # all-MiniLM-L6-v2 is a small (22 MB), fast model that
    # converts text into 384-dimensional vectors locally.
    # Downloaded once by sentence-transformers; no API key needed.
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },

    # ── LLM for Mem0 internal fact extraction ─────────────────
    # Mem0 calls this LLM to parse conversations and extract
    # structured facts like "user's name is Alice" or
    # "user prefers Python over JavaScript".
    # We reuse Ollama so everything stays local.
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0,      # 0 = fully deterministic for fact extraction
            "max_tokens": 2000,
        },
    },
}

# Initialise Mem0 — loads (or creates) the FAISS index from disk.
print("⏳  Loading memory store …", flush=True)
memory = Memory.from_config(MEM0_CONFIG)
print("✅  Memory store ready.\n")

# A fixed user ID.  Mem0 supports multi-user storage; we use
# one constant ID for this single-user demo.
USER_ID = "user_001"


# ════════════════════════════════════════════════════════════
# SECTION 3 — MEMORY HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def store_memory(user_message: str, agent_reply: str) -> None:
    """
    Save a conversation turn (user + assistant) to Mem0.

    Mem0's add() method:
      - Runs the fact-extraction LLM to pull out key information.
      - Embeds the text with the HuggingFace model.
      - Writes the embedding + text to the FAISS index on disk.

    Parameters
    ----------
    user_message : the raw text the user typed
    agent_reply  : the agent's response (ANSI codes already stripped)
    """
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": agent_reply},
    ]
    memory.add(messages, user_id=USER_ID)


def retrieve_memories(query: str, top_k: int = 5) -> str:
    """
    Search Mem0 for memories most relevant to the current query.

    Uses cosine similarity: the query is embedded with the same
    HuggingFace model, then compared against every stored vector.
    The top-k closest matches are returned as a numbered list.

    Parameters
    ----------
    query : the user's current message (used as the search string)
    top_k : how many memories to retrieve (default 5)

    Returns
    -------
    A formatted string ready for insertion into a system prompt,
    or an empty string if no memories are stored yet.
    """
    results = memory.search(query=query, filters={"user_id": USER_ID}, limit=top_k)
    memories_list = results.get("results", [])

    if not memories_list:
        return ""   # agent will reply without memory context

    # Format as a numbered list for readability in the system prompt
    formatted = "\n".join(
        f"{i+1}. {m['memory']}"
        for i, m in enumerate(memories_list)
        if m.get("memory")
    )
    return formatted


# ════════════════════════════════════════════════════════════
# SECTION 4 — ANSI STRIP UTILITY
# ════════════════════════════════════════════════════════════
# Strands can emit ANSI colour/cursor codes in its output.
# We strip them before saving to Mem0 so stored memories
# contain clean plain text rather than terminal escape sequences.

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

def strip_ansi(text: str) -> str:
    """Remove all ANSI escape codes from a string."""
    return ANSI_ESCAPE.sub("", text)


# ════════════════════════════════════════════════════════════
# SECTION 5 — MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════
# Lower temperature (0.3) makes the agent more deterministic
# for tool calls and fact recall, while still being conversational.

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,
)


# ════════════════════════════════════════════════════════════
# SECTION 6 — AGENT FACTORY  (memory-aware, tool-equipped)
# ════════════════════════════════════════════════════════════
#
# Why recreate the Agent on every turn?
#
# Strands Agents store in-session history internally, but their
# system_prompt is fixed at construction time.  By rebuilding the
# agent each turn we can inject a fresh, query-specific memory
# block into the system prompt — giving the LLM relevant context
# without filling its window with every past memory.
#
# In-session conversational history is NOT lost: the conversation
# is stored in Mem0 and the most relevant turns are injected back
# in the system prompt.

BASE_SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with access to tools and a long-term memory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• calculator    — evaluate any mathematical expression
• get_weather   — look up current weather for a city
• calculate_age — compute someone's age from a birth date

Always prefer using a tool when the user's request clearly maps to one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{memory_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIOUR RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Address the user by name whenever you know it.
- When the user shares personal information, acknowledge it naturally.
- When asked to recall something, look in the memory context above.
- Never say you cannot remember things that appear in your memories.
- Be concise but friendly.
"""


def build_agent(query: str) -> Agent:
    """
    Build a Strands Agent whose system prompt is enriched with
    the memories most relevant to the current query.

    Steps
    -----
    1. Search Mem0 for memories related to `query`.
    2. Format them into the BASE_SYSTEM_PROMPT template.
    3. Create and return an Agent with all three tools attached.
    """
    recalled = retrieve_memories(query)

    memory_block = (
        f"What you know about the user from past conversations:\n{recalled}"
        if recalled
        else "(No memories stored yet — this may be the first conversation.)"
    )

    system_prompt = BASE_SYSTEM_PROMPT.format(memory_block=memory_block)

    return Agent(
        model=ollama_model,
        tools=[calculator, get_weather, calculate_age],
        system_prompt=system_prompt,
    )


# ════════════════════════════════════════════════════════════
# SECTION 7 — INTERACTIVE REPL
# ════════════════════════════════════════════════════════════
# The REPL (Read-Eval-Print Loop) lets the user chat with the
# full agent interactively.  Special commands let them inspect
# or manage stored memories without leaving the chat session.

print("=" * 62)
print("  Full Agent — Strands SDK + Ollama  (llama3.2:3b)")
print("  Tools : calculator · weather · age calculator")
print("  Memory: Mem0 + FAISS  (persisted locally)")
print("=" * 62)
print()
print("  TOOL EXAMPLES")
print("    🧮  'What is 17 * 48 + sqrt(256)?'")
print("    🌦  'What's the weather in Tokyo?'")
print("    🎂  'How old is someone born in 1995 on June 15?'")
print()
print("  MEMORY EXAMPLES")
print("    'My name is Alice and I love hiking'")
print("    'What is my name?'           ← tests recall")
print("    'What are my hobbies?'        ← tests recall")
print()
print("  SPECIAL COMMANDS")
print("    memory  → list all stored memories")
print("    clear   → wipe all memories")
print("    help    → show this guide again")
print("    quit    → exit")
print("=" * 62)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        # Ctrl-C / Ctrl-D exits cleanly without a traceback
        print("\nGoodbye!  Your memories are saved for next time.")
        break

    # ── Exit commands ─────────────────────────────────────────
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!  Your memories are saved for next time.")
        break

    # ── Skip blank lines ──────────────────────────────────────
    if not user_input:
        continue

    # ── Special command: show help banner ─────────────────────
    if user_input.lower() == "help":
        print()
        print("  TOOL EXAMPLES")
        print("    🧮  'What is 17 * 48 + sqrt(256)?'")
        print("    🌦  'What's the weather in Tokyo?'")
        print("    🎂  'How old is someone born in 1995 on June 15?'")
        print()
        print("  MEMORY EXAMPLES")
        print("    'My name is Alice and I love hiking'")
        print("    'What is my name?'           ← tests recall")
        print()
        print("  SPECIAL COMMANDS")
        print("    memory · clear · help · quit")
        print()
        continue

    # ── Special command: list all stored memories ─────────────
    if user_input.lower() == "memory":
        all_mem = memory.get_all(filters={"user_id": USER_ID})
        items   = all_mem.get("results", [])
        if items:
            print(f"\n📦  Stored memories ({len(items)} total):")
            for i, m in enumerate(items, 1):
                print(f"  {i}. {m.get('memory', '—')}")
        else:
            print("\n  (no memories stored yet)")
        print()
        continue

    # ── Special command: clear all memories ───────────────────
    if user_input.lower() == "clear":
        memory.delete_all(filters={"user_id": USER_ID})
        print("🗑  All memories cleared.\n")
        continue

    # ── Normal chat turn ──────────────────────────────────────
    # Step 1 — Build the agent with a memory-enriched system prompt.
    #          This retrieves the top-5 most relevant past memories
    #          and injects them so the LLM can recall user details.
    agent = build_agent(user_input)

    # Step 2 — Call the agent.
    #          Strands automatically:
    #            • decides whether to call a tool or answer directly
    #            • executes any tool calls (calculator / weather / age)
    #            • streams the final answer to stdout
    print("\nAgent: ", end="", flush=True)
    response = agent(user_input)

    # Step 3 — Extract clean text from the response object.
    #          response.message["content"] contains the full reply.
    reply_text = strip_ansi(
        str(
            response.message.get("content", "")
            if hasattr(response, "message")
            else str(response)
        )
    )

    print()  # blank line between turns for readability

    # Step 4 — Persist this turn to Mem0 / FAISS.
    #          Mem0 extracts key facts and stores their embeddings
    #          so future sessions can recall them.
    try:
        store_memory(user_input, reply_text)
    except Exception as mem_err:
        # Memory errors must never crash the chat loop.
        print(f"  ⚠  Memory save warning: {mem_err}\n")
