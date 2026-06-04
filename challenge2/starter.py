# ============================================================
# Challenge 2 — Tools Agent using Strands SDK + Ollama
# Model : llama3.2:3b  (runs 100% locally, no API keys needed)
# Tools : calculator · weather · age calculator
# ============================================================

# ── Imports ──────────────────────────────────────────────────
from datetime import date          # used by the age calculator tool
from strands import Agent, tool    # 'tool' decorator turns a function into an agent tool
from strands.models.ollama import OllamaModel


# ════════════════════════════════════════════════════════════
# SECTION 1 — TOOL DEFINITIONS
# ════════════════════════════════════════════════════════════
# The @tool decorator does three things:
#   1. Registers the function so the agent knows it exists.
#   2. Uses the function's name as the tool name.
#   3. Uses the docstring as the tool description the LLM reads
#      to decide *when* and *how* to call the tool.
#
# The type annotations are important — Strands uses them to
# build the JSON schema that the model receives, so it knows
# exactly what arguments to pass.

# ── Tool 1 : Calculator ──────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Use this tool whenever the user asks you to perform any
    arithmetic or mathematical calculation such as addition,
    subtraction, multiplication, division, powers, or
    percentages.

    Args:
        expression: A valid Python math expression as a string.
                    Examples: "2 + 2", "100 * 0.15", "2 ** 10"

    Returns:
        The result as a string, or an error message if the
        expression is invalid.
    """
    try:
        # eval() is intentionally scoped to math-safe builtins
        # so arbitrary code cannot be injected.
        allowed_names = {"__builtins__": {}}
        import math
        allowed_names.update(vars(math))   # allow sin, cos, sqrt …
        result = eval(expression, allowed_names)  # noqa: S307
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
        city: The name of the city to get weather for.
              Examples: "London", "New York", "Tokyo"

    Returns:
        A weather summary string with temperature and conditions.

    Note:
        This is a simulated weather tool for demonstration
        purposes. In a real agent you would call a live API
        such as OpenWeatherMap here.
    """
    # Simulated weather data — swap this dict for a real API
    # call (e.g. requests.get("https://api.openweathermap.org/…"))
    # without changing anything else.
    weather_data = {
        "london":        {"temp": 15, "condition": "Cloudy",       "humidity": 72},
        "new york":      {"temp": 22, "condition": "Sunny",        "humidity": 55},
        "tokyo":         {"temp": 28, "condition": "Partly Cloudy","humidity": 68},
        "paris":         {"temp": 18, "condition": "Rainy",        "humidity": 80},
        "sydney":        {"temp": 20, "condition": "Clear",        "humidity": 60},
        "dubai":         {"temp": 38, "condition": "Hot & Sunny",  "humidity": 40},
        "toronto":       {"temp": 12, "condition": "Windy",        "humidity": 65},
        "berlin":        {"temp": 14, "condition": "Overcast",     "humidity": 75},
        "mumbai":        {"temp": 32, "condition": "Humid",        "humidity": 85},
        "los angeles":   {"temp": 26, "condition": "Sunny",        "humidity": 45},
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
    else:
        # Graceful fallback for cities not in our mock dataset
        return (
            f"Weather data for '{city}' is not available in the "
            f"demo dataset. Available cities: "
            + ", ".join(c.title() for c in weather_data)
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
        birth_year:  The four-digit year of birth. Example: 1990
        birth_month: The month of birth as a number (1–12).
                     Defaults to 1 (January) if not provided.
        birth_day:   The day of birth (1–31).
                     Defaults to 1 if not provided.

    Returns:
        A string describing the person's age in years and months,
        plus the day of the week they were born.
    """
    try:
        today     = date.today()
        birthdate = date(birth_year, birth_month, birth_day)

        if birthdate > today:
            return "That birth date is in the future — please check the year."

        # Calculate full years
        years = today.year - birthdate.year
        # Subtract 1 if the birthday hasn't occurred yet this year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            years -= 1

        # Remaining months after the last birthday
        months = today.month - birthdate.month
        if today.day < birthdate.day:
            months -= 1
        if months < 0:
            months += 12

        # Day of the week the person was born
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
# SECTION 2 — MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════
# Same OllamaModel setup as Challenge 1, but now we'll attach
# tools so the model can call our functions above.

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,   # lower than challenge 1 — tool-use benefits
                       # from more deterministic, focused responses
)


# ════════════════════════════════════════════════════════════
# SECTION 3 — AGENT CREATION (with tools)
# ════════════════════════════════════════════════════════════
# The key difference from Challenge 1: we pass a `tools` list.
# Strands sends each tool's schema to the model so it knows
# what functions are available and how to invoke them.

agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, calculate_age],   # ← the three tools
    system_prompt=(
        "You are a helpful assistant with access to three tools:\n"
        "  • calculator      — evaluate any math expression\n"
        "  • get_weather     — look up current weather for a city\n"
        "  • calculate_age   — compute someone's age from a birth date\n\n"
        "Always use a tool when the user's question clearly maps to one. "
        "For everything else, answer from your own knowledge. "
        "Be concise and friendly."
    ),
)


# ════════════════════════════════════════════════════════════
# SECTION 4 — INTERACTIVE REPL
# ════════════════════════════════════════════════════════════
# The same REPL pattern from Challenge 1, but we print a richer
# banner that lists the available tools as hints to the user.

print("=" * 58)
print("  Tools Agent — Strands SDK + Ollama  (llama3.2:3b)")
print("=" * 58)
print("  Available tools:")
print("    🧮  Calculator   e.g. 'What is 17 * 48?'")
print("    🌦  Weather      e.g. 'What's the weather in Tokyo?'")
print("    🎂  Age Calc     e.g. 'How old is someone born in 1990?'")
print()
print("  Type 'quit' or 'exit' to stop.")
print("=" * 58)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        # Ctrl-C or Ctrl-D exits cleanly
        print("\nGoodbye!")
        break

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not user_input:
        continue

    print("\nAgent: ", end="", flush=True)
    agent(user_input)
    print()  # blank line between turns
