# ============================================================
# Challenge 5 — Local MCP Server
# ============================================================
#
# This file IS the MCP server.  It is launched automatically
# as a subprocess by starter.py — you do NOT need to run it
# manually.
#
# HOW AN MCP SERVER WORKS
# ────────────────────────
# 1. The server registers tools using the @mcp.tool() decorator.
# 2. It starts with mcp.run(transport="stdio"), which means it
#    reads JSON-RPC requests from stdin and writes responses to
#    stdout.
# 3. The MCP client (in starter.py) sends:
#      • initialize  → handshake
#      • tools/list  → get all available tools + their schemas
#      • tools/call  → invoke a specific tool with arguments
# 4. Stdout of this process must stay clean — only MCP protocol
#    JSON is written there.  Use stderr for debug messages.
#
# SETUP
# ──────
#   pip install mcp
# ============================================================

# ── MCP Python SDK ────────────────────────────────────────────
# FastMCP is the high-level MCP server class — it handles all the
# JSON-RPC plumbing and exposes a simple @mcp.tool() decorator API.
from mcp.server.fastmcp import FastMCP

# ── Standard library ─────────────────────────────────────────
import platform     # for get_system_info tool
import sys          # for stderr debug output (never stdout!)


# ════════════════════════════════════════════════════════════
# SECTION 1 — CREATE THE MCP SERVER INSTANCE
# ════════════════════════════════════════════════════════════
#
# FastMCP(name) creates the server.  The name is sent to the
# client during the initialize handshake so the agent knows
# which server it is connected to.

mcp = FastMCP("LocalDemoServer")


# ════════════════════════════════════════════════════════════
# SECTION 2 — TOOL DEFINITIONS
# ════════════════════════════════════════════════════════════
#
# @mcp.tool() works similarly to Strands' @tool decorator:
#   - The function name becomes the tool name.
#   - The docstring becomes the tool description (sent to the LLM).
#   - Type annotations build the JSON schema the model receives.
#
# Return values must be JSON-serialisable (str, int, float,
# dict, list, etc.).


# ── MCP Tool 1 : add ─────────────────────────────────────────
@mcp.tool()
def add(a: float, b: float) -> str:
    """
    Add two numbers together and return the result.

    Use this tool when the user asks to add, sum, or combine
    two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        A human-readable string showing the calculation and result.

    Examples:
        add(10, 32)  → "10.0 + 32.0 = 42.0"
        add(3.14, 2) → "3.14 + 2.0 = 5.140000000000001"
    """
    result = a + b
    # Debug output goes to stderr — never stdout (stdout is MCP protocol)
    print(f"[MCP Server] add({a}, {b}) = {result}", file=sys.stderr)
    return f"{a} + {b} = {result}"


# ── MCP Tool 2 : get_weather ─────────────────────────────────
@mcp.tool()
def get_weather(city: str) -> str:
    """
    Return the current weather conditions for a given city.

    Use this tool whenever the user asks about the weather,
    temperature, forecast, or climate for a specific location.

    Args:
        city: The name of the city to look up.
              Examples: "London", "New York", "Tokyo"

    Returns:
        A weather summary with temperature, conditions, and humidity,
        or a helpful message if the city is not in the dataset.

    Note:
        This uses a simulated dataset for demonstration.  Replace
        the weather_data dictionary with a live API call to connect
        real weather data without changing the tool interface.
    """
    # Simulated weather data — replace with a real API call if needed:
    #   import requests
    #   r = requests.get(f"https://api.openweathermap.org/data/2.5/weather"
    #                    f"?q={city}&appid=YOUR_KEY&units=metric")
    #   data = r.json()
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
    print(f"[MCP Server] get_weather('{city}')", file=sys.stderr)

    if key in weather_data:
        w = weather_data[key]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature : {w['temp']}°C\n"
            f"  Conditions  : {w['condition']}\n"
            f"  Humidity    : {w['humidity']}%"
        )

    available = ", ".join(c.title() for c in weather_data)
    return (
        f"No weather data found for '{city}'. "
        f"Available cities: {available}"
    )


# ── MCP Tool 3 : get_system_info ─────────────────────────────
@mcp.tool()
def get_system_info() -> str:
    """
    Retrieve basic information about the host machine.

    Use this tool when the user asks about the current system,
    operating system, hardware, Python version, or machine details.

    Returns:
        A formatted string containing:
        - Operating system name and version
        - CPU architecture
        - Python version
        - Hostname
    """
    print("[MCP Server] get_system_info()", file=sys.stderr)
    info = {
        "os":           platform.system(),
        "os_version":   platform.version(),
        "architecture": platform.machine(),
        "python":       platform.python_version(),
        "hostname":     platform.node(),
    }
    lines = [
        "System Information:",
        f"  OS          : {info['os']} {info['os_version']}",
        f"  Architecture: {info['architecture']}",
        f"  Python      : {info['python']}",
        f"  Hostname    : {info['hostname']}",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# SECTION 3 — START THE MCP SERVER
# ════════════════════════════════════════════════════════════
#
# mcp.run(transport="stdio") starts the event loop that:
#   - Reads JSON-RPC messages from stdin (line by line)
#   - Dispatches them to the matching tool function
#   - Writes JSON-RPC responses back to stdout
#
# This blocks forever until the parent process (starter.py)
# closes the stdin pipe, which happens when the MCPClient
# context manager exits.

if __name__ == "__main__":
    print("[MCP Server] Starting on stdio transport …", file=sys.stderr)
    mcp.run(transport="stdio")
