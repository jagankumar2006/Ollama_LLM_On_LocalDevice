# ============================================================
# Challenge 1 - Simple AI Agent using Strands SDK + Ollama
# Model: llama3.2:3b (runs 100% locally, no API keys needed)
# ============================================================

# 1. Import the Agent class from the Strands SDK.
#    Agent is the core class that manages the conversation loop
#    and connects your code to the underlying language model.
from strands import Agent

# 2. Import OllamaModel — the Strands adapter that lets the
#    Agent talk to a locally-running Ollama server.
from strands.models.ollama import OllamaModel

# 3. Create the model instance.
#    host  → URL where Ollama is listening (default port 11434)
#    model_id → the exact model name you pulled with ollama pull
#    temperature → 0.7 gives a good balance: creative but focused
#                  (0.0 = deterministic, 1.0 = very random)
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# 4. Create the Agent and attach the model.
#    You can also pass a system_prompt here to give the agent
#    a persona or set of rules it should always follow.
agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful AI assistant. "
        "Answer clearly and concisely."
    ),
)

# 5. Print a friendly banner so we know the agent is ready.
print("=" * 50)
print("  Simple AI Agent — Strands SDK + Ollama")
print("  Model : llama3.2:3b  (running locally)")
print("  Type  : 'quit' or 'exit' to stop")
print("=" * 50)
print()

# 6. A simple REPL (Read-Eval-Print Loop) so you can chat
#    with the agent interactively from your terminal.
while True:
    # Read input from the user.
    user_input = input("You: ").strip()

    # Allow the user to exit gracefully.
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    # Skip empty lines — no point sending a blank message.
    if not user_input:
        continue

    # 7. Send the message to the agent.
    #    The agent automatically:
    #      - formats the message for the model
    #      - calls the Ollama API at localhost:11434
    #      - streams/prints the response to stdout
    #      - keeps conversation history so follow-up questions work
    print("\nAgent: ", end="", flush=True)
    agent(user_input)
    print()  # blank line between turns for readability
