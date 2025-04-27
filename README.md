# SynergyLab

A powerful Python framework for orchestrating intelligent AI systems with multiple LLMs, thinking agents, and collaborative multi-agent architectures. SynergyLab seamlessly integrates with various LLM providers including OpenAI, Anthropic, HuggingFace, Cohere, Mistral, and Google, enabling sophisticated AI capabilities through a unified interface.

## Key Capabilities

- 🔄 **Unified LLM Integration** - Single interface for multiple providers with easy switching
- 🧠 **Cognitive Agents** - Agents with advanced thinking capabilities (Chain of Thought, Tree of Thought, Reflection)
- 🤝 **Multi-Agent Systems** - Build collaborative teams of specialized AI agents
- 🛠️ **Extensible Tool Framework** - Expand agent capabilities through custom tools
- 💾 **Conversation Memory** - Sophisticated context management for natural interactions
- ⚙️ **Robust Configuration** - JSON-based config with environment variable support
- 🔒 **Error Handling** - Graceful error management with informative messages

## Usage

### Basic LLM Integration

```python
from generator.llms import ModelUtility, ModelConfig

# Create a model utility
model_util = ModelUtility()

# Add model configurations
model_util.add_model(ModelConfig(
    model_name="gpt-4",
    provider="openai",
    temperature=0.7,
    max_tokens=1000
))

# Generate text
response = model_util.generate_text(
    "Explain the concept of artificial intelligence in simple terms."
)
print(response)
```

### Using Multiple Models

```python
# Add another model
model_util.add_model(ModelConfig(
    model_name="claude-3-opus-20240229",
    provider="anthropic",
    temperature=0.5,
    max_tokens=2000
))

# Switch between models
model_util.set_active_model("claude-3-opus-20240229")

# Generate text with the new model
response = model_util.generate_text(
    "What are the potential applications of large language models?"
)
print(response)
```

### Configuration Management

```python
# Save configurations to a file
model_util.save_configs_to_file("model_configs.json")

# Load configurations from a file
model_util = ModelUtility()
model_util.load_configs_from_file("model_configs.json")
```

### Creating an Agent with Thinking Capabilities

```python
from generator.agents import create_thinking_agent, AgentRole, ThinkingMode

# Create a thinking agent
agent = create_thinking_agent(
    name="Sherlock",
    role=AgentRole.RESEARCHER,
    model_utility=model_util,
    thinking_model="gpt-4",  # Model to use for thinking
    response_model="gpt-4",  # Model to use for responses
    thinking_mode=ThinkingMode.CHAIN,  # Use chain-of-thought thinking
    verbose=True
)

# Ask the agent a question requiring thinking
user_question = "What are three potential solutions to climate change, and what are their pros and cons?"

# Get the agent's thinking process
thinking = agent.think(user_question)
print("Agent's thinking process:")
print(thinking)

# Get the agent's response
response = agent.respond(user_question)
print("Agent's response:")
print(response)
```

### Using Different Thinking Modes

```python
# Chain of Thought thinking
chain_agent = create_thinking_agent(
    name="ChainThinker",
    role=AgentRole.ASSISTANT,
    model_utility=model_util,
    thinking_mode=ThinkingMode.CHAIN
)

# Tree of Thought thinking
tree_agent = create_thinking_agent(
    name="TreeThinker",
    role=AgentRole.ASSISTANT,
    model_utility=model_util,
    thinking_mode=ThinkingMode.TREE
)

# Reflective thinking
reflect_agent = create_thinking_agent(
    name="ReflectiveThinker",
    role=AgentRole.ASSISTANT,
    model_utility=model_util,
    thinking_mode=ThinkingMode.REFLECT
)
```

### Creating a Multi-Agent System

```python
from generator.agents import MultiAgentSystem

# Create specialized agents
tech_agent = create_thinking_agent(
    name="TechExpert",
    role="technology expert",
    model_utility=model_util,
    thinking_mode=ThinkingMode.CHAIN,
    system_prompt="You are a technology expert who specializes in explaining technical concepts clearly."
)

ethics_agent = create_thinking_agent(
    name="EthicsExpert",
    role="ethics specialist",
    model_utility=model_util,
    thinking_mode=ThinkingMode.REFLECT,
    system_prompt="You are an ethics specialist who focuses on the ethical implications of technologies and decisions."
)

coordinator = create_thinking_agent(
    name="Coordinator",
    role=AgentRole.PLANNER,
    model_utility=model_util,
    thinking_mode=ThinkingMode.NONE
)

# Create a multi-agent system with the coordinator
system = MultiAgentSystem(coordinator)
system.add_agent(tech_agent)
system.add_agent(ethics_agent)

# Ask the system a question
response, agent = system.route_message("What are the implications of widespread facial recognition technology?")
print(f"Question routed to: {agent.name}")
print(f"Response: {response}")
```

### Using Agents with Tools

```python
from generator.agents import create_thinking_agent, AgentRole, ThinkingMode
from generator.tools import ToolManager

# Create an agent with tools enabled
tool_agent = create_thinking_agent(
    name="ToolUser",
    role=AgentRole.ASSISTANT,
    model_utility=model_util,
    thinking_mode=ThinkingMode.CHAIN,
    use_tools=True,  # Enable tools
    verbose=True
)

# Ask the agent to perform tasks requiring tools
tool_question = "What is 25 * 48 + 172 / 4?"
response = tool_agent.respond(tool_question)
print(f"Response: {response}")

# Get tool usage history
tool_usage = tool_agent.get_tool_usage_history()
for usage in tool_usage:
    print(f"{usage['tool_name']}({', '.join([f'{k}={v}' for k, v in usage['parameters'].items()])})")
```

### Creating Custom Tools

```python
from generator.tools import tool, ToolCategory, ToolResult

# Create a custom tool using the @tool decorator
@tool(
    description="Greet a person by name.",
    category=ToolCategory.CUSTOM,
    return_description="A greeting message.",
    examples=["greet_person(name='John')"]
)
def greet_person(name: str, formal: bool = False) -> str:
    """
    Greet a person by name.

    Args:
        name: The name of the person to greet
        formal: Whether to use a formal greeting (default: False)

    Returns:
        A greeting message
    """
    if formal:
        return f"Good day, {name}. How may I assist you?"
    else:
        return f"Hi {name}! How can I help you today?"

# Create a tool manager and register your tool
from generator.tools import ToolManager
tool_manager = ToolManager()

# Now create an agent that can use your tool
tool_agent = create_thinking_agent(
    name="Greeter",
    role=AgentRole.ASSISTANT,
    model_utility=model_util,
    use_tools=True
)

# The agent can now use your custom greeting tool
response = tool_agent.respond("Can you greet John formally?")
print(response)
```

## Supported Components

### LLM Providers

- OpenAI (`"openai"`)
- Anthropic (`"anthropic"`)
- HuggingFace (`"huggingface"`)
- Cohere (`"cohere"`)
- Mistral (`"mistral"`)
- Google (`"google"`)

### Agent Roles

- Assistant (`AgentRole.ASSISTANT`)
- Researcher (`AgentRole.RESEARCHER`)
- Coder (`AgentRole.CODER`)
- Critic (`AgentRole.CRITIC`)
- Planner (`AgentRole.PLANNER`)
- Executor (`AgentRole.EXECUTOR`)
- Custom (`AgentRole.CUSTOM` or any string)

### Thinking Modes

- None (`ThinkingMode.NONE`) - Direct response without explicit thinking
- Chain of Thought (`ThinkingMode.CHAIN`) - Linear step-by-step thinking
- Tree of Thought (`ThinkingMode.TREE`) - Consider multiple approaches
- Reflection (`ThinkingMode.REFLECT`) - Self-reflective thinking

### Built-in Tool Categories

- Web (`ToolCategory.WEB`) - Web search, webpage fetching
- File (`ToolCategory.FILE`) - File operations (read, write, list)
- Math (`ToolCategory.MATH`) - Mathematical calculations
- Time (`ToolCategory.TIME`) - Date and time operations
- Search (`ToolCategory.SEARCH`) - Search operations
- Database (`ToolCategory.DATABASE`) - Database operations
- API (`ToolCategory.API`) - API calls
- Custom (`ToolCategory.CUSTOM`) - Custom tools

## Installation

1. Clone this repository
2. Install dependencies for the providers you plan to use:

```bash
# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For HuggingFace
pip install huggingface_hub

# For Cohere
pip install cohere

# For Mistral
pip install mistralai

# For Google
pip install google-generativeai
```

## Configuration Format

The JSON configuration file should follow this format:

```json
{
  "model-name": {
    "model_name": "model-name",
    "provider": "provider-name",
    "temperature": 0.7,
    "max_tokens": 1000,
    "parameters": {
      "additional_param1": "value1",
      "additional_param2": "value2"
    }
  }
}
```

## Environment Variables

You can set API keys as environment variables using the format:

```
PROVIDER_API_KEY=your-api-key
```

Examples:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `HUGGINGFACE_API_KEY`
- `COHERE_API_KEY`
- `MISTRAL_API_KEY`
- `GOOGLE_API_KEY`

## Examples

See the `examples/` directory for full working examples:

- Basic LLM Usage
- Thinking Agents
- Multi-Agent Systems
- Agent with Tools
- Custom Tools

## Documentation

See the `docs/` directory for comprehensive documentation:

- Model Utility
- Agent System
- Conversation Memory
- Multi-Agent Systems
