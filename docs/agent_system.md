# Agent System Documentation

The Agent System provides a framework for creating AI agents capable of using tools to accomplish tasks. This document explains how to create and configure agents, define custom tools, and manage interactions between agents and tools.

## Overview

The Agent System consists of several key components:

1. **Agents**: AI assistants that can use tools to complete tasks
2. **Tools**: Functions that agents can call to perform specific actions
3. **ToolManager**: A registry that manages tool availability
4. **Prompts**: Templates that define agent behavior and tool usage
5. **Memory**: Systems for storing and retrieving conversation history

## Creating an Agent

### Basic Agent Creation

```python
from generator.agents import Agent
from generator.llms import ModelUtility, ModelConfig

# Set up a model utility
model_util = ModelUtility()
model_util.add_model(ModelConfig(
    model_name="gpt-4",
    provider="openai",
    temperature=0.7
))

# Create a basic agent
agent = Agent(
    name="ResearchAssistant",
    description="An AI assistant that helps with research tasks",
    model_utility=model_util
)

# Use the agent to respond to a query
response = agent.chat("What is the capital of France?")
print(response)
```

### Agent with Custom System Prompt

```python
from generator.agents import Agent
from generator.llms import ModelUtility

# Create a model utility
model_util = ModelUtility()
# ... configure models ...

# Create an agent with a custom system prompt
system_prompt = """You are a coding assistant with expertise in Python.
Your goal is to help users write clean, efficient, and well-documented code.
Always suggest best practices and provide explanations for your code suggestions."""

agent = Agent(
    name="PythonCodeExpert",
    description="Python coding assistant",
    model_utility=model_util,
    system_prompt=system_prompt
)

# Use the agent
response = agent.chat("How do I implement a binary search in Python?")
print(response)
```

## Configuring Agents with Tools

### Registering Default Tools

```python
from generator.agents import Agent
from generator.tools import ToolManager
from generator.default_tools import web_search, calculator, date_time

# Create a tool manager and register some default tools
tool_manager = ToolManager()
tool_manager.register_tool(web_search)
tool_manager.register_tool(calculator)
tool_manager.register_tool(date_time)

# Create an agent with tools
agent = Agent(
    name="ResearchAssistant",
    description="Research assistant with web search capabilities",
    model_utility=model_util,
    tool_manager=tool_manager
)

# Now the agent can use tools
response = agent.chat("What is the population of Tokyo and what percentage is it of Japan's total population?")
print(response)
```

### Creating and Registering Custom Tools

```python
from generator.agents import Agent
from generator.tools import ToolManager, Tool
import random
import requests

# Define a custom tool for weather information
def get_weather(location, unit="celsius"):
    """Get current weather information for a location.

    Args:
        location (str): City name or location to get weather for
        unit (str, optional): Temperature unit ("celsius" or "fahrenheit")

    Returns:
        dict: Weather information including temperature, conditions, and humidity
    """
    # In a real implementation, this would call a weather API
    # This is a mock implementation for demonstration
    weather_conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]
    temp = random.randint(0, 30) if unit == "celsius" else random.randint(32, 86)

    return {
        "location": location,
        "temperature": f"{temp}°{'C' if unit == 'celsius' else 'F'}",
        "condition": random.choice(weather_conditions),
        "humidity": f"{random.randint(30, 90)}%"
    }

# Create a Tool object from the function
weather_tool = Tool(
    name="get_weather",
    description="Get current weather information for a location",
    function=get_weather
)

# Register the custom tool
tool_manager = ToolManager()
tool_manager.register_tool(weather_tool)

# Create an agent with the custom tool
agent = Agent(
    name="WeatherAssistant",
    description="Assistant that can provide weather information",
    model_utility=model_util,
    tool_manager=tool_manager
)

# Test the agent with the custom tool
response = agent.chat("What's the weather like in Paris today?")
print(response)
```

## Tool Management

### The ToolManager Class

The `ToolManager` class is responsible for registering, retrieving, and managing tools:

```python
from generator.tools import ToolManager, Tool

# Create a tool manager
tool_manager = ToolManager()

# Register tools
tool_manager.register_tool(my_tool)
tool_manager.register_tools([tool1, tool2, tool3])  # Register multiple tools at once

# Get a tool by name
my_tool = tool_manager.get_tool("my_tool_name")

# Check if a tool exists
if tool_manager.has_tool("my_tool_name"):
    print("Tool is available!")

# Get all registered tools
all_tools = tool_manager.get_all_tools()

# Get tool descriptions for prompting
tool_descriptions = tool_manager.get_tool_descriptions()
```

### Tool Definition Structure

A Tool is defined with the following attributes:

```python
from generator.tools import Tool

my_tool = Tool(
    name="tool_name",  # Unique identifier for the tool
    description="Description of what the tool does",  # Used in prompts
    function=my_function,  # The actual function to call
    parameters={  # JSON Schema for parameters
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of parameter 1"
            },
            "param2": {
                "type": "integer",
                "description": "Description of parameter 2"
            }
        },
        "required": ["param1"]
    }
)
```

If using a simple function with type hints and docstrings, the `parameters` field can be automatically generated:

```python
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Sum of the two numbers
    """
    return a + b

# Create tool with auto-generated parameters from function signature
calculator_tool = Tool.from_function(add_numbers)
```

## Agent-Tool Interaction

When an agent uses a tool, the following process occurs:

1. The agent determines which tool to use based on the user's query
2. The agent formats the tool call with appropriate parameters
3. The `ToolManager` executes the tool with the provided parameters
4. The tool result is returned to the agent
5. The agent incorporates the tool's response into its final answer

### Tool History

You can access a history of tool usages to track what tools an agent has used:

```python
# Create and use an agent with tools
agent = Agent(name="MultiToolAgent", model_utility=model_util, tool_manager=tool_manager)
agent.chat("What's the weather like in Tokyo and what's the population?")

# Get tool usage history
tool_history = agent.get_tool_history()

# Print tool usage details
print(f"Total tool usages: {len(tool_history)}")
for i, usage in enumerate(tool_history):
    print(f"Tool usage {i+1}:")
    print(f"  Tool: {usage['tool_name']}")
    print(f"  Parameters: {usage['parameters']}")
    print(f"  Result: {usage['result']}")
    print(f"  Timestamp: {usage['timestamp']}")
```

## Advanced Agent Configurations

### Agents with Memory

```python
from generator.agents import Agent
from generator.memory import ConversationMemory

# Create a memory system
memory = ConversationMemory(max_messages=10)

# Create an agent with memory
agent = Agent(
    name="MemoryEnabledAgent",
    description="Agent that remembers conversation history",
    model_utility=model_util,
    tool_manager=tool_manager,
    memory=memory
)

# Have a multi-turn conversation
agent.chat("My name is Alice.")
response = agent.chat("What's my name?")  # Agent should remember "Alice"
print(response)
```

### Multi-Agent Systems

```python
from generator.agents import Agent

# Create specialized agents
research_agent = Agent(
    name="Researcher",
    description="Agent specialized in research",
    model_utility=model_util,
    tool_manager=tool_manager_research
)

coding_agent = Agent(
    name="Coder",
    description="Agent specialized in writing code",
    model_utility=model_util,
    tool_manager=tool_manager_coding
)

# Use agents for different parts of a workflow
research_result = research_agent.chat("Find information about sorting algorithms")
coding_result = coding_agent.chat(f"Using this information: {research_result}, implement a quicksort algorithm in Python")

print(coding_result)
```

## Best Practices

### Tool Design Principles

1. **Single Responsibility**: Each tool should do one thing well
2. **Clear Documentation**: Provide clear descriptions and parameter documentation
3. **Robust Error Handling**: Handle errors gracefully and return informative error messages
4. **Parameter Validation**: Validate inputs to prevent misuse
5. **Security Considerations**: Be cautious with tools that access sensitive data or systems

### Agent Configuration Tips

1. **Appropriate System Prompts**: Define clear roles and instructions
2. **Tool Selection**: Only provide tools relevant to the agent's purpose
3. **Temperature Settings**: Use lower temperatures (0.2-0.4) for more deterministic tool use
4. **Memory Management**: Clear memory when switching contexts
5. **Monitoring**: Monitor and log tool usage for debugging

## Example: Complete Agent Workflow

```python
import os
from generator.agents import Agent
from generator.llms import ModelUtility, ModelConfig
from generator.tools import ToolManager, Tool
from generator.default_tools import web_search, calculator
from generator.memory import ConversationMemory

# Set up API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create model utility
model_util = ModelUtility()
model_util.add_model(ModelConfig(
    model_name="gpt-4",
    provider="openai",
    temperature=0.4  # Lower temperature for more consistent tool use
))

# Define custom tools
def generate_report(data, format="markdown"):
    """Generate a formatted report from data.

    Args:
        data (str): The data to include in the report
        format (str): Format of the report (markdown, html, or plain)

    Returns:
        str: Formatted report
    """
    if format == "markdown":
        return f"# Generated Report\n\n## Data\n\n{data}"
    elif format == "html":
        return f"<h1>Generated Report</h1><h2>Data</h2><p>{data}</p>"
    else:
        return f"GENERATED REPORT\n\nDATA:\n{data}"

# Create tools
report_tool = Tool.from_function(generate_report)

# Set up tool manager
tool_manager = ToolManager()
tool_manager.register_tools([web_search, calculator, report_tool])

# Create memory system
memory = ConversationMemory()

# Create agent
research_assistant = Agent(
    name="ResearchReportAssistant",
    description="An assistant that can research topics and generate reports",
    model_utility=model_util,
    tool_manager=tool_manager,
    memory=memory,
    system_prompt="""You are a research assistant capable of finding information
    and generating formatted reports. Use web_search to find information,
    calculator for calculations, and generate_report to create final reports."""
)

# Use the agent for a research task
user_query = "Research the impact of climate change on coral reefs and generate a report."
response = research_assistant.chat(user_query)
print(response)

# Check what tools were used
tool_history = research_assistant.get_tool_history()
print(f"\nTools used: {len(tool_history)}")
for usage in tool_history:
    print(f"- {usage['tool_name']}")
```

## Troubleshooting

### Common Issues and Solutions

1. **Tool Not Being Used**

   - Check tool description clarity
   - Verify the agent's system prompt mentions tool usage
   - Lower the temperature setting

2. **Incorrect Parameter Formatting**

   - Review parameter schema definitions
   - Check for clear parameter descriptions
   - Use type hints and docstrings for auto-generated parameters

3. **Agent Hallucinating Tool Results**

   - Ensure tools return concrete, specific results
   - Check that tool outputs are properly formatted
   - Use appropriate model settings (lower temperature)

4. **Memory Issues**
   - Verify memory is properly initialized
   - Check memory capacity limits
   - Clear memory if context becomes too large
