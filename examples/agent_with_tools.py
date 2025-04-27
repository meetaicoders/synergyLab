"""
Example 4: Agent with Tools

This example demonstrates how to create an agent that can use various tools to perform tasks,
including web search, file operations, mathematical calculations, and more.
"""
import os
import sys

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig
from generator.agents import create_thinking_agent, AgentRole, ThinkingMode
from generator.tools import ToolManager, ToolCategory

def main():
    """Main function demonstrating agent with tools."""
    print("=== Example 4: Agent with Tools ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add model configuration (assuming API key is set in environment variable)
    model_util.add_model(ModelConfig(
        model_name="gpt-4",
        provider="openai",
        temperature=0.7,
        max_tokens=1000
    ))
    
    # Step 1: Create a Tool Manager and explore available tools
    print("\n=== Step 1: Exploring Available Tools ===")
    tool_manager = ToolManager()
    
    # Get all available tool categories
    tool_categories = {cat.value for cat in ToolCategory}
    print(f"Available tool categories: {', '.join(tool_categories)}")
    
    # List tools in each category
    for category in ToolCategory:
        tools = tool_manager.get_tools_by_category(category)
        if tools:
            print(f"\nTools in category '{category.value}':")
            for tool_name in tools:
                tool_desc = tool_manager.get_tool_description(tool_name)
                print(f"  - {tool_name}: {tool_desc.description}")
    
    # Step 2: Create an agent with tools enabled
    print("\n=== Step 2: Creating Agent with Tools ===")
    tool_agent = create_thinking_agent(
        name="ToolUser",
        role=AgentRole.ASSISTANT,
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        use_tools=True,  # Enable tools
        system_prompt=(
            "You are a helpful assistant with access to various tools. "
            "When asked to perform tasks, use the most appropriate tool. "
            "Always think step-by-step about which tool to use and why."
        ),
        verbose=True
    )
    print("Created agent with tools enabled")
    
    # Step 3: Use the agent to perform tasks with different tools
    print("\n=== Step 3: Using Tools to Perform Tasks ===")
    
    # Example 1: Mathematical calculation
    math_question = "What is the result of 25 * 48 + 172 / 4?"
    print(f"\nTask 1 (Math): {math_question}")
    response = tool_agent.respond(math_question)
    print(f"Response: {response}")
    
    # Example 2: Get current time
    time_question = "What is the current date and time?"
    print(f"\nTask 2 (Time): {time_question}")
    response = tool_agent.respond(time_question)
    print(f"Response: {response}")
    
    # Example 3: Format data as JSON
    json_question = "Can you format this data as JSON: name=John Smith, age=35, occupation=Engineer, skills=[Python, Java, AI]"
    print(f"\nTask 3 (JSON): {json_question}")
    response = tool_agent.respond(json_question)
    print(f"Response: {response}")
    
    # Example 4: File operations (create, read, list)
    file_question = "Please create a file called 'sample_data.txt' with the content 'This is a sample file created by the agent.' Then read the content back to confirm."
    print(f"\nTask 4 (File): {file_question}")
    response = tool_agent.respond(file_question)
    print(f"Response: {response}")
    
    list_question = "List all text files in the current directory."
    print(f"\nTask 5 (List Files): {list_question}")
    response = tool_agent.respond(list_question)
    print(f"Response: {response}")
    
    # Example 5: Web search (this uses a mock implementation)
    search_question = "What are the latest developments in renewable energy?"
    print(f"\nTask 6 (Web Search): {search_question}")
    response = tool_agent.respond(search_question)
    print(f"Response: {response}")
    
    # Example 6: Multi-step task requiring multiple tools
    complex_question = "Calculate the square root of 144, add 10 to it, and save the result to a file called 'calculation_result.txt'."
    print(f"\nTask 7 (Multi-step): {complex_question}")
    response = tool_agent.respond(complex_question)
    print(f"Response: {response}")
    
    # Step 4: Review tool usage history
    print("\n=== Step 4: Reviewing Tool Usage History ===")
    tool_usage = tool_agent.get_tool_usage_history()
    print(f"Total tool usages: {len(tool_usage)}")
    
    for i, usage in enumerate(tool_usage, 1):
        print(f"\nTool Usage {i}:")
        print(f"Tool: {usage['tool_name']}")
        print(f"Parameters: {usage['parameters']}")
        if usage['error']:
            print(f"Error: {usage['error']}")
        else:
            # For brevity, limit result output
            result_str = str(usage['result'])
            if len(result_str) > 100:
                result_str = result_str[:100] + "..."
            print(f"Result: {result_str}")


if __name__ == "__main__":
    main()
    print("\n=== End of Example 4 ===")
    print("Note: This example requires a valid OPENAI_API_KEY to be set in environment variables")
    print("Some tools may have limitations in this demo implementation (e.g., web search is mocked)") 