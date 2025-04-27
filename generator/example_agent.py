"""
Example usage of the Agent class with thinking capabilities.
"""
from llms import ModelUtility, ModelConfig
from agents import Agent, AgentRole, ThinkingMode, create_thinking_agent, MultiAgentSystem
from tools import ToolCategory, ToolManager

def main():
    # Create a model utility and add some models
    model_util = ModelUtility()
    
    # Add model configurations (assuming API keys are set in environment variables)
    model_util.add_model(ModelConfig(
        model_name="gpt-4",
        provider="openai",
        temperature=0.7,
        max_tokens=1000
    ))
    
    model_util.add_model(ModelConfig(
        model_name="claude-3-opus-20240229",
        provider="anthropic",
        temperature=0.5,
        max_tokens=2000
    ))
    
    # Example 1: Single thinking agent
    print("=== Example 1: Single Thinking Agent ===")
    
    # Create a thinking agent
    agent = create_thinking_agent(
        name="Sherlock",
        role=AgentRole.RESEARCHER,
        model_utility=model_util,
        thinking_model="gpt-4",  # Use GPT-4 for thinking
        response_model="gpt-4",  # Use GPT-4 for responding
        thinking_mode=ThinkingMode.CHAIN,  # Use chain-of-thought thinking
        verbose=True
    )
    
    # Ask the agent a question requiring thinking
    user_question = "What are three potential solutions to climate change, and what are their pros and cons?"
    
    # Get the agent's thinking process
    thinking = agent.think(user_question)
    print("\nAgent's thinking process:")
    print(thinking)
    
    # Get the agent's response
    response = agent.respond(user_question)
    print("\nAgent's response:")
    print(response)
    
    # Example 2: Agent with different thinking modes
    print("\n=== Example 2: Different Thinking Modes ===")
    
    # Create agents with different thinking modes
    chain_agent = create_thinking_agent(
        name="ChainThinker",
        role=AgentRole.ASSISTANT,
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        verbose=True
    )
    
    tree_agent = create_thinking_agent(
        name="TreeThinker",
        role=AgentRole.ASSISTANT,
        model_utility=model_util,
        thinking_mode=ThinkingMode.TREE,
        verbose=True
    )
    
    reflect_agent = create_thinking_agent(
        name="ReflectiveThinker",
        role=AgentRole.ASSISTANT,
        model_utility=model_util,
        thinking_mode=ThinkingMode.REFLECT,
        verbose=True
    )
    
    # Complex question requiring analytical thinking
    complex_question = "How might quantum computing affect cryptography and cybersecurity in the next decade?"
    
    for agent in [chain_agent, tree_agent, reflect_agent]:
        print(f"\n--- {agent.name} with {agent.thinking_mode} thinking ---")
        thinking = agent.think(complex_question)
        print(f"Thinking process snippet: {thinking[:200]}...")
        response = agent.respond(complex_question, with_thinking=True)
        print(f"Response snippet: {response[:200]}...")
    
    # Example 3: Multi-agent system
    print("\n=== Example 3: Multi-Agent System ===")
    
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
    multi_agent_question = "What are the implications of widespread facial recognition technology?"
    
    # Get and display the response
    response, agent = system.route_message(multi_agent_question)
    print(f"\nQuestion routed to: {agent.name}")
    print(f"Response snippet: {response[:200]}...")
    
    # Save agent conversation history
    agent.save_conversation("conversation_history.json")
    print("\nConversation history saved to conversation_history.json")
    
    # Example 4: Agent with tools
    print("\n=== Example 4: Agent with Tools ===")
    
    # Create a tool manager to explore available tools
    tool_manager = ToolManager()
    
    # Get available tool categories
    tool_categories = {cat.value for cat in ToolCategory}
    print(f"Available tool categories: {', '.join(tool_categories)}")
    
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
    tool_questions = [
        "What is 25 * 48 + 172 / 4?",
        "What is the current date and time?",
        "Can you format this data as JSON: name=John Smith, age=35, occupation=Engineer",
        "Create a file called sample.txt with the content 'This is a sample file created by the agent.'"
    ]
    
    for question in tool_questions:
        print(f"\nQuestion: {question}")
        response = tool_agent.respond(question)
        print(f"Response: {response}")
    
    # Example 5: Researcher agent with web search tools
    print("\n=== Example 5: Researcher Agent with Web Search Tools ===")
    
    researcher_agent = create_thinking_agent(
        name="WebResearcher",
        role=AgentRole.RESEARCHER,
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        use_tools=True,
        system_prompt=(
            "You are a helpful research assistant who uses search tools to find information. "
            "When asked a question, you should use search_web to find information, "
            "and then synthesize it into a comprehensive answer."
        ),
        verbose=True
    )
    
    research_question = "What are the latest developments in renewable energy technology?"
    print(f"\nResearch question: {research_question}")
    response = researcher_agent.respond(research_question)
    print(f"Response: {response}")
    
    # Display tool usage history
    tool_usage = researcher_agent.get_tool_usage_history()
    print(f"\nTool usage history: {len(tool_usage)} tool calls")
    for i, usage in enumerate(tool_usage, 1):
        print(f"{i}. {usage['tool_name']}({', '.join([f'{k}={v}' for k, v in usage['parameters'].items()])})")


if __name__ == "__main__":
    main() 