"""
Example 2: Thinking Agents

This example demonstrates how to create and use agents with different thinking modes
and capabilities, showing how they approach problem-solving differently.
"""
import os
import sys

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig
from generator.agents import create_thinking_agent, AgentRole, ThinkingMode

def main():
    """Main function demonstrating thinking agents."""
    print("=== Example 2: Thinking Agents ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add model configuration (assuming API key is set in environment variable)
    model_util.add_model(ModelConfig(
        model_name="gpt-4",
        provider="openai",
        temperature=0.7,
        max_tokens=1000
    ))
    
    # Example 1: Agent with Chain of Thought thinking
    print("\n=== Part 1: Chain of Thought Thinking ===")
    
    chain_agent = create_thinking_agent(
        name="ChainThinker",
        role=AgentRole.RESEARCHER,
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        verbose=True
    )
    
    # A question requiring analytical thinking
    question = "What are three approaches to reducing carbon emissions, and what are the trade-offs of each?"
    
    print(f"\nQuestion: {question}")
    print("\nThinking process (Chain of Thought):")
    # First, let's see the thinking process
    chain_thinking = chain_agent.think(question)
    print(chain_thinking)
    
    # Now, get the response which will include thinking steps before answering
    print("\nResponse with Chain of Thought:")
    chain_response = chain_agent.respond(question)
    print(chain_response)
    
    # Example 2: Agent with Tree of Thought thinking
    print("\n\n=== Part 2: Tree of Thought Thinking ===")
    
    tree_agent = create_thinking_agent(
        name="TreeThinker",
        role=AgentRole.RESEARCHER,
        model_utility=model_util,
        thinking_mode=ThinkingMode.TREE,
        verbose=True
    )
    
    print(f"\nQuestion: {question}")
    print("\nThinking process (Tree of Thought):")
    # Look at the tree of thought process
    tree_thinking = tree_agent.think(question)
    print(tree_thinking)
    
    # Get the response from tree of thought agent
    print("\nResponse with Tree of Thought:")
    tree_response = tree_agent.respond(question)
    print(tree_response)
    
    # Example 3: Agent with Self-Reflection
    print("\n\n=== Part 3: Self-Reflection Thinking ===")
    
    reflect_agent = create_thinking_agent(
        name="ReflectiveThinker",
        role=AgentRole.RESEARCHER,
        model_utility=model_util,
        thinking_mode=ThinkingMode.REFLECT,
        verbose=True
    )
    
    print(f"\nQuestion: {question}")
    print("\nThinking process (Self-Reflection):")
    # Look at the self-reflection process
    reflect_thinking = reflect_agent.think(question)
    print(reflect_thinking)
    
    # Get the response from reflective agent
    print("\nResponse with Self-Reflection:")
    reflect_response = reflect_agent.respond(question)
    print(reflect_response)
    
    # Example 4: Compare responses to a different type of question
    print("\n\n=== Part 4: Comparing Thinking Modes on a Creative Question ===")
    
    creative_question = "Design a sustainable city transportation system for the year 2050."
    
    # Compare the three thinking modes on this question
    for agent_name, agent in [
        ("Chain of Thought", chain_agent),
        ("Tree of Thought", tree_agent),
        ("Self-Reflection", reflect_agent)
    ]:
        print(f"\n--- {agent_name} Thinking ---")
        thinking = agent.think(creative_question)
        print(f"Thinking excerpt: {thinking[:300]}...")
        
        response = agent.respond(creative_question)
        print(f"Response excerpt: {response[:300]}...")
    
    # Save conversation history to see the full thinking process
    reflect_agent.save_conversation("reflective_agent_conversation.json")
    print("\nReflective agent conversation saved to 'reflective_agent_conversation.json'")


if __name__ == "__main__":
    main()
    print("\n=== End of Example 2 ===")
    print("Note: This example requires a valid OPENAI_API_KEY to be set in environment variables") 