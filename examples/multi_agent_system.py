"""
Example 3: Multi-Agent System

This example demonstrates how to create a multi-agent system with specialized agents that
work together to solve complex problems, with a coordinator agent routing questions
to the appropriate specialist.
"""
import os
import sys

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig
from generator.agents import create_thinking_agent, AgentRole, ThinkingMode, MultiAgentSystem

def main():
    """Main function demonstrating a multi-agent system."""
    print("=== Example 3: Multi-Agent System ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add model configuration (assuming API key is set in environment variable)
    model_util.add_model(ModelConfig(
        model_name="gpt-4",
        provider="openai",
        temperature=0.7,
        max_tokens=1000
    ))
    
    # Step 1: Create specialized agents for different domains
    print("\n=== Step 1: Creating Specialized Agents ===")
    
    # Technical expert specializing in technology
    tech_agent = create_thinking_agent(
        name="TechExpert",
        role="technology expert",
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        system_prompt=(
            "You are a technology expert specializing in explaining complex technical concepts clearly. "
            "Your expertise includes computer science, artificial intelligence, and emerging technologies. "
            "You provide accurate, detailed, and well-structured explanations on technical topics."
        )
    )
    print("Created Technical Expert agent")
    
    # Ethics specialist focusing on ethical implications
    ethics_agent = create_thinking_agent(
        name="EthicsExpert",
        role="ethics specialist",
        model_utility=model_util,
        thinking_mode=ThinkingMode.REFLECT,  # Reflective thinking suits ethical considerations
        system_prompt=(
            "You are an ethics specialist who focuses on the ethical implications of technologies and decisions. "
            "Your expertise includes ethical frameworks, bias analysis, fairness considerations, and societal impacts. "
            "You provide nuanced, balanced perspectives on ethical questions."
        )
    )
    print("Created Ethics Expert agent")
    
    # Business analyst focusing on market and business implications
    business_agent = create_thinking_agent(
        name="BusinessAnalyst",
        role="business analyst",
        model_utility=model_util,
        thinking_mode=ThinkingMode.TREE,  # Tree of thought for considering multiple business scenarios
        system_prompt=(
            "You are a business analyst specializing in market trends, business models, and economic impacts. "
            "Your expertise includes market analysis, business strategy, ROI evaluation, and industry trends. "
            "You provide practical, data-driven insights on business-related questions."
        )
    )
    print("Created Business Analyst agent")
    
    # Step 2: Create a coordinator agent to route questions
    print("\n=== Step 2: Creating Coordinator Agent ===")
    coordinator = create_thinking_agent(
        name="Coordinator",
        role=AgentRole.PLANNER,
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        system_prompt=(
            "You are a coordinator responsible for routing questions to the most appropriate specialist. "
            "You have three specialists available:\n"
            "1. TechExpert - Technology and technical concepts\n"
            "2. EthicsExpert - Ethical implications and considerations\n"
            "3. BusinessAnalyst - Business strategy and market analysis\n"
            "Analyze each question carefully and route it to the most suitable specialist."
        )
    )
    print("Created Coordinator agent")
    
    # Step 3: Create a multi-agent system with the coordinator
    print("\n=== Step 3: Creating Multi-Agent System ===")
    system = MultiAgentSystem(coordinator)
    system.add_agent(tech_agent)
    system.add_agent(ethics_agent)
    system.add_agent(business_agent)
    print("Created Multi-Agent System with 3 specialized agents and a coordinator")
    
    # Step 4: Ask the system different types of questions
    print("\n=== Step 4: Testing Multi-Agent System with Different Questions ===")
    
    # Technical question
    tech_question = "Explain how transformers have improved natural language processing."
    print(f"\nQuestion 1 (Technical): {tech_question}")
    response, agent = system.route_message(tech_question)
    print(f"Question routed to: {agent.name}")
    print(f"Response: {response}")
    
    # Ethical question
    ethics_question = "What are the ethical concerns around using AI for automated hiring decisions?"
    print(f"\nQuestion 2 (Ethical): {ethics_question}")
    response, agent = system.route_message(ethics_question)
    print(f"Question routed to: {agent.name}")
    print(f"Response: {response}")
    
    # Business question
    business_question = "How might AI as a service (AIaaS) change business models in the software industry?"
    print(f"\nQuestion 3 (Business): {business_question}")
    response, agent = system.route_message(business_question)
    print(f"Question routed to: {agent.name}")
    print(f"Response: {response}")
    
    # Interdisciplinary question
    mixed_question = "What are the business opportunities and ethical considerations of deploying facial recognition in retail stores?"
    print(f"\nQuestion 4 (Interdisciplinary): {mixed_question}")
    response, agent = system.route_message(mixed_question)
    print(f"Question routed to: {agent.name}")
    print(f"Response: {response}")
    
    # Step 5: Review conversation history
    print("\n=== Step 5: Reviewing Conversation History ===")
    print(f"Number of interactions: {len(system.conversation_history)}")
    for i, interaction in enumerate(system.conversation_history, 1):
        print(f"\nInteraction {i}:")
        print(f"User question: {interaction['user_input']}")
        print(f"Routed to: {interaction['agent']}")
        print(f"Response excerpt: {interaction['response'][:150]}...")


if __name__ == "__main__":
    main()
    print("\n=== End of Example 3 ===")
    print("Note: This example requires a valid OPENAI_API_KEY to be set in environment variables") 