"""
Example 5: Creating Custom Tools

This example demonstrates how to create custom tools and register them for use with agents.
"""
import os
import sys
import random
import time
from typing import List, Dict, Optional

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig
from generator.agents import create_thinking_agent, AgentRole, ThinkingMode
from generator.tools import tool, ToolCategory, ToolManager, ToolResult

# Step 1: Define custom tools using the @tool decorator

@tool(
    description="Generate a random password with specified complexity.",
    category=ToolCategory.CUSTOM,
    return_description="A randomly generated password.",
    examples=["generate_password(length=12, include_special=True)"]
)
def generate_password(length: int = 12, include_numbers: bool = True, 
                      include_special: bool = True, include_uppercase: bool = True) -> str:
    """
    Generate a random password with specified complexity.
    
    Args:
        length: Length of the password to generate
        include_numbers: Whether to include numbers in the password
        include_special: Whether to include special characters
        include_uppercase: Whether to include uppercase letters
        
    Returns:
        A randomly generated password
    """
    lowercase_chars = "abcdefghijklmnopqrstuvwxyz"
    uppercase_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if include_uppercase else ""
    number_chars = "0123456789" if include_numbers else ""
    special_chars = "!@#$%^&*()-_=+[]{}|;:,.<>?" if include_special else ""
    
    all_chars = lowercase_chars + uppercase_chars + number_chars + special_chars
    
    if not all_chars:
        return "Error: No character sets selected"
    
    # Ensure at least one character from each selected set
    password = []
    if include_uppercase:
        password.append(random.choice(uppercase_chars))
    password.append(random.choice(lowercase_chars))
    if include_numbers:
        password.append(random.choice(number_chars))
    if include_special:
        password.append(random.choice(special_chars))
    
    # Fill the rest with random characters
    remaining_length = length - len(password)
    password.extend(random.choice(all_chars) for _ in range(remaining_length))
    
    # Shuffle the password
    random.shuffle(password)
    return ''.join(password)


@tool(
    description="Recommend a book based on a genre or topic.",
    category=ToolCategory.CUSTOM,
    return_description="A book recommendation including title, author, and description.",
    examples=["recommend_book(genre='science fiction')", "recommend_book(topic='machine learning')"]
)
def recommend_book(genre: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, str]:
    """
    Recommend a book based on a genre or topic.
    
    Args:
        genre: Book genre (e.g., 'science fiction', 'mystery', 'fantasy')
        topic: Book topic (e.g., 'machine learning', 'history', 'cooking')
        
    Returns:
        Dictionary with book details including title, author, and description
    """
    # Mock database of books
    books = {
        "science fiction": [
            {
                "title": "Dune",
                "author": "Frank Herbert",
                "description": "A sweeping epic set in a distant future on the desert planet of Arrakis, dealing with political intrigue, ecology, and human evolution."
            },
            {
                "title": "The Three-Body Problem",
                "author": "Liu Cixin",
                "description": "A mind-bending story about humanity's first contact with an alien civilization on the brink of destruction."
            }
        ],
        "mystery": [
            {
                "title": "The Silent Patient",
                "author": "Alex Michaelides",
                "description": "A psychological thriller about a woman who shoots her husband and then stops speaking, and the therapist determined to unravel her story."
            }
        ],
        "fantasy": [
            {
                "title": "The Name of the Wind",
                "author": "Patrick Rothfuss",
                "description": "An epic fantasy about a legendary wizard telling the story of how he became the most notorious wizard his world has ever seen."
            }
        ],
        "machine learning": [
            {
                "title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
                "author": "Aurélien Géron",
                "description": "A practical guide to machine learning that provides concrete examples and minimal theory."
            }
        ],
        "history": [
            {
                "title": "Sapiens: A Brief History of Humankind",
                "author": "Yuval Noah Harari",
                "description": "A broad exploration of the history of human beings from the stone age to the twenty-first century."
            }
        ],
        "cooking": [
            {
                "title": "Salt, Fat, Acid, Heat",
                "author": "Samin Nosrat",
                "description": "A guide to mastering the four elements of good cooking with insights that apply to all styles of cooking."
            }
        ]
    }
    
    if genre and genre.lower() in books:
        book_list = books[genre.lower()]
        return random.choice(book_list)
    elif topic and topic.lower() in books:
        book_list = books[topic.lower()]
        return random.choice(book_list)
    else:
        # If no match, return a random book
        all_books = [book for category in books.values() for book in category]
        return random.choice(all_books)


@tool(
    description="Track the progress of a task with time spent.",
    category=ToolCategory.CUSTOM,
    return_description="Task status including completion percentage and time spent.",
    examples=["track_task(task_name='Write documentation', percent_complete=75)"]
)
def track_task(task_name: str, percent_complete: int, time_spent_minutes: int = 0) -> Dict[str, any]:
    """
    Track the progress of a task with time spent.
    
    Args:
        task_name: Name of the task to track
        percent_complete: Percentage of task completion (0-100)
        time_spent_minutes: Time spent on the task in minutes
        
    Returns:
        Dictionary with task status details
    """
    # Validate inputs
    if percent_complete < 0 or percent_complete > 100:
        raise ValueError("Percent complete must be between 0 and 100")
    
    # Calculate estimated time to completion
    estimated_total_time = 0
    estimated_remaining_time = 0
    if percent_complete > 0 and time_spent_minutes > 0:
        estimated_total_time = (time_spent_minutes / percent_complete) * 100
        estimated_remaining_time = estimated_total_time - time_spent_minutes
    
    # Generate status based on completion percentage
    status = "Not Started"
    if percent_complete > 0:
        status = "In Progress"
    if percent_complete >= 100:
        status = "Completed"
    
    # Create task record
    task_record = {
        "task_name": task_name,
        "percent_complete": percent_complete,
        "status": status,
        "time_spent_minutes": time_spent_minutes,
        "estimated_remaining_minutes": round(estimated_remaining_time),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return task_record


def main():
    """Main function demonstrating custom tools."""
    print("=== Example 5: Creating Custom Tools ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add model configuration (assuming API key is set in environment variable)
    model_util.add_model(ModelConfig(
        model_name="gpt-4",
        provider="openai",
        temperature=0.7,
        max_tokens=1000
    ))
    
    # Step 1: Check if our custom tools are registered
    print("\n=== Step 1: Verifying Custom Tool Registration ===")
    tool_manager = ToolManager()
    
    custom_tools = tool_manager.get_tools_by_category(ToolCategory.CUSTOM)
    print(f"Custom tools registered: {', '.join(custom_tools)}")
    
    # Step 2: Test custom tools directly through the tool manager
    print("\n=== Step 2: Testing Custom Tools Directly ===")
    
    print("\nTesting password generator:")
    pwd_result = tool_manager.execute_tool("generate_password", length=16, include_special=True)
    print(f"Result: {pwd_result.result}")
    
    print("\nTesting book recommender:")
    book_result = tool_manager.execute_tool("recommend_book", genre="science fiction")
    print(f"Result: {book_result.result}")
    
    print("\nTesting task tracker:")
    task_result = tool_manager.execute_tool("track_task", 
                                           task_name="Implement custom tools", 
                                           percent_complete=75,
                                           time_spent_minutes=120)
    print(f"Result: {task_result.result}")
    
    # Step 3: Create an agent that can use custom tools
    print("\n=== Step 3: Creating Agent with Custom Tools ===")
    tool_agent = create_thinking_agent(
        name="CustomToolUser",
        role=AgentRole.ASSISTANT,
        model_utility=model_util,
        thinking_mode=ThinkingMode.CHAIN,
        use_tools=True,
        system_prompt=(
            "You are a helpful assistant with access to various tools including custom ones. "
            "When asked to perform tasks, use the most appropriate tool. "
            "You have access to tools for generating passwords, recommending books, and tracking tasks."
        ),
        verbose=True
    )
    print("Created agent with custom tools enabled")
    
    # Step 4: Use the agent with custom tools
    print("\n=== Step 4: Using Agent with Custom Tools ===")
    
    # Example 1: Password generator
    password_question = "Generate a secure password that's 20 characters long and includes special characters."
    print(f"\nTask 1 (Password): {password_question}")
    response = tool_agent.respond(password_question)
    print(f"Response: {response}")
    
    # Example 2: Book recommendation
    book_question = "Can you recommend a good science fiction book for me to read?"
    print(f"\nTask 2 (Book): {book_question}")
    response = tool_agent.respond(book_question)
    print(f"Response: {response}")
    
    # Example 3: Task tracking
    task_question = "Track my progress on writing documentation. I'm 60% done and have spent 90 minutes on it so far."
    print(f"\nTask 3 (Task): {task_question}")
    response = tool_agent.respond(task_question)
    print(f"Response: {response}")
    
    # Example 4: Complex request using multiple tools
    complex_question = "I need a secure password for my account, and I'd also like a book recommendation for when I'm done with my documentation task which is 80% complete after 3 hours of work."
    print(f"\nTask 4 (Complex): {complex_question}")
    response = tool_agent.respond(complex_question)
    print(f"Response: {response}")
    
    # Step 5: Review tool usage history
    print("\n=== Step 5: Reviewing Tool Usage History ===")
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
    print("\n=== End of Example 5 ===")
    print("Note: This example requires a valid OPENAI_API_KEY to be set in environment variables") 