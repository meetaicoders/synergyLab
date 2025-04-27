# Conversation Memory System

The Conversation Memory System provides a framework for storing, retrieving, and managing conversation history in AI applications. This document explains how to use the memory system to maintain context in conversations and improve AI assistant responses.

## Overview

The memory system consists of several key components:

1. **ConversationMemory**: Core class for storing and managing conversation history
2. **Message**: Represents a single message in a conversation
3. **Persistence**: Methods for saving and loading conversation history
4. **Memory Management**: Tools for managing memory size and retention

## Basic Usage

### Creating and Using Conversation Memory

```python
from generator.memory import ConversationMemory, Message, Role

# Create a new conversation memory
memory = ConversationMemory()

# Add messages to the memory
memory.add_message(Message(role=Role.USER, content="Hello, how can you help me today?"))
memory.add_message(Message(role=Role.ASSISTANT, content="I can help with a variety of tasks. What do you need assistance with?"))

# Get all messages
all_messages = memory.get_messages()
for message in all_messages:
    print(f"{message.role.value}: {message.content}")

# Get the most recent message
last_message = memory.get_last_message()
print(f"Last message: {last_message.content}")
```

### Message Structure

Each message in the conversation memory consists of:

1. **Role**: Who sent the message (USER, ASSISTANT, SYSTEM, FUNCTION)
2. **Content**: The text content of the message
3. **Timestamp**: When the message was created
4. **Metadata**: Optional additional information

```python
from generator.memory import Message, Role
from datetime import datetime

# Create a message with metadata
message = Message(
    role=Role.USER,
    content="What's the weather like in Paris?",
    timestamp=datetime.now(),
    metadata={"client_id": "user123", "location": "Paris", "language": "en"}
)

# Access message attributes
print(f"Role: {message.role.value}")
print(f"Content: {message.content}")
print(f"Timestamp: {message.timestamp}")
print(f"Metadata: {message.metadata}")
```

## Memory Configuration

### Setting Memory Limits

You can configure memory limits to prevent excessive token usage:

```python
# Create memory with a maximum of 10 messages
memory = ConversationMemory(max_messages=10)

# Create memory with a maximum of 4000 tokens (approximate)
memory = ConversationMemory(max_tokens=4000)

# Create memory with both message and token limits
memory = ConversationMemory(max_messages=20, max_tokens=8000)
```

### Windowing and Summary Options

```python
# Create memory with windowing (keeps most recent messages)
memory = ConversationMemory(
    max_messages=10,
    window_type="recent"  # Keeps most recent messages when limit is reached
)

# Create memory with summarization
# (Summarizes older messages when limit is reached)
memory = ConversationMemory(
    max_messages=10,
    window_type="summary",
    summarize_function=my_summary_function
)

# Custom summarization function
def my_summary_function(messages):
    """Create a summary of a list of messages.

    Args:
        messages (list): List of Message objects to summarize

    Returns:
        str: Summary of the messages
    """
    # Your summarization logic here
    return "Summary of conversation about topic X..."
```

## Memory Operations

### Searching and Filtering Messages

```python
# Search memory for messages containing specific text
results = memory.search("weather")
for msg in results:
    print(f"{msg.role.value}: {msg.content}")

# Filter messages by role
user_messages = memory.filter_by_role(Role.USER)
assistant_messages = memory.filter_by_role(Role.ASSISTANT)

# Get messages from a specific time range
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)
recent_messages = memory.filter_by_time(start_time=yesterday)
```

### Conversation Management

```python
# Clear the memory
memory.clear()

# Remove the oldest message
memory.remove_oldest()

# Remove a specific message
memory.remove_message(message_id)

# Get conversation statistics
stats = memory.get_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"User messages: {stats['user_messages']}")
print(f"Assistant messages: {stats['assistant_messages']}")
print(f"Total tokens: {stats['total_tokens']}")
```

## Persistence

### Saving and Loading Conversations

```python
# Save conversation to file
memory.save_to_file("conversation.json")

# Load conversation from file
memory = ConversationMemory.load_from_file("conversation.json")

# Save conversation to database
memory.save_to_database(database_connection, conversation_id="conv123")

# Load conversation from database
memory = ConversationMemory.load_from_database(database_connection, conversation_id="conv123")
```

### Memory Format Conversion

```python
# Convert memory to format compatible with specific LLM providers

# Convert to OpenAI message format
openai_messages = memory.to_openai_format()

# Convert to Anthropic message format
anthropic_messages = memory.to_anthropic_format()

# Convert to generic format
generic_messages = memory.to_dict()
```

## Advanced Usage

### Message Threading

```python
# Create a threaded conversation
memory = ConversationMemory(threaded=True)

# Add a message to the main thread
memory.add_message(Message(role=Role.USER, content="Let's discuss two topics."))

# Start a thread for first topic
thread1_id = memory.create_thread("Topic 1")
memory.add_message(
    Message(role=Role.USER, content="Tell me about AI."),
    thread_id=thread1_id
)
memory.add_message(
    Message(role=Role.ASSISTANT, content="AI is a field of computer science..."),
    thread_id=thread1_id
)

# Start a thread for second topic
thread2_id = memory.create_thread("Topic 2")
memory.add_message(
    Message(role=Role.USER, content="Tell me about machine learning."),
    thread_id=thread2_id
)

# Get messages from a specific thread
thread1_messages = memory.get_thread_messages(thread1_id)
```

### Memory with Metadata

```python
# Create memory with session metadata
memory = ConversationMemory(
    metadata={
        "user_id": "user123",
        "session_id": "session456",
        "preferences": {
            "language": "en",
            "timezone": "UTC-5"
        }
    }
)

# Update memory metadata
memory.update_metadata({"last_active": datetime.now().isoformat()})

# Access memory metadata
user_id = memory.metadata.get("user_id")
```

### Memory Integration with Agents

```python
from generator.agents import Agent
from generator.memory import ConversationMemory
from generator.llms import ModelUtility

# Create a memory system
memory = ConversationMemory(max_messages=20)

# Create an agent with the memory system
model_util = ModelUtility()
# ... configure model_util ...

agent = Agent(
    name="MemoryEnabledAgent",
    description="Agent with conversation memory",
    model_utility=model_util,
    memory=memory
)

# Use the agent in a conversation
agent.chat("Hello, my name is Alice.")
agent.chat("What's my name?")  # Should remember "Alice"

# Access the agent's memory
agent_memory = agent.memory
print(f"Messages in memory: {len(agent_memory.get_messages())}")
```

## Memory Analytics

### Basic Analytics

```python
# Calculate conversation metrics
metrics = memory.analyze()
print(f"Average message length: {metrics['avg_message_length']}")
print(f"User participation rate: {metrics['user_participation_rate']}")
print(f"Response time average: {metrics['avg_response_time']}")
```

### Topic Extraction

```python
# Extract conversation topics (requires NLP capabilities)
topics = memory.extract_topics()
print("Conversation topics:")
for topic, relevance in topics:
    print(f"- {topic}: {relevance:.2f}")
```

## Best Practices

1. **Memory Limits**: Set appropriate message and token limits to prevent excessive resource usage
2. **Regular Pruning**: Implement policies to remove old or low-relevance information
3. **Efficient Storage**: Use database backing for long-term conversation storage
4. **Privacy Considerations**: Implement retention policies and user controls for conversation data
5. **Context Management**: Balance between keeping enough context and staying within token limits

## Example: Complete Memory Workflow

```python
import json
from datetime import datetime
from generator.memory import ConversationMemory, Message, Role

# Create a conversation memory system
memory = ConversationMemory(
    max_messages=50,
    max_tokens=8000,
    metadata={"user_id": "user123", "session_start": datetime.now().isoformat()}
)

# Add system message to set context
memory.add_message(Message(
    role=Role.SYSTEM,
    content="You are a helpful assistant that remembers conversation history."
))

# Simulate a conversation
memory.add_message(Message(role=Role.USER, content="Hi, my name is Alex."))
memory.add_message(Message(role=Role.ASSISTANT, content="Hello Alex! How can I help you today?"))
memory.add_message(Message(role=Role.USER, content="I'm planning a trip to Japan next month."))
memory.add_message(Message(
    role=Role.ASSISTANT,
    content="That sounds exciting! What would you like to know about Japan for your trip?"
))
memory.add_message(Message(
    role=Role.USER,
    content="What's the best time to visit Tokyo and what are some must-see attractions?"
))

# Add a function call message
memory.add_message(Message(
    role=Role.FUNCTION,
    content=json.dumps({
        "attractions": [
            "Tokyo Skytree",
            "Senso-ji Temple",
            "Meiji Shrine",
            "Tokyo Disneyland"
        ],
        "best_time": "Late March to May (spring) or October to November (fall)"
    }),
    metadata={"function_name": "get_travel_info"}
))

# Get conversation summary for a new assistant
recent_context = memory.get_recent_context(5)  # Get last 5 messages
full_history = memory.get_messages()  # Get all messages

# Save conversation for later
memory.save_to_file("alex_travel_conversation.json")

# Create a new memory instance and load the saved conversation
new_memory = ConversationMemory.load_from_file("alex_travel_conversation.json")

# Continue the conversation
new_memory.add_message(Message(
    role=Role.ASSISTANT,
    content="Tokyo is best visited during spring (late March to May) for cherry blossoms or fall (October to November) for pleasant weather and autumn colors. Must-see attractions include Tokyo Skytree, Senso-ji Temple, Meiji Shrine, and Tokyo Disneyland. Would you like more specific recommendations based on your interests?"
))

# Check the conversation statistics
stats = new_memory.get_stats()
print(f"Conversation messages: {stats['total_messages']}")
print(f"Estimated tokens: {stats['total_tokens']}")
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Overflow**

   - Symptoms: Slow performance, excessive token usage
   - Solutions:
     - Reduce max_messages or max_tokens
     - Implement summarization for older messages
     - Use thread management for multiple topics

2. **Context Loss**

   - Symptoms: Agent forgets earlier parts of conversation
   - Solutions:
     - Periodically summarize key information
     - Use metadata to store critical information
     - Implement importance-based retention

3. **Performance Issues**
   - Symptoms: Slow response times with large conversation history
   - Solutions:
     - Use database backing for large histories
     - Implement efficient indexing for search operations
     - Use windowing to focus on recent or relevant messages
