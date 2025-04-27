"""
Example: Streaming Text Generation

This example demonstrates how to use the streaming capabilities of the ModelUtility
class to get text generation results in real-time as they're being generated.
"""
import os
import sys
import asyncio
import time

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig


def print_chunk(chunk: str):
    """Print a chunk of text without line breaks, flushing to show progress."""
    print(chunk, end="", flush=True)


async def main():
    """Main function demonstrating streaming text generation."""
    print("=== Example: Streaming Text Generation ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add model configuration
    print("\nAdding model configuration")
    model_util.add_model(ModelConfig(
        model_name="gpt-3.5-turbo",  # Using a faster model for streaming
        provider="openai",
        temperature=0.7,
        max_tokens=150
    ))
    
    # Use the streaming interface with a callback
    print("\n1. Streaming with real-time display of chunks:")
    print("\nResponse: ", end="")
    
    try:
        prompt = "Write a short paragraph explaining how streaming works in language models."
        start_time = time.time()
        
        # The callback will print chunks as they arrive
        result = await model_util.generate_text_streaming(prompt, callback=print_chunk)
        
        end_time = time.time()
        print(f"\n\nTotal time: {end_time - start_time:.2f} seconds")
        print(f"Total length: {len(result)} characters")
    except Exception as e:
        print(f"\nError during streaming: {str(e)}. Is your API key set?")
    
    # Compare with non-streaming version
    print("\n\n2. Using regular non-streaming generation for comparison:")
    try:
        start_time = time.time()
        result = await model_util.generate_text_async(prompt)
        end_time = time.time()
        
        print(f"Response: {result}")
        print(f"\nTotal time: {end_time - start_time:.2f} seconds")
        print(f"Total length: {len(result)} characters")
    except Exception as e:
        print(f"Error generating text: {str(e)}. Is your API key set?")
    
    
    # Custom chunk handling example
    print("\n\n3. Custom chunk handling (word counting):")
    try:
        prompt = "Explain the benefits of asynchronous programming in Python."
        
        word_count = 0
        
        def count_words(chunk: str):
            nonlocal word_count
            # Count words in the chunk (rough approximation)
            new_words = len(chunk.split())
            word_count += new_words
            print(f"Received chunk ({len(chunk)} chars, ~{new_words} words). Total words so far: {word_count}")
        
        result = await model_util.generate_text_streaming(prompt, callback=count_words)
        
        print(f"\nFinal response length: {len(result)} characters")
        print(f"Final word count: {len(result.split())}")  # More accurate final count
    except Exception as e:
        print(f"Error during streaming: {str(e)}. Is your API key set?")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Set your API key before running this example.")
    else:
        asyncio.run(main())
        print("\n=== End of Example ===") 