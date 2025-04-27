"""
Example: Async Performance Comparison

This example demonstrates the performance benefits of using async operations
for LLM generation by comparing synchronous and asynchronous approaches.
"""
import os
import sys
import time
import asyncio

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig


def sync_process_prompts(model_util, prompts):
    """Process a list of prompts synchronously."""
    start_time = time.time()
    
    responses = []
    for prompt in prompts:
        response = model_util.generate_text(prompt)
        responses.append(response)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return responses, duration


async def async_process_prompts(model_util, prompts):
    """Process a list of prompts asynchronously."""
    start_time = time.time()
    
    responses = await model_util.generate_text_batch_async(prompts)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return responses, duration


async def main_async():
    """Main async function to run the performance comparison."""
    print("=== Async Performance Comparison ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add a model configuration
    print("\nAdding model configuration")
    model_util.add_model(ModelConfig(
        model_name="gpt-3.5-turbo",  # Using a faster model for testing
        provider="openai",
        temperature=0.7,
        max_tokens=150
    ))
    
    # Test prompts
    prompts = [
        "Write a short poem about artificial intelligence.",
        "Explain how neural networks work in one paragraph.",
        "List three benefits of asynchronous programming.",
        "Describe the concept of prompt engineering in AI.",
        "Summarize the history of natural language processing."
    ]
    
    print(f"\nWill process {len(prompts)} prompts using both sync and async methods")
    
    # Run synchronous version
    print("\n1. Running synchronous prompt processing...")
    try:
        sync_responses, sync_duration = sync_process_prompts(model_util, prompts)
        print(f"Synchronous processing completed in {sync_duration:.2f} seconds")
    except Exception as e:
        print(f"Error in synchronous processing: {str(e)}")
        return
    
    # Run asynchronous version
    print("\n2. Running asynchronous prompt processing...")
    try:
        async_responses, async_duration = await async_process_prompts(model_util, prompts)
        print(f"Asynchronous processing completed in {async_duration:.2f} seconds")
    except Exception as e:
        print(f"Error in asynchronous processing: {str(e)}")
        return
    
    # Calculate speed improvement
    if sync_duration > 0:
        speedup = sync_duration / async_duration
        print(f"\nSpeed improvement: {speedup:.2f}x faster with async processing")
        print(f"Time saved: {sync_duration - async_duration:.2f} seconds ({(1 - async_duration/sync_duration) * 100:.1f}%)")
    
    print("\n=== Results Summary ===")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Sync processing time: {sync_duration:.2f} seconds")
    print(f"Async processing time: {async_duration:.2f} seconds")


def main():
    """Entry point that runs the async main function."""
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Set your API key before running this example.")
        return
    
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
    print("\n=== End of Example ===") 