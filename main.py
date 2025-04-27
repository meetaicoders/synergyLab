"""Main function demonstrating async LLM usage."""

import asyncio
from generator.llms import ModelUtility, ModelConfig

async def async_main():
    print("=== Example: Async LLM Usage ===")
    
    # Create a model utility
    model_util = ModelUtility()
    
    # Add an OpenAI model configuration
    print("\n1. Adding OpenAI model configuration")
    model_util.add_model(ModelConfig(
        model_name="gpt-4",
        provider="openai",
        temperature=0.7,
        max_tokens=1000
    ))
    
    # Generate text with the active model asynchronously
    print("\n2. Generating text with the active model (OpenAI) asynchronously")
    try:
        response = await model_util.generate_text_async(
            "Explain the concept of large language models in simple terms."
        )
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error generating text: {str(e)}. Is your API key set?")
    
    # Demonstrate batch processing multiple prompts concurrently
    print("\n3. Processing multiple prompts concurrently")
    try:
        prompts = [
            "What are large language models?",
            "How do transformers work?",
            "Explain the concept of attention in neural networks."
        ]
        
        responses = await model_util.generate_text_batch_async(prompts)
        
        for i, response in enumerate(responses):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Response {i+1}:\n{response}")
            print("-" * 50)
    except Exception as e:
        print(f"Error with batch processing: {str(e)}. Is your API key set?")

def main():
    # Run the async main function using asyncio
    asyncio.run(async_main())

if __name__ == "__main__":
    main()