"""
Example 1: Basic LLM Usage

This example demonstrates basic usage of the ModelUtility class for interacting
with various LLM providers including OpenAI, Anthropic, and others.
"""
import os
import sys

# Add the parent directory to the path so we can import from generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.llms import ModelUtility, ModelConfig

def main():
    """Main function demonstrating basic LLM usage."""
    print("=== Example 1: Basic LLM Usage ===")
    
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
    
    # Add an Anthropic model configuration
    print("\n2. Adding Anthropic model configuration")
    model_util.add_model(ModelConfig(
        model_name="claude-3-opus-20240229",
        provider="anthropic",
        temperature=0.5,
        max_tokens=2000
    ))
    
    # Generate text with the active model (first added model is active by default)
    print("\n3. Generating text with the active model (OpenAI)")
    try:
        response = model_util.generate_text(
            "Explain the concept of large language models in simple terms."
        )
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error generating text: {str(e)}. Is your API key set?")
    
    # Switch to a different model
    print("\n4. Switching to Anthropic model")
    model_util.set_active_model("claude-3-opus-20240229")
    
    # Generate text with the new active model
    print("\n5. Generating text with Claude")
    try:
        response = model_util.generate_text(
            "What are the key differences between transformers and previous neural network architectures?"
        )
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error generating text: {str(e)}. Is your API key set?")
    
    # Save model configurations to a file
    config_file = "model_configs.json"
    print(f"\n6. Saving model configurations to {config_file}")
    model_util.save_configs_to_file(config_file)
    print(f"Configurations saved to {config_file}")
    
    # Load model configurations from a file
    print(f"\n7. Loading model configurations from {config_file}")
    new_model_util = ModelUtility()
    try:
        new_model_util.load_configs_from_file(config_file)
        print(f"Configurations loaded from {config_file}")
        print(f"Available models: {list(new_model_util.configs.keys())}")
    except Exception as e:
        print(f"Error loading configurations: {str(e)}")


if __name__ == "__main__":
    main()
    print("\n=== End of Example 1 ===")
    print("Note: This example requires valid API keys to be set in environment variables:")
    print("- OPENAI_API_KEY for OpenAI models")
    print("- ANTHROPIC_API_KEY for Anthropic models") 