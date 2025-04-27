"""
Example usage of the ModelUtility class.
"""
from llms import ModelUtility, ModelConfig

def main():
    # Create a model utility
    model_util = ModelUtility()
    
    # Add some model configurations
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
    
    # Save configurations to a file
    model_util.save_configs_to_file("model_configs.json")
    
    # Load configurations from a file
    # model_util.load_configs_from_file("model_configs.json")
    
    # Set the active model
    model_util.set_active_model("gpt-4")
    
    # Generate text with the active model
    try:
        response = model_util.generate_text(
            "Explain the concept of artificial intelligence in simple terms.",
            temperature=0.8  # Override temperature parameter
        )
        print("Response from OpenAI:")
        print(response)
        print()
    except ValueError as e:
        print(f"Error: {e}")
    
    # Switch to another model
    model_util.set_active_model("claude-3-opus-20240229")
    
    # Generate text with the new active model
    try:
        response = model_util.generate_text(
            "What are the potential applications of large language models?",
            max_tokens=500  # Override max_tokens parameter
        )
        print("Response from Anthropic:")
        print(response)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 