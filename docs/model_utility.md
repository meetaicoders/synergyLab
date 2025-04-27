# Model Utility Documentation

The `ModelUtility` class provides a centralized way to manage language model configurations and interactions with different LLM providers. This document covers how to use the ModelUtility to work with various LLM providers, configuration options, and best practices.

## Overview

The `ModelUtility` class is designed to:

1. Manage multiple model configurations in one place
2. Provide a unified interface for interacting with different LLM providers
3. Allow easy switching between models
4. Save and load model configurations for persistence
5. Handle provider-specific parameters and authentication

## Basic Usage

### Importing and Initializing

```python
from generator.llms import ModelUtility, ModelConfig

# Create a model utility instance
model_util = ModelUtility()
```

### Adding Model Configurations

```python
# Add an OpenAI model
model_util.add_model(ModelConfig(
    model_name="gpt-4",
    provider="openai",
    temperature=0.7,
    max_tokens=1000
))

# Add an Anthropic model
model_util.add_model(ModelConfig(
    model_name="claude-3-opus-20240229",
    provider="anthropic",
    temperature=0.5,
    max_tokens=2000
))
```

### Generating Text

```python
# Generate text using the active model (default is the first added model)
response = model_util.generate_text("Explain the concept of quantum computing.")

# Or specify a prompt as a list of messages
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the concept of quantum computing."}
]
response = model_util.generate_text(messages)

print(response)
```

### Switching Between Models

```python
# Switch to a different model by name
model_util.set_active_model("claude-3-opus-20240229")

# Now generation will use the Anthropic model
response = model_util.generate_text("What are the key differences between classical and quantum computing?")
```

## Supported Providers

The `ModelUtility` currently supports the following providers:

### OpenAI

```python
# OpenAI configuration
model_util.add_model(ModelConfig(
    model_name="gpt-4",  # or "gpt-3.5-turbo", etc.
    provider="openai",
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
))
```

Authentication: Requires `OPENAI_API_KEY` environment variable or explicit key via `api_key` parameter.

### Anthropic

```python
# Anthropic configuration
model_util.add_model(ModelConfig(
    model_name="claude-3-opus-20240229",  # or other Claude models
    provider="anthropic",
    temperature=0.5,
    max_tokens=2000,
    top_p=0.9,
    top_k=50
))
```

Authentication: Requires `ANTHROPIC_API_KEY` environment variable or explicit key via `api_key` parameter.

### Other Providers

Additional providers can be implemented by extending the provider-specific adapter classes.

## Model Configuration Options

The `ModelConfig` class accepts the following parameters:

| Parameter           | Description                                 | Default              |
| ------------------- | ------------------------------------------- | -------------------- |
| `model_name`        | Name of the model to use (e.g., "gpt-4")    | Required             |
| `provider`          | LLM provider ("openai", "anthropic", etc.)  | Required             |
| `temperature`       | Controls randomness (0-1)                   | 0.7                  |
| `max_tokens`        | Maximum tokens in the response              | 1000                 |
| `top_p`             | Nucleus sampling parameter                  | 1.0                  |
| `api_key`           | API key for the provider                    | None (uses env vars) |
| `api_base`          | Base URL for API calls                      | Provider default     |
| `timeout`           | Timeout for API requests in seconds         | 60                   |
| `additional_params` | Dict of additional provider-specific params | {}                   |

## Saving and Loading Configurations

You can save model configurations to a file and load them later:

```python
# Save configurations to a file
model_util.save_configurations("model_configs.json")

# Load configurations from a file
model_util.load_configurations("model_configs.json")

# Get a list of available model names
model_names = model_util.get_available_models()
print(f"Available models: {model_names}")
```

## Error Handling

The `ModelUtility` provides error handling for common issues:

```python
try:
    response = model_util.generate_text("What is the capital of France?")
    print(response)
except Exception as e:
    print(f"Error generating text: {e}")
```

Common errors include:

- Authentication errors (missing or invalid API keys)
- Rate limiting or quota exceeded
- Invalid model configurations
- Network errors or timeouts

## Best Practices

1. **Environment Variables**: Store API keys in environment variables instead of hardcoding them
2. **Error Handling**: Always include error handling for API calls
3. **Configuration Management**: Save configurations to avoid redefining them in each session
4. **Token Management**: Be mindful of token limits, especially for lengthy conversations
5. **Cost Management**: Monitor API usage to control costs, particularly with more powerful models

## Example: Complete Workflow

```python
import os
from generator.llms import ModelUtility, ModelConfig

# Set up API keys (alternatively use environment variables)
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Initialize model utility
model_util = ModelUtility()

# Add multiple model configurations
model_util.add_model(ModelConfig(
    model_name="gpt-3.5-turbo",
    provider="openai",
    temperature=0.7,
    max_tokens=500
))

model_util.add_model(ModelConfig(
    model_name="gpt-4",
    provider="openai",
    temperature=0.3,  # Lower temperature for more deterministic responses
    max_tokens=1000
))

model_util.add_model(ModelConfig(
    model_name="claude-3-opus-20240229",
    provider="anthropic",
    temperature=0.5,
    max_tokens=2000
))

# Save configurations for future use
model_util.save_configurations("model_configs.json")

# List available models
models = model_util.get_available_models()
print(f"Available models: {models}")

# Use GPT-3.5-Turbo for a simple query (it's the default first model)
response = model_util.generate_text("Write a short poem about AI.")
print(f"GPT-3.5 Response: {response}")

# Switch to GPT-4 for a more complex query
model_util.set_active_model("gpt-4")
response = model_util.generate_text("Explain the ethical implications of AI-generated content.")
print(f"GPT-4 Response: {response}")

# Switch to Claude for another query
model_util.set_active_model("claude-3-opus-20240229")
response = model_util.generate_text("Compare and contrast different approaches to AI alignment.")
print(f"Claude Response: {response}")
```

## Advanced Usage

### Custom Message Formatting

You can use structured messages for more control:

```python
messages = [
    {"role": "system", "content": "You are an expert mathematician."},
    {"role": "user", "content": "Explain the Riemann Hypothesis."},
    {"role": "assistant", "content": "The Riemann Hypothesis is a conjecture about the distribution of prime numbers."},
    {"role": "user", "content": "Continue explaining in more technical detail."}
]

response = model_util.generate_text(messages)
```

### Streaming Responses

For providers that support streaming:

```python
for chunk in model_util.generate_text_stream("Explain quantum computing step by step."):
    print(chunk, end="", flush=True)
```

### Custom Provider Settings

Use `additional_params` for provider-specific settings:

```python
model_util.add_model(ModelConfig(
    model_name="gpt-4",
    provider="openai",
    temperature=0.7,
    additional_params={
        "logit_bias": {50256: -100},  # Reduce likelihood of specific tokens
        "user": "user-123"  # For tracking purposes
    }
))
```
