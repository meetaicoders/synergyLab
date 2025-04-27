"""
A supporting module handling different LLMs and their configurations.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import json
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, Callable, List



class ModelConfig:
    """
    A class to store model configuration parameters.
    """
    def __init__(
        self,
        model_name: str,
        provider: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary"""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "parameters": self.parameters
        }


class ModelUtility:
    """
    A utility class to handle different model integrations and providers.
    Supports OpenAI, Anthropic, HuggingFace, and others.
    """
    
    SUPPORTED_PROVIDERS = ["openai", "anthropic", "huggingface", "cohere", "mistral", "google"]
    
    def __init__(self):
        self.configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.clients: Dict[str, Any] = {}
        self.async_clients: Dict[str, Any] = {}
    
    def add_model(self, config: ModelConfig) -> None:
        """
        Add a model configuration to the utility.
        
        Args:
            config: ModelConfig object with model settings
        """
        self.configs[config.model_name] = config
        if self.active_model is None:
            self.active_model = config.model_name
    
    def set_active_model(self, model_name: str) -> bool:
        """
        Set the active model for generation.
        
        Args:
            model_name: Name of the model to set active
            
        Returns:
            bool: True if successful, False if model not found
        """
        if model_name in self.configs:
            self.active_model = model_name
            return True
        return False
    
    def get_client(self, model_name: Optional[str] = None) -> Any:
        """
        Get or create a client for the specified model.
        
        Args:
            model_name: Name of the model to get client for, uses active_model if None
            
        Returns:
            Client instance for the specified provider
        
        Raises:
            ValueError: If model configuration is not found or provider not supported
        """
        model_name = model_name or self.active_model
        if model_name is None:
            raise ValueError("No active model set")
            
        if model_name not in self.configs:
            raise ValueError(f"Model {model_name} not configured")
            
        if model_name in self.clients:
            return self.clients[model_name]
            
        config = self.configs[model_name]
        client = self._create_client(config)
        self.clients[model_name] = client
        return client
    
    async def get_async_client(self, model_name: Optional[str] = None) -> Any:
        """
        Get or create an async client for the specified model.
        
        Args:
            model_name: Name of the model to get client for, uses active_model if None
            
        Returns:
            Async client instance for the specified provider
        
        Raises:
            ValueError: If model configuration is not found or provider not supported
        """
        model_name = model_name or self.active_model
        if model_name is None:
            raise ValueError("No active model set")
            
        if model_name not in self.configs:
            raise ValueError(f"Model {model_name} not configured")
            
        if model_name in self.async_clients:
            return self.async_clients[model_name]
            
        config = self.configs[model_name]
        client = await self._create_async_client(config)
        self.async_clients[model_name] = client
        return client
    
    def _create_client(self, config: ModelConfig) -> Any:
        """
        Create a client for the specified provider.
        
        Args:
            config: ModelConfig for the model
            
        Returns:
            Client instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()
        
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider {provider} not supported")
            
        if provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=config.api_key, base_url=config.api_base)
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
                
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=config.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
                
        elif provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                return InferenceClient(token=config.api_key)
            except ImportError:
                raise ImportError("Hugging Face package not installed. Install with 'pip install huggingface_hub'")
                
        elif provider == "cohere":
            try:
                import cohere
                return cohere.Client(api_key=config.api_key)
            except ImportError:
                raise ImportError("Cohere package not installed. Install with 'pip install cohere'")
                
        elif provider == "mistral":
            try:
                from mistralai.client import MistralClient
                return MistralClient(api_key=config.api_key)
            except ImportError:
                raise ImportError("Mistral AI package not installed. Install with 'pip install mistralai'")
                
        elif provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.api_key)
                return genai
            except ImportError:
                raise ImportError("Google AI package not installed. Install with 'pip install google-generativeai'")
    
    async def _create_async_client(self, config: ModelConfig) -> Any:
        """
        Create an async client for the specified provider.
        
        Args:
            config: ModelConfig for the model
            
        Returns:
            Async client instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()
        
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider {provider} not supported")
            
        if provider == "openai":
            try:
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key=config.api_key, base_url=config.api_base)
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
                
        elif provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
                return AsyncAnthropic(api_key=config.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
                
        elif provider == "huggingface":
            try:
                from huggingface_hub import AsyncInferenceClient
                return AsyncInferenceClient(token=config.api_key)
            except ImportError:
                raise ImportError("Hugging Face package not installed. Install with 'pip install huggingface_hub'")
                
        elif provider == "cohere":
            try:
                import cohere
                # Note: Cohere doesn't have an official async client, so we'll use the sync client
                # and wrap calls in asyncio.to_thread in the generate method
                return cohere.Client(api_key=config.api_key)
            except ImportError:
                raise ImportError("Cohere package not installed. Install with 'pip install cohere'")
                
        elif provider == "mistral":
            try:
                from mistralai.async_client import MistralAsyncClient
                return MistralAsyncClient(api_key=config.api_key)
            except ImportError:
                raise ImportError("Mistral AI package not installed. Install with 'pip install mistralai'")
                
        elif provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.api_key)
                # Google AI doesn't have a separate async client class
                return genai
            except ImportError:
                raise ImportError("Google AI package not installed. Install with 'pip install google-generativeai'")
    
    def generate_text(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the specified model.
        
        Args:
            prompt: The prompt to send to the model
            model_name: Name of the model to use, uses active_model if None
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If model is not configured or an error occurs during generation
        """
        model_name = model_name or self.active_model
        if model_name is None:
            raise ValueError("No active model set")
            
        if model_name not in self.configs:
            raise ValueError(f"Model {model_name} not configured")
            
        config = self.configs[model_name]
        client = self.get_client(model_name)
        
        # Merge config parameters with any overrides
        params = config.parameters.copy()
        params.update(kwargs)
        
        try:
            return self._generate_with_provider(client, config, prompt, params)
        except Exception as e:
            raise ValueError(f"Error generating text with {config.provider}: {str(e)}")
    
    async def generate_text_async(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the specified model asynchronously.
        
        Args:
            prompt: The prompt to send to the model
            model_name: Name of the model to use, uses active_model if None
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If model is not configured or an error occurs during generation
        """
        model_name = model_name or self.active_model
        if model_name is None:
            raise ValueError("No active model set")
            
        if model_name not in self.configs:
            raise ValueError(f"Model {model_name} not configured")
            
        config = self.configs[model_name]
        client = await self.get_async_client(model_name)
        
        # Merge config parameters with any overrides
        params = config.parameters.copy()
        params.update(kwargs)
        
        try:
            return await self._generate_with_provider_async(client, config, prompt, params)
        except Exception as e:
            raise ValueError(f"Error generating text with {config.provider}: {str(e)}")
    
    async def generate_text_streaming(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the model with streaming, processing chunks as they arrive.
        
        Args:
            prompt: The prompt to send to the model
            model_name: Name of the model to use, uses active_model if None
            callback: Optional callback function to process each chunk as it arrives
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Complete generated text response after streaming finishes
            
        Raises:
            ValueError: If model is not configured or an error occurs during generation
        """
        model_name = model_name or self.active_model
        if model_name is None:
            raise ValueError("No active model set")
            
        if model_name not in self.configs:
            raise ValueError(f"Model {model_name} not configured")
            
        config = self.configs[model_name]
        client = await self.get_async_client(model_name)
        
        # Merge config parameters with any overrides
        params = config.parameters.copy()
        params.update(kwargs)
        
        # Always enable streaming
        params["stream"] = True
        
        try:
            full_response = ""
            async for chunk in self._generate_with_provider_streaming(client, config, prompt, params):
                if callback:
                    callback(chunk)
                full_response += chunk
            return full_response
        except Exception as e:
            raise ValueError(f"Error streaming text with {config.provider}: {str(e)}")
    
    async def _generate_with_provider_streaming(
        self,
        client: Any,
        config: ModelConfig,
        prompt: str,
        params: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with the specified provider using streaming.
        
        Args:
            client: The async client to use for generation
            config: The model configuration
            prompt: The prompt to send to the model
            params: Additional parameters for the model
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            ValueError: If provider doesn't support streaming or other errors occur
        """
        provider = config.provider.lower()
        
        if provider == "openai":
            stream = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                stream=True,
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens", "stream"]}
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        elif provider == "anthropic":
            with client.messages.stream(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens", "stream"]}
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        elif provider == "mistral":
            stream = await client.chat_stream(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens", "stream"]}
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        else:
            # For providers without native streaming support, simulate it
            # by getting the full response and yielding it character by character
            full_response = await self._generate_with_provider_async(client, config, prompt, {
                k: v for k, v in params.items() if k != "stream"
            })
            
            # Simulate streaming by yielding chunks of the response
            chunk_size = 4  # Adjust for realistic streaming
            for i in range(0, len(full_response), chunk_size):
                yield full_response[i:i+chunk_size]
                await asyncio.sleep(0.01)  # Small delay to simulate network latency
    
    def _generate_with_provider(
        self, 
        client: Any, 
        config: ModelConfig, 
        prompt: str, 
        params: Dict[str, Any]
    ) -> str:
        """
        Generate text with the specified provider.
        
        Args:
            client: The client to use for generation
            config: The model configuration
            prompt: The prompt to send to the model
            params: Additional parameters for the model
            
        Returns:
            Generated text response
        """
        provider = config.provider.lower()
        
        if provider == "openai":
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.choices[0].message.content
            
        elif provider == "anthropic":
            response = client.messages.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.content[0].text
            
        elif provider == "huggingface":
            response = client.text_generation(
                prompt,
                model=config.model_name,
                temperature=params.get("temperature", config.temperature),
                max_new_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response
            
        elif provider == "cohere":
            response = client.generate(
                prompt=prompt,
                model=config.model_name,
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.generations[0].text
            
        elif provider == "mistral":
            response = client.chat(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.choices[0].message.content
            
        elif provider == "google":
            model = client.GenerativeModel(
                model_name=config.model_name,
                generation_config={
                    "temperature": params.get("temperature", config.temperature),
                    "max_output_tokens": params.get("max_tokens", config.max_tokens),
                    **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
                }
            )
            response = model.generate_content(prompt)
            return response.text
    
    async def _generate_with_provider_async(
        self, 
        client: Any, 
        config: ModelConfig, 
        prompt: str, 
        params: Dict[str, Any]
    ) -> str:
        """
        Generate text with the specified provider asynchronously.
        
        Args:
            client: The async client to use for generation
            config: The model configuration
            prompt: The prompt to send to the model
            params: Additional parameters for the model
            
        Returns:
            Generated text response
        """
        provider = config.provider.lower()
        
        if provider == "openai":
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.choices[0].message.content
            
        elif provider == "anthropic":
            response = await client.messages.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.content[0].text
            
        elif provider == "huggingface":
            response = await client.text_generation(
                prompt,
                model=config.model_name,
                temperature=params.get("temperature", config.temperature),
                max_new_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response
            
        elif provider == "cohere":
            # Cohere doesn't have an official async client, so we wrap the sync client in asyncio.to_thread
            response = await asyncio.to_thread(
                client.generate,
                prompt=prompt,
                model=config.model_name,
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.generations[0].text
            
        elif provider == "mistral":
            response = await client.chat(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.get("temperature", config.temperature),
                max_tokens=params.get("max_tokens", config.max_tokens),
                **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.choices[0].message.content
            
        elif provider == "google":
            # Google AI doesn't have a native async API, so we wrap the sync API in asyncio.to_thread
            model = client.GenerativeModel(
                model_name=config.model_name,
                generation_config={
                    "temperature": params.get("temperature", config.temperature),
                    "max_output_tokens": params.get("max_tokens", config.max_tokens),
                    **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]}
                }
            )
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text
    
    def load_configs_from_file(self, file_path: str) -> None:
        """
        Load model configurations from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If the file is not found
            json.JSONDecodeError: If the file is not valid JSON
        """
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            for model_name, config in config_data.items():
                self.add_model(ModelConfig(
                    model_name=model_name,
                    provider=config["provider"],
                    api_key=config.get("api_key"),
                    api_base=config.get("api_base"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens"),
                    parameters=config.get("parameters", {})
                ))
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {file_path}", "", 0)
    
    def save_configs_to_file(self, file_path: str, exclude_api_keys: bool = True) -> None:
        """
        Save model configurations to a JSON file.
        
        Args:
            file_path: Path to save the JSON configuration file
            exclude_api_keys: Whether to exclude API keys from the saved file
        """
        config_data = {}
        for model_name, config in self.configs.items():
            config_dict = config.to_dict()
            if not exclude_api_keys and config.api_key:
                config_dict["api_key"] = config.api_key
            if config.api_base:
                config_dict["api_base"] = config.api_base
            config_data[model_name] = config_dict
            
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    async def generate_text_batch_async(
        self,
        prompts: List[str],
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts to process
            model_name: Name of the model to use, uses active_model if None
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of generated text responses in the same order as the prompts
            
        Raises:
            ValueError: If model is not configured or an error occurs during generation
        """
        tasks = [self.generate_text_async(prompt, model_name, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

