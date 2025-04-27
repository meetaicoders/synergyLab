"""
Module for implementing AI agents with thinking capabilities.
This module provides classes for creating agents that can think through steps
before taking actions, using different LLM providers.
"""
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import json
import time
import logging
from enum import Enum
from .llms import ModelUtility, ModelConfig
from .tools import ToolManager, ToolResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of different agent roles."""
    ASSISTANT = "assistant"
    RESEARCHER = "researcher"
    CODER = "coder"
    CRITIC = "critic"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CUSTOM = "custom"


class ThinkingMode(Enum):
    """Enumeration of different thinking modes."""
    NONE = "none"  # No thinking, direct response
    CHAIN = "chain"  # Chain of thought thinking
    TREE = "tree"  # Tree of thought thinking
    REFLECT = "reflect"  # Self-reflection thinking


class Message:
    """Class representing a message in a conversation."""
    def __init__(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary."""
        msg = cls(data["role"], data["content"], data.get("metadata", {}))
        msg.timestamp = data.get("timestamp", time.time())
        return msg


class ThoughtProcess:
    """Class for managing the thinking process of an agent."""
    def __init__(self, mode: ThinkingMode = ThinkingMode.CHAIN):
        self.mode = mode
        self.thoughts: List[str] = []
        self.reasoning_steps: List[Dict[str, Any]] = []
        self.decision_points: List[Dict[str, Any]] = []
    
    def add_thought(self, thought: str) -> None:
        """Add a thought to the thinking process."""
        self.thoughts.append(thought)
    
    def add_reasoning_step(self, step: str, rationale: str) -> None:
        """Add a reasoning step to the thinking process."""
        self.reasoning_steps.append({
            "step": step,
            "rationale": rationale,
            "timestamp": time.time()
        })
    
    def add_decision_point(self, options: List[str], chosen: str, reason: str) -> None:
        """Add a decision point to the thinking process."""
        self.decision_points.append({
            "options": options,
            "chosen": chosen,
            "reason": reason,
            "timestamp": time.time()
        })
    
    def get_formatted_thinking(self) -> str:
        """Get the formatted thinking process."""
        if not self.thoughts and not self.reasoning_steps:
            return ""
        
        formatted = "Thinking process:\n"
        
        if self.thoughts:
            formatted += "\nThoughts:\n"
            for i, thought in enumerate(self.thoughts, 1):
                formatted += f"{i}. {thought}\n"
        
        if self.reasoning_steps:
            formatted += "\nReasoning Steps:\n"
            for i, step in enumerate(self.reasoning_steps, 1):
                formatted += f"{i}. {step['step']}\n   Rationale: {step['rationale']}\n"
        
        if self.decision_points:
            formatted += "\nDecision Points:\n"
            for i, decision in enumerate(self.decision_points, 1):
                formatted += f"{i}. Options: {', '.join(decision['options'])}\n"
                formatted += f"   Chosen: {decision['chosen']}\n"
                formatted += f"   Reason: {decision['reason']}\n"
        
        return formatted
    
    def clear(self) -> None:
        """Clear the thinking process."""
        self.thoughts = []
        self.reasoning_steps = []
        self.decision_points = []


class Agent:
    """
    Class for implementing an AI agent with thinking capabilities.
    
    The agent can think through steps before taking actions, using
    different LLM providers and thinking modes.
    """
    def __init__(
        self,
        name: str,
        role: Union[AgentRole, str],
        model_utility: ModelUtility,
        thinking_model: Optional[str] = None,
        response_model: Optional[str] = None,
        thinking_mode: ThinkingMode = ThinkingMode.CHAIN,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        use_tools: bool = False,
        verbose: bool = False
    ):
        """
        Initialize an agent.
        
        Args:
            name: Name of the agent
            role: Role of the agent (AgentRole enum or string)
            model_utility: ModelUtility instance for LLM access
            thinking_model: Name of the model to use for thinking (if None, uses active model)
            response_model: Name of the model to use for responses (if None, uses active model)
            thinking_mode: Mode of thinking to use
            system_prompt: System prompt to use for the agent
            tools: List of tools available to the agent
            use_tools: Whether to enable tool usage
            verbose: Whether to log verbose information
        """
        self.name = name
        self.role = role.value if isinstance(role, AgentRole) else role
        self.model_utility = model_utility
        self.thinking_model = thinking_model
        self.response_model = response_model
        self.thinking_mode = thinking_mode
        self.verbose = verbose
        self.use_tools = use_tools
        
        # Initialize tool manager if tools are enabled
        self.tool_manager = ToolManager() if use_tools else None
        
        # Initialize thought process
        self.thought_process = ThoughtProcess(mode=thinking_mode)
        
        # Set up system prompt based on role if not provided
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Set up conversation history
        self.conversation: List[Message] = []
        if self.system_prompt:
            self.conversation.append(Message("system", self.system_prompt))
        
        # Set up tools
        self.tools = tools or []
        
        # Add tool handlers
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Tool usage history
        self.tool_usage: List[Dict[str, Any]] = []
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt based on the agent's role."""
        tools_instruction = ""
        if self.use_tools:
            tools_instruction = (
                "You have access to tools that can help you perform tasks. "
                "When you need to use a tool, respond with the tool name and parameters. "
                "For example: 'I will use search_web(query=\"latest news\")' or "
                "'I need to calculate(expression=\"2+2\")'. "
                "Wait for the tool result before continuing."
            )
        
        if self.role == AgentRole.ASSISTANT.value:
            return (
                "You are a helpful, harmless, and honest AI assistant. "
                "Always think step-by-step before answering questions. "
                f"{tools_instruction}"
            )
        elif self.role == AgentRole.RESEARCHER.value:
            return (
                "You are a thorough and methodical researcher. "
                "Your goal is to explore topics deeply and provide comprehensive information. "
                f"{tools_instruction}"
            )
        elif self.role == AgentRole.CODER.value:
            return (
                "You are an expert programmer. "
                "Your goal is to write clean, efficient, and well-documented code. "
                f"{tools_instruction}"
            )
        elif self.role == AgentRole.CRITIC.value:
            return (
                "You are a thoughtful critic. "
                "Your goal is to provide constructive feedback and identify potential issues. "
                f"{tools_instruction}"
            )
        elif self.role == AgentRole.PLANNER.value:
            return (
                "You are a strategic planner. "
                "Your goal is to break down complex tasks into manageable steps. "
                f"{tools_instruction}"
            )
        elif self.role == AgentRole.EXECUTOR.value:
            return (
                "You are an efficient executor. "
                "Your goal is to carry out tasks accurately and effectively. "
                f"{tools_instruction}"
            )
        else:
            return (
                f"You are an AI assistant named {self.name} with the role of {self.role}. "
                "Think step-by-step when solving problems. "
                f"{tools_instruction}"
            )
    
    def _format_conversation_for_prompt(self) -> List[Dict[str, str]]:
        """Format the conversation history for a prompt."""
        return [{"role": msg.role, "content": msg.content} for msg in self.conversation]
    
    def _generate_thinking_prompt(self, user_input: str) -> str:
        """Generate a prompt for the thinking process."""
        if self.thinking_mode == ThinkingMode.NONE:
            return user_input
        
        if self.thinking_mode == ThinkingMode.CHAIN:
            return (
                f"I need to respond to this user request: '{user_input}'\n\n"
                "Before responding, I'll think step-by-step to make sure my answer is correct.\n\n"
                "Let me break this down:\n"
                "1. "
            )
        elif self.thinking_mode == ThinkingMode.REFLECT:
            return (
                f"I need to respond to this user request: '{user_input}'\n\n"
                "I'll first reflect on what this request is asking for, then think through how to best respond.\n\n"
                "Reflection:"
            )
        elif self.thinking_mode == ThinkingMode.TREE:
            return (
                f"I need to respond to this user request: '{user_input}'\n\n"
                "I'll consider multiple approaches to answering this question.\n\n"
                "Approach 1: "
            )
        else:
            return user_input
    
    def _extract_thinking(self, thinking_response: str) -> str:
        """Extract the thinking part from a response."""
        # For simplicity, we'll just return the whole response for now
        # In a more complex implementation, you might want to parse out
        # the final answer vs. the thinking steps
        return thinking_response
    
    def _generate_response_prompt(self, user_input: str, thinking: str) -> str:
        """Generate a prompt for the final response."""
        tools_info = ""
        if self.use_tools and self.tool_manager:
            tools_json = self.tool_manager.to_json_schema()
            tools_info = (
                f"\nYou have access to the following tools: {json.dumps(tools_json, indent=2)}\n"
                "If you need to use a tool, indicate it clearly in your response "
                "by saying 'I will use TOOL_NAME(param1=value1, param2=value2)' "
                "and wait for the result before continuing."
            )
        
        return (
            f"I need to respond to this user request: '{user_input}'\n\n"
            f"I've already thought through the problem as follows:\n{thinking}\n\n"
            f"{tools_info}\n\n"
            "Based on this thinking, my final response should be:\n"
        )
    
    def think(self, user_input: str) -> str:
        """
        Think through a problem without generating a final response.
        
        Args:
            user_input: The user's input to think about
            
        Returns:
            The thinking process as a string
        """
        if self.thinking_mode == ThinkingMode.NONE:
            return ""
        
        # Clear previous thoughts
        self.thought_process.clear()
        
        # Generate thinking prompt
        thinking_prompt = self._generate_thinking_prompt(user_input)
        
        # Generate thinking using the thinking model
        try:
            thinking_response = self.model_utility.generate_text(
                thinking_prompt,
                model_name=self.thinking_model,
                temperature=0.7  # Slightly higher temperature for creative thinking
            )
            
            # Extract and store thinking
            thinking = self._extract_thinking(thinking_response)
            self.thought_process.add_thought(thinking)
            
            if self.verbose:
                logger.info(f"Agent {self.name} thinking: {thinking}")
            
            return thinking
        except Exception as e:
            logger.error(f"Error during thinking: {str(e)}")
            return ""
    
    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from text.
        
        Args:
            text: Text to extract tool calls from
            
        Returns:
            List of tool calls with name and parameters
        """
        tool_calls = []
        
        # Basic regex pattern to match tool calls like: tool_name(param1="value", param2=123)
        # This is a simplified version; in practice you'd want a more robust parser
        pattern = r'(\w+)\s*\(\s*(.*?)\s*\)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            tool_name = match[0]
            params_str = match[1]
            
            # Check if this is a valid tool
            if not self.tool_manager or not self.tool_manager.get_tool(tool_name):
                continue
            
            # Parse parameters
            params = {}
            param_pattern = r'(\w+)\s*=\s*([^,)]+)'
            param_matches = re.findall(param_pattern, params_str)
            
            for param_match in param_matches:
                param_name = param_match[0]
                param_value = param_match[1].strip()
                
                # Convert string representations to appropriate types
                if param_value.startswith('"') and param_value.endswith('"'):
                    # String
                    params[param_name] = param_value[1:-1]
                elif param_value.startswith("'") and param_value.endswith("'"):
                    # String with single quotes
                    params[param_name] = param_value[1:-1]
                elif param_value.lower() == 'true':
                    # Boolean True
                    params[param_name] = True
                elif param_value.lower() == 'false':
                    # Boolean False
                    params[param_name] = False
                elif param_value.isdigit():
                    # Integer
                    params[param_name] = int(param_value)
                elif self._is_float(param_value):
                    # Float
                    params[param_name] = float(param_value)
                else:
                    # Default to string
                    params[param_name] = param_value
            
            tool_calls.append({
                "name": tool_name,
                "parameters": params
            })
        
        return tool_calls
    
    def _is_float(self, s: str) -> bool:
        """Check if a string can be converted to a float."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tool calls.
        
        Args:
            tool_calls: List of tool calls with name and parameters
            
        Returns:
            List of tool results
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            parameters = tool_call["parameters"]
            
            if not self.tool_manager:
                results.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "success": False,
                    "error": "Tools are not enabled for this agent",
                    "result": None
                })
                continue
            
            try:
                # Execute the tool
                result = self.tool_manager.execute_tool(tool_name, **parameters)
                
                # Add to tool usage history
                self.tool_usage.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": result.result if result.success else None,
                    "error": result.error,
                    "timestamp": time.time()
                })
                
                # Add to results
                results.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "success": result.success,
                    "error": result.error,
                    "result": result.result
                })
                
                if self.verbose:
                    logger.info(f"Agent {self.name} executed tool {tool_name}: {str(result)}")
            
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {str(e)}"
                logger.error(error_message)
                
                # Add to tool usage history
                self.tool_usage.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": None,
                    "error": error_message,
                    "timestamp": time.time()
                })
                
                # Add to results
                results.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "success": False,
                    "error": error_message,
                    "result": None
                })
        
        return results
    
    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format tool results as a string to include in the conversation.
        
        Args:
            results: List of tool results
            
        Returns:
            Formatted tool results
        """
        if not results:
            return ""
        
        formatted = "Tool Results:\n\n"
        
        for result in results:
            tool_name = result["tool_name"]
            parameters_str = ", ".join([f"{k}={repr(v)}" for k, v in result["parameters"].items()])
            
            formatted += f"{tool_name}({parameters_str})\n"
            
            if result["success"]:
                formatted += f"Result: {result['result']}\n"
            else:
                formatted += f"Error: {result['error']}\n"
            
            formatted += "\n"
        
        return formatted
    
    def respond(self, user_input: str, with_thinking: bool = True) -> str:
        """
        Generate a response to a user input.
        
        Args:
            user_input: The user's input
            with_thinking: Whether to use the thinking process
            
        Returns:
            The agent's response
        """
        # Add user message to conversation
        self.conversation.append(Message("user", user_input))
        
        # Think about the response if requested and thinking mode is not NONE
        thinking = ""
        if with_thinking and self.thinking_mode != ThinkingMode.NONE:
            thinking = self.think(user_input)
        
        # Generate response prompt
        if thinking and with_thinking:
            response_prompt = self._generate_response_prompt(user_input, thinking)
        else:
            # Format conversation for prompt
            messages = self._format_conversation_for_prompt()
            response_prompt = user_input
        
        # Generate response using the response model
        try:
            response = self.model_utility.generate_text(
                response_prompt if thinking else user_input,
                model_name=self.response_model,
                messages=messages if not thinking else None
            )
            
            # Check for tool calls in the response if tools are enabled
            if self.use_tools and self.tool_manager:
                tool_calls = self._extract_tool_calls(response)
                
                if tool_calls:
                    # Execute tool calls
                    tool_results = self._execute_tool_calls(tool_calls)
                    
                    # Format tool results
                    formatted_results = self._format_tool_results(tool_results)
                    
                    # Add assistant's response with tool calls to conversation
                    self.conversation.append(Message("assistant", response))
                    
                    # Add tool results to conversation
                    if formatted_results:
                        self.conversation.append(Message("system", formatted_results))
                    
                    # Generate follow-up response using the tool results
                    followup_messages = self._format_conversation_for_prompt()
                    followup_response = self.model_utility.generate_text(
                        "Continue your response based on the tool results.",
                        model_name=self.response_model,
                        messages=followup_messages
                    )
                    
                    # Add follow-up response to conversation
                    self.conversation.append(Message("assistant", followup_response))
                    
                    # Combine original response, tool results, and follow-up
                    response = f"{response}\n\n{formatted_results}\n\n{followup_response}"
                else:
                    # No tool calls, just add the response to conversation
                    self.conversation.append(Message("assistant", response))
            else:
                # No tools, just add the response to conversation
                self.conversation.append(Message("assistant", response))
            
            if self.verbose:
                logger.info(f"Agent {self.name} response: {response}")
            
            return response
        except Exception as e:
            logger.error(f"Error during response: {str(e)}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            self.conversation.append(Message("assistant", error_msg))
            return error_msg
    
    def register_tool(self, tool_name: str, handler: Callable) -> None:
        """
        Register a tool handler.
        
        Args:
            tool_name: Name of the tool
            handler: Function to handle the tool call
        """
        self.tool_handlers[tool_name] = handler
        # Add tool to the tools list if not already present
        if not any(tool["name"] == tool_name for tool in self.tools):
            self.tools.append({"name": tool_name, "handler": handler.__name__})
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a registered tool.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool call
            
        Raises:
            ValueError: If the tool is not registered
        """
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Tool {tool_name} not registered")
        
        try:
            result = self.tool_handlers[tool_name](**kwargs)
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            raise
    
    def save_conversation(self, file_path: str) -> None:
        """
        Save the conversation to a JSON file.
        
        Args:
            file_path: Path to save the conversation to
        """
        conv_data = [msg.to_dict() for msg in self.conversation]
        with open(file_path, 'w') as f:
            json.dump(conv_data, f, indent=2)
    
    def load_conversation(self, file_path: str) -> None:
        """
        Load a conversation from a JSON file.
        
        Args:
            file_path: Path to load the conversation from
            
        Raises:
            FileNotFoundError: If the file is not found
            json.JSONDecodeError: If the file is not valid JSON
        """
        try:
            with open(file_path, 'r') as f:
                conv_data = json.load(f)
            
            self.conversation = [Message.from_dict(msg) for msg in conv_data]
        except FileNotFoundError:
            raise FileNotFoundError(f"Conversation file not found: {file_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON in conversation file: {file_path}", "", 0)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history, keeping only the system prompt."""
        if not self.conversation:
            return
        
        # Keep only the system prompt if present
        if self.conversation and self.conversation[0].role == "system":
            self.conversation = [self.conversation[0]]
        else:
            self.conversation = []
    
    def get_tool_usage_history(self) -> List[Dict[str, Any]]:
        """
        Get the tool usage history.
        
        Returns:
            List of tool usage records
        """
        return self.tool_usage


class MultiAgentSystem:
    """
    A system for managing multiple agents that can interact with each other.
    """
    def __init__(self, coordinator_agent: Optional[Agent] = None):
        """
        Initialize a multi-agent system.
        
        Args:
            coordinator_agent: Optional coordinator agent to manage the other agents
        """
        self.agents: Dict[str, Agent] = {}
        self.coordinator = coordinator_agent
        self.conversation_history: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the system.
        
        Args:
            agent: Agent to add
        """
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get an agent by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            The agent if found, None otherwise
        """
        return self.agents.get(name)
    
    def route_message(self, user_input: str) -> Tuple[str, Agent]:
        """
        Route a message to the appropriate agent.
        
        Args:
            user_input: The user's input
            
        Returns:
            Tuple of (response, agent) who handled the message
        """
        if not self.agents:
            raise ValueError("No agents registered")
        
        if self.coordinator:
            # Use coordinator to decide which agent should handle the message
            routing_prompt = (
                f"I need to route this user request to the right agent: '{user_input}'\n\n"
                f"Available agents: {', '.join(self.agents.keys())}\n\n"
                "Which agent should handle this request? Just respond with the agent name."
            )
            
            try:
                target_agent_name = self.coordinator.model_utility.generate_text(
                    routing_prompt,
                    model_name=self.coordinator.response_model
                ).strip()
                
                if target_agent_name in self.agents:
                    agent = self.agents[target_agent_name]
                else:
                    # Default to the first agent if the coordinator returns an invalid name
                    agent = next(iter(self.agents.values()))
            except Exception as e:
                logger.error(f"Error during routing: {str(e)}")
                # Default to the first agent if there's an error
                agent = next(iter(self.agents.values()))
        else:
            # Default to the first agent if there's no coordinator
            agent = next(iter(self.agents.values()))
        
        # Have the agent respond to the message
        response = agent.respond(user_input)
        
        # Record the interaction
        self.conversation_history.append({
            "user_input": user_input,
            "agent": agent.name,
            "response": response,
            "timestamp": time.time()
        })
        
        return response, agent


def create_thinking_agent(
    name: str,
    role: Union[AgentRole, str],
    model_utility: ModelUtility,
    thinking_model: Optional[str] = None,
    response_model: Optional[str] = None,
    thinking_mode: ThinkingMode = ThinkingMode.CHAIN,
    system_prompt: Optional[str] = None,
    use_tools: bool = False,
    verbose: bool = False
) -> Agent:
    """
    Create an agent with thinking capabilities.
    
    Args:
        name: Name of the agent
        role: Role of the agent (AgentRole enum or string)
        model_utility: ModelUtility instance for LLM access
        thinking_model: Name of the model to use for thinking (if None, uses active model)
        response_model: Name of the model to use for responses (if None, uses active model)
        thinking_mode: Mode of thinking to use
        system_prompt: System prompt to use for the agent
        use_tools: Whether to enable tool usage
        verbose: Whether to log verbose information
        
    Returns:
        An Agent instance with thinking capabilities
    """
    return Agent(
        name=name,
        role=role,
        model_utility=model_utility,
        thinking_model=thinking_model,
        response_model=response_model,
        thinking_mode=thinking_mode,
        system_prompt=system_prompt,
        use_tools=use_tools,
        verbose=verbose
    )
