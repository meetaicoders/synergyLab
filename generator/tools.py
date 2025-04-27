"""
Module for implementing tools that agents can use to perform various actions.
This module provides a framework for creating, registering, and using tools
with the agent system.
"""
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union, Type, get_type_hints
import inspect
import json
import time
import logging
import os
import re
import requests
import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Enumeration of different tool categories."""
    WEB = "web"
    FILE = "file"
    MATH = "math"
    TIME = "time"
    SEARCH = "search"
    DATABASE = "database"
    API = "api"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Class representing a parameter for a tool."""
    name: str
    type: Type
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDescription:
    """Class representing the description of a tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    return_description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Class representing the result of a tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Get the string representation of the result."""
        if not self.success:
            return f"Error: {self.error}"
        return str(self.result)


T = TypeVar('T', bound=Callable)


def tool(
    description: str,
    category: Union[ToolCategory, str],
    return_description: str = "",
    examples: List[str] = None
) -> Callable[[T], T]:
    """
    Decorator for registering a function as a tool.
    
    Args:
        description: Description of the tool
        category: Category of the tool
        return_description: Description of the return value
        examples: Examples of using the tool
        
    Returns:
        The decorated function
    """
    def decorator(func: T) -> T:
        # Extract parameter information from function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        parameters = []
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_type = type_hints.get(name, Any)
            has_default = param.default is not param.empty
            default_value = param.default if has_default else None
            required = not has_default
            
            # Extract parameter description from docstring
            param_desc = ""
            if func.__doc__:
                # Match parameters in docstring (assuming Google style docstrings)
                pattern = rf"\s+{name}:\s*(.*?)(?=\n\s+\w+:|$)"
                match = re.search(pattern, func.__doc__, re.DOTALL)
                if match:
                    param_desc = re.sub(r'\s+', ' ', match.group(1).strip())
            
            parameters.append(ToolParameter(
                name=name,
                type=param_type,
                description=param_desc,
                required=required,
                default=default_value
            ))
        
        # Store tool information in function attributes
        tool_category = category.value if isinstance(category, ToolCategory) else category
        func._tool_info = ToolDescription(
            name=func.__name__,
            description=description,
            category=tool_category,
            parameters=parameters,
            return_description=return_description,
            examples=examples or []
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                return ToolResult(
                    success=True,
                    result=result,
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Error executing tool {func.__name__}: {str(e)}")
                return ToolResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
        
        # Transfer tool info to wrapper
        wrapper._tool_info = func._tool_info
        
        return wrapper
    
    return decorator


def wraps(wrapped):
    """
    Simple implementation of functools.wraps for maintaining function metadata.
    """
    def decorator(wrapper):
        wrapper.__name__ = wrapped.__name__
        wrapper.__doc__ = wrapped.__doc__
        wrapper.__module__ = wrapped.__module__
        wrapper.__qualname__ = wrapped.__qualname__
        wrapper.__annotations__ = wrapped.__annotations__
        return wrapper
    return decorator


class ToolRegistry:
    """
    A registry for tools that agents can use.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance.tools = {}
            cls._instance.categories = {}
        return cls._instance
    
    def register_tool(self, func: Callable) -> Callable:
        """
        Register a function as a tool.
        
        Args:
            func: Function to register
            
        Returns:
            The registered function
        """
        if not hasattr(func, '_tool_info'):
            raise ValueError(f"Function {func.__name__} is not decorated with @tool")
        
        tool_name = func.__name__
        tool_info = func._tool_info
        
        self.tools[tool_name] = func
        
        # Add to category
        category = tool_info.category
        if isinstance(category, ToolCategory):
            category = category.value
            
        if category not in self.categories:
            self.categories[category] = []
        
        if tool_name not in self.categories[category]:
            self.categories[category].append(tool_name)
        
        return func
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool function if found, None otherwise
        """
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: Union[ToolCategory, str]) -> List[str]:
        """
        Get all tools in a category.
        
        Args:
            category: Category to get tools for
            
        Returns:
            List of tool names in the category
        """
        category_value = category.value if isinstance(category, ToolCategory) else category
        return self.categories.get(category_value, [])
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of all tools
        """
        return self.tools
    
    def get_tool_description(self, name: str) -> Optional[ToolDescription]:
        """
        Get the description of a tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            ToolDescription if found, None otherwise
        """
        tool = self.get_tool(name)
        if tool and hasattr(tool, '_tool_info'):
            return tool._tool_info
        return None
    
    def get_all_descriptions(self) -> Dict[str, ToolDescription]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            Dictionary of tool descriptions
        """
        return {name: self.get_tool_description(name) for name in self.tools}
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert all tool descriptions to JSON Schema format (compatible with OpenAI function calling).
        
        Returns:
            Dictionary of tool descriptions in JSON Schema format
        """
        functions = []
        
        for name, tool in self.tools.items():
            if hasattr(tool, '_tool_info'):
                tool_info = tool._tool_info
                
                # Convert parameters to properties
                properties = {}
                required = []
                
                for param in tool_info.parameters:
                    # Map Python types to JSON Schema types
                    if param.type == str:
                        type_spec = {"type": "string"}
                    elif param.type == int:
                        type_spec = {"type": "integer"}
                    elif param.type == float:
                        type_spec = {"type": "number"}
                    elif param.type == bool:
                        type_spec = {"type": "boolean"}
                    elif param.type == list or param.type == List:
                        type_spec = {"type": "array"}
                    elif param.type == dict or param.type == Dict:
                        type_spec = {"type": "object"}
                    else:
                        type_spec = {"type": "string"}
                    
                    properties[param.name] = {
                        **type_spec,
                        "description": param.description
                    }
                    
                    if param.required:
                        required.append(param.name)
                
                function_def = {
                    "name": name,
                    "description": tool_info.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
                
                if required:
                    function_def["parameters"]["required"] = required
                
                functions.append(function_def)
        
        return {"functions": functions}


# Create a global registry instance
registry = ToolRegistry()


# ===== WEB TOOLS =====

@tool(
    description="Search the web for information on a specific query.",
    category=ToolCategory.WEB,
    return_description="Search results as a list of snippets with URLs.",
    examples=["search_web(query='latest news about AI')"]
)
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information on a specific query.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        List of search results with title, snippet, and URL
    """
    try:
        # Mock implementation for demonstration
        logger.info(f"Searching web for: {query}")
        time.sleep(1)  # Simulate network delay
        
        # Mock results
        results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a snippet for result {i+1} related to {query}...",
                "url": f"https://example.com/result{i+1}"
            }
            for i in range(num_results)
        ]
        
        return results
    except Exception as e:
        logger.error(f"Error in search_web: {str(e)}")
        raise


@tool(
    description="Fetch the content of a webpage at the specified URL.",
    category=ToolCategory.WEB,
    return_description="The text content of the webpage.",
    examples=["fetch_webpage(url='https://example.com')"]
)
def fetch_webpage(url: str, include_html: bool = False) -> str:
    """
    Fetch the content of a webpage.
    
    Args:
        url: The URL of the webpage to fetch
        include_html: Whether to include HTML tags (default: False)
        
    Returns:
        The text content of the webpage
    """
    try:
        # For demonstration purposes, we'll use a simple approach
        # In a real implementation, you might want to use a more robust solution
        # like newspaper3k or trafilatura
        logger.info(f"Fetching webpage: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            if include_html:
                return response.text
            
            # Simple extraction of text (in a real implementation, use a better parser)
            # This is a very naive implementation for demonstration
            text = re.sub('<.*?>', ' ', response.text)
            text = re.sub('\\s+', ' ', text)
            return text.strip()
            
        except requests.RequestException as e:
            raise ValueError(f"Error fetching webpage: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in fetch_webpage: {str(e)}")
        raise


# ===== FILE TOOLS =====

@tool(
    description="Read the content of a file.",
    category=ToolCategory.FILE,
    return_description="The content of the file as a string.",
    examples=["read_file(file_path='data.txt')"]
)
def read_file(file_path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The content of the file as a string
    """
    try:
        logger.info(f"Reading file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return content
    except Exception as e:
        logger.error(f"Error in read_file: {str(e)}")
        raise


@tool(
    description="Write content to a file.",
    category=ToolCategory.FILE,
    return_description="True if the file was written successfully.",
    examples=["write_file(file_path='output.txt', content='Hello, world!')"]
)
def write_file(file_path: str, content: str, append: bool = False) -> bool:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        append: Whether to append to the file (default: False)
        
    Returns:
        True if the file was written successfully
    """
    try:
        logger.info(f"Writing to file: {file_path}")
        
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
            
        return True
    except Exception as e:
        logger.error(f"Error in write_file: {str(e)}")
        raise


@tool(
    description="List files in a directory.",
    category=ToolCategory.FILE,
    return_description="List of files in the directory.",
    examples=["list_files(directory_path='./data')"]
)
def list_files(directory_path: str, pattern: str = "*") -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory_path: Path to the directory
        pattern: Pattern to match files against (default: "*")
        
    Returns:
        List of files in the directory
    """
    try:
        import glob
        logger.info(f"Listing files in: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        if not os.path.isdir(directory_path):
            raise ValueError(f"Not a directory: {directory_path}")
            
        # Join directory path with pattern
        search_path = os.path.join(directory_path, pattern)
        
        # Get list of files
        files = glob.glob(search_path)
        
        # Convert to relative paths if requested
        return [os.path.basename(f) for f in files]
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        raise


# ===== MATH TOOLS =====

@tool(
    description="Evaluate a mathematical expression.",
    category=ToolCategory.MATH,
    return_description="The result of the mathematical expression.",
    examples=["calculate('2 + 2 * 3')", "calculate('sin(pi/2)')"]
)
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate
        
    Returns:
        The result of the expression
    """
    try:
        # Using safer eval approach
        import math
        import numpy as np
        
        logger.info(f"Calculating: {expression}")
        
        # Define safe functions and constants
        safe_dict = {
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            
            # Math module
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            
            # NumPy functions if available
            "average": np.average if 'np' in locals() else None,
            "mean": np.mean if 'np' in locals() else None,
            "median": np.median if 'np' in locals() else None,
            "std": np.std if 'np' in locals() else None,
            "var": np.var if 'np' in locals() else None,
        }
        
        # Remove None values
        safe_dict = {k: v for k, v in safe_dict.items() if v is not None}
        
        # Evaluate the expression with safe functions only
        return eval(expression, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        logger.error(f"Error in calculate: {str(e)}")
        raise


# ===== TIME TOOLS =====

@tool(
    description="Get the current date and time.",
    category=ToolCategory.TIME,
    return_description="The current date and time.",
    examples=["get_current_time()", "get_current_time(timezone='US/Pacific')"]
)
def get_current_time(timezone: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.
    
    Args:
        timezone: The timezone to use (default: "UTC")
        format: The format to return the date and time in (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        The current date and time in the specified format and timezone
    """
    try:
        logger.info(f"Getting current time in timezone: {timezone}")
        
        # In a real implementation, you'd use pytz or similar
        # For simplicity in this example, we'll just return the current UTC time
        current_time = datetime.datetime.utcnow()
        
        if timezone.lower() != "utc":
            logger.info(f"Note: Only UTC is supported in this example. Returning UTC time.")
        
        return current_time.strftime(format)
    except Exception as e:
        logger.error(f"Error in get_current_time: {str(e)}")
        raise


# ===== UTILITY TOOLS =====

@tool(
    description="Format data as JSON.",
    category="utility",
    return_description="The formatted JSON string.",
    examples=["format_json(data={'name': 'John', 'age': 30})"]
)
def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as JSON.
    
    Args:
        data: The data to format as JSON
        indent: Number of spaces for indentation (default: 2)
        
    Returns:
        The formatted JSON string
    """
    try:
        logger.info("Formatting data as JSON")
        return json.dumps(data, indent=indent)
    except Exception as e:
        logger.error(f"Error in format_json: {str(e)}")
        raise


@tool(
    description="Parse JSON data.",
    category="utility",
    return_description="The parsed JSON data.",
    examples=["parse_json(json_string='{\"name\": \"John\", \"age\": 30}')"]
)
def parse_json(json_string: str) -> Any:
    """
    Parse JSON data.
    
    Args:
        json_string: The JSON string to parse
        
    Returns:
        The parsed JSON data
    """
    try:
        logger.info("Parsing JSON data")
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error in parse_json: {str(e)}")
        raise


# Register all tools
for name, obj in list(globals().items()):
    if callable(obj) and hasattr(obj, '_tool_info'):
        registry.register_tool(obj)


class ToolManager:
    """
    A class for managing tools that an agent can use.
    """
    def __init__(self):
        self.registry = registry
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool function if found, None otherwise
        """
        return self.registry.get_tool(name)
    
    def get_tools_by_category(self, category: Union[ToolCategory, str]) -> List[str]:
        """
        Get all tools in a category.
        
        Args:
            category: Category to get tools for
            
        Returns:
            List of tool names in the category
        """
        return self.registry.get_tools_by_category(category)
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of all tools
        """
        return self.registry.get_all_tools()
    
    def get_tool_description(self, name: str) -> Optional[ToolDescription]:
        """
        Get the description of a tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            ToolDescription if found, None otherwise
        """
        return self.registry.get_tool_description(name)
    
    def get_all_descriptions(self) -> Dict[str, ToolDescription]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            Dictionary of tool descriptions
        """
        return self.registry.get_all_descriptions()
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert all tool descriptions to JSON Schema format (compatible with OpenAI function calling).
        
        Returns:
            Dictionary of tool descriptions in JSON Schema format
        """
        return self.registry.to_json_schema()
    
    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool.
        
        Args:
            name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            ToolResult containing the result or error
        """
        tool_func = self.get_tool(name)
        if not tool_func:
            return ToolResult(
                success=False,
                error=f"Tool not found: {name}"
            )
        
        start_time = time.time()
        try:
            result = tool_func(**kwargs)
            execution_time = time.time() - start_time
            
            # If result is already a ToolResult, just update execution time
            if isinstance(result, ToolResult):
                result.execution_time = execution_time
                return result
            
            # Otherwise, wrap the result in a ToolResult
            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing tool {name}: {str(e)}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
