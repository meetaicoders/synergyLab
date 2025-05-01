import asyncio
from typing import Dict, List, Set, Any, Callable, Awaitable, Optional, Union
import logging


class Agent:
    """Base Agent class that can be run synchronously or asynchronously."""
    
    def __init__(self, name: str):
        self.name = name
        
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous execution method."""
        return context
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronous execution method."""
        return self.run(context)


class FunctionAgent(Agent):
    """Agent that wraps a function."""
    
    def __init__(self, name: str, func: Union[Callable, Awaitable]):
        super().__init__(name)
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)
        
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_async:
            raise RuntimeError(f"Agent {self.name} contains async function but was called synchronously")
        return self.func(context)
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_async:
            return await self.func(context)
        else:
            return self.func(context)


class AgentFlow:
    """DAG-based flow of agents with async execution capabilities."""
    
    def __init__(self, logging_level=logging.INFO):
        self.agents: Dict[str, Agent] = {}
        self.dependencies: Dict[str, Set[str]] = {}  # agent -> dependencies
        self.dependents: Dict[str, Set[str]] = {}    # agent -> dependents
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
    
    def add_agent(self, agent: Agent) -> 'AgentFlow':
        """Add an agent to the flow."""
        if not isinstance(agent, Agent):
            raise TypeError("Only Agent instances accepted")
        
        if agent.name in self.agents:
            raise ValueError(f"Agent with name '{agent.name}' already exists")
            
        self.agents[agent.name] = agent
        self.dependencies[agent.name] = set()
        self.dependents[agent.name] = set()
        return self
    
    def add_dependency(self, dependent_name: str, dependency_name: str) -> 'AgentFlow':
        """Specify that agent A depends on agent B."""
        if dependent_name not in self.agents:
            raise ValueError(f"Agent '{dependent_name}' not registered")
        if dependency_name not in self.agents:
            raise ValueError(f"Agent '{dependency_name}' not registered")
        
        self.dependencies[dependent_name].add(dependency_name)
        self.dependents[dependency_name].add(dependent_name)
        return self
    
    def _detect_cycles(self) -> Optional[List[str]]:
        """Check for dependency cycles in the graph."""
        visited = set()
        path = set()
        
        def dfs(node):
            if node in path:
                return [node]
            if node in visited:
                return None
                
            visited.add(node)
            path.add(node)
            
            for dep in self.dependencies[node]:
                cycle = dfs(dep)
                if cycle:
                    if cycle[0] == node:
                        return cycle
                    else:
                        return [node] + cycle
            
            path.remove(node)
            return None
            
        for agent_name in self.agents:
            cycle = dfs(agent_name)
            if cycle:
                return cycle
        return None
    
    def _topological_sort(self) -> List[str]:
        """Sort agents in topological order."""
        cycle = self._detect_cycles()
        if cycle:
            cycle_str = " -> ".join(cycle + [cycle[0]])
            raise ValueError(f"Dependency cycle detected: {cycle_str}")
        
        sorted_agents = []
        no_deps = [name for name, deps in self.dependencies.items() if not deps]
        visited = set(no_deps)
        
        while no_deps:
            agent_name = no_deps.pop(0)
            sorted_agents.append(agent_name)
            
            for dependent in list(self.dependents[agent_name]):
                self.dependencies[dependent].remove(agent_name)
                if not self.dependencies[dependent] and dependent not in visited:
                    no_deps.append(dependent)
                    visited.add(dependent)
        
        if len(sorted_agents) != len(self.agents):
            raise ValueError("Could not resolve all dependencies")
            
        return sorted_agents
    
    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the flow synchronously."""
        context = initial_context or {}
        sorted_agents = self._topological_sort()
        
        for agent_name in sorted_agents:
            agent = self.agents[agent_name]
            self.logger.debug(f"Running agent: {agent_name}")
            try:
                context = agent.run(context)
            except Exception as e:
                self.logger.error(f"Error in agent {agent_name}: {str(e)}")
                raise
        
        return context
    
    async def arun(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the flow asynchronously, executing independent agents in parallel."""
        context = initial_context or {}
        sorted_levels = self._group_by_levels()
        
        for level in sorted_levels:
            self.logger.debug(f"Running level with {len(level)} agents")
            # Agents at the same level can run in parallel
            results = await asyncio.gather(
                *[self.agents[name].arun(context.copy()) for name in level],
                return_exceptions=True
            )
            
            # Merge results from this level
            for i, agent_name in enumerate(level):
                if isinstance(results[i], Exception):
                    self.logger.error(f"Error in agent {agent_name}: {str(results[i])}")
                    raise results[i]
                
                # Update main context with this agent's results
                context.update(results[i])
        
        return context
    
    def _group_by_levels(self) -> List[List[str]]:
        """Group agents by execution level for parallel processing."""
        levels = []
        remaining = set(self.agents.keys())
        dependencies = {name: deps.copy() for name, deps in self.dependencies.items()}
        
        while remaining:
            # Find all nodes with no dependencies
            current_level = [name for name in remaining if not dependencies[name]]
            
            if not current_level:
                raise ValueError("Dependency cycle detected during level grouping")
            
            levels.append(current_level)
            remaining -= set(current_level)
            
            # Remove edges from the graph
            for name in remaining:
                dependencies[name] -= set(current_level)
        
        return levels