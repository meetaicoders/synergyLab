class Agent:
    def __init__(self, name):
        self.name = name
        
    def run(self, context):
        return context

class AgentFlow:
    def __init__(self):
        self.agents = {}
        self.dependencies = {}
        self.reverse_deps = {}

    def add_agent(self, agent):
        if not isinstance(agent, Agent):
            raise TypeError("Only Agent instances accepted")
        self.agents[agent.name] = agent
        self.dependencies.setdefault(agent, [])
        return self

    def add_dependency(self, agent_a, agent_b):
        if agent_a not in self.agents.values() or agent_b not in self.agents.values():
            raise ValueError("Unregistered agents detected")
        self.dependencies[agent_a].append(agent_b)
        self.reverse_deps.setdefault(agent_b, []).append(agent_a)
        return self

    def _topological_sort(self):
        in_degree = {agent: 0 for agent in self.agents.values()}
        for agent in self.agents.values():
            for dep in self.dependencies[agent]:
                in_degree[agent] += 1  # FIXED: Track agent's in-degree

        queue = [agent for agent, deg in in_degree.items() if deg == 0]
        sorted_agents = []
        while queue:
            current = queue.pop(0)
            sorted_agents.append(current)
            for neighbor in self.reverse_deps.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_agents) != len(self.agents):
            raise ValueError("Dependency cycle detected")
        return sorted_agents  # FIXED: No reversal needed

    def run(self, initial_context=None):
        context = initial_context or {}
        for agent in self._topological_sort():
            context = agent.run(context)
        return context