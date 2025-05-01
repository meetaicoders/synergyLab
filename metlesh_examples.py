from mertlesh_sonic import Agent, AgentFlow, FunctionAgent
import asyncio
# Example usage:
async def example():
    # Create agents
    async def process_data(ctx):
        ctx['processed'] = ctx['data'] * 2
        return ctx
        
    def final_step(ctx):
        ctx['result'] = f"Final: {ctx['processed']}"
        return ctx
    
    # Build flow
    flow = AgentFlow()
    flow.add_agent(Agent("loader"))
    flow.add_agent(FunctionAgent("processor", process_data))
    flow.add_agent(FunctionAgent("finalizer", final_step))
    
    # Set dependencies
    flow.add_dependency("processor", "loader")
    flow.add_dependency("finalizer", "processor")
    
    # Run the flow
    initial_ctx = {"data": 21}
    result = await flow.arun(initial_ctx)
    print(result)  # {'data': 21, 'processed': 42, 'result': 'Final: 42'}
if __name__ == "__main__":
    # Run the example async
    asyncio.run(example())




# import requests

# class Preprocessor(Agent):
#     def run(self, context):
#         print(f"[{self.name}] Preprocessing text...")
#         context["clean_text"] = context["raw_text"].lower().strip()
#         return context

# class LLMCaller(Agent):
#     def __init__(self, name, model="gpt-3.5-turbo"):
#         super().__init__(name)
#         self.model = model
    
#     def run(self, context):
#         print(f"[{self.name}] Calling {self.model}...")
#         # Simulated API call
#         context["response"] = f"AI-generated content based on: {context['clean_text']}"
#         return context

# class Postprocessor(Agent):
#     def run(self, context):
#         print(f"[{self.name}] Formatting output...")
#         context["final_output"] = context["response"].upper()
#         return context

# class Notifier(Agent):
#     def run(self, context):
#         print(f"[{self.name}] Sending notification...")
#         # Simulated notification
#         context["notification_sent"] = True
#         return context

# # Create workflow
# flow = AgentFlow()

# pre = Preprocessor("TextPreprocessor")
# llm = LLMCaller("GPT4Caller", model="gpt-4")
# post = Postprocessor("OutputFormatter")
# notify = Notifier("EmailNotifier")

# # Define dependencies
# (flow.add_agent(pre)
#  .add_agent(llm)
#  .add_agent(post)
#  .add_agent(notify)
#  .add_dependency(llm, pre)    # LLM depends on preprocessor
#  .add_dependency(post, llm)   # Postprocess after LLM
#  .add_dependency(notify, post) # Notify after postprocessing
# )

# # Execute pipeline
# result = flow.run({
#     "raw_text": "  EXAMPLE INPUT TEXT  ",
#     "user_email": "user@example.com"
# })

# print("\nFinal Context:", result)