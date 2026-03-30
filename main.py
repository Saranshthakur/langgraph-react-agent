import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

# Tool wrappers and tool classes
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults

# LLM
from langchain_groq import ChatGroq

# LangGraph components
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# Load API keys from .env file
load_dotenv()


# =========================================================
# 1. INITIALIZE TOOLS
# =========================================================
# These are the external tools your agent can use.
# This is where TOOL USAGE starts.

# Tool 1: Arxiv search tool
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

# Tool 2: Wikipedia search tool
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

# Tool 3: Tavily web search tool
tavily_tool = TavilySearchResults()

# Put all tools in one list so the LLM can access them
tools = [arxiv_tool, wiki_tool, tavily_tool]


# =========================================================
# 2. INITIALIZE LLM
# =========================================================
# This is the main language model that will decide:
# - whether to answer directly
# - or call a tool first
llm = ChatGroq(model="qwen-qwq-32b")

# Bind tools to the LLM
# This is where the LLM gets tool-calling ability.
llm_with_tools = llm.bind_tools(tools)


# =========================================================
# 3. DEFINE STATE
# =========================================================
# State stores conversation messages.
# LangGraph passes this state between nodes.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# =========================================================
# 4. DEFINE LLM NODE
# =========================================================
# This node sends conversation messages to the LLM.
# The LLM can either:
# - give a direct answer
# - or request a tool call
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# =========================================================
# 5. BUILD GRAPH
# =========================================================
# Create the workflow graph
builder = StateGraph(State)

# Add node 1: LLM node
builder.add_node("tool_calling_llm", tool_calling_llm)

# Add node 2: Tool execution node
# This node actually runs the tools when the LLM requests them
builder.add_node("tools", ToolNode(tools))

# Start the graph by calling the LLM first
builder.add_edge(START, "tool_calling_llm")

# Conditional routing:
# If the LLM decides to call a tool, go to "tools"
# If not, stop and return the answer
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)

# After tool execution, go back to the LLM
# This creates the ReAct-style loop:
# LLM -> Tool -> LLM
builder.add_edge("tools", "tool_calling_llm")


# =========================================================
# 6. COMPILE GRAPH
# =========================================================
# Graph without memory
graph = builder.compile()

# Graph with memory
# Memory allows multi-turn conversation by saving previous messages
memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)


# =========================================================
# 7. RUN INTERACTIVE CHAT
# =========================================================
print("\nReAct-style LangGraph Agent is running. Type 'exit' to quit.\n")

# Keep same thread_id so memory works across multiple user questions
thread_id = "1"

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = graph_memory.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    final_msg = result["messages"][-1]
    print(f"Agent: {final_msg.content}\n")
