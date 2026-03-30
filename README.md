# ReAct-style LangGraph Agent

A tool-augmented LLM agent built using LangGraph, LangChain, and ChatGroq.  
This project implements a **ReAct-style reasoning loop (Reason + Act)** and supports **multi-turn conversations using memory**.

---

## 🚀 Features

- ReAct-style reasoning (LLM → Tool → LLM loop)
- Dynamic tool selection using function calling
- Integrated tools:
  - Arxiv (research papers)
  - Wikipedia (general knowledge)
  - Tavily (web search)
- Stateful conversations using memory (MemorySaver)
- Graph-based orchestration using LangGraph
- Modular and extensible agent design

---

## 🧠 How this agent works (ReAct-style reasoning)

This agent does not directly answer queries. Instead, it follows an iterative reasoning loop:

Thought → Action → Observation → Thought → Final Answer

### Step-by-step workflow

1. **User Input**
   - Example: "latest research on LLM agents"

2. **Reason (LLM decides)**
   - The LLM analyzes the query and decides:
     - Answer directly, or
     - Call an external tool

3. **Act (Tool Execution)**
   - The agent calls one of the tools:
     - Arxiv → research papers
     - Wikipedia → general knowledge
     - Tavily → web search

4. **Observe (Tool Output)**
   - The tool returns relevant information

5. **Reason Again**
   - The LLM refines its response using tool output

6. **Final Answer**
   - The agent produces a more accurate response

---

## 🔁 ReAct loop in this project
LLM → Tool → LLM → Tool → Final Answer


This loop is implemented using:

```python
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
