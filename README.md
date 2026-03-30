# ReAct-style LangGraph Agent

A tool-augmented LLM agent built using LangGraph, LangChain, and ChatGroq.  
This project implements a ReAct-style reasoning loop where the model can dynamically decide when to call external tools and when to respond directly.

---

## Overview

This project builds a **stateful, tool-calling AI agent** that can:

- Answer user queries
- Decide when external data is needed
- Call tools like Arxiv, Wikipedia, and Tavily
- Use tool results to improve responses
- Maintain conversation context across multiple turns

The system follows a **ReAct-style reasoning pattern**:

LLM → Tool → LLM → Final Answer

---

## Features

- ReAct-style agent workflow (Reason + Act)
- Dynamic tool selection using LLM decision-making
- Integration with:
  - Arxiv (research papers)
  - Wikipedia (general knowledge)
  - Tavily (web search)
- Conditional execution using LangGraph
- Multi-turn conversation with memory (MemorySaver)
- Modular and extensible architecture

---

## Tech Stack

### AI / LLM Engineering
- LangGraph
- LangChain
- ChatGroq (Qwen-32B)
- Tool Calling / Function Calling
- ReAct-style Agentic Workflow

### Data & Integration
- Arxiv API
- Wikipedia API
- Tavily Search API

### Programming
- Python
