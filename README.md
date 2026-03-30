# LangGraph ReAct-Style Multi-Tool LLM Agent

Built a ReAct-style LLM agent using LangGraph and ChatGroq with dynamic tool calling across Arxiv, Wikipedia, and Tavily for real-time knowledge retrieval and response refinement.

## Overview

This project implements a tool-augmented AI agent that follows a ReAct-style reasoning loop:

LLM → Tool → LLM → Final Response

The agent can:
- Decide when to answer directly
- Decide when to call external tools
- Retrieve information from academic, encyclopedic, and web sources
- Loop through tool results to improve final responses
- Persist conversation state with memory checkpointing

## Features

- ReAct-style agent workflow using LangGraph
- Dynamic tool calling with LangChain tools
- External knowledge retrieval from:
  - Arxiv
  - Wikipedia
  - Tavily Search
- Graph-based orchestration with conditional routing
- Stateful conversation flow
- Memory checkpointing for multi-turn interactions
- Modular and extensible architecture

## Tech Stack

- Python
- LangGraph
- LangChain
- ChatGroq
- Arxiv API Wrapper
- Wikipedia API Wrapper
- Tavily Search API

## Project Architecture

1. User sends a query
2. LLM evaluates whether tool usage is needed
3. If needed, the appropriate tool is called
4. Tool results are passed back to the LLM
5. LLM refines the response
6. Final answer is returned to the user

## Repository Structure

```bash
langgraph-react-style-agent/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── screenshots/
