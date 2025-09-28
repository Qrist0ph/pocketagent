# LangGraph Multi-Agent Reference Implementation

This repository provides a **reference implementation of a multi-agent system** built with [LangGraph](https://github.com/langchain-ai/langgraph).  
It is intended as a **learning resource** for agent architectures and can also be used for **rapid prototyping** and **proof-of-concepts**.
The MCP Integration is based on [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters).

The Agent can be exposed
 * via Command Line
 * via REST API (FastApi)
 * via Chat Website (Streamlit)
---

## Architecture Overview

```mermaid
flowchart TD
    START --> router
    router -->|intent: weather| weather_agent
    router -->|intent: travel| travel_agent
    router -->|intent: chitchat| smalltalk
    router -->|intent: rag| rag_agent
    router -->|intent: return_agent| return_agent
    router -->|intent: mcp_agent| mcp_agent

    weather_agent --> weather_tools
    weather_agent --> END
    travel_agent --> travel_tools
    travel_agent --> END
    rag_agent --> condense
    condense --> retrieve
    retrieve --> generate
    generate --> END
    smalltalk --> END
    return_agent --> return_output
    return_output --> END

    %% Add clickable links to all nodes except START and END
    click router "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/router_node.py" "Go to router_node.py" _blank
    click weather_agent "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/agents.py#L40" "Weather Agent" _blank
    click travel_agent "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/agents.py#L44" "Travel Agent" _blank
    click smalltalk "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/smalltalk_node.py#L16" "Smalltalk Agent" _blank
    click rag_agent "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/ragbot.py" "RAG Agent" _blank
    click return_agent "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/wizardagent.py#L28" "Return Agent" _blank
    click mcp_agent "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent_cli.py#L37" "MCP Agent" _blank
    click weather_tools "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/agents.py#L40" "Weather Tools" _blank
    click travel_tools "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent/agents.py#L44" "Travel Tools" _blank
    click return_output "https://github.com/Qrist0ph/pocketagent/blob/main/src/pocketagent_app.py#L25" "Return Output" _blank

    %% Node coloring
    style weather_agent fill:#b6fcd5,stroke:#333,stroke-width:2px
    style travel_agent fill:#b6fcd5,stroke:#333,stroke-width:2px
    style weather_tools fill:#b6fcd5,stroke:#333,stroke-width:2px
    style travel_tools fill:#b6fcd5,stroke:#333,stroke-width:2px
    style smalltalk fill:#ffa500,stroke:#333,stroke-width:2px
    style rag_agent fill:#87ceeb,stroke:#333,stroke-width:2px
    style condense fill:#87ceeb,stroke:#333,stroke-width:2px
    style retrieve fill:#87ceeb,stroke:#333,stroke-width:2px
    style generate fill:#87ceeb,stroke:#333,stroke-width:2px
    style return_agent fill:#ffb6c1,stroke:#333,stroke-width:2px
    style return_output fill:#ffb6c1,stroke:#333,stroke-width:2px
    style mcp_agent fill:#ff4c4c,stroke:#333,stroke-width:2px
```

---

## Agent Concepts

- **Intention-Based Routing**  
  - Detects intent and routes to the appropriate sub-agent.  
  👉 [router_node.py](src/pocketagent/router_node.py)

- **RAG Agent**  
  - Retrieval-Augmented Generation.  
  - Answers product-related questions.  
  - Data source: `product_faq.md`  
  - Vectorized with **FAISS**.  
  👉 [ragbot.py](src/pocketagent/ragagent/ragagent.py)

- **Form / Workflow Wizard**  
  - Collects information like a form or guided workflow.  
  - Detects when all required fields are filled.  
  - Example: Return process → collects *email* + *order ID*.  
  👉 [wizardagent.py](src/pocketagent/returnagent/wizardagent.py)

- **Small Talk Agent**  
  - Just does some small talk.  
  👉 [smalltalk_node.py](src/pocketagent/smalltalk_agent/smalltalk_node.py)  


- **Tool Agents**  
  - Call external (mock) tools.  
  - Examples: Weather queries, hotel lookup.  
  👉 [weather_tools.py](src/pocketagent/TravelAgent/agent.py)  


- **MCP Agent**  
  - Queries a [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) server for available tools.  
  - Includes a minimal MCP server implementation.  
  👉 [pocketagent_cli.py](src/mcpagent/mcpagent.py)

---

# Getting Started

# Run on Linux & Mac & Windows WSL

## Initialize Environment
```bash
git clone https://github.com/Qrist0ph/pocketagent.git
```
```bash
cd pocketagent
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
export OPENAI_API_KEY=sk-123456789abcdef
```

## Run via Command Line
```bash
python3 src/pocketagent_cli.py 
```
<img width="1215" height="353" alt="{25700465-2DB2-4064-A03D-BDD008EE8F26}" src="https://github.com/user-attachments/assets/d1dd738c-118d-4b6e-895e-4027dbe145f4" />


## Run API:

```bash
uvicorn src.api.index:app --reload
```

Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

<img width="1528" height="592" alt="{8705C21B-E321-4785-8438-206A98E24C31}" src="https://github.com/user-attachments/assets/7c8d2bce-fc5a-47a6-9acb-c9fab726f265" />

## Run Streamlit :

```bash
streamlit run src/streamlit/chatbot.py 
```

http://localhost:8501/

<img width="797" height="369" alt="{EB20555E-A51A-4287-B1A8-AC588E4F52DB}" src="https://github.com/user-attachments/assets/8ee10061-2a4f-4c60-85b2-8c049a48d29e" />

## Agent2Agent Server

```bash
python3 src/a2a/__main__.py 
```bash

http://localhost:10000/.well-known/agent-card.json