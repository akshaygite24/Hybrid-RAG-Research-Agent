# Hybrid RAG Research Agent

An agentic AI research assistant that combines document retrieval and live web search to answer complex research queries. Built with Langchain, Groq, Pinecone and streamlit.

---

## What is does

Most AI chatbots just generate answer from memory. This system actually retrieves information first, then answers.
It:
- Breaks complex querues ubti focused sub-questions
- Decides which tool to use for each sub-question (PDF search or web search)
- Combines results from multiple sources
- Reviews and improves the final answer using a critic agent
- Remembers conversation history across turns

---

## Architecture

```
User Query
    ↓
Planner Agent — breaks query into sub-questions
    ↓
Research Agent — routes each sub-question to the right tool
    ├── RAG Tool — searches uploaded PDF documents (Pinecone)
    ├── Web Search Tool — searches the internet (DuckDuckGo)
    └── Direct LLM — answers simple/general questions
    ↓
Aggregator — combines all answers
    ↓
Critic Agent — reviews and improves the answer
    ↓
Final Response
```

---

## Features

- **Query Decomposition** — complex questions are broken into focused sub-questions
- **Hybrid Tool Routing** — agent dynamically decides between PDF search, web search, or direct LLM
- **Semantic Search** — uses vector embeddings to find relevant document chunks
- **Live Web Search** — retrieves current information via DuckDuckGo
- **Critic Agent** — self-reflection loop that improves answer quality
- **Conversational Memory** — remembers previous turns in the conversation
- **Smart Detection** — automatically skips planner/critic for simple conversational queries
- **PDF Ingestion** — upload and ingest any PDF document via the UI


## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database | Pinecone |
| Web Search | DuckDuckGo |
| Agent Framework | LangChain |
| UI | Streamlit |

---

## Project Structure

```
Hybrid-RAG-Research-Agent/
│
├── agents/
│   ├── __init__.py
│   ├── research_agent.py    # main ReAct agent with tool routing
│   ├── planner_agent.py     # query decomposition
│   └── critic_agent.py      # answer review and improvement
│
├── tools/
│   ├── rag_tool.py          # PDF search tool (BaseTool)
│   └── web_search_tool.py   # DuckDuckGo search tool (BaseTool)
│
├── rag/
│   ├── ingest.py            # PDF loading, chunking, Pinecone ingestion
│   └── retriever.py         # semantic search and chunk retrieval
│
├── prompts/                 # prompt templates (future)
│
├── utils/
│   └── llm.py               # Groq LLM setup
│
├── pipeline.py              # orchestrates full agent pipeline
├── app.py                   # Streamlit UI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/akshaygite24/Hybrid-RAG-Research-Agent.git
cd Hybrid-RAG-Research-Agent
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=research-agent
```

Get your API keys:
- Groq: https://console.groq.com
- Pinecone: https://app.pinecone.io

### 5. Run the app

```bash
streamlit run app.py
```

---

## How to Use

1. Upload a PDF documetn using the sidebar
2. Wait for ingestion to complete
3. Ask any research question in the chat
4. The agent will automatically search your document and the web
5. Toggle **Use Planner** and **Use Critic** in the sidebar to control the pipeline
---

## How it Works

**Planner Agent** breaks your query into 2-3 focused sub-questions using structured JSON output.

**Research Agent** uses the ReAct reasoning pattern (Thought → Action → Observation) to decide which tool to call for each sub-question.

**RAG Tool** converts your question into a vector embedding, searches Pinecone for the most semantically similar document chunks, and sends them to the LLM to generate an answer.

**Web Search Tool** queries DuckDuckGo for real-time information and summarizes the results.

**Critic Agent** reviews the combined answer for gaps, weak explanations, and clarity issues, then rewrites it to improve quality.

---
## Future Improvements

- [ ] Add support for multiple PDF documents
- [ ] Implement hybrid search (dense + sparse) in Pinecone
- [ ] Add streaming responses for real-time output
- [ ] Move hardcoded prompts to `prompts/` folder
- [ ] Add evaluation metrics for answer quality
- [ ] Support for more file types (DOCX, TXT, CSV)
- [ ] Add authentication for multi-user support

---

## Author
**Akshay Gite**