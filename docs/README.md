# FusionQA

## 🚀 Project Overview
FusionQA is a **hybrid Retrieval-Augmented Generation (RAG) system** that combines **vector search** and **graph search** for question answering. It enables flexible querying from PDF documents by generating both a **Neo4j knowledge graph** and a **FAISS vector database**, and fusing their contexts using LLMs.

### [🔗 Summarizer API documentation link](https://tuhindutta.github.io/FusionQA/api_doc.html)
### [🔗 Docker Hub image and user guide](https://hub.docker.com/r/tkdutta/fusion-qa)

---
FusionQA provides:

* **Vector Search** → Semantic search using embeddings.
* **Graph Search** → Querying structured knowledge in a Neo4j graph.
* **Hybrid Search** → Fuses vector + graph contexts for richer answers.
* **Configurable Roles & Filters** → System roles (e.g., tutor, banking assistant) and graph node/relationship filtering.

It is designed for **flexibility** (plugging in prebuilt vector DBs/graphs), **modularity** (separate sub-packages), and **scalability** (API-driven via FastAPI + Docker).

---

## 📂 Directory Structure

```
summarizer
│   app.py                # Main FastAPI app
│   generate_graph.py     # Script to generate graph from documents
│   generate_vector.py    # Script to generate vector DB
│
├───data_processing       # Document loading & preprocessing
│       loader.py
│       utils.py
│
├───documents             # PDF files to process
│
├───graph                 # Neo4j graph-related functionality
│       prepare.py
│       token_match.py
│
├───llm                    # LLM modules
│       base_llm.py
│       embedding.py       # Chunking, embeddings, similarity search
│       graph_llm.py       # Graph-based reasoning
│       hybrid_llm.py      # Fusion of graph + vector contexts
│       vector_llm.py      # Vector-based reasoning
│
├───vector_db              # Vector DB storage (FAISS + metadata)
└───vector_store           # Sample/placeholder vector store
        index.faiss
        index.pkl
```

---

## 🧩 Sub-Packages & Modules

### 1. `data_processing`

* Loads documents from `documents/`.
* Cleans and preprocesses content before further processing.

### 2. `llm`

* **`base_llm.py`** → Base class for LLMs, extended by other modules.
* **`embedding.py`** → Splits documents into chunks, creates embeddings, stores in vector DB, performs similarity searches.
* **`vector_llm.py`** → Vector-based retrieval + response generation.
* **`graph_llm.py`** → Uses `graph.prepare.Graph` to retrieve graph context.
* **`hybrid_llm.py`** → Combines vector and graph contexts for fused reasoning.

### 3. `graph`

* Connects to **Neo4j graph DB**.
* Retrieves schema, nodes, and relationships.
* Matches query input to graph elements.
* Provides graph generation utilities.

### 4. `vector_store`

* Contains a **sample FAISS-based vector store**.
* Used if no external DB is available.

### 5. `vector_db`

* Stores generated FAISS index and metadata from documents.

### 6. Scripts

* **`generate_graph.py`** → Builds graph from documents.
* **`generate_vector.py`** → Builds vector DB from documents.
* **`app.py`** → Launches the FastAPI app, exposes API.

---

## 🔎 LLMs & Models Used

* **Graph generation, graph/vector responses, node/relationship retrieval** → `llama-3.3-70b-versatile`
* **Cypher query generation** → `openai/gpt-oss-120b`
* **Semantic embeddings for RAG** → `sentence-transformers/all-mpnet-base-v2`

---

## 🔄 Workflow

1. Place clean PDF documents inside `documents/`.
2. Generate **graph** & **vector DB** using API endpoints.
3. Refresh the vector DB using the refresh endpoint.
4. *(Optional)* Place ready-made vector DB files directly in `vector_db/` and refresh, skipping steps 1–2. Graph can also be created independently.
5. *(Optional)* Set **system role** (e.g., banking assistant, tutor).
6. *(Optional)* Restrict **node/relationship types** for graph queries.
7. Perform queries → vector search, graph search, or hybrid.

---

## 📦 Major Dependencies

* **FastAPI** → API framework (`app.py`).
* **Uvicorn** → ASGI server.
* **FAISS** → Vector similarity search engine.
* **Neo4j** → Graph database backend.
* **LangChain** → Orchestrates LLM workflows.
* **Sentence-Transformers** → Embedding model for RAG.
* **Transformers + Torch** → LLM + embeddings support.
* **pypdf** → PDF parsing.
* **Pydantic** → Data validation.

(See `requirements.txt` for full list.)

---

## ✨ Summary

FusionQA is a **modular, LLM-powered, hybrid RAG framework** that:

* Ingests and preprocesses PDFs.
* Builds both vector and graph representations.
* Supports standalone or fused querying.
* Exposes functionality via FastAPI.
* Deployable locally or via Docker.

This makes it suitable for domains like **finance, education, research, and knowledge management**, where combining **semantic search** with **structured graph reasoning** provides superior results.

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes. For more details and updates, visit the [GitHub Repository](https://github.com/tuhindutta/FusionQA).
