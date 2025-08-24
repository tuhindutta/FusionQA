from fastapi import FastAPI
import os
from pydantic import BaseModel
import subprocess
from llm.embedding import VectorStore
from llm.hybrid_llm import HybridLlm


QUERY_MODEL = "llama-3.3-70b-versatile"
CYPHER_MODEL = "openai/gpt-oss-120b"
CYPHER_MODEL_API_KEY = os.getenv("GROQ_API_KEY")
QUERY_MODEL_API_KEY = os.getenv("GROQ_API_KEY")
CYPHER_MODEL_API = "groq"
QUERY_MODEL_API = "groq"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

for var_name, value in {
    "CYPHER_MODEL_API_KEY": CYPHER_MODEL_API_KEY,
    "QUERY_MODEL_API_KEY": QUERY_MODEL_API_KEY,
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USERNAME,
    "NEO4J_PASSWORD": NEO4J_PASSWORD
}.items():
    if not value:
        raise EnvironmentError(f"Missing environment variable: {var_name}")

vs = VectorStore()
vs.load()

app = FastAPI(
    title="📘 Summarizer API",
    description="""
A **Retrieval-Augmented Generation (RAG) API** for:
- 📂 Document ingestion
- 🔍 Vector, graph & hybrid search (Vector + Graph)
- 🤖 Query answering with LLM integration

### Features
- Upload & manage documents
- Query responses powered by LLM
- API-ready vector and graph database

> Use the `/docs` endpoint for interactive Swagger UI or `/redoc` for alternative documentation.
"""
)

llm_hybrid = HybridLlm(CYPHER_MODEL, CYPHER_MODEL_API_KEY, QUERY_MODEL, QUERY_MODEL_API_KEY,
                       vs, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, None,
                       CYPHER_MODEL_API, QUERY_MODEL_API,
                       history_tracking=False)

class SystemRole(BaseModel):
    prompt: str

class QueryRequest(BaseModel):
    query: str
    use_vector: bool = True
    use_graph: bool = False

class GraphAllowedNodesRels(BaseModel):
    allowed_nodes: list = []
    allowed_relationships: list = []

class PageWiseGraphAllowedNodesRels(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    allowed_nodes: list = []
    allowed_relationships: list = []

@app.post("/set-system-role")
def set_system_role(request: SystemRole):
    try:
        llm_hybrid.system_role_prompt = request.prompt
        llm_hybrid.get_llms()
        return {"status": "success", "message": "System role set"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/refresh-vector-db")
def refresh_vector_db():
    try:
        vs.load()
        return {"status": "success", "message": "Vector DB refreshed"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}

@app.post("/query")
def query_rag(request: QueryRequest):
    """Run a RAG query using Vector or Hybrid mode."""
    if not llm_hybrid.llms_loaded:
        llm_hybrid.get_llms() 
    if request.use_vector and request.use_graph:
        llm = llm_hybrid
        llm_type = "hybrid"
        system_role = llm.system_role_prompt
    elif request.use_vector:
        llm = llm_hybrid.vector_llm
        llm_type = "vector"
        system_role = llm.system_role_prompt
    elif request.use_graph:
        llm = llm_hybrid.graph_llm
        llm_type = "graph"
        system_role = None
    query = request.query
    response = llm.query_llm(query)
    return {
        "system_role": system_role,
        "llm_type": llm_type,
        "using_sample_vector": vs.using_sample_vector,
        "query": query,
        "response": response
    }

@app.post("/vector/extract")
def generate_vector():
    try:
        subprocess.run(["python", "generate_vector.py"], check=True)
        vs.load()
        return {"status": "success", "message": "Vector generated"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}
    
    
@app.post("/vector/extract-page-wise")
def generate_vector_pagewise():
    try:
        subprocess.run(["python", "generate_vector.py", "--page-wise"], check=True)
        vs.load()
        return {"status": "success", "message": "Vector generated with page-wise splitting"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/graph/extract")
def generate_graph(request: PageWiseGraphAllowedNodesRels):
    try:
        chunk_size = str(request.chunk_size)
        chunk_overlap = str(request.chunk_overlap)

        args = ["python", "generate_graph.py", "--chunk-size", chunk_size,
                "--chunk-overlap", chunk_overlap]

        nodes = request.allowed_nodes
        rels = request.allowed_relationships

        if len(rels) > 0:
            args.extend(["--allowed_nodes", *nodes])
        if len(nodes) > 0:
            args.extend(["--allowed_relationships", *rels])

        subprocess.run(args, check=True)
        vs.load()
        return {"status": "success", "message": "Graph generated"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Subprocess failed: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}
    
    
@app.post("/graph/extract-page-wise")
def generate_graph_pagewise(request: GraphAllowedNodesRels):
    try:
        args = ["python", "generate_graph.py", "--page-wise"]
        
        nodes = request.allowed_nodes
        rels = request.allowed_relationships

        if len(rels) > 0:
            args.extend(["--allowed_nodes", *nodes])
        if len(nodes) > 0:
            args.extend(["--allowed_relationships", *rels])

        subprocess.run(args, check=True)

        vs.load()
        return {"status": "success", "message": "Graph generated with page-wise splitting"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Subprocess failed: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}

    
@app.post("/graph/set-include-types")
def set_include_types(request: GraphAllowedNodesRels):
    try:
        llm_hybrid.allowed_nodes = request.allowed_nodes
        llm_hybrid.allowed_relationships = request.allowed_relationships
        llm_hybrid.get_llms()
        return {"status": "success", "message": "Graph include types set"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}