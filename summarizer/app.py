from fastapi import FastAPI
from pydantic import BaseModel
import os
import subprocess
from llm.embedding import VectorStore
from llm.vector_llm import VectorLlm
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

app = FastAPI(title="RAG API", description="Vector & Hybrid RAG API")


llm_vector = VectorLlm(QUERY_MODEL, QUERY_MODEL_API_KEY, vs, history_tracking=False)

llm_hybrid = HybridLlm(CYPHER_MODEL, CYPHER_MODEL_API_KEY, QUERY_MODEL, QUERY_MODEL_API_KEY,
                       vs, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
                       CYPHER_MODEL_API, QUERY_MODEL_API,
                       history_tracking=False)

class QueryRequest(BaseModel):
    query: str
    hybrid: bool = False

class GraphIncludeTypes(BaseModel):
    include_types: list = []

class GraphAllowedNodesRels(BaseModel):
    allowed_nodes: list = []
    allowed_relationships: list = []

@app.post("/query")
def query_rag(request: QueryRequest):
    """Run a RAG query using Vector or Hybrid mode."""
    llm = llm_hybrid if request.hybrid else llm_vector
    query = request.query
    response = llm.query_llm(query)
    return {
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
def generate_graph(request: GraphAllowedNodesRels):
    try:
        args = ["python", "generate_graph.py"]

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
def generate_graph_pagewise(request: GraphIncludeTypes):
    try:
        llm_hybrid.include_types = request.include_types
        llm_hybrid.get_llms()
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}