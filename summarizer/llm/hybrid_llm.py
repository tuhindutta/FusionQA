from llm.embedding import VectorStore
from llm.vector_llm import VectorLlm
from llm.graph_llm import GraphLlm


class HybridLlm:

    def __init__(self, cypher_model_name:str, cypher_model_api_key:str,
                 query_model_name:str, query_model_api_key:str,
                 vector_store:VectorStore,
                 neo4j_url, neo4j_username, neo4j_password, system_role_prompt:str=None,
                 cypher_model_api:str="groq", query_model_api:str="groq",
                 allowed_nodes:list=None, allowed_relationships:list=None,
                 history_tracking=False):
        
        
        self.cypher_model_name = cypher_model_name
        self.cypher_model_api_key = cypher_model_api_key
        self.query_model_name = query_model_name
        self.query_model_api_key = query_model_api_key
        self.vector_store = vector_store
        self.system_role_prompt = system_role_prompt
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.cypher_model_api = cypher_model_api
        self.query_model_api = query_model_api
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.history_tracking = history_tracking
        self.llms_loaded = False
        
        
    def get_llms(self):
        self.vector_llm = VectorLlm(self.query_model_name, self.query_model_api_key,
                                    self.vector_store, self.system_role_prompt,
                                    self.history_tracking)
        self.graph_llm = GraphLlm(self.cypher_model_name, self.cypher_model_api_key,
                                  self.query_model_name, self.query_model_api_key,
                                  self.neo4j_url, self.neo4j_username, self.neo4j_password,
                                  self.cypher_model_api, self.query_model_api,
                                  self.allowed_nodes, self.allowed_relationships)
        self.llms_loaded = True

    def query_llm(self, query:str):
        graph_response = self.graph_llm.query_llm(query)
        graph_context = graph_response.get('result')
        prompt = f"""Knowledge Graph Context: {graph_context}
Use the above KG context also for the query:\n{query}"""
        response = {"graph_response": graph_response,
                    "response": self.vector_llm.query_llm(prompt)}
        return response