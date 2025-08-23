from langchain_neo4j import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from graph.prepare import Graph


CYPHER_PROMPT = """
You are an expert at generating Neo4j Cypher queries.

The following IDs belong to these labels:
{node_mappings}

The following is the relationship schema:
{relationships}

Important rules:
- Only query existing nodes and relationships.
- Never use CREATE, MERGE, DELETE, or SET.
- Always assume the entity exists in the graph.
- Use case-insensitive matching when comparing names.
- While generating cypher queries, use the property '.id' instead of '.name'.

Now, based on the above node mappings and relationships schema, generate the correct Cypher query for the following and don not return blank:
{query}
"""


class GraphLlm(Graph):

    def __init__(self, cypher_model_name:str, cypher_model_api_key:str, query_model_name:str, query_model_api_key:str,
                 url, username, password, cypher_model_api:str, query_model_api:str, 
                 allowed_nodes:list=None, allowed_relationships:list=None):
        super().__init__(url, username, password, cypher_model_api_key,
                         allowed_nodes, allowed_relationships,
                        cypher_model_name, cypher_model_api)
        self.include_types = ((allowed_nodes or []) + (allowed_relationships or [])) or []
        llms = {"openai": ChatOpenAI, "groq": ChatGroq, "deepseek": ChatDeepSeek}
        self.QAllm = llms.get(query_model_api)(
            api_key=query_model_api_key,
            model_name=query_model_name,
            temperature=0.3
        )
        self.create_chain()        

    def create_chain(self):
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.llm,
            qa_llm=self.QAllm,
            include_types=self.include_types,
            validate_cypher=True,
            top_k=100,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=PromptTemplate.from_template(CYPHER_PROMPT)
        )

    def query_llm(self, query:str):
        mappings = self.get_id_label_mapping(query)
        relationships = self.get_relationships()
        prompt = {"allowed_nodes":self.allowed_nodes, "node_mappings":mappings,
                  "allowed_relationships":self.allowed_relationships,
                  "relationships":relationships, "query": query}
        res = self.chain.invoke(prompt)
        return res