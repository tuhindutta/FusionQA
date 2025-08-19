from langchain_neo4j import GraphCypherQAChain
from graph.prepare import Graph
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek


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


# def rename_ids(query: str, mappings: dict) -> str:
#     modified_query = query

#     for key, value in mappings.items():
#         # Escape special regex chars in key
#         key_pattern = re.escape(key)

#         # Replace whole word matches (case insensitive)
#         modified_query = re.sub(
#             rf"\b{key_pattern}\b",  # word boundary
#             value,
#             modified_query,
#             flags=re.IGNORECASE
#         )
#     return modified_query


class GraphLlm(Graph):

    def __init__(self, cypher_model_name:str, cypher_model_api_key:str, query_model_name:str, query_model_api_key:str,
                 url, username, password, cypher_model_api:str, query_model_api:str, 
                 allowed_nodes:list=None, allowed_relationships:list=None):
        super().__init__(url, username, password, cypher_model_api_key, allowed_nodes, allowed_relationships,
                        cypher_model_name, cypher_model_api)
        self.include_types = ((allowed_nodes or []) + (allowed_relationships or [])) or []
        llms = {"openai": ChatOpenAI, "groq": ChatGroq, "deepseek": ChatDeepSeek}
        self.query_llm = llms.get(query_model_api)(
            api_key=query_model_api_key,
            model_name=query_model_name,
            temperature=0.3
        )
        self.create_chain()        

    def create_chain(self):
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.llm,
            qa_llm=self.query_llm,
            include_types=self.include_types,
            validate_cypher=True,
            top_k=100,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=PromptTemplate.from_template(CYPHER_PROMPT)
        )

    def query(self, query:str):
        mappings = self.get_id_label_mapping(query)
        relationships = self.get_relationships()
        prompt = {"node_mappings":mappings, "relationships":relationships, "query": query}
        print(prompt)
        res = self.chain.invoke(prompt)
        return res