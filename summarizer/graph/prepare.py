from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from tqdm import tqdm
from llm.embedding import Documents
from data_processing.loader import json_dumper
from graph.token_match import TokenMatch
from neo4j import GraphDatabase




class Graph:
    graph_id_label_map_file = "./graph_id_label_map.json"
    # relationship_file = "./relationship.json"

    def __init__(self, url, username, password, api_key,
                 allowed_nodes:list=None, allowed_relationships:list=None,
                 llm_model="o4-mini-2025-04-16", llm_api="openai"):
        self.url = url
        self.username = username
        self.password = password
        self.allowed_nodes = allowed_nodes or []
        self.allowed_relationships = allowed_relationships or []
        self.em = TokenMatch()
        self.graph = Neo4jGraph(
            url=url,
            username=username,
            password=password
        )
        llms = {"openai": ChatOpenAI, "groq": ChatGroq, "deepseek": ChatDeepSeek}
        self.llm = llms.get(llm_api)(
            api_key=api_key,
            model_name=llm_model,
            temperature=1
        )
        self.llm_transformer = LLMGraphTransformer(llm=self.llm,
                                                   allowed_nodes=self.allowed_nodes,
                                                   allowed_relationships=self.allowed_relationships)

    @staticmethod
    def combine_filename_document(filename, document):
        txt = f"The complete following document is for *Title: {filename}*\n{document}"
        return txt

    async def prepare_graph(self, split_pagewise=False):
        self.graph_documents = []
        doc = Documents()
        if split_pagewise:
            doc.prepare_splitted_document_chunks_pagewise()
        else:
            doc.prepare_splitted_document_chunks()
        for document in tqdm(doc.chunked_docs):
            filename = document.metadata.get('source')
            document.page_content = self.combine_filename_document(filename, document.page_content)
            # print(document.page_content)
        graph_doc = await self.llm_transformer.aconvert_to_graph_documents(doc.chunked_docs)
        self.graph.add_graph_documents(graph_doc)
        self.graph_documents.append(graph_doc)

    def get_id_label_mapping_from_db(self):
        auth = (self.username, self.password)
        driver = GraphDatabase.driver(self.url, auth=auth)
        query = """
        MATCH (n)
        WHERE coalesce(n.id, n.name, n.ID) IS NOT NULL
        RETURN DISTINCT coalesce(n.id, n.name, n.ID) AS node_id, labels(n) AS labels
        """
        mapping = []
        with driver.session() as session:
            results = session.run(query)
            for record in results:
                mapping.append({
                    "id": record["node_id"],
                    "labels": record["labels"]
                })
        return mapping

    def get_id_label_mapping(self, text:str):
        id_label_mapping = self.get_id_label_mapping_from_db()
        mapping = {i.get('id'):i.get('labels') for i in id_label_mapping}
        if len(self.allowed_nodes) > 0:
            mapping = {i:j for i,j in mapping.items() if any([node in j for node in self.allowed_nodes])}
        nodes_required = self.em.extract(text, list(mapping.keys()))
        mapping = {i:j for i,j in mapping.items() if i in nodes_required}
        mappings_edited = ""
        for key, value in mapping.items():
            for val in value:
                mappings_edited += f'{key} → ({val} {{name: "{key}"}})\n'
        mappings_edited = mappings_edited.strip()
        return mappings_edited
    
    def get_relationships(self):
        relationships = [f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})" for rel in self.graph.get_structured_schema['relationships']]
        if len(self.allowed_nodes) > 0:
            relationships = [i for i in relationships if any([node in i for node in self.allowed_nodes])]
        relationships = '\n'.join(relationships)        
        return relationships.strip()
    

        # if save:
        #     json_dumper(self.id_label_mapping, self.graph_id_label_map_file)

    # def get_relationships(self, save=False):
    #     query = """
    #     MATCH ()-[r]->()
    #     RETURN DISTINCT type(r) AS relationship
    #     """
    #     results = self.graph.query(query)
    #     relationships = [record["relationship"] for record in results]
    #     self.relationships = relationships
    #     if save:
    #         json_dumper(self.relationships, self.relationship_file)

