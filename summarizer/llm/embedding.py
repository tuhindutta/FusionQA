import faiss
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from data_processing.loader import Loader, PdfFiles



class Documents:

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prepare()

    def prepare(self):
        docs = []
        for path in PdfFiles.pdfs:
            line_start_idx = 0
            ld = Loader(path)
            doc = ld.start_from_line(line_start_idx)
            filename = ld.filename
            docs.append([filename, doc])
        self.docs = docs

    def prepare_splitted_document_chunks(self):
        documents = [Document(page_content = document, metadata={"source": filename}) for filename, document in self.docs]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)
        self.chunked_docs = all_splits

    def prepare_splitted_document_chunks_pagewise(self):
        documents = [[filename, document.split("||PAGE_BREAK||")] for filename, document in self.docs]
        vec_documents = []
        for filename, pages in documents:
            for page_no, page in enumerate(pages, start=1):
                doc = Document(page_content = page, metadata={"source": filename,
                                                              "page_number": page_no})
                vec_documents.append(doc)
        self.chunked_docs = vec_documents


    
class VectorStore:

    def __init__(self, huggingface_embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.huggingface_embedding_model = huggingface_embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.huggingface_embedding_model)
        self.distance = 5

    @property
    def vector_store_loc(self):
        if len(os.listdir('./vector_db')) > 0:
            loc = './vector_db'
        else:
            loc = './vector_store'
        return loc

    def create_empty_store(self):
        embedding_dim = len(self.embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_documents(self, documents:Documents):
        self.create_empty_store()
        self.vector_store.add_documents(documents.chunked_docs)
        self.vector_store.save_local('./vector_db')

    def load(self):
        loading_from = self.vector_store_loc
        self.vector_store = FAISS.load_local(
            loading_from,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.using_sample_vector = loading_from == './vector_store'

    def similarity_search(self, query):
        results = self.vector_store.similarity_search_with_score(query)
        results = [doc.page_content for doc, score in results if score <= self.distance]
        return results