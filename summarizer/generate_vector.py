import argparse
from llm.embedding import Documents, VectorStore


parser = argparse.ArgumentParser(
        description="Generate vector store"
    )
parser.add_argument(
        "--page-wise", action="store_true",
        help="Create vector chunks page wise"
    )

args = parser.parse_args()

documents = Documents()

if args.page_wise:
    documents.prepare_splitted_document_chunks_pagewise()
else:
    documents.prepare_splitted_document_chunks()

vs = VectorStore()
vs.add_documents(documents)
