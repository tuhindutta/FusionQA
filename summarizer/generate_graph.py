import argparse
import asyncio
import os
from graph.prepare import Graph
# from dotenv import load_dotenv
# load_dotenv()


parser = argparse.ArgumentParser(
        description="Generate knowledge graph"
    )
parser.add_argument(
        "--page-wise", action="store_true",
        help="Create vector chunks page wise"
    )
parser.add_argument("--allowed_nodes", nargs='+', default=[],
                    help="Allowed nodes")
parser.add_argument("--allowed_relationships", nargs='+', default=[],
                    help="Allowed relationships")


args = parser.parse_args()

async def main():

    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    api = os.getenv("GROQ_API_KEY")

    gph = Graph(url, username, password, api,
                args.allowed_nodes, args.allowed_relationships,
                "llama-3.3-70b-versatile", 'groq')

    await gph.prepare_graph(split_pagewise=args.page_wise)
    gph.get_id_label_mapping(save=True)


if __name__ == "__main__":
    asyncio.run(main())