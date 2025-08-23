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
parser.add_argument("--chunk-size", type=int, default=1000,
                    help="Chunk size for splitting text.")
parser.add_argument("--chunk-overlap", type=int, default=200,
                    help="Chunk overlap size.",)


args = parser.parse_args()

async def main():

    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    api = os.getenv("GROQ_API_KEY")

    gph = Graph(url, username, password, api,
                args.allowed_nodes, args.allowed_relationships,
                "llama-3.3-70b-versatile", 'groq')

    await gph.prepare_graph(args.page_wise, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    asyncio.run(main())