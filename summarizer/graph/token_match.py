import ast
import os
from llm.base_llm import LLM
# from dotenv import load_dotenv
# load_dotenv()


class TokenMatch:

    def __init__(self):
        prompt = """
You are a language expert.
From the given text, extract all and any possible matching keywords present in the provided keywords list and return in a python list and nothing else.
"""
        api = os.getenv("GROQ_API_KEY")
        self.extractor_llm = LLM("llama-3.1-8b-instant", api, prompt)

    def extract(self, txt:str, *keywords:str):
        prompt = f"text: {txt}\nkeywords:{keywords}"
        response = self.extractor_llm.query_llm(prompt)
        response = ast.literal_eval(response)
        return response