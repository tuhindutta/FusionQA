from llm.embedding import VectorStore
from llm.base_llm import LLM


class VectorLlm(LLM):

    def __init__(self, model_name:str, GROQ_API_KEY:str, vector_store:VectorStore,
                 system_role_prompt:str=None, history_tracking=False):
        self.system_role_prompt = system_role_prompt
#         system_prompt = """You are a banking expert with knowledge about regulatory compliances about different entities.
# Answer user questions clearly, concisely, and professionally explaining in simple terms assuming the user does not have a vivid knowledge of BFSI domain.
# Do not exaggerate or fabricate any information. If the context does not provide sufficient information to answer a question, respond with
# "I don't have enough information to answer that.". You may answer unrelated questions only if you are confident in your response.
# """
        system_prompt = f"""{system_role_prompt}
Answer user questions clearly, concisely, and professionally explaining in simple terms assuming the user does not have a vivid knowledge of BFSI domain.
Do not exaggerate or fabricate any information. If the context does not provide sufficient information to answer a question, respond with
"I don't have enough information to answer that.". You may answer unrelated questions only if you are confident in your response.
"""
        super().__init__(model_name, GROQ_API_KEY, system_prompt, vector_store, history_tracking)