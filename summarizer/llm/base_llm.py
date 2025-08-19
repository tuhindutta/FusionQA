import datetime
import requests
from llm.embedding import VectorStore


class LLM:

    def __init__(self, model_name:str, GROQ_API_KEY:str, system_prompt:str,
                 vector_store:VectorStore=None, history_tracking=False):
        self.model_name = model_name    # "llama-3.3-70b-versatile"
        self.__vector_store = vector_store
        self.system_prompt = system_prompt
        self.history_tracking = history_tracking
        self.temperature = 0.3
        self.first_call = True
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")
        self.__api_key = GROQ_API_KEY
        self.chat_history = []

    def build_prompt(self, user_input, context, formatted_chat_history):
        today_date = str(datetime.datetime.now().strftime('%B %d, %Y'))
        if self.first_call or not (not self.first_call and self.history_tracking):
            system_prompt = self.system_prompt + '\n'
        else:
            system_prompt = ""
        self.first_system_prompt = system_prompt

        system_prompt += f"Today's date: {today_date}\nUse the following context to answer queries:\n{context}" if self.__vector_store else ""
        # print(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"chat history: {formatted_chat_history}\n\nuser input: {user_input}"})
        return messages

    def query_llm(self, user_input):
        url = "https://api.groq.com/openai/v1/chat/completions"

        formatted_chat_history = '\n\n'.join(self.chat_history) if self.history_tracking else ''
        if isinstance(self.__vector_store, VectorStore):
            context = "\n".join(self.__vector_store.similarity_search(user_input))
        else:
            context = ""
        messages = self.build_prompt(user_input, context, formatted_chat_history)
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json"
        }       
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            # "max_tokens": 800          
        }
        response = requests.post(url, headers=headers, json=payload)
        output = response.json()['choices'][0]['message']['content'].strip()
        # output = 'test'
        self.first_call = False
        if self.history_tracking:
            self.chat_history.append(f"query: {self.first_system_prompt}{user_input}\nresponse: {output}")
        return output
    





# class LLM:

#     def __init__(self, GROQ_API_KEY:str, vector_store:VectorStore):
#         self.__vector_store = vector_store
#         if not GROQ_API_KEY:
#             raise ValueError("GROQ_API_KEY is not set")
#         self.__api_key = GROQ_API_KEY
#         self.chat_history = []

#     def build_prompt(self, user_input, context, formatted_chat_history):
#         today_date = str(datetime.datetime.now().strftime('%B %d, %Y'))
#         # formatted_chat_history = formatted_chat_history+'\n\n' if formatted_chat_history!='' else formatted_chat_history
#         system_prompt = f"""You are a banking expert with knowledge about regulatory compliances about different entities.
# Answer user questions clearly, concisely, and professionally explaining in simple terms assuming the user does not have a vivid knowledge of BFSI domain.
# Do not exaggerate or fabricate any information. If the context does not provide sufficient information to answer a question, respond with
# "I don't have enough information to answer that.". You may answer unrelated questions only if you are confident in your response.
# Date: {today_date}
# Use the following context to answer queries:
# {context}
# """
#         messages = [{"role": "system", "content": system_prompt}]
#         messages.append({"role": "user", "content": f"chat history: {formatted_chat_history}\n\nuser input: {user_input}"})
#         return messages

#     def query_llm(self, user_input):
#         url = "https://api.groq.com/openai/v1/chat/completions"
#         formatted_chat_history = '\n\n'.join(self.chat_history)
#         context = "\n".join(self.__vector_store.similarity_search(user_input))
#         messages = self.build_prompt(user_input, context, formatted_chat_history)
#         headers = {
#             "Authorization": f"Bearer {self.__api_key}",
#             "Content-Type": "application/json"
#         }       
#         payload = {
#             "model": "llama-3.3-70b-versatile",
#             "messages": messages,
#             "temperature": 0.3,
#             # "max_tokens": 800          
#         }
#         response = requests.post(url, headers=headers, json=payload)
#         output = response.json()['choices'][0]['message']['content'].strip()
#         self.chat_history.append(f"query: {user_input}\nresponse: {output}")
#         return output