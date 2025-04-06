#from swarm.llm.format import Message
from swarm.graph import Node
from typing import List, Any, Optional
from swarm.environment.agents.agent_registry import AgentRegistry
import abc
import base64
import io
import os
import time
#import google.generativeai as genai
#from google.generativeai.types import generation_types
import numpy as np
from PIL import Image
import requests
import tiktoken
import asyncio
import openai
import httpx
# 中转
ERROR_CALLING_LLM = 'Error calling LLM'

def process_prompt(prompt):
    # 使用tiktoken的encoding_for_model函数来获取编码器
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    #encoder = tiktoken.get_encoding("gpt-3.5-turbo")

    # 对prompt进行tokenize
    tokens = encoding.encode(prompt)
    token_count = len(tokens)
    print('token_count:', token_count)
    
    if token_count > 14385:
        # 截取前16000个tokens
        #cut_num = 16000 - token_count
        cutted_token = tokens[:14000]
        cutted_text = encoding.decode(cutted_token)
        cutted_text_len = len(cutted_text)
        truncated_prompt = prompt[:cutted_text_len]
        print(f"Token count exceeded limit. Truncated to 16000 tokens.")
        return truncated_prompt
    else:
        return prompt

@AgentRegistry.register("COT_LLM_zz")
class COT_LLM_zz(Node): 
    def __init__(self, 
                 model_name: str,
                 max_retry: int = 3,
                 temperature: float = 0.0,
                 operation_description: str = "Make one step of CoT",
                 id=None
                ):
        super().__init__(operation_description, id, True)
        
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name
        self.RETRY_WAITING_SECONDS = 20
        #self.llm = LLMRegistry.get(model_name)

    @property
    def node_name(self):
        return self.__class__.__name__

    async def _execute(self, text_prompt: str) -> tuple[str, Any]:
        #print('---NodeZZ_______execute----')
        #print('---Current_node:', self.id)
        #print('---Current_node_temperature:', self.temperature)
        #print('---node_input:', text_prompt[-1000:])
        # headers = {
        # 'Content-Type': 'application/json',
        # 'Authorization': f'Bearer {self.openai_api_key}',}
        # #text_prompt = process_prompt(text_prompt)
        # payload = {
        #     'model': self.model,
        #     'temperature': self.temperature,
        #     'messages': [{
        #         'role': 'user',
        #         'content': [
        #             {'type': 'text', 'text': text_prompt},
        #         ],
        #     }],
        #     'max_tokens': 1000,
        # }

        # Gpt-4v supports multiple images, just need to insert them in the content
        # list.
        """ images = []
        for image in images:
            payload['messages'][0]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
                },
            }) """
        
        # client_gpt4 = OpenAI(base_url="https://gtfast.xiaoerchaoren.com:8937//v1", 
        #     #api_key=os.environ.get("OPENAI_API_KEY", r"sk-IuF0jpgmmFbZB7HrE5891a25822a48FdA3195484B2299c82"),
        #     api_key=os.environ.get("OPENAI_API_KEY", r"sk-mUFF0PuQ9RcIL5R8DbCe3e33B46046EeA4C32aF454Ab1495"),
        #     http_client=httpx.Client(
        #         base_url="https://gtfast.xiaoerchaoren.com:8937//v1",
        #         follow_redirects=True,
        #     ))
        openai.api_key  = "sk-UOArhyzuKw4Xaiga3e40F22502B44a6c93CaAaC336A3A1F1"
        openai.api_base = "http://15.204.101.64:4000/v1"
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a super-intelligent agent who can expertly operate an Android phone on behalf of a user."},
                        {"role": "user", "content": text_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000
                    )
                #return response.choices[0].message.content
                print('---GPT-4o agent output..., temperature:', self.temperature)
                #print('---Node prompt:', text_prompt)
                #response = []
                raw_response = response
                content = response.choices[0].message.content
                executions = {
                    "raw_response": raw_response,
                    "output": content,
                }
                #print('-------current_node_output:', content)
                return executions
                # else:
                #     print(
                #         'Error calling OpenAI API with error message: '
                #         + response.json()['error']['message']
                #     )
                #     return ERROR_CALLING_LLM, None
                #     time.sleep(wait_seconds)
                #     wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None


# COT_LLM_test = COT_LLM_zz(model_name='gpt-4o')

# text = 'how are you?'

# async def main():
#     response = await COT_LLM_test._execute(text)
    
#     print(response['output'])

# asyncio.run(main())