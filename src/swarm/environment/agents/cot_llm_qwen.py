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
#from openai import OpenAI
import openai
import httpx

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

@AgentRegistry.register("COT_LLM_qwen")
class COT_LLM_qwen(Node): 
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
        #identifier = f"Execution from {self.node_name}"
        #print(f'{identifier} - Entering _execute')
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
        #print('==================text prompt==============:', text_prompt)
        
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:18081/v1"
        #client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = openai.ChatCompletion.create(
                        model="Qwen2-7B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text_prompt},
                        ],
                        temperature=self.temperature,
                        top_p=0.8,
                        max_tokens=1000,
                    )

                # response = client_gpt4.chat.completions.create(
                #     model="gpt-4o",
                #     messages=[
                #         #{"role": "system", "content": "You are a helpful assistant."},
                #         {"role": "user", "content": text_prompt}
                #     ],
                #     temperature=self.temperature,
                #     max_tokens=1000
                #     )
                #return response.choices[0].message.content
                
                raw_response = response
                content = response.choices[0].message['content']
                executions = {
                    "raw_response": raw_response,
                    "output": content,
                }
                
                print('---Qwen agent output..., temperature:', self.temperature)
                #print('Qwen agent answer:', content)
                #print('Qwen agent prompt:', text_prompt)
                #print('======================================')
                #print('-------qwen_node_output:', content)
                #print(f'{identifier} - Returning response:', executions)
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
                print('{identifier} - Error calling Qwen, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None


# COT_LLM_Qwen = COT_LLM_qwen(model_name='gpt-4o')

# text = 'who are you?'

# async def main():
#     response = await COT_LLM_Qwen._execute(text)
#     print('print1', response)

# asyncio.run(main())