from cot_llm_zz import COT_LLM_zz
from cot_llm_qwen import COT_LLM_qwen
import asyncio

# 创建 COT_LLM_zz 实例
# COT_LLM_Zz = COT_LLM_zz(model_name='gpt-4o')
text = 'who are you?'

# async def main_zz():
#     response = await COT_LLM_Zz._execute(text)
#     print('print1', response)

# asyncio.run(main_zz())

# 创建 COT_LLM_qwen 实例
COT_LLM_Qwen = COT_LLM_qwen(model_name='gpt-4o')

async def main_qwen():
    response = await COT_LLM_Qwen._execute(text)
    print('print2', response)

asyncio.run(main_qwen())
