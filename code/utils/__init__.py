import os
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

openai.api_key = ""


def get_chatgpt_info(model_name: str, 
                     input_context: str, 
                     temperature: float, 
                     persona_info="You are an excellent question responder.") -> str:
    
    # print(persona_info)
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": persona_info,
            },
            {
                "role": "user",
                "content": input_context,
            },
        ],
        temperature=temperature,
    )
    
    return response["choices"][0]["message"]["content"]


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = json.load(fr)

    return data_pool


def read_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = [json.loads(line) for line in fr.readlines()]

    return data_pool


def write_json(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(fp=fw, obj=dataset, indent=4, ensure_ascii=False)


def jsonl2json(file):
    write_json(file, read_jsonl(file))
