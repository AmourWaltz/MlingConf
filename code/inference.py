r"""
Author: XUE Boyang      Filename: inference.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Generate answers on QA val set in few-shot.
"""
import os
import time
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import operator
from functools import reduce

from tqdm import tqdm
import json
import random
from ipdb import set_trace

import numpy as np

import torch
import transformers
from transformers import set_seed, GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils import *


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

language_list = ["en", "zh", "ja", "fr", "th"]

@dataclass
class ModelArguments:
    model_name: str = field(default="gpt-3.5-turbo", metadata={"help": "Model name.", "choices": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "llama3", "vicuna", "gpt2"]})
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": ["triviaqa", "gsm8k", "common", "sciq"]})
    data_suffix: str = field(default="2k_1s", metadata={"help": "Data file suffix."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=True, metadata={"help": "Continue from the previous generations."})


@dataclass
class InferenceArguments:
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling or not."})
    output_dir: str = field(default="./exp/{}/inference", metadata={"help": "Directory to save results."})
    suffix: str = field(default="infer", metadata={"help": "File name to save the results."})
    num_sampling: int = field(default=5, metadata={"help": "Number of samples."})
    temperature: float = field(default=0.8, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Top p for sampling."})
    top_k: int = field(default=40, metadata={"help": "Top k for sampling."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for sampling."})
    max_length: int = field(default=16, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty."})
    num_examples: int = field(default=8, metadata={"help": "Number of examples to generate."})


@dataclass
class DeviceArguments:
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    seed: int = field(default=3407, metadata={"help": "Random seed."})
    gpu_num: int = field(default=1, metadata={"help": "Number of GPUs."})
    local_rank: int = field(default=0, metadata={"help": "Local rank."})
    global_rank: int = field(default=0, metadata={"help": "Global rank."})
    world_size: int = field(default=0, metadata={"help": "World size."})


# Parse arguments.
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments, DeviceArguments))
model_args, data_args, infer_args, device_args = parser.parse_args_into_dataclasses()


# Resize tokenizer and embedding.
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# Format the few-shot examplar of list to string.
def format_examplar(few_shot_examples, examplar_split):
    few_shot_examplar_list = {}
    # import pdb; pdb.set_trace()
    for language in language_list:
        few_shot_examplar_list[language] = []

    for few_shot_example in few_shot_examples:
        for language in language_list:
            # if data_args.dataset in ["triviaqa", "common", "sciq"]:
            few_shot_examplar_list[language].append("*** {} ***: {}\n*** {} ***: {}".format(examplar_split[language][0], few_shot_example["question"][language], 
                                                        examplar_split[language][1], few_shot_example["answer"][language]))
            # elif data_args.dataset == "gsm8k":
            #     few_shot_examplar_list[language].append("{}: {}\n{}: {}".format(examplar_split[language][0], few_shot_example["question"][language], 
            #                                                 examplar_split[language][1], few_shot_example["answer"][language]))
    for language in language_list:
        few_shot_examplar_list[language] = "\n\n".join(few_shot_examplar_list[language])

    # import pdb; pdb.set_trace()
    return few_shot_examplar_list


# Split the generation to get the answer part.
def output_split(output, tokenizer, split_len, prompt_split):
    logits = output.scores
    probs = [torch.softmax(log, dim=-1) for log in logits]

    generated_ids = output.sequences
    if data_args.dataset == "gsm8k":
        response = tokenizer.decode(generated_ids[0][split_len:], 
                                skip_special_tokens=True).split(prompt_split)[0].lstrip()
    else:
        response = tokenizer.decode(generated_ids[0][split_len:], 
                                skip_special_tokens=True).split(prompt_split)[0].replace("\n", "").lstrip()
    token_ids, token_probs = [], []
    # print(response)
    for i, token_id in enumerate(generated_ids[0][split_len:]):
        token_prob = probs[i][0, token_id].item()
        token = tokenizer.decode(token_id)
        # print(f"Token ID: {token_id}, Probability: {token_prob}, Token: {token}")
        if token == prompt_split:
            break
        token_ids.append(token_id)
        token_probs.append(token_prob)
        
    # import pdb; pdb.set_trace()
    norm_prob = float(np.array(pow(reduce(operator.mul, token_probs), 1/len(token_probs))))
    return response, norm_prob


def dataset_loader():
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), 
                                  "mling_{}.json".format(data_args.dataset))
    logging.info(f"Loading data from {data_path} ...")
    dataset = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, 
                                                  f"infer_temp.json")))
    instruction = prompt_template["instruction"]
    prompt_split = prompt_template["output_split"]
    few_shot_split = prompt_template["few_shot_split"]
    prompt_input = prompt_template["standard_prompt"]

    few_shot_examplar_list = format_examplar(dataset[:infer_args.num_examples], few_shot_split)
    dataset = dataset[infer_args.num_examples:]

    samples = []
    for data in dataset:
        sample = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"],
            "input": {}
        }
        for language in language_list:
            sample["input"][language] = prompt_input[language]. \
                format(instruction=instruction[language], 
                        examples=few_shot_examplar_list[language], 
                        question=sample["question"][language])

        samples.append(sample)

    # import pdb; pdb.set_trace()
    return samples, prompt_split


def llama_generate(dataset, prompt_split):
    # import pdb; pdb.set_trace()
    # Info: Device settings: random seed, using cuda or not, distributed setting.
    set_seed(device_args.seed)

    device_args.num_gpu = torch.cuda.device_count()
    device_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.float16,
        device_map="balanced" # device_map: "auto", "balanced", "balanced_low_0", "sequential"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Resize tokenizer and embedding.
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token == "":
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token == "":
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token == "":
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Sample the data.
    logging.info("Start generating ...")
    with tqdm(total=data_len) as t:
        for idx, batch in enumerate(dataset):            
            # import pdb; pdb.set_trace()
            # time.sleep(1)
            generations, norm_probs = {}, {}
            for lang in language_list:
                input_ids = tokenizer(batch["input"][lang], return_tensors="pt")["input_ids"].to(device_args.device)
                with torch.no_grad():
                    generation_config = GenerationConfig(
                                            do_sample=False,
                                            num_beams=infer_args.num_beams,
                                            repetition_penalty=infer_args.repetition_penalty)

                    generation = model.generate(input_ids,
                                            generation_config=generation_config,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_new_tokens=infer_args.max_length,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id, 
                                            bos_token_id=tokenizer.bos_token_id)
                    # import pdb; pdb.set_trace()
                    generation, norm_prob = output_split(generation, tokenizer, len(input_ids[0]), prompt_split)

                generations[lang] = generation
                norm_probs[lang] = norm_prob

            # print(data_point)
            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "output": generations,
                "probs": norm_probs
            }
            # print(instance)

            # Real-time saving the results.
            with open(infer_args.save_path, "a+") as fw: 
                instance_write = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(instance_write + '\n')

            t.set_postfix()
            t.update(1)


def gpt_generate(dataset):
    # Generate answer on input QA val set on multiple languages.
    logging.info("Start generating ...")
    with tqdm(total=len(dataset)) as t:
        # for data in data_pool[:3]:
        for idx, batch in enumerate(dataset):
            input_context = batch["input"]
            # print(input_context)
            # set_trace()
            generations, norm_probs = {}, {}
            for lang in language_list:
                generated_text, token_probs = get_chatgpt_info(model_args.model_name, input_context[lang], infer_args.temperature, max_tokens=infer_args.max_length, logprobs=False)
                # print(token_probs)
                norm_prob = float(np.array(pow(reduce(operator.mul, token_probs), 1/len(token_probs))))
                generations[lang] = generated_text
                norm_probs[lang] = norm_prob
            
            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "output": generations,
                "probs": norm_probs
            }

            with open(infer_args.save_path, mode="a+") as fw: 
                data_rec = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)


if __name__=="__main__":
    dataset, prompt_split = dataset_loader()

    # Set up logging.
    infer_args.output_dir = os.path.join(infer_args.output_dir.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{infer_args.suffix}")
    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    log_path = os.path.join(infer_args.output_dir, f"generate.log")
    # print(f"Log path: {infer_args.log_path}")
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Format the output file.
    # import pdb; pdb.set_trace()
    infer_args.save_path = os.path.join(infer_args.output_dir, "generate.json")
    if data_args.continue_generate:
        exist_num = len(read_jsonl(infer_args.save_path))
        # Split the dataset if needed.
        dataset = dataset[exist_num::]
    else:
        # dataset = dataset[:3]
        open(infer_args.save_path, "w").close()

    data_len = len(dataset)
    logging.info(f"The number of dataset: {data_len}")
    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\nInference Arguments: {infer_args}")

    start_time = time.time()
    if model_args.model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
        gpt_generate(dataset)
    elif model_args.model_name in ["llama3", "vicuna", "gpt2"]:
        llama_generate(dataset, prompt_split)

    # import pdb; pdb.set_trace()
    elapsed_time = format_seconds(time.time() - start_time)
    logging.info(f"Total elapsed time: {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s")

    # Convert jsonl to json format.
    logging.info("Generating is done.")
    jsonl2json(infer_args.save_path, infer_args.save_path)
    logging.info(f"Save to {infer_args.save_path}")
