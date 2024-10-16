r"""
Author: XUE Boyang      Filename: confidence.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Confidence estimation script for the MlingConf project.
"""
import os
import time
import copy
import re
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
    model_name: str = field(default="gpt-3.5-turbo", metadata={"help": "Model name.", "choices": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "llama3", "vicuna", "gpt2", "llama2", "vicuna2"]})
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./exp/{}/inference", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": ["triviaqa", "gsm8k", "common", "sciq"]})
    data_suffix: str = field(default="infer", metadata={"help": "Data file suffix."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=True, metadata={"help": "Continue from the previous generations."})


@dataclass
class InferenceArguments:
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling or not."})
    output_dir: str = field(default="./exp/{}/confidence", metadata={"help": "Directory to save results."})
    suffix: str = field(default="confidence", metadata={"help": "File name to save the results."})
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
def ptrue_output_split(output, tokenizer, split_len, language):
    logits = output.scores
    probs = [torch.softmax(log, dim=-1) for log in logits]

    # if language == "en":
    #     import pdb; pdb.set_trace()

    generated_ids = output.sequences
    # print(generated_ids)
    # response = tokenizer.decode(generated_ids[0][split_len:], 
    #                         skip_special_tokens=True).split()[0].replace("\n", "").lstrip()
    response = tokenizer.decode(generated_ids[0][split_len:], 
                            skip_special_tokens=True)
    print(response)
    token_ids, token_probs, tokens = [], [], []
    # print(response)
    # import pdb; pdb.set_trace()
    for i, token_id in enumerate(generated_ids[0][split_len:]):
        token_prob = probs[i][0, token_id].item()
        if i == 0:
            first_prob = token_prob
        token = tokenizer.decode(token_id)
        if token.strip().lower() == lang_aware_dict[language][0].lower() or token.strip().lower() == lang_aware_dict[language][1].lower():
            norm_prob = token_prob
            if token.lower() == lang_aware_dict[language][1].lower():
                norm_prob = 1 - norm_prob
            # print(language)
            # print(response)
            # print(token)
            return token, norm_prob

        if token.strip().lower() in lang_aware_dict[language][0].lower() or token.strip().lower() in lang_aware_dict[language][1].lower():
            token_probs.append(token_prob)
            tokens.append(token)
            token_ids.append(token_id)
        else:
            # print(f"Token ID: {token_id}, Probability: {token_prob}, Token: {token}")
            token_ids, token_probs, tokens = [], [], []
            continue
        
        generated_token = "".join(tokens).strip()
        if generated_token.lower() == lang_aware_dict[language][0].lower() or generated_token.lower() == lang_aware_dict[language][1].lower():
            break
    
    # print(response)
    # print(generated_token)
    # import pdb; pdb.set_trace()
    if token_probs:
        norm_prob = float(np.array(pow(reduce(operator.mul, token_probs), 1/len(token_probs))))
        if generated_token.lower() == lang_aware_dict[language][1].lower():
            norm_prob = 1 - norm_prob
    else:
        norm_prob = 0.5
        generated_token = None

    return generated_token, norm_prob


ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
INVALID_ANS = 0.5
def verb_output_split(output, tokenizer, split_len, language):
    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS
    generated_ids = output.sequences
    # print(generated_ids)
    # response = tokenizer.decode(generated_ids[0][split_len:], 
    #                         skip_special_tokens=True).split()[0].replace("\n", "").lstrip()
    response = tokenizer.decode(generated_ids[0][split_len:], 
                            skip_special_tokens=True).split("\n")[0].replace("#", "").lstrip()
    response = extract_answer(response)
    # print(response)
    # import pdb; pdb.set_trace()
    
    return response


def dataset_loader():
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), 
                                  "{}_{}_infer/generate.json".format(model_args.model_name, data_args.dataset))
    logging.info(f"Loading data from {data_path} ...")
    dataset = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, 
                                                  f"confidence_temp.json")))
    ptrue_instruction = prompt_template["ptrue_prompt"]["instruction"]
    ptrue_prompt_input = prompt_template["ptrue_prompt"]["standard_prompt"]

    verb_instruction = prompt_template["verbal_prompt"]["instruction"]
    verb_prompt_input = prompt_template["verbal_prompt"]["standard_prompt"]

    samples = []
    for data in dataset:
        sample = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"],
            "output": data["output"],
            "probs": data["probs"],
            "ptrue_input": {},
            "verb_input": {}
        }
        for language in language_list:
            sample["ptrue_input"][language] = ptrue_prompt_input["en"]. \
                format(instruction=ptrue_instruction["en"], 
                        question=sample["question"][language],
                        answer=sample["output"][language])
            
            sample["verb_input"][language] = verb_prompt_input. \
                format(instruction=verb_instruction, 
                        question=sample["question"][language],
                        answer=sample["output"][language])
            
        samples.append(sample)

    # import pdb; pdb.set_trace()
    return samples


def llama_conf_generate(dataset):
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
            ptrue_generations, verb_generations = {}, {}
            for lang in language_list:
                ptrue_input_ids = tokenizer(batch["ptrue_input"][lang], return_tensors="pt")["input_ids"].to(device_args.device)
                verb_input_ids = tokenizer(batch["verb_input"][lang], return_tensors="pt")["input_ids"].to(device_args.device)
                with torch.no_grad():
                    generation_config = GenerationConfig(
                                            do_sample=False,
                                            num_beams=infer_args.num_beams,
                                            repetition_penalty=infer_args.repetition_penalty)

                    ptrue_generation = model.generate(ptrue_input_ids,
                                            generation_config=generation_config,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_new_tokens=infer_args.max_length,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id, 
                                            bos_token_id=tokenizer.bos_token_id)
                    
                    verb_generation = model.generate(verb_input_ids,
                                            generation_config=generation_config,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_new_tokens=12,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id, 
                                            bos_token_id=tokenizer.bos_token_id)
                    # import pdb; pdb.set_trace()
                    ptrue_generation, ptrue_norm_prob = ptrue_output_split(ptrue_generation, tokenizer, len(ptrue_input_ids[0]), language="en")
                    verb_generation = verb_output_split(verb_generation, tokenizer, len(verb_input_ids[0]), language=lang)

                ptrue_generations[lang] = {
                    "conf": ptrue_norm_prob,
                    "judge": ptrue_generation
                }
                
                verb_generations[lang] = verb_generation

            # print(data_point)
            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "output": batch["output"],
                "probs": batch["probs"],
                "ptrue": ptrue_generations,
                "verb": verb_generations
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
        # for data in dataset[:3]:
        for batch in dataset:
            # print(input_context)
            # set_trace()
            ptrue_generations, verb_generations = {}, {}
            for lang in language_list:
                ptrue_input = batch["ptrue_input"][lang]
                verb_input = batch["verb_input"][lang]
                ptrue_generation, ptrue_probs = get_chatgpt_info(model_args.model_name, ptrue_input, infer_args.temperature, max_tokens=infer_args.max_length, logprobs=False)
                verb_generation, _ = get_chatgpt_info(model_args.model_name, verb_input, infer_args.temperature, max_tokens=infer_args.max_length, logprobs=False)

                if ptrue_generation.lower() == "true":
                    ptrue_prob = ptrue_probs[0]
                elif ptrue_generation.lower() == "false":
                    ptrue_prob = 1 - ptrue_probs[0]
                else:
                    ptrue_prob = 0.5
                  
                ptrue_generations[lang] = {
                    "conf": ptrue_prob,
                    "judge": ptrue_generation
                }
                verb_generations[lang] = float(verb_generation)

            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "output": batch["output"],
                "probs": batch["probs"],
                "ptrue": ptrue_generations,
                "verb": verb_generations
            }

            # Real-time saving the results.
            with open(infer_args.save_path, "a+") as fw: 
                instance_write = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(instance_write + '\n')

            t.set_postfix()
            t.update(1)


if __name__=="__main__":
    dataset = dataset_loader()

    # Set up logging.
    infer_args.output_dir = os.path.join(infer_args.output_dir.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{infer_args.suffix}")
    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    log_path = os.path.join(infer_args.output_dir, f"confidence.log")
    # print(f"Log path: {infer_args.log_path}")
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Format the output file.
    # import pdb; pdb.set_trace()
    infer_args.save_path = os.path.join(infer_args.output_dir, "confidence.json")
    if data_args.continue_generate:
        exist_num = len(read_jsonl(infer_args.save_path))
        # Split the dataset if needed.
        dataset = dataset[exist_num+1::]
    else:
        # dataset = dataset[:3]
        open(infer_args.save_path, "w").close()

    data_len = len(dataset)
    logging.info(f"The number of dataset: {data_len}")
    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\nInference Arguments: {infer_args}")

    start_time = time.time()
    if model_args.model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
        gpt_generate(dataset)
    elif model_args.model_name in ["llama3", "vicuna", "gpt2", "llama2", "vicuna2"]:
        llama_conf_generate(dataset)

    # import pdb; pdb.set_trace()
    elapsed_time = format_seconds(time.time() - start_time)
    logging.info(f"Total elapsed time: {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s")

    # Convert jsonl to json format.
    logging.info("Generating is done.")
    jsonl2json(infer_args.save_path, infer_args.save_path)
    logging.info(f"Save to {infer_args.save_path}")
