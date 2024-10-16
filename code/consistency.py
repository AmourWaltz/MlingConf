r"""
Author: XUE Boyang      Filename: consistency.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Check the consistency of the translation results.
"""
import os
import re
import json
import random
import argparse
import logging

from tqdm import tqdm, trange
from ipdb import set_trace

import utils
from utils import read_jsonl, read_json, write_json, write_jsonl, jsonl2json, lang_map, lang_true_false_dict

import six
from google.cloud import translate_v2 as translate


parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, default="check", choices=["translate", "filter", "refilter", "check"])
parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"])
parser.add_argument('--dataset', type=str, default="common", choices=["trivia", "gsm8k", "common", "sciq"])
parser.add_argument('--prompt_path', type=str, default="./prompt/")
parser.add_argument('--inp_path', type=str, default="./data/{}/")
parser.add_argument('--out_path', type=str, default="./data/{}/")
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--continue_generation', type=bool, default=True)

args = parser.parse_args()


def check_translation_results(model_name, data_pool, dataset):  
    language_list = ["English", "Chinese", "Japanese", "French", "Thai"]
    # language_list = ["English", "Chinese", "Japanese"]

    pos, neg = lang_true_false_dict["English"]

    # Load prompt template
    template_file = os.path.join(args.prompt_path, "check_temp.json")
    template = json.load(open(template_file))

    out_file = os.path.join(args.out_path.format(dataset), f"mling_{dataset}_check.json")
    if args.continue_generation:
        if os.path.exists(out_file):
            data_pool = data_pool[len(read_jsonl(out_file)):]
        else:
            open(out_file, "w").close()
    else:
        open(out_file, "w").close()

    logging.basicConfig(
        filename=out_file.replace(".json", ".log"),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    instances = []
    with tqdm(total=len(data_pool)) as t:
        for data in data_pool:
            instance = {
                "question_id": data["question_id"],
                "question": {},
                "answer": {},
                "check_results": {},
                "consistency": {}
                }
            
            for idx in range(len(language_list)):
                instance["consistency"][language_list[idx]] = 0

            for id_lang, language_1 in enumerate(language_list):
                for language_2 in language_list[id_lang+1:]:
                    
                    persona_info = template["persona_info"].format(language_1, language_2)
                    text_1 = f"{data[lang_map[language_1]]['question']}\n{data[lang_map[language_1]]['answer']}"
                    text_2 = f"{data[lang_map[language_2]]['question']}\n{data[lang_map[language_2]]['answer']}"
                    input_context = template["prompt_input"].format(text_1, text_2)
                    check_result = utils.get_chatgpt_info(model_name, input_context, temperature=0.0, 
                                                            persona_info=persona_info)
                    instance["check_results"][f"{language_1}_{language_2}"] = check_result

                    check_result = check_result.replace(".", "")
                    if neg.lower() in check_result.lower():
                        instance["consistency"][language_1] += 1
                        instance["consistency"][language_2] += 1
                    elif pos.lower() in check_result.lower():
                        pass
                    else:
                        logging.info("Sample {}: {}_{} fails ......".format(data["question_id"], language_1, language_2))
            
                instance["question"][lang_map[language_1]] = data[lang_map[language_1]]["question"]
                instance["answer"][lang_map[language_1]] = data[lang_map[language_1]]["answer"]

            instances.append(instance)

            with open(out_file, "a+") as fw:
                ins_rec = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(ins_rec + '\n')

            t.set_postfix()
            t.update(1)
            
            # for key in ["question", "answer"]:
            #     check_results = []
            #     for id_lang, language_1 in enumerate(language_list):
            #         for language_2 in language_list[id_lang+1:]:
                        
            #             persona_info = template["persona_info"].format(language_1, language_2)
            #             input_context = template["prompt_input"].format(data[para.lang_map[language_1]][key], data[para.lang_map[language_2]][key])
            #             check_result = utils.get_chatgpt_info(model_name, input_context, temperature=0.0, 
            #                                                     persona_info=persona_info)
            #             check_results.append(f"{language_1} and {language_2}: {check_result}")
                
            #         instance[key][para.lang_map[language_1]] = data[para.lang_map[language_1]][key]
            #         # print(language_1)
            #     instance["check_results"][key] = check_results

            # instances.append(instance)

    jsonl2json(out_file, out_file)

    return instances


def check_consistency(data_pool, dataset):
    out_file = os.path.join(args.out_path.format(dataset), f"mling_{dataset}_check_consistency.json")
    # print(out_file)
    logging.basicConfig(
        filename=out_file.replace(".json", ".log"),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    language_list = ["English", "Chinese", "Japanese", "French", "Thai"]
    num_wrong = 0
    samples = []
    for data in data_pool:
        if max(data["consistency"].values()) >= 2:
            num_wrong += 1
            logging.info(f"Sample {data['question_id']} fails the consistency check.")
            for language in language_list:
                logging.info(f"{language}: {data['consistency'][language]}")
        else:
            samples.append({
                "question_id": data["question_id"],
                "question": data["question"],
                "answer": data["answer"],
                "consistency": data["consistency"]
            })
    print(f"Total number of wrong samples: {num_wrong}.")
    print(f"Total number of correct samples: {len(samples)}.")
    write_json(out_file, samples)


if __name__=="__main__":
    if args.stage == "consistency":
        # datasets = ["trivia", "gsm8k", "common", "sciq"]
        # for dataset in datasets:
        dataset = args.dataset
        print(f"Check the translation results for {dataset}.")
        inp_file = os.path.join(args.out_path.format(dataset), "mling_{}.json".format(dataset))
        data_pool = read_json(inp_file)
        check_translation_results(args.model_name, data_pool, dataset)
    elif args.stage == "check":
        datasets = ["trivia", "gsm8k", "common", "sciq"]
        for dataset in datasets:
            print(f"Check the translation results for {dataset}.")
            inp_file = os.path.join(args.out_path.format(dataset), "mling_{}_check.json".format(dataset))
            data_pool = read_json(inp_file)
            check_consistency(data_pool, dataset)


    