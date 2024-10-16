r"""
Author: XUE Boyang      Filename: translate.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Translate input QA into multilanguage.
"""
import os
import re
import json
import random
import argparse

from tqdm import tqdm, trange
from ipdb import set_trace

import utils
from utils import read_jsonl, read_json, write_json, write_jsonl, lang_map

import six


parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, default="refilter", choices=["translate", "filter", "refilter", "check"])
parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"])
parser.add_argument('--dataset', type=str, default="common", choices=["trivia", "gsm8k", "common", "sciq"])
parser.add_argument('--prompt_path', type=str, default="./prompt/")
parser.add_argument('--inp_path', type=str, default="./data/{}/")
parser.add_argument('--out_path', type=str, default="./data/{}/")
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--continue_generation', type=bool, default=True)

args = parser.parse_args()


def sampling(k):
    inp_file = os.path.join(args.inp_path, "val.json")
    out_file= os.path.join(args.out_path, "val_2k.json")

    data_pool = utils.read_json(inp_file)
    data_sample = random.sample(data_pool, k)
    utils.write_json(out_file, data_sample)


def translate_text(model_name, temperature, language, filename="val", dataset="common"):
    # Read QA dataset.
    inp_file = os.path.join(args.inp_path.format(dataset), "{}_en.json".format(filename))
    data_pool = utils.read_json(inp_file)
    # data_pool = read_json(inp_file)[:3]

    # Construct output file.
    out_file= os.path.join(args.out_path.format(dataset), "{}_{}.json".format(filename, lang_map[language]))
    if args.continue_generation and os.path.exists(out_file):
        data_pool = data_pool[len(read_jsonl(out_file)):]
    else:
        open(out_file, "w").close()

    # Load prompt template
    template_file = os.path.join(args.prompt_path, "trans_temp.json")
    template = json.load(open(template_file))    

    # set_trace()
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool[:3]:
        for data in data_pool:
            ins = dict()

            # set_trace()
            for key in ["question", "answer"]:
                text = data[key]
                input_context = template["prompt_input"].format(language, text)
                generated_text = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                                  persona_info=template["persona_info"])
                ins[key] = generated_text.split(template["response_split"])[-1].strip()

            with open(out_file, "a+") as fw: 
                ins_rec = json.dumps(obj=ins, ensure_ascii=False)
                fw.write(ins_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)


def travia_few_shot_examples(model_name, temperature):
    inp_file = os.path.join(args.inp_path, "val.json")
    val_file = os.path.join(args.out_path, "val_2k.json")
    template_file = os.path.join(args.prompt_path, "trans_temp.json")

    # Sample the few-shot examples.
    # Read QA dataset.
    data_pool = utils.read_json(inp_file)
    val_pool = utils.read_json(val_file)
    ques_ids = [data["question_id"] for data in val_pool]

    samps = []
    # set_trace()
    while len(samps) < 5:
        samp = random.sample(data_pool, 1)[0]
        if samp["question_id"] in ques_ids:
            continue
        samps.append(samp)

    # set_trace()
    # Warrant the duplicated questions are removed
    val_pool = val_pool + samps
    key_ids = [data["question_id"] for data in val_pool]
    assert len(list(set(key_ids))) == len(key_ids)

    # set_trace()
    template = json.load(open(template_file))

    # set_trace()
    # for language in ["English"]:
    for language in list(lang_map.keys())[2:]:
        out_file= os.path.join(args.out_path, "val_2k_{}.json".format(lang_map[language]))
        if language != "English":
            with tqdm(total=len(samps)) as t:
                for data in samps:
                    ins = dict()
                    ins["question_id"] = data["question_id"]

                    # set_trace()
                    for key in ["question", "answer"]:
                        text = data[key]
                        input_context = template["prompt_input"].format(language, text)
                        generated_text = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                                        persona_info=template["persona_info"])
                        ins[key] = generated_text

                    with open(out_file, "a+") as fw: 
                        ins_rec = json.dumps(obj=ins, ensure_ascii=False)
                        fw.write(ins_rec + '\n')

                    t.set_postfix()
                    t.update(1)
        else:
            for data in val_pool:
                with open(out_file, "a+") as fw: 
                    ins_rec = json.dumps(obj=data, ensure_ascii=False)
                    fw.write(ins_rec + '\n')


def combine_and_filter(temperature, dataset):
    lang_list = ["English", "Chinese", "Japanese", "French", "Thai"]
    data_pool = {}
    for lang in lang_list:
        if dataset == "trivia":
            inp_file = os.path.join(args.out_path.format(dataset), "val_2k_{}.json".format(lang_map[lang]))
            data_pool[lang_map[lang]] = read_jsonl(inp_file)
        else:
            inp_file = os.path.join(args.out_path.format(dataset), "val_{}.json".format(lang_map[lang]))
            data_pool[lang_map[lang]] = read_json(inp_file)

    out_file = os.path.join(args.out_path.format(dataset), f"mling_{dataset}.json")
    # Load prompt template
    template_file = os.path.join(args.prompt_path, "trans_temp.json")
    template = json.load(open(template_file))    

    assert len(data_pool["en"]) == len(data_pool["zh"]) == len(data_pool["ja"]) == len(data_pool["fr"]) == len(data_pool["th"])
    ins_set = []
    with tqdm(total=len(data_pool["en"])) as t:
        for idx in range(len(data_pool["en"])):
            ins = {}
            ins["question_id"] = str(idx+1)
            for language in lang_list:
                lang = lang_map[language]
                ins[lang] = {}
                # if idx == 15:
                #     set_trace()
                if lang == "en":
                    ins[lang] = {
                        "question": data_pool[lang][idx]["question"],
                        "answer": data_pool[lang][idx]["answer"]
                    }
                else:
                    for key in ["question", "answer"]:
                        text = data_pool[lang][idx][key].split(template["response_split"])[-1].split(template["input_split"])[-1]. \
                                                        split("### 输出 ###:")[-1].split("### 出力 ###:")[-1].strip()
                        if text == "":
                            # import pdb; pdb.set_trace()
                            input_context = template["prompt_input"].format(language, data_pool["en"][idx][key])
                            generated_text = utils.get_chatgpt_info("gpt-4", input_context, temperature, 
                                                            persona_info=template["persona_info"])
                            text = generated_text
                        ins[lang][key] = text
                
            ins_set.append(ins)

            t.set_postfix()
            t.update(1)

    write_json(out_file, ins_set)


def recheck_and_filter(dataset):
    data_file = os.path.join(args.out_path.format(dataset), f"mling_{dataset}.json")
    save_file = os.path.join(args.out_path.format(dataset), f"mling_{dataset}_refilter.json")
    data_pool = read_json(data_file)

    # Load prompt template
    template_file = os.path.join(args.prompt_path, "trans_temp.json")
    template = json.load(open(template_file))    

    with tqdm(total=len(data_pool)) as t:
        for idx, data in enumerate(data_pool):
            for language in ["Chinese", "Japanese", "French", "Thai"]:
                for key in ["question", "answer"]:
                    separators = [template["input_split"], template["response_split"], "### 出力 ###:", "### 输出 ###:"]
                    regex = "|".join(separators)
                    text = re.split(regex, data[lang_map[language]][key])[-1].strip()
                    # text = data[lang_map[language]][key].split(template["response_split"])[-1].strip()
                    if text == "":
                        # import pdb; pdb.set_trace()
                        input_context = template["prompt_input"].format(language, data_pool[idx]["en"][key])
                        generated_text = utils.get_chatgpt_info("gpt-4", input_context, temperature=0.0, 
                                                        persona_info=template["persona_info"])
                        data_pool[idx][lang_map[language]][key] = generated_text
            t.set_postfix()
            t.update(1)

    write_json(save_file, data_pool)


if __name__=="__main__":
    # Sampling 1k QA data points for quick test.
    # sampling(k=2000)

    # Translate the QA dataset into multilanguage.
    if args.stage == "translate":
        lang_list = ["Chinese", "Japanese", "French", "Thai"]
        for language in lang_list[3:]:
            print(f"Translate the {args.dataset} dataset into {language}.")
            translate_text(args.model_name, args.temperature, language, filename="val", dataset=args.dataset)
        if args.dataset == "trivia":
            travia_few_shot_examples(args.model_name, args.temperature)
    elif args.stage == "filter":
        combine_and_filter(args.temperature, dataset=args.dataset)
    elif args.stage == "refilter":
        recheck_and_filter(args.dataset)



    