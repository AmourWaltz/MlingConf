r"""
Author: XUE Boyang      Filename: refine.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Self-Reinement using confidence score on various languages.
"""
import argparse
import collections
import json
import os
import re
import string
import random

from tqdm import tqdm, trange
from ipdb import set_trace
from rouge import Rouge

import utils
import para


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--prompt_path', type=str, default="./prompt/trivia/")
parser.add_argument('--inp_path', type=str, default="./exp/trivia/confidence/v2/verb")
parser.add_argument('--out_path', type=str, default="./exp/trivia/refinement")
parser.add_argument('--eval_path', type=str, default="./exp/trivia/evaluate/v2")
parser.add_argument('--threshold', type=float, default=0.75)
parser.add_argument('--temperature', type=float, default=0.8)

args = parser.parse_args()


def refine_judge(conf_score):
    if args.threshold != 0.0:
        if conf_score <= args.threshold:
            return True
        else:
            False
    else:
        # set_trace()
        if conf_score <= random.random():
            return True
        else:
            return False


def confidence_refine(model_name, temperature, language):
    # Load input QA dataset with confidence scores.
    inp_file = os.path.join(args.inp_path, "val_2k_{}_pred_eval_en.json".format(para.lang_map[language]))
    data_pool = utils.read_json(inp_file)
    # data_pool = utils.read_json(inp_file)[:1]

    # Construct output file.
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # Load the first evaluation files and preserve the results that doesn't need to be refined.
    eval_file =  os.path.join(args.eval_path, "val_2k_{}_pred_eval.json".format(para.lang_map[language]))
    eval_pool = utils.read_json(eval_file)
    eval_dict = {}
    for samp in eval_pool:
        eval_dict[samp["question_id"]] = {
            "EM score": samp["EM score"],
            "F1 score": samp["F1 score"],
            "Rouge-L score": samp["Rouge-L score"],
            "Rouge-1 score": samp["Rouge-1 score"],
            "NLI result": samp["NLI result"],
            "NLI score": samp["NLI score"]
        }

    out_file = os.path.join(args.out_path, "val_2k_{}_conf_refine_{}.json".format(para.lang_map[language], 
                                        args.threshold if args.threshold != 0.0 else "sample"))

    open(out_file, "w").close()

    # Load evaluation prompt template in different languages.
    template_file = os.path.join(args.prompt_path, "refine_temp.json")
    template = json.load(open(template_file))

    # NLI by GPT-3.5-turbo on input QA val set on multiple languages.
    # set_trace()
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool:
        for data in data_pool:
            confidence = float(data["probability"])
            data["first answer"] = data["prediction"]
            if refine_judge(conf_score=confidence):
                input_context = template["prompt_input"][para.lang_map[language]].format(data["question"], data["prediction"])
                # print(input_context)
                # set_trace()
                generated_texts = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                                        persona_info=template["persona_info"][para.lang_map["English"]])
                data["prediction"] = generated_texts
            else:
                data.update(eval_dict[data["question_id"]])

            with open(out_file, mode="a+") as fw: 
                data_rec = json.dumps(obj=data, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)


def self_refinement():
    # Evaluate the generated answers with gold answers on F1 measure, Exact Match, Rouge-L, Rouge-1, and NLI.
    # Language list: ["Chinese", "English", "Japanese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    lang_list = ["Thai", "Italian"]
    for language in lang_list:
    # for language in list(para.lang_map.keys())[2:]:
        print(f"Self-refienment using confidence score on {language} val dataset.")
        confidence_refine(args.model, args.temperature, language)


if __name__ == '__main__':
    self_refinement()
    pass
