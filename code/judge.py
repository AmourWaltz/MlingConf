r"""
Author: XUE Boyang      Filename: judge.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Model self-evaluate if the answer is true or false.
"""
import os
import json
import argparse

from tqdm import tqdm, trange
from ipdb import set_trace

import conf
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help=["gpt-3.5-turbo", "gpt-4"])
parser.add_argument('--prompt_path', type=str, default="./prompt/trivia/")
parser.add_argument('--inp_path', type=str, default="./exp/trivia/generate")
parser.add_argument('--out_path', type=str, default="./exp/trivia/judge")
parser.add_argument('--temperature', type=float, default=0.8)

args = parser.parse_args()


def prediction_judge(model_name, temperature, language):
    # Load input QA val dataset with generated answers.
    inp_file = os.path.join(args.inp_path, "val_2k_{}_gpt-3.5-turbo_T0.8.json".format(conf.lang_map[language]))
    data_pool = utils.read_json(inp_file)

    # Construct output file.
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    out_file = os.path.join(args.out_path, "val_2k_{}_{}_T{}.json".format(conf.lang_map[language], 
                                                                      model_name, temperature))
    open(out_file, "w").close()

    # Load prompt template in different languages.
    template_file = os.path.join(args.prompt_path, "judge_temp.json")
    template = json.load(open(template_file))

    # Judge the answer true or false on input QA val set on multiple languages.
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool[:3]:
        for data in data_pool:
            input_context = template["prompt_input"][conf.lang_map[language]].format(data["question"], data["prediction"])
            # print(input_context)
            # set_trace()
            generated_text = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                                persona_info=template["persona_info"][conf.lang_map[language]])
            data["judgement"] = generated_text
            # print(generated_text)
            # set_trace()

            with open(out_file, mode="a+") as fw: 
                data_rec = json.dumps(obj=data, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)
    

if __name__=="__main__":
    # Language list: ["Chinese", "English", "Japenese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    lang_list = ["Arabic"]
    for language in lang_list:
    # for language in list(conf.lang_map.keys())[1:]:
        print(f"Judge the generated answers on TriviaQA {language} val dataset [True] or [False] using {args.model} model.")
        prediction_judge(args.model, args.temperature, language)

