r"""
Author: XUE Boyang      Filename: infer.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Generate answers on QA val set in few-shot.
"""
import os
import json
import argparse

from tqdm import tqdm, trange
from ipdb import set_trace

import conf
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="text-davinci-003", help=["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
parser.add_argument('--prompt_path', type=str, default="./prompt/trivia/")
parser.add_argument('--inp_path', type=str, default="./data/trivia/")
parser.add_argument('--out_path', type=str, default="./exp/trivia/")
parser.add_argument('--temperature', type=float, default=0.8)

args = parser.parse_args()


def few_shot_prompt(few_shot_samples, prompt):
    for sample in few_shot_samples:
        prompt += '# Question #: ' + sample["question"] + "\n# Answer #: " + sample["answer"] + "\n\n"

    return prompt


def generate(model_name, temperature, language, filename="val_2k"):
    # Load input QA val dataset.
    inp_file = os.path.join(args.inp_path, "{}_{}.json".format(filename, conf.lang_map[language]))
    few_shot_samples, data_pool = utils.read_jsonl(inp_file)[:5], utils.read_jsonl(inp_file)[5:]

    # Construct output file.
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    out_file = os.path.join(args.out_path, "{}_{}_{}_T{}.json".format(filename, conf.lang_map[language], 
                                                                      model_name, temperature))
    open(out_file, "w").close()

    # Load prompt template and construct few-shot examples.
    template_file = os.path.join(args.prompt_path, "infer_temp.json")
    template = json.load(open(template_file))

    icl_prompt = few_shot_prompt(few_shot_samples, template["instruction"])

    # Generate answer on input QA val set on multiple languages.
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool[:3]:
        for data in data_pool:
            input_context = template["prompt_input"].format(icl_prompt, data["question"])
            # print(input_context)
            # set_trace()
            generated_text = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                                persona_info=template["persona_info"])
            data["prediction"] = generated_text

            with open(out_file, mode="a+") as fw: 
                data_rec = json.dumps(obj=data, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)
    

if __name__=="__main__":
    lang_list = ["Chinese", "Japenese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    for language in list(conf.lang_map.keys()):
    # for language in lang_list[:2]:
        print(f"Generate answers on TriviaQA {language} val dataset using {args.model} model.")
        generate(args.model, args.temperature, language)

