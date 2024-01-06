r"""
Author: XUE Boyang      Filename: trans.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Translate input QA into multilanguage.
"""
import os
import json
import random
import argparse

from tqdm import tqdm, trange
from ipdb import set_trace

import conf
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="gpt-4")
parser.add_argument('--prompt_path', type=str, default="./prompt/trivia/")
parser.add_argument('--inp_path', type=str, default="./data/trivia/")
parser.add_argument('--out_path', type=str, default="./data/trivia/")
parser.add_argument('--temperature', type=float, default=0.9)

args = parser.parse_args()


def sampling(k):
    inp_file = os.path.join(args.inp_path, "val.json")
    out_file= os.path.join(args.out_path, "val_2k.json")

    data_pool = utils.read_json(inp_file)
    data_sample = random.sample(data_pool, k)
    utils.write_json(out_file, data_sample)


def translate(model_name, temperature, language, filename="val"):
    # Read QA dataset.
    inp_file = os.path.join(args.inp_path, "{}.json".format(filename))
    data_pool = utils.read_json(inp_file)
    # data_pool = read_json(inp_file)[:3]

    # Construct output file.
    out_file= os.path.join(args.out_path, "{}_{}.json".format(filename, conf.lang_map[language]))
    open(out_file, "w").close()

    # Load prompt template
    template_file = os.path.join(args.prompt_path, "trans_temp.json")
    template = json.load(open(template_file))    

    # set_trace()
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool[:3]:
        for data in data_pool:
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



def few_shot_examples(model_name, temperature):
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
    for language in list(conf.lang_map.keys())[2:]:
        out_file= os.path.join(args.out_path, "val_2k_{}.json".format(conf.lang_map[language]))
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



if __name__=="__main__":
    # Sampling 1k QA data points for quick test.
    # sampling(k=2000)

    print(list(conf.lang_map.keys()))
    # lang_list = ["Chinese", "Japenese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    # for language in lang_list:
    for language in list(conf.lang_map.keys())[1:]:
        print(f"Translate the TriviaQA dataset into {language}.")
        translate(args.model_name, args.temperature, language, filename="val_2k")

    # Few shot examples construction
    few_shot_examples(args.model_name, args.temperature)

    