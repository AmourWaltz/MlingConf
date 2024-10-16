r"""
Author: XUE Boyang      Filename: evaluation.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Calculate the ECE and AUROC score on several dataset.
"""
import os
import re
import json
import string
import argparse

from tqdm import tqdm, trange
from ipdb import set_trace
import sklearn
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import operator
from functools import reduce

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help=["gpt-3.5-turbo", "gpt-4", "gpt-4o", "llama3", "vicuna", "gpt2"])
parser.add_argument('--dataset', type=str, default="triviaqa", help=["triviaqa", "gsm8k", "common", "sciq"])
parser.add_argument('--data_path', type=str, default="./exp/{}/confidence/")

args = parser.parse_args()


"""
Evaluate QA outputs: TriviaQA, SciQ, CommonSenseQA, etc.
"""
# Normalize the answer.
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    # return white_space_fix(remove_punc(lower(s)))
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Compute the exact match score in different ways.
def compute_exact(a_gold, a_pred):
    eval_type = "EM_RP"

    if eval_type == "EM":
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    elif eval_type == "EM_R":
        return int(normalize_answer(a_gold) in normalize_answer(a_pred))
    elif eval_type == "EM_P":
        return int(normalize_answer(a_pred) in normalize_answer(a_gold))
    elif eval_type == "EM_RP":
        if args.dataset == "triviaqa":
            return int(normalize_answer(a_gold) in normalize_answer(a_pred)) or int(normalize_answer(a_pred) in normalize_answer(a_gold))
        elif args.dataset == "gsm8k":
            return int(normalize_answer(a_gold) == normalize_answer(a_pred))


"""
Evaluate math problems: GSM8K
"""
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS
    return int(extract_answer(model_completion) == gt_answer)

def ece(y_true, y_prob, n_bins=10):
    # Ensure the input in [0, 1]
    y_prob = np.clip(y_prob, 0, 1)
    
    # Calculate the bucket index of each sample
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    # Calculate the average absolute error of each bucket
    ece_cumulative = []
    for i in range(n_bins):
        # For each bucket, calculate the average of |prob_true - prob_pred|
        mask = (bin_indices == i)
        if np.sum(mask) > 0:  # Ensure the bucket is not empty
            prob_true = y_true[mask].mean()  # The average of the true probability in the bucket
            prob_pred = y_prob[mask].mean()  # The average of the predicted probability in the bucket
            ece_cumulative.append(abs(prob_true - prob_pred))
    
    # Calibration Error
    ece_value = np.mean(ece_cumulative)
    return ece_value


def confidence_score(model_name, temperature, language, baseline="prob", instruction=False):
    # Load input QA val dataset with generated answers.
    inp_file = os.path.join(args.inp_path, "val_2k_{}_gpt-4_T0.9.json".format(para.lang_map[language]))
    data_pool = utils.read_json(inp_file)
    # data_pool = utils.read_json(inp_file)[168:] # Quick test

    # Construct output file.
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    out_file = os.path.join(args.out_path, "val_2k_{}_{}_T{}.json".format(
                para.lang_map[language], model_name, temperature) if instruction != "en" else \
                "val_2k_{}_{}_T{}_{}.json".format(para.lang_map[language], model_name, temperature, instruction))

    # open(out_file, "w").close()

    # Load prompt template in different languages.
    template_file = os.path.join(args.prompt_path, "conf_{}_temp.json".format(baseline))
    template = json.load(open(template_file))

    # Judge the answer true or false on input QA val set on multiple languages.
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool[1008:]:
        for data in data_pool:
            input_context = template["prompt_input"][para.lang_map[language if instruction != "en" else "English"]]. \
                                                                            format(data["answer"], data["prediction"])
            # print(input_context)
            # set_trace()
            if baseline in ["judge", "verb", "word"]:
                generated_texts = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                persona_info=template["persona_info"][para.lang_map[language if instruction != "en" else "English"]])
                data["probability"] = generated_texts
            elif baseline == "prob":
                generated_texts, generated_probs = utils.get_chatgpt_info(model_name, input_context, temperature, logprobs=True,
                                persona_info=template["persona_info"][para.lang_map[language if instruction != "en" else "English"]])
                
                data["probability"] = generated_probs
                data["judgement"] = generated_texts

            with open(out_file, mode="a+") as fw: 
                data_rec = json.dumps(obj=data, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)


def aware_conf_auroc(language, instruction=False):
    # Load input QA val dataset with self-aware confidence score to calculate auroc score.
    data_file = os.path.join("./exp/trivia/confidence/v3/prob", "val_2k_{}_gpt-3.5-turbo_T0.8_en.json".format(
                        para.lang_map[language]) if instruction != "en" else \
                        "val_2k_{}_gpt-3.5-turbo_T0.8_{}.json".format(para.lang_map[language], instruction))
    data_pool = utils.read_json(data_file)

    label_file = os.path.join(args.label_path, "val_2k_{}_pred_eval.json".format(
                        para.lang_map[language]) if instruction != "en" else \
                        "val_2k_{}_pred_eval_{}.json".format(para.lang_map[language], instruction))
    labels_pool = utils.read_json(label_file)

    out_file = os.path.join(args.out_path, "val_2k_{}_gpt-3.5-turbo_T0.9_conf_aware.json".format(para.lang_map[language]) \
        if instruction != "en" else "val_2k_{}_gpt-3.5-turbo_T0.9_conf_aware_{}.json".format(para.lang_map[language], instruction))
    
    ins_set, inputs, labels, fails = [], [], [], 0
    # set_trace()
    for idx, data in enumerate(data_pool):
        # assert data["question_id"] == labels_pool[idx]["question_id"]

        judge = data["judgement"].replace(".", "")
        if judge == para.lang_aware_dict[language if instruction != "en" else "English"][0]:
            val_list = list(data["probability"].values())
            # Two methods: min and avg
            # val = val_list[0] # min(val_list)
            val = min(val_list)
            score_vector = [val, 1-val]
        elif judge == para.lang_aware_dict[language if instruction != "en" else "English"][1]:
            val_list = list(data["probability"].values())
            # Two methods: min and avg
            # val = val_list[0] # min(val_list)
            val = min(val_list)
            score_vector = [1-val, val]
        else:
            # print("Sample {}: {} fails ......".format(data["question_id"], data["judgement"]))
            fails += 1
            continue

        inputs.append(score_vector)
        labels.append(labels_pool[idx]["NLI score"])
        ins = {
            "question_id": data["question_id"],
            "question": data["question"],
            "gold answer": data["prediction"],
            "model answer": data["answer"],
            "NLI score": labels_pool[idx]["NLI score"],
            "self-aware result": judge,
            "self-aware score": score_vector,
        }
        ins_set.append(ins)

    # set_trace()
    print("Total {} samples fail ......".format(fails))
    trues = np.array(inputs)[:, 0] # Extract the logit of "True" label

    assert len(labels) == len(inputs)
    true_auroc = roc_auc_score(np.array(labels), np.array(trues))
    print(true_auroc)

    utils.write_json(out_file, ins_set)

    return true_auroc


def logit_conf_auroc(language, instruction=False):
    # Load input QA val dataset with self-aware confidence score to calculate auroc score.
    data_file = os.path.join("./exp/trivia/generate/v3/", "val_2k_{}_gpt-4_T0.9.json".format(
                        para.lang_map[language]))
    data_pool = utils.read_json(data_file)

    label_file = os.path.join(args.label_path, "val_2k_{}_pred_eval.json".format(
                        para.lang_map[language]) if instruction != "en" else \
                        "val_2k_{}_pred_eval_{}.json".format(para.lang_map[language], instruction))
    labels_pool = utils.read_json(label_file)

    out_file = os.path.join(args.out_path, "val_2k_{}_gpt-3.5-turbo_T0.9_conf_aware.json".format(para.lang_map[language]) \
        if instruction != "en" else "val_2k_{}_gpt-3.5-turbo_T0.9_conf_aware_{}.json".format(para.lang_map[language], instruction))
    data_scores = utils.read_json(out_file)

    inputs, labels, idy, length = [], [], 0, 0
    # set_trace()
    for idx, data in enumerate(data_pool):
        assert data["question_id"] == labels_pool[idx]["question_id"]
        if data["question_id"] != data_scores[idx-idy]["question_id"]:
            idy += 1
            continue
        
        val_list = list(data["probability"].values())
        # Tree methods: norm, min and avg
        # val = val_list[0] # min(val_list)
        val_min = np.array(val_list).min()
        val_avg = np.array(val_list).mean()
        val_norm = float(np.array(pow(reduce(operator.mul, val_list), 1/len(val_list))))
        # set_trace()
        inputs.append([val_min, val_avg, val_norm])
        labels.append(labels_pool[idx]["NLI score"])

        data_scores[idx-idy]["logit min score"] = val_min
        data_scores[idx-idy]["logit avg score"] = val_avg
        data_scores[idx-idy]["logit norm score"] = val_norm

        length += len(val_list)

    # set_trace()
    assert len(labels) == len(inputs)
    min_auroc = roc_auc_score(torch.tensor(labels), torch.tensor(inputs)[:, 0])
    avg_auroc = roc_auc_score(torch.tensor(labels), torch.tensor(inputs)[:, 1])
    norm_auroc = roc_auc_score(torch.tensor(labels), torch.tensor(inputs)[:, 2])

    print(min_auroc, avg_auroc, norm_auroc)

    utils.write_json(out_file, data_scores)

    return min_auroc, avg_auroc, norm_auroc, length/len(inputs)


def verb_conf_auroc(language, instruction=False, verb="word"):
    # Load input QA val dataset with self-aware confidence score to calculate auroc score.
    data_file = os.path.join(f"./exp/trivia/confidence/v3/{verb}/", "val_2k_{}_gpt-3.5-turbo_T0.8_en.json".format(
                        para.lang_map[language]))
    data_pool = utils.read_json(data_file)

    label_file = os.path.join(args.label_path, "val_2k_{}_pred_eval.json".format(
                        para.lang_map[language]) if instruction != "en" else \
                        "val_2k_{}_pred_eval_{}.json".format(para.lang_map[language], instruction))
    labels_pool = utils.read_json(label_file)

    out_file = os.path.join(args.out_path, "val_2k_{}_gpt-3.5-turbo_T0.9_conf_aware.json".format(para.lang_map[language]) \
        if instruction != "en" else "val_2k_{}_gpt-3.5-turbo_T0.9_conf_aware_{}.json".format(para.lang_map[language], instruction))
    data_scores = utils.read_json(out_file)

    inputs, labels, idy = [], [], 0
    # set_trace()
    for idx, data in enumerate(data_pool):
        # assert data["question_id"] == labels_pool[idx]["question_id"]
        if data["question_id"] != data_scores[idx-idy]["question_id"]:
            idy += 1
            continue
        
        if verb == "verb":
            val_verb = float(data["probability"])
        elif verb == "word":
            val_verb = para.word_conf_score[data["probability"].lower().replace(".", "")]

        # set_trace()
        inputs.append(val_verb)
        labels.append(labels_pool[idx]["NLI score"])

        data_scores[idx-idy]["self-verb score"] = val_verb

    # set_trace()
    assert len(labels) == len(inputs)
    verb_auroc = roc_auc_score(torch.tensor(labels), torch.tensor(inputs))
    print(verb_auroc)

    utils.write_json(out_file, data_scores)
    return verb_auroc


"""Modules"""
def conf_gene():
    for language in list(para.lang_map.keys()):
        # print(f"Baseline estimation of P(True) of answers on TriviaQA {language} val dataset using {args.model} model.")
        print(f"Baseline self-verbalized confidence score of answers on TriviaQA {language} val dataset using {args.model} model.")
        confidence_score(args.model, args.temperature, language, args.baseline, instruction="en")


def auroc_calc():
    # Format self-aware and self-verbalized confidence scores to calculate AUROC.
    ins_set = []
    # lang_list = ["Chinese"]
    # for language in lang_list:
    for language in list(para.lang_map.keys()):
        print(f"AUROC calculation on TriviaQA {language} val dataset using {args.model} model.")
        aware_auroc = aware_conf_auroc(language, instruction="en")
        min_auroc, avg_auroc, norm_auroc, avg_len = logit_conf_auroc(language, instruction="en")
        verb_auroc = verb_conf_auroc(language, instruction="en", verb="verb")
        word_auroc = verb_conf_auroc(language, instruction="en", verb="word")
        ins = {
            "language": language,
            "logit min auroc": min_auroc,
            "logit avg auroc": avg_auroc,
            "logit norm auroc": norm_auroc,
            "self aware auroc": aware_auroc,
            "self verb auroc": verb_auroc,
            "self word auroc": word_auroc,
            "average length": avg_len
        }
        ins_set.append(ins)

    utils.write_json(os.path.join(args.out_path, "overall_results.json"), ins_set)



if __name__=="__main__":
    auroc_calc()
    pass