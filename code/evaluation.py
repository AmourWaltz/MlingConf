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
parser.add_argument('--dataset', type=str, default="sciq", help=["triviaqa", "gsm8k", "common", "sciq"])
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
        return int(normalize_answer(a_gold) in normalize_answer(a_pred)) or int(normalize_answer(a_pred) in normalize_answer(a_gold))


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


def accu_calc(data_pool, language):
    num_pos = 0
    for data in data_pool:
        num_pos += data["accuracy"][language]
    
    return round(num_pos / len(data_pool)*100, 2)
        

def conf_ece_calu(data_pool, language):
    ptrue_inputs, prob_inputs, verb_inputs, labels, fails = [], [], [], [], 0
    # set_trace()
    for _, data in enumerate(data_pool):
        # assert data["question_id"] == labels_pool[idx]["question_id"]
        ptrue_val = data["ptrue"][language]["conf"]
        prob_val = data["probs"][language]
        # import pdb; pdb.set_trace()
        verb_val = data["verb"][language]
        # Two methods: min and avg
        ptrue_inputs.append(ptrue_val)
        prob_inputs.append(prob_val)
        verb_inputs.append(verb_val)
        labels.append(data["accuracy"][language])

    # set_trace()
    # print("Total {} samples fail ......".format(fails))
    ptrue_trues = np.array(ptrue_inputs) # Extract the logit of "True" label
    prob_trues = np.array(prob_inputs) # Extract the logit of "True" label
    verb_trues = np.array(verb_inputs)# Extract the logit of "True" label

    # assert len(labels) == len(ptrue_inputs) == len(prob_inputs) == len(verb_inputs)
    ptrue_ece = round(ece(np.array(labels), np.array(ptrue_trues))*100, 2)
    prob_ece = round(ece(np.array(labels), np.array(prob_trues))*100, 2)
    verb_ece = round(ece(np.array(labels), np.array(verb_trues))*100, 2)

    print(ptrue_ece, prob_ece, verb_ece)
    # import pdb; pdb.set_trace()
    return ptrue_ece, prob_ece, verb_ece


def accuracy_obtain():
    data_path = os.path.join(args.data_path.format(args.dataset), f"{args.model}_{args.dataset}_confidence/confidence.json")
    data_pool = read_json(data_path)
    for data in data_pool:
        data["accuracy"] = {}
        temp = data["verb"]
        data["verb"] = {}
        for idx, lang in enumerate(language_list):
            lang = lang_map[lang]
            if args.dataset == "gsm8k":
                data["accuracy"][lang] = is_correct(data["output"][lang], data["answer"][lang])
            else:
                data["accuracy"][lang] = compute_exact(data["output"][lang], data["answer"][lang])
            # import pdb; pdb.set_trace()
            # print(temp[idx], data["question_id"], lang)
            # if isinstance(temp[idx], str):
            #     print(temp[idx])
            #     if temp[idx].strip() == "":
            #         data["verb"][lang] = 0.5

            data["verb"][lang] = float(temp[idx])
    # import pdb; pdb.set_trace()
    write_json(data_path, data_pool)


def conf_auroc(data_pool, language):    
    ptrue_inputs, prob_inputs, verb_inputs, labels, fails = [], [], [], [], 0
    # set_trace()
    for _, data in enumerate(data_pool):
        # assert data["question_id"] == labels_pool[idx]["question_id"]
        ptrue_val = data["ptrue"][language]["conf"]
        prob_val = data["probs"][language]
        # import pdb; pdb.set_trace()
        verb_val = data["verb"][language]
        # Two methods: min and avg
        # val = val_list[0] # min(val_list)
        ptrue_score_vector = [ptrue_val, 1-ptrue_val]
        prob_score_vector = [prob_val, 1-prob_val]
        verb_score_vector = [verb_val, 1-verb_val]

        ptrue_inputs.append(ptrue_score_vector)
        prob_inputs.append(prob_score_vector)
        verb_inputs.append(verb_score_vector)
        labels.append(data["accuracy"][language])

    # set_trace()
    # print("Total {} samples fail ......".format(fails))
    ptrue_trues = np.array(ptrue_inputs)[:, 0] # Extract the logit of "True" label
    prob_trues = np.array(prob_inputs)[:, 0] # Extract the logit of "True" label
    verb_trues = np.array(verb_inputs)[:, 0] # Extract the logit of "True" label

    assert len(labels) == len(ptrue_inputs) == len(prob_inputs) == len(verb_inputs)
    ptrue_auroc = round(roc_auc_score(np.array(labels), np.array(ptrue_trues))*100, 2)
    prob_auroc = round(roc_auc_score(np.array(labels), np.array(prob_trues))*100, 2)
    verb_auroc = round(roc_auc_score(np.array(labels), np.array(verb_trues))*100, 2)
    print(ptrue_auroc, prob_auroc, verb_auroc)
    # import pdb; pdb.set_trace()
    return ptrue_auroc, prob_auroc, verb_auroc


def auroc_calc():
    # Format self-aware and self-verbalized confidence scores to calculate AUROC.
    ins_set = []
    data_path = os.path.join(args.data_path.format(args.dataset), f"{args.model}_{args.dataset}_confidence/confidence.json")
    log_path = os.path.join(args.data_path.format(args.dataset), f"{args.model}_{args.dataset}_confidence/log.json")
    accuracy_obtain()
    data_pool = read_json(data_path)

    for language in language_list:
        print(f"AUROC calculation on {language} dataset using {args.model} model.")
        language = lang_map[language]
        true_auroc, prob_auroc, verb_auroc = conf_auroc(data_pool, language)
        true_ece, prob_ece, verb_ece = conf_ece_calu(data_pool, language)
        accu = accu_calc(data_pool, language)
        ins = {
            "language": language,
            "prob auroc": prob_auroc,
            "true auroc": true_auroc,
            "verb auroc": verb_auroc,
            "prob ece": prob_ece,
            "true ece": true_ece,
            "verb ece": verb_ece,
            "accuracy": accu
        }
        ins_set.append(ins)

    write_json(log_path, ins_set)


if __name__=="__main__":
    auroc_calc()
    pass