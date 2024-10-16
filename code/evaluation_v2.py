r"""
Author: XUE Boyang      Filename: eval.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Evaluation script for generated answers of QA dataset.
"""
import argparse
import collections
import json
import os
import re
import string

from tqdm import tqdm, trange
from ipdb import set_trace
from rouge import Rouge

import utils
import para


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--prompt_path', type=str, default="./prompt/trivia/")
parser.add_argument('--data_path', type=str, default="./exp/trivia/generate/v3")
parser.add_argument('--label_path', type=str, default="./exp/trivia/evaluate/v2")
parser.add_argument('--save_path', type=str, default="./exp/trivia/evaluate/v3")
parser.add_argument('--temperature', type=float, default=0.8)

args = parser.parse_args()

rouger = Rouge()

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


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(data_pool):
    exact_scores = {}
    f1_scores = {}
    rougeL_scores = {}
    rouge1_scores = {}

    for data in data_pool:
        pred_ans = data['prediction']
        gold_ans = normalize_answer(data['answer'])

        # Take max over all gold answers
        em_score = compute_exact(gold_ans, pred_ans)
        exact_scores[data["question_id"]] = em_score

        f1_score = compute_f1(gold_ans, pred_ans)
        f1_scores[data["question_id"]] = f1_score

        rougeL_score = rouger.get_scores(pred_ans, gold_ans)[0]["rouge-l"]["f"]
        rougeL_scores[data["question_id"]] = rougeL_score

        rouge1_score = rouger.get_scores(pred_ans, gold_ans)[0]["rouge-1"]["f"]
        rouge1_scores[data["question_id"]] = rouge1_score
        # print(data['question_id'])

    return exact_scores, f1_scores, rougeL_scores, rouge1_scores


def make_eval_dict(exact_scores, f1_scores, rougeL_scores, rouge1_scores, nli_scores):
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores) / len(exact_scores)),
        ('f1', 100.0 * sum(f1_scores) / len(f1_scores)),
        ('rougeL', 100.0 * sum(rougeL_scores) / len(rougeL_scores)),
        ('rouge1', 100.0 * sum(rouge1_scores) / len(rouge1_scores)),
        ('nli', 100.0 * sum(nli_scores) / len(nli_scores))
    ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def nli_to_score(data_pool, language):
    correct, incorrect, failed = 0, 0, 0
    pos, neg = para.lang_true_false_dict[language]

    # print(pos, neg)
    for data in data_pool:
        res = data["NLI result"]
        res = res.replace(".", "")
        if neg in res:
            data["NLI score"] = 0
            incorrect += 1
        elif pos in res:
            data["NLI score"] = 1
            correct += 1
        else:
            print("Sample {}: {} fails ......".format(data["question_id"], data["NLI result"]))
            failed += 1
            

    assert(correct + incorrect + failed  == len(data_pool))

    print("{} correct samples, {} incorrect samples, {} failed samples. Accuracy: {}".\
          format(correct, incorrect, failed, correct / len(data_pool)))
    return data_pool


def pred_ans_evals(model_name, temperature, language, instruction=False):
    # Load input QA dataset with generated answers.
    inp_file = os.path.join(args.data_path, "val_2k_{}_gpt-4_T0.9.json".format(para.lang_map[language]))
    # inp_file = os.path.join(args.data_path, "val_2k_{}_conf_refine_0.75.json".format(para.lang_map[language]))
    # data_pool = utils.read_json(inp_file)[:3]
    data_pool = utils.read_json(inp_file)

    # Construct output file.
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    out_file = os.path.join(args.save_path, "val_2k_{}_pred_eval.json".format(
                            para.lang_map[language]) if instruction != "en" else \
                            "val_2k_{}_pred_eval_{}.json".format(para.lang_map[language], instruction))

    open(out_file, "w").close()
    
    # Load evaluation prompt template in different languages.
    template_file = os.path.join(args.prompt_path, "eval_temp.json")
    template = json.load(open(template_file))

    # Compute F1 Measure, Exact Match, and Entailment by GPT-3.5-turbo.
    exact_raw, f1_raw, rougeL_raw, rouge1_raw = get_raw_scores(data_pool)

    # NLI by GPT-3.5-turbo on input QA val set on multiple languages.
    # set_trace()
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool:
        for data in data_pool:
            if "NLI result" not in data.keys():
                input_context = template["prompt_input"][para.lang_map[language if instruction != "en" else "English"]].format(
                                                                                            data["answer"], data["prediction"])
                # print(template["persona_info"][conf.lang_map[language if instruction != "en" else "English"]])
                # print(input_context)
                # set_trace()
                nli_res = utils.get_chatgpt_info(model_name, input_context, temperature, 
                                persona_info=template["persona_info"][para.lang_map[language if instruction != "en" else "English"]])

            ins = {
                "question_id": data["question_id"],
                "generated answer": data["prediction"],
                "gold answer": data["answer"],
                "EM score": exact_raw[data["question_id"]],
                "F1 score": f1_raw[data["question_id"]],
                "Rouge-L score": rougeL_raw[data["question_id"]],
                "Rouge-1 score": rouge1_raw[data["question_id"]],
                "NLI result": nli_res if "NLI result" not in data.keys() else data["NLI result"]
            }

            with open(out_file, mode="a+") as fw: 
                data_rec = json.dumps(obj=ins, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)


def overall_calc(lang_list, instruction=False):
    out_file = os.path.join(args.save_path, "val_2k_overall_eval.json")

    for language in lang_list:
    # for language in list(conf.lang_map.keys()):
        # Load evaluation value file.
        inp_file = os.path.join(args.save_path, "val_2k_{}_pred_eval.json".format(
                            para.lang_map[language]) if instruction != "en" else \
                            "val_2k_{}_pred_eval_{}.json".format(para.lang_map[language], instruction))
        
        data_pool = utils.read_json(inp_file)

        # Overall prediction evaluation of F1, EM, R-L, R-1 and NLI scores.
        # Two types of instructions are used: English and [current language].
        if para.lang_map["English"] in inp_file and language != "English":
            data_pool = nli_to_score(data_pool, language="English")
        else:
            data_pool = nli_to_score(data_pool, language)

        utils.write_json(filename=inp_file, dataset=data_pool)

        # set_trace()
        exact_raw = [data["EM score"] for data in data_pool]
        f1_raw = [data["F1 score"] for data in data_pool]
        rougeL_raw = [data["Rouge-L score"] for data in data_pool]
        rouge1_raw = [data["Rouge-1 score"] for data in data_pool]
        nli_raw = [data["NLI score"] for data in data_pool]

        out_eval = make_eval_dict(exact_raw, f1_raw, rougeL_raw, rouge1_raw, nli_raw)
        overall = {
                "Langauge": language if para.lang_map["English"] not in inp_file else f"{language} English",
                "EM score": round(out_eval["exact"], 2),
                "F1 score": round(out_eval["f1"], 2),
                "RL score": round(out_eval["rougeL"], 2),
                "R1 score": round(out_eval["rouge1"], 2),
                "NLI score": round(out_eval["nli"], 2)
            }
        
        print(overall)
        with open(out_file, mode="a+") as fw: 
            data_rec = json.dumps(obj=overall, ensure_ascii=False)
            fw.write(data_rec + '\n')



def accu_eval():
    # Evaluate the generated answers with gold answers on F1 measure, Exact Match, Rouge-L, Rouge-1, and NLI.
    # Language list: ["Chinese", "English", "Japanese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    # lang_list = ["French", "Italian", "Korean", "Thai"]
    
    for language in list(para.lang_map.keys())[3:]:
    # for language in lang_list:
        print(f"Evaluating F1, EM, R-L, R-1, and NLI scores on TriviaQA {language} val dataset.")
        pred_ans_evals(args.model, args.temperature, language)

    overall_calc(list(para.lang_map.keys()), instruction="en")
    # overall_calc(lang_list)


if __name__ == '__main__':
    accu_eval()
    pass

