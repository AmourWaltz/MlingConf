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

import utils
import conf


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--prompt_path', type=str, default="./prompt/trivia/")
parser.add_argument('--inp_path', type=str, default="./exp/trivia/generate")
parser.add_argument('--out_path', type=str, default="./exp/trivia/evaluate")
parser.add_argument('--temperature', type=float, default=0.8)
    
args = parser.parse_args()


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

    for data in data_pool:
        pred_ans = data['prediction']
        gold_ans = normalize_answer(data['answer'])

        # Take max over all gold answers
        em_score = compute_exact(gold_ans, pred_ans)
        exact_scores[data["question_id"]] = em_score

        f1_score = compute_f1(gold_ans, pred_ans)
        f1_scores[data["question_id"]] = f1_score

    return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores, nli_scores):
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores) / len(exact_scores)),
        ('f1', 100.0 * sum(f1_scores) / len(f1_scores)),
        ('nli', 100.0 * sum(nli_scores) / len(nli_scores))
    ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def nli_to_score(data_pool, language):
    correct, incorrect, failed = 0, 0, 0
    pos, neg = conf.lang_true_false_dict[language]

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
            print("Sample {} fails ......".format(data["question_id"]))
            failed += 1
            

    assert(correct + incorrect + failed  == len(data_pool))

    print("{} correct samples, {} incorrect samples, {} failed samples. Accuracy: {}".format(correct, incorrect, failed, correct / len(data_pool)))
    return data_pool


def pred_ans_evals(model_name, temperature, language, instruction=False):
    # Load input QA dataset with generated answers.
    inp_file = os.path.join(args.inp_path, "val_2k_{}_gpt-3.5-turbo_T0.8.json".format(conf.lang_map[language]))
    # data_pool = utils.read_json(inp_file)[:3]
    data_pool = utils.read_json(inp_file)

    # Construct output file.
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    out_file = os.path.join(args.out_path, "val_2k_{}_pred_eval.json".format(
                            conf.lang_map[language]) if instruction != "en" else \
                            "val_2k_{}_pred_eval_{}.json".format(conf.lang_map[language], instruction))

    open(out_file, "w").close()
    
    # Load evaluation prompt template in different languages.
    template_file = os.path.join(args.prompt_path, "eval_temp.json")
    template = json.load(open(template_file))

    # Compute F1 Measure, Exact Match, and Entailment by GPT-3.5-turbo.
    exact_raw, f1_raw = get_raw_scores(data_pool)

    # NLI by GPT-3.5-turbo on input QA val set on multiple languages.
    # set_trace()
    with tqdm(total=len(data_pool)) as t:
        # for data in data_pool:
        for data in data_pool:
            input_context = template["prompt_input"][conf.lang_map[language if instruction != "en" else "English"]].format(
                                                                                        data["answer"], data["prediction"])
            # print(template["persona_info"][conf.lang_map[language if instruction != "en" else "English"]])
            # print(input_context)
            # set_trace()
            nli_res = utils.get_chatgpt_info(model_name, input_context, temperature, 
                            persona_info=template["persona_info"][conf.lang_map[language if instruction != "en" else "English"]])

            ins = {
                "question_id": data["question_id"],
                "generaed answer": data["prediction"],
                "gold answer": data["answer"],
                "EM score": exact_raw[data["question_id"]],
                "F1 score": f1_raw[data["question_id"]],
                "NLI result": nli_res
            }

            with open(out_file, mode="a+") as fw: 
                data_rec = json.dumps(obj=ins, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)

    utils.jsonl2json(out_file)


def overall_calc(lang_list, instruction=False):
    out_file = os.path.join(args.out_path, "val_2k_overall_eval.json")

    for language in lang_list:
    # for language in list(conf.lang_map.keys()):
        # Load evaluation value file.
        inp_file = os.path.join(args.out_path, "val_2k_{}_pred_eval.json".format(
                            conf.lang_map[language]) if instruction != "en" else \
                            "val_2k_{}_pred_eval_{}.json".format(conf.lang_map[language], instruction))
        
        data_pool = utils.read_json(inp_file)

        # Overall prediction evaluation of F1, EM and NLI scores.
        # Two types of instructions are used: English and [current language].
        if conf.lang_map["English"] in inp_file and language != "English":
            data_pool = nli_to_score(data_pool, language="English")
        else:
            data_pool = nli_to_score(data_pool, language)

        utils.write_json(filename=inp_file, dataset=data_pool)

        # set_trace()
        exact_raw = [data["EM score"] for data in data_pool]
        f1_raw = [data["F1 score"] for data in data_pool]
        nli_raw = [data["NLI score"] for data in data_pool]

        out_eval = make_eval_dict(exact_raw, f1_raw, nli_raw)
        overall = {
                "Langauge": language if conf.lang_map["English"] not in inp_file else f"{language} English",
                "EM score": round(out_eval["exact"], 2),
                "F1 score": round(out_eval["f1"], 2),
                "NLI score": round(out_eval["nli"], 2)
            }
        
        print(overall)
        with open(out_file, mode="a+") as fw: 
            data_rec = json.dumps(obj=overall, ensure_ascii=False)
            fw.write(data_rec + '\n')


if __name__ == '__main__':
    # Evaluate the generated answers with gold answers on F1 measure, Exact Match, and NLI.
    # Language list: ["Chinese", "English", "Japenese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    # lang_list = ["Japenese", "Korean", "French", "Arabic", "German", "Indonesian", "Thai", "Italian"]
    
    # # for language in list(conf.lang_map.keys()):
    # for language in lang_list:
    #     print(f"Evaluating F1, EM and NLI scores on TriviaQA {language} val dataset.")
    #     pred_ans_evals(args.model, args.temperature, language)

    lang_list = ["Thai"]
    overall_calc(lang_list)
