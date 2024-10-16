r"""
Author: XUE Boyang      Filename: preparation.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Parse and save input QA val dataset.
"""
import argparse
import json

import datasets
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="common", choices=["trivia", "gsm8k", "common", "sciq"])
parser.add_argument('--out_file', type=str, default="./data/{}/val_en.json")
args = parser.parse_args()

seed_value = 10


def get_trivia_dataset(split='validation'):
    print('Preprocessing TriviaQA dataset')
    val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    id_mem = set()

    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])

        return batch

    val_data = val_data.map(remove_dups, batch_size=1, batched=True, 
                            load_from_cache_file=False, remove_columns=["search_results", "question_source", "entity_pages"])
    # import pdb; pdb.set_trace()

    # Warrant the duplicated data was removed
    assert pd.Series([_['question_id'] for _ in val_data]).value_counts().max() == 1

    ins_set = []
    for data in val_data:
        ins = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"]["value"]
        }
        ins_set.append(ins)

    with open(args.out_file.format(args.dataset), "w") as fw:
        json.dump(fp=fw, obj=ins_set, indent=4, ensure_ascii=False)


def get_sciq_dataset(split="validation"):
    print('Preprocessing SciQ dataset')
    val_data = datasets.load_dataset("sciq", split=split)
    ins_set = []
    for data in val_data:
        # import pdb; pdb.set_trace()
        ins = {
            "question": data["question"],
            "answer": data["correct_answer"]
        }
        ins_set.append(ins)

    with open(args.out_file.format(args.dataset), "w") as fw:
        json.dump(fp=fw, obj=ins_set, indent=4, ensure_ascii=False)


def get_gsm8k_dataset():
    print('Preprocessing GSM8K dataset')
    with open("./data/gsm8k/test.jsonl", "r") as fr:
        val_data = [json.loads(line) for line in fr.readlines()]

    ins_set = []
    for data in val_data:
        ins = {
            "question": data["question"],
            "answer": data["answer"]
        }
        ins_set.append(ins)

    with open(args.out_file.format(args.dataset), "w") as fw:
        json.dump(fp=fw, obj=ins_set, indent=4, ensure_ascii=False)


def get_common_dataset():
    print('Preprocessing CommonsenseQA dataset')
    with open("./data/common/dev_rand_split.jsonl", "r") as fr:
        val_data = [json.loads(line) for line in fr.readlines()]

    ins_set = []
    for data in val_data:
        choice_list = []
        for choice in data["question"]["choices"]:
            choice_list.append(choice["text"])
            if choice["label"] == data["answerKey"]:
                answer = choice["text"]
            choices = "\n".join(choice_list)

        question = data["question"]["stem"] + "Select one from the following choices\n\n" + choices
        ins = {
            "question": question,
            "answer": answer
        }
        ins_set.append(ins)

    with open(args.out_file.format(args.dataset), "w") as fw:
        json.dump(fp=fw, obj=ins_set, indent=4, ensure_ascii=False)


if __name__=="__main__":
    if args.dataset == "trivia":
        get_trivia_dataset(split='validation')
    elif args.dataset == "gsm8k":
        get_gsm8k_dataset()
    elif args.dataset == "common":
        get_common_dataset()
    elif args.dataset == "sciq":
        get_sciq_dataset(split='validation')

