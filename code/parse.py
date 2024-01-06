r"""
Author: XUE Boyang      Filename: parse.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Parse and save input QA val dataset.
"""
import argparse
import json

import datasets
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--out_file', type=str, default="./data/trivia/val_en.json")
args = parser.parse_args()

seed_value = 10


def get_dataset(split='validation'):
    print('Preprocessing dataset')
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

    with open(args.out_file, "w") as fw:
        json.dump(fp=fw, obj=ins_set, indent=4, ensure_ascii=False)


if __name__=="__main__":
    get_dataset()

