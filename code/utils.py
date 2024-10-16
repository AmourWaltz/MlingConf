# import openai
import os
from tqdm import tqdm, trange
from ipdb import set_trace
import json
import math

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = open("./../Materials/openai_api.txt", "r").readline().strip()

client = OpenAI(api_key=open("./code/openai_api.txt", "r").readline().strip())

# print(open("./code/openai_api.txt", "r").readline().strip())
# openai.api_key = open("./code/openai_api.txt", "r").readline().strip()

language_list = ["English", "Chinese", "Japanese", "French", "Thai"]

lang_map = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "French": "fr",
    "Arabic": "ar",
    "German": "de",
    "Korean": "ko",
    "Indonesian": "id",
    "Thai": "th",
    "Italian": "it"
}


lang_true_false_dict = {
    "English": ["Yes", "No"],
    "Chinese": ["是", "否"],
    "Japanese": ["はい", "いいえ"],
    "French": ["Oui", "Non"],
    "Arabic": ["نعم", "لا"],
    "German": ["Ja", "Nein"],
    "Korean": ["예", "아니요"],
    "Indonesian": ["Ya", "Tidak"],
    "Thai": ["ใช่", "ไม่ใช่"],
    "Italian": ["Sì", "No"]
}


lang_aware_dict = {
    "en": ["True", "False"],
    "zh": ["正确", "错误"],
    "ja": ["真", "誤り"],
    "fr": ["Vrai", "Faux"],
    "ar": ["صحيح", "خطأ"],
    "de": ["Richtig", "Falsch"],
    "ko": ["참", "거짓"],
    "id": ["Benar", "Salah"],
    "th": ["จริง", "เท็จ"],
    "it": ["Vero", "Falso"]
}

lang_ques_ans_dict = {
    "en": ["Question", "Answer"],
    "Chinese": ["问题", "答案"],
    "Japanese": ["質問", "答え"],
    "French": ["Question", "Réponse"],
    "Arabic": ["سؤال", "إجابة"],
    "German": ["Frage", "Antwort"],
    "Korean": ["질문", "답변"],
    "Indonesian": ["Pertanyaan", "Jawaban"],
    "Thai": ["คำถาม", "คำตอบ"],
    "Italian": ["Domanda", "Risposta"]
}


word_conf_score = {
    "lowest": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "highest": 1.0,
    "unknown": 0.5
}



def get_chatgpt_info(model_name: str, 
                     input_context: str, 
                     temperature: float, 
                     max_tokens: int,
                     logprobs=True,
                     persona_info="You are an excellent question responder.") -> str:
    
    # print(persona_info)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": persona_info,
            },
            {
                "role": "user",
                "content": input_context,
            },
        ],
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        # logprobs=True,
        # top_logprobs=1,
    )
    response_message = response.choices[0].message.content
    # print(f"logprobs: {logprobs}")
    # print(response_message)
    # print(response)
    
    if logprobs:
        # set_trace()
        logprobs = []
        for token_logprob in response.choices[0].logprobs.content:
            logprobs.append(token_logprob.logprob)

        return response_message, parse_logprobs(logprobs)
    else:
        return response_message, [0.5] # only output texts


def parse_logprobs(token_list):
    return [math.exp(x) for x in token_list]
    # log_dict = {}
    # for token in token_list:
    #     log_dict[token["token"]] = math.exp(token["logprob"])

    # return log_dict


"""
Json utils
"""
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = json.load(fr)

    return data_pool


def read_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = [json.loads(line) for line in fr.readlines()]

    return data_pool


def write_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        fw.writelines([json.dumps(obj=ins, ensure_ascii=False) + '\n' for ins in dataset])


def write_json(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(fp=fw, obj=dataset, indent=4, ensure_ascii=False)


def jsonl2json(file1, file2):
    write_json(file2, read_jsonl(file1))


def json2jsonl(file1, file2):
    dataset = read_json(file1)
    write_jsonl(file2, dataset)


def json_merge(files, out_file):
    data = []
    for file in files:
        data += read_json(file)
    write_json(out_file, data)


def read_jsons(files):
    data = []
    for file in files:
        data += read_json(file)
    return data


# Time utils
def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)  # 1h = 3600s
    minutes, seconds = divmod(remainder, 60)  # 1m = 60s
    return [int(hours), int(minutes), int(seconds)]


def get_current_time():
    tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')


"""
Model parameters utils
"""
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )
