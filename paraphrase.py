import deepl
import itertools
import json
import numpy as np
import os
import pandas as pd
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

load_dotenv()
translator = deepl.Translator(os.getenv("AUTH_KEY"))

model = SentenceTransformer('sentence-transformers/LaBSE')

def format_sentence(s: str) -> str:
    s = re.sub("(?<=\?|\!|\.)\s.*", "", s.replace("paraphrasedoutput: ", ""))
    return s


def add_paraphrase(p_list: list, p0: str) -> bool:
    p0_stripped = re.sub(r'[^A-Za-z0-9 ]+', '', p0).lower()
    for p in p_list:
        p = re.sub(r'[^A-Za-z0-9 ]+', '', p).lower()
        if p == p0_stripped:
            return False
    return True        



def en_paraphrase(df: pd.DataFrame, n_para: int=3):
    model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
    tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

    device = torch.device("cpu")
    model = model.to(device)

    for i, row in df.iterrows():
        phrase = row["en_source"]
        encoding = tokenizer.encode_plus(phrase, max_length=128, padding=True, return_tensors="pt")
        input_ids, attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        model.eval()
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            early_stopping=True,
            num_beams=15,
            num_return_sequences=n_para
        )
        en_phrases = [row["en_source"]]
        j = 1
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sent_f = format_sentence(sent)
            if add_paraphrase(en_phrases, sent_f):
                en_phrases.append(sent_f)
                df.loc[i, f"en_para_{j}"] = sent_f
                j += 1
    return df


def read_aligned(f_name: str, lang: str="en") -> pd.DataFrame:
    if lang == "en":
        df = pd.read_csv(f"aligned/en_{f_name}.txt", header=None, sep="\n|\r\n")
    elif lang == "sl":
        df = pd.read_csv(f"aligned/sl_{f_name}.txt", header=None, sep="\n|\r\n", encoding="utf-8")
    return df

def split_text(f_name: str, lang: str="en") -> pd.DataFrame:
    """
    Takes file suffix of text file. 
    Handles different punctuation marks. 
    If there's an end mark inside of a row, it shifts the remainder to the next row.
    Ideally in the end every row represents a sentence.
    #TODO: exclude Mr., Mrs. and such special cases. 
    """
    df = read_aligned(f_name, lang)
    df_clean = df.apply(lambda x: x.replace(r'^\s*\-*\s*', '', regex=True).replace(r"\.{2,}\s*$", ".", regex=True).replace(r"\.{2,}", '', regex=True))
    df_split = pd.DataFrame(columns=["line"])
    j = 0
    for i, row in df_clean.iterrows():
        line_split = re.split(r'(?<=[\.\!\?])\s', row[0])
        df_split.loc[j, "line"] = line_split[0]
        j += 1
        for l in line_split[1:]:
            df_split.loc[j, "line"] = l
            j += 1
    df_split.to_csv(f"formatted/{lang}_{f_name}_fmt.txt", header=None, index=None, mode="a")
    return

def create_trans_df(sl_df: pd.DataFrame, en_df: pd.DataFrame, f_name: str) -> pd.DataFrame:
    """
    Takes file suffix to read aligned slovene and english files.
    Creates and returns dataframe with the slovene source phrase, english source phrase and its slovene translation.
    """
    para_df = pd.DataFrame(index=[0], columns=["sl_source", "en_source", "sl_trans"])
    for i, row in en_df.iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        if re.sub(r'\W+', '', sl_df.loc[i,0]) != re.sub(r'\W+', '', sl_trans):
            para_df.loc[i] = [sl_df.loc[i, 0], row[0], sl_trans]
    return para_df.dropna(how="all") 

def compare_df(f_name:str):
    with open(f'interim/paras_{f_name}.json', encoding="utf-8") as json_file:
        d = json.load(json_file)
    df_sims = pd.DataFrame.from_dict(d, orient="index")
    for i, row in df_sims.iterrows():
        s = [row["sl_source"], row["sl_trans"]]
        df_sims.loc[i, "cos_sim"] = compare_phrases(s)
    df_sims = df_sims.sort_values("cos_sim")
    with open(f"processed/paras_{f_name}_sims.json", "w", encoding="utf-8") as f:
        json.dump(df_sims.to_dict(orient="index"), f, ensure_ascii=False)

def compare_phrases(phr_list: list[str]) -> float:
    e = model.encode(phr_list)
    return util.pytorch_cos_sim(e[0], e[1]).item()
    

def slice_files(f_name: str, batch_size: int=100):
    """
    Function for slicing .json files into ones containing one batch of data.
    Needed because processing larger files takes too much time at once. 
    """
    f = f"data/interim/paras_{f_name}.json"
    with open(f, encoding="utf-8") as json_file:
        df_whole = pd.DataFrame.from_dict(json.load(json_file), orient="index")
        i = 1
        sliced = False
        while not sliced:
            with open(f"data/sliced/paras_{f_name}_{i}.json", "w", encoding="utf-8") as sliced_f:
                if len(df_whole.index) > batch_size:
                    df = df_whole.iloc[:batch_size]
                    df_whole = df_whole.iloc[batch_size:]
                else:
                    df = df_whole
                    sliced = True
                json.dump(df.to_dict(orient="index"), sliced_f, ensure_ascii=False)
                i += 1
    
def group_sl_paraphrases(f_name:str) -> pd.DataFrame:
    df_slo_paras = pd.DataFrame(columns=["phrase", "paraphrase", "sem_sim"])
    g = {}
    with open(f'data/translated/trans_{f_name}.json', encoding="utf-8") as json_file:
        d = json.load(json_file)
    i = 0
    for _, v in d.items():
        para_list = [v[k] for k, val in v.items() if "sl" in k]
        for combo in itertools.permutations(para_list, 2):
            df_slo_paras.loc[i, :] = [combo[0], combo[1], 0]
            i += 1
    return df_slo_paras

f_suffix = "zh"

# prep for aligning
# split_text(f_suffix)
# split_text(f_suffix, lang="sl")

sl_paras = group_sl_paraphrases(f_suffix)
for i, row in sl_paras.iterrows():
    sl_paras.loc[i, "sem_sim"] = compare_phrases([row["phrase"], row["paraphrase"]])
pass

with open(f"data/slo_parafraze/sim_computed/{f_suffix}_pairs_sim.json", "w", encoding="utf-8") as j:
    json.dump(sl_paras.to_dict(orient="index"), j, ensure_ascii=False)

pass



# translating
with open(f'data/processed/paras_{f_suffix}.json', encoding="utf-8") as json_file:
    df = pd.DataFrame.from_dict(json.load(json_file), orient="index")
df = df.assign(sl_para_1=None, sl_para_2=None, sl_para_3=None)
for i, row in df.iterrows():
    sl_phrases = [row["sl_source"], row["sl_trans"]]
    for j in range(1,4):
        if row[f"en_para_{j}"] == row[f"en_para_{j}"]:
            t = translator.translate_text(row[f"en_para_{j}"], source_lang="EN", target_lang="SL").text
            if add_paraphrase(sl_phrases, t):
                sl_phrases.append(t)
                df.loc[i, f"sl_para_{j}"] = t

pass
with open(f"data/translated/trans_{f_suffix}.json", "w", encoding="utf-8") as j:
    json.dump(df.to_dict(orient="index"), j, ensure_ascii=False)

i = 1
df_processed = pd.DataFrame()
while i <= 4:    
    with open(f'data/sliced/paras_{f_suffix}_{i}.json', encoding="utf-8") as json_file:
        df = pd.DataFrame.from_dict(json.load(json_file), orient="index")
        df_processed = pd.concat([df_processed, en_paraphrase(df)])
        with open(f"data/processed/paras_{f_suffix}_{i}.json", "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="index"), f, ensure_ascii=False)
    i += 1
with open(f"data/processed/paras_{f_suffix}.json", "w", encoding="utf-8") as f:
    json.dump(df_processed.to_dict(orient="index"), f, ensure_ascii=False)
pass



# run english paraphrase generator
with open(f'interim/paras_{f_suffix}.json', encoding="utf-8") as json_file:
    df = pd.DataFrame.from_dict(json.load(json_file), orient="index")
df_para = en_paraphrase(df)
with open(f"processed/paras_{f_suffix}_added.json", "w", encoding="utf-8") as f:
    json.dump(df_para.to_dict(orient="index"), f, ensure_ascii=False)
pass



