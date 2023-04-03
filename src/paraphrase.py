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

from formatting import pair_sl_paraphrases
from modeling import compute_phrase_sem_sim, generate_en_paraphrases, translate_paraphrases

path = os.getcwd()

f_suffix = "ut"
fmt_path = os.path.join(path, f"data\\formatted\\en_{f_suffix}_fmt.txt")

df_en_aligned = pd.read_csv(fmt_path, header=None, sep="\n|\r\n").rename(columns={0:"en_source"})
df_sl_aligned = pd.read_csv(fmt_path, header=None, sep="\n|\r\n").rename(columns={0:"sl_source"})

df_en_paras = generate_en_paraphrases(df_en_aligned)
df_sl_orig = pd.concat([df_en_paras, df_sl_aligned], axis=1)

df_full_trans = translate_paraphrases(df_sl_orig)


# prep for aligning
# split_text(f_suffix)
# split_text(f_suffix, lang="sl")

sl_paras = pair_sl_paraphrases(f_suffix)
for i, row in sl_paras.iterrows():
    sl_paras.loc[i, "sem_sim"] = compute_phrase_sem_sim([row["phrase"], row["paraphrase"]])
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



