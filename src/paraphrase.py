import json
import os
import pandas as pd

from modeling import get_paraphrase_chatgpt

def get_paraphrase_df(df: pd.DataFrame):
    df = df.assign(en_para_1=None, en_para_2=None)
    for i, row in df.iterrows():
        if len(row["en_source"].split(" ")) == 1:
            continue
        paraph_list = get_paraphrase_chatgpt(row["en_source"])
        for j, phrase in enumerate(paraph_list):
            df.loc[i, f"en_para_{j+1}"] = phrase
    df = df.dropna(subset=["en_para_1"])
    return df


path = os.getcwd()
data_path = os.path.join(path, "data")

f_suffix = "zz"
interim_path = os.path.join(data_path, f"interim\\paras_{f_suffix}.json")
df_interim = pd.read_json(interim_path, orient="index", encoding="utf-8")
df_en_para = get_paraphrase_df(df_interim)
df_en_para.to_pickle(f"{f_suffix}_paras_chatgpt.pkl")
with open(f"data/gpt_paraphrase/{f_suffix}_para_gpt.json", "w", encoding="utf-8") as j:
    json.dump(df_en_para.to_dict(orient="index"), j, ensure_ascii=False)
