import os
import pandas as pd

from modeling import get_paraphrase_chatgpt

def get_paraphrase_df(df: pd.DataFrame):
    df = df.assign(en_para_1=None, en_para_2=None, en_para_3=None)
    for i, row in df.iterrows():
        paraph_list = get_paraphrase_chatgpt(row["en_source"])
        for j, phrase in enumerate(paraph_list):
            df.loc[i, f"en_para_{j+1}"] = phrase
    return df


path = os.getcwd()
data_path = os.path.join(path, "data")

f_suffix = "ab"
interim_path = os.path.join(data_path, f"interim\\paras_{f_suffix}.json")
df_interim = pd.read_json(interim_path, orient="index", encoding="utf-8")
df_en_para = get_paraphrase_df(df_interim)