import deepl
import json
import os
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()
translator = deepl.Translator(os.getenv("AUTH_KEY"))

def read_aligned(f_name: str, lang: str="en") -> pd.DataFrame:
    if lang == "en":
        df = pd.read_csv(f"aligned/en_{f_name}.txt", header=None, sep="\n|\r\n")
    elif lang == "sl":
        df = pd.read_csv(f"aligned/sl_{f_name}.txt", header=None, sep="\n|\r\n", encoding="utf-8")
    return df

def split_text(f_name: str, lang: str="en") -> pd.DataFrame:
    df = read_aligned(f_name, lang)
    df_split = pd.DataFrame(columns=["line"])
    j = 0
    for i, row in df.iterrows():
        line_split = re.split(r'(?<=[\.\!\?])\s', row[0])
        df_split.loc[j, "line"] = line_split[0]
        j += 1
        for l in line_split[1:]:
            df_split.loc[j, "line"] = l
            j += 1
    return df_split
        
def create_trans_df(f_name: str) -> pd.DataFrame:
    """
    Takes file suffix to read aligned slovene and english files.
    Creates and returns dataframe with the slovene source phrase, english source phrase and its slovene translation.
    """
    sl_f = read_aligned(f_name, lang="sl")
    en_f = read_aligned(f_name)
    para_df = pd.DataFrame(index=[0], columns=["sl_source", "en_source", "sl_trans"])

    for i, row in en_f.iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        para_df.loc[i] = [sl_f.loc[i, 0], row[0], sl_trans]
    return para_df 

<<<<<<< HEAD
f_suffix = "zh"
=======

f_suffix = "ab"
#df_split = split_text(f_suffix)

>>>>>>> dev
para_df = create_trans_df(f_suffix)
para_df_clean = para_df.applymap(lambda x: re.sub("^-", "", x).strip())
# remove entries with identical translations
para_df_unique = para_df_clean.loc[para_df_clean["sl_source"] != para_df_clean["sl_trans"], :]
with open(f"interim/paras_{f_suffix}.json", "w", encoding="utf-8") as f:
    json.dump(para_df_unique.to_dict(orient="index"), f, ensure_ascii=False)
