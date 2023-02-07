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
    """
    Takes file suffix of text file. 
    Handles different punctuation marks. 
    If there's an end mark inside of a row, it shifts the remainder to the next row.
    Ideally in the end every row represents a sentence.
    #TODO: exclude Mr., Mrs. and such special cases. 
    """
    df = read_aligned(f_name, lang)
    df_clean = df.apply(lambda x: x.replace(r'^\s*\"*\s*\-*\s*', '', regex=True).replace(r"\.{2,}\s*$", ".", regex=True).replace(r"\.{2,}", "", regex=True))
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
    for i, row in en_df.loc[495:, :].iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        if sl_df.loc[i, 0] != sl_trans:
            para_df.loc[i] = [sl_df.loc[i, 0], row[0], sl_trans]
    return para_df.dropna(how="all") 


f_suffix = "ut"

sl_df = pd.read_csv(f"formatted/sl_{f_suffix}_fmt.txt", header=None, sep="\n|\r\n", encoding="utf-8")
en_df = pd.read_csv(f"formatted/en_{f_suffix}_fmt.txt", header=None, sep="\n|\r\n")

para_df = create_trans_df(sl_df, en_df, f_suffix)
# remove entries with identical translations
with open(f"interim/paras_{f_suffix}.json", "w", encoding="utf-8") as f:
    json.dump(para_df.to_dict(orient="index"), f, ensure_ascii=False)



