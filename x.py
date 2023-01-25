import deepl
import json
import os
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()
translator = deepl.Translator(os.getenv("AUTH_KEY"))

def create_df(f_name: str) -> pd.DataFrame:
    """
    Takes file suffix to read aligned slovene and english files.
    Creates and returns dataframe with the slovene source phrase, english source phrase and its slovene translation.
    """
    sl_f = pd.read_csv(f"aligned/sl_{f_name}.txt", header=None, sep="\n", encoding="utf-8")
    en_f = pd.read_csv(f"aligned/en_{f_name}.txt", header=None, sep="\n")
    para_df = pd.DataFrame(index=[0], columns=["sl_source", "en_source", "sl_trans"])
    for i, row in en_f.iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        para_df.iloc[len(para_df.index)] = [sl_f.loc[i, 0], row[0], sl_trans]
    return para_df 


f_suffix = "ah"
para_df = create_df(f_suffix)
para_df_clean = para_df.applymap(lambda x: re.sub("^-", "", x).strip())
# remove entries with identical translations
para_df_unique = para_df_clean.loc[para_df_clean["sl_source"] != para_df_clean["sl_trans"], :]
with open(f"interim/paras_{f_suffix}.json", "w", encoding="utf-8") as f:
    json.dump(para_df_unique, f)


