import deepl
import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
translator = deepl.Translator(os.getenv("AUTH_KEY"))

def create_dict(f_name: str) -> dict[str, dict]:
    """
    Takes file prefix to read aligned slovene and english files.
    Creates and returns dictionary where keys are slovene source phrases, values are
    dictionary with english source phrases and their translations.
    """
    sl_f = pd.read_csv(f"aligned/sl_{f_name}.txt", header=None, sep="\n", encoding="utf-8")
    en_f = pd.read_csv(f"aligned/en_{f_name}.txt", header=None, sep="\n")
    para_dict = dict()
    for i, row in en_f.loc[:10, :].iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        para_dict[i] = {"sl_source": sl_f.loc[i, 0].strip(), "eng_source": row[0].strip(), "sl_trans": sl_trans}
    return para_dict

f_prefix = "ab"
para_dict = create_dict(f_prefix)
# remove entries with identical translations
para_dict2 = {k: v for k, v in para_dict.items() if k != v["sl_trans"]}
with open(f"interim/paras_{f_prefix}.json", "w", encoding="utf-8") as f:
    json.dump(para_dict, f)

