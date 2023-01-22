import deepl
import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
auth_key = os.getenv("AUTH_KEY")
translator = deepl.Translator(auth_key)

eng = pd.read_csv("aligned/en_ab.txt", header=None, sep='\n')
orig = pd.read_csv("aligned/sl_ab.txt", header=None, encoding="utf-8", sep='\n')
para_dict = dict()
for i, row in eng.loc[:5,:].iterrows():
    para_dict[orig.loc[i,0]] = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
with open("paras.json", "w") as f:
    json.dump(para_dict, f)
pass