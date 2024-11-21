import os
import re

import deepl
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI, api_key, completions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from formatting import format_generated_paraphrase, paraphrase_check

load_dotenv()
translator = deepl.Translator(os.getenv("DEEPL_KEY"))
api_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=api_key)

def get_paraphrase_chatgpt(phrase: str, num_p: int=2, max_tokens: int=100, lang: str="eng"):

    if lang=="eng":
        p = f"Create {num_p} slightly colloquial paraphrases of {phrase}"
    elif lang == "si": 
        p = f"Ustvari {num_p} parafraze {phrase}"
    
    # for text-davinci-003
    response = completions.create(model="text-davinci-003", prompt=p, max_tokens=max_tokens)
    r_clean = re.sub("\\n\d?\.?", "", response.choices[0].text)
    r_list = re.split("(?<=(\.|\!|\?))\W", r_clean)
    t_list = [re.sub("^\W+", "", x) for x in r_list if len(x) > 1]
    return t_list

def generate_en_paraphrases(df: pd.DataFrame, n_para: int=3):
    """Generating paraphrases with old model."""
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
            sentence = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sentence_f = format_generated_paraphrase(sentence)
            if paraphrase_check(en_phrases, sentence_f):
                en_phrases.append(sentence_f)
                df.loc[i, f"en_para_{j}"] = sentence_f
                j += 1
    return df

def translate_paraphrases(df: pd.DataFrame, api: str) -> pd.DataFrame:
    """Translates given english phrases into Slovene. Usage of two APIs possible, either deepl or chatGPT.
    Returns dataframe containing both original english phrases and slovenian translations."""
    df_trans = df.copy()
    for i, row in df_trans.iterrows():
        phrase_list = [row["sl_source"]]
        en_para_list = [f"en_para_{k}" for k in range(1,5)]
        if not "sl_trans" in df.columns:
            en_para_list.append("en_source")
        else:
            phrase_list.append(row["sl_trans"])
        for k, en_phr in enumerate(en_para_list):
            if row[en_phr] == row[en_phr]:
                if api == "deepl":
                    sl_trans = translator.translate_text(row[en_phr], source_lang="EN", target_lang="SL").text
                elif api == "chat_gpt":
                    p = f"Translate {row[en_phr]} into slovene."
                    response = completions.create(model="text-davinci-003", prompt=p, max_tokens=50)
                    sl_trans = re.sub("\\n\d?\.?", "", response.choices[0].text)
                if not sl_trans in phrase_list:
                    df_trans.loc[i, f"sl_para_{k+1}"] = sl_trans
                    phrase_list.append(sl_trans)
                    
    return df_trans
