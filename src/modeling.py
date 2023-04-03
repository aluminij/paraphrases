import deepl
import json
import os
import pandas as pd
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

from formatting import format_generated_paraphrase, paraphrase_check

load_dotenv()
translator = deepl.Translator(os.getenv("AUTH_KEY"))

model = SentenceTransformer('sentence-transformers/LaBSE')


def generate_en_paraphrases(df: pd.DataFrame, n_para: int=3):
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


def translate_paraphrases(df: pd.DataFrame) -> pd.DataFrame:
    df_trans = df.copy()
    for i, row in df_trans.iterrows():
        phrase_list = [row["sl_source"]]
        j = 1
        for en_phr in ["en_source", "en_para_1", "en_para_2", "en_para_3"]:
            sl_trans = translator.translate_text(row[en_phr], source_lang="EN", target_lang="SL").text
            if not sl_trans in phrase_list:
                df_trans.loc[i, f"sl_para_{j}"] = sl_trans
                phrase_list.append(sl_trans)
                j += 1
    return df_trans



def create_trans_df(sl_df: pd.DataFrame, en_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates and returns dataframe with the slovene source phrase, english source phrase and its slovene translation.
    """
    para_df = pd.DataFrame(index=[0], columns=["sl_source", "en_source", "sl_trans"])
    for i, row in en_df.iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        if re.sub(r'\W+', '', sl_df.loc[i,0]) != re.sub(r'\W+', '', sl_trans):
            para_df.loc[i] = [sl_df.loc[i, 0], row[0], sl_trans]
    return para_df.dropna(how="all") 

def compute_df_sem_sim(f_name: str):
    """ 
    Computes semantic similarity for input .json file."""
    with open(f'interim/paras_{f_name}.json', encoding="utf-8") as json_file:
        d = json.load(json_file)
    df_sims = pd.DataFrame.from_dict(d, orient="index")
    for i, row in df_sims.iterrows():
        s = [row["sl_source"], row["sl_trans"]]
        df_sims.loc[i, "cos_sim"] = compute_phrase_sem_sim(s)
    df_sims = df_sims.sort_values("cos_sim")
    with open(f"processed/paras_{f_name}_sims.json", "w", encoding="utf-8") as f:
        json.dump(df_sims.to_dict(orient="index"), f, ensure_ascii=False)

def compute_phrase_sem_sim(phr_list: list[str]) -> float:
    """
    Compute semantic similarity for two phrases.
    """
    e = model.encode(phr_list)
    return util.pytorch_cos_sim(e[0], e[1]).item()



