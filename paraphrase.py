import deepl
import json
import os
import pandas as pd
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

load_dotenv()
translator = deepl.Translator(os.getenv("AUTH_KEY"))


def en_paraphrase(df: pd.DataFrame, n_para: int=3):
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
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sent = sent.replace("paraphrasedoutput: ", "")
            if sent not in en_phrases:
                en_phrases.append(sent)
                df.loc[i, f"en_para_{j}"] = sent
                j += 1
    return df



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
    df_clean = df.apply(lambda x: x.replace(r'^\s*\-*\s*', '', regex=True).replace(r"\.{2,}\s*$", ".", regex=True).replace(r"\.{2,}", '', regex=True))
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
    for i, row in en_df.iterrows():
        sl_trans = translator.translate_text(row[0], source_lang="EN", target_lang="SL").text
        if re.sub(r'\W+', '', sl_df.loc[i,0]) != re.sub(r'\W+', '', sl_trans):
            para_df.loc[i] = [sl_df.loc[i, 0], row[0], sl_trans]
    return para_df.dropna(how="all") 


def compare_phrases(f_name: str):
    model = SentenceTransformer('sentence-transformers/LaBSE')
    with open(f'interim/paras_{f_name}.json', encoding="utf-8") as json_file:
        d = json.load(json_file)
    df_sims = pd.DataFrame.from_dict(d, orient="index")
    for i, row in df_sims.iterrows():
        s = [row["sl_source"], row["sl_trans"]]
        e = model.encode(s)
        df_sims.loc[i, "cos_sim"] = util.pytorch_cos_sim(e[0], e[1]).item()
    df_sims = df_sims.sort_values("cos_sim")
    with open(f"processed/paras_{f_name}_sims.json", "w", encoding="utf-8") as f:
        json.dump(df_sims.to_dict(orient="index"), f, ensure_ascii=False)

    

f_suffix = "zz"

# prep for aligning
# split_text(f_suffix)
# split_text(f_suffix, lang="sl")

# translating
# sl_df = pd.read_csv(f"formatted/sl_{f_suffix}_fmt.txt", header=None, sep="\n|\r\n", encoding="utf-8")
# en_df = pd.read_csv(f"formatted/en_{f_suffix}_fmt.txt", header=None, sep="\n|\r\n")
# para_df = create_trans_df(sl_df, en_df, f_suffix)

# with open(f"interim/paras_{f_suffix}.json", "w", encoding="utf-8") as f:
#     json.dump(para_df.to_dict(orient="index"), f, ensure_ascii=False)

# run english paraphrase generator
with open(f'interim/paras_{f_suffix}.json', encoding="utf-8") as json_file:
    df = pd.DataFrame.from_dict(json.load(json_file), orient="index")
df_para = en_paraphrase(df)
with open(f"processed/paras_{f_suffix}_added.json", "w", encoding="utf-8") as f:
    json.dump(df_para.to_dict(orient="index"), f, ensure_ascii=False)
pass



