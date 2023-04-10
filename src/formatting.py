import itertools
import json
import pandas as pd
import re


def format_generated_paraphrase(s: str) -> str:
    s = re.sub("(?<=\?|\!|\.)\s.*", "", s.replace("paraphrasedoutput: ", ""))
    return s

def paraphrase_check(p_list: list, p0: str) -> bool:
    """
    Check if two phrases are different. Strips of symbols and cases.
    """
    p0_stripped = re.sub(r'[^A-Za-z0-9 ]+', '', p0).lower()
    for p in p_list:
        p = re.sub(r'[^A-Za-z0-9 ]+', '', p).lower()
        if p == p0_stripped:
            return False
    return True        


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

def slice_files(f_name: str, batch_size: int=100):
    """
    Function for slicing .json files into ones containing one batch of data.
    Needed because processing larger files takes too much time at once. 
    """
    f = f"data/interim/paras_{f_name}.json"
    with open(f, encoding="utf-8") as json_file:
        df_whole = pd.DataFrame.from_dict(json.load(json_file), orient="index")
        i = 1
        sliced = False
        while not sliced:
            with open(f"data/sliced/paras_{f_name}_{i}.json", "w", encoding="utf-8") as sliced_f:
                if len(df_whole.index) > batch_size:
                    df = df_whole.iloc[:batch_size]
                    df_whole = df_whole.iloc[batch_size:]
                else:
                    df = df_whole
                    sliced = True
                json.dump(df.to_dict(orient="index"), sliced_f, ensure_ascii=False)
                i += 1
    
def pair_sl_paraphrases(f_name:str) -> pd.DataFrame:
    df_slo_paras = pd.DataFrame(columns=["phrase", "paraphrase", "sem_sim"])
    g = {}
    with open(f'data/translated/trans_{f_name}.json', encoding="utf-8") as json_file:
        d = json.load(json_file)
    i = 0
    for _, v in d.items():
        para_list = [v[k] for k, val in v.items() if ("sl" in k) and (v[k] == v[k])]
        for combo in itertools.permutations(para_list, 2):
            df_slo_paras.loc[i, :] = [combo[0], combo[1], 0]
            i += 1
    return df_slo_paras



