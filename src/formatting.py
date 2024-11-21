import json
import jsonlines
import os
import pandas as pd
import re


def format_generated_paraphrase(s: str) -> str:
    s = re.sub("(?<=\?|\!|\.)\s.*", "", s.replace("paraphrasedoutput: ", ""))
    return s

def paraphrase_check(p_list: list, p0: str) -> bool:
    """
    Check if two phrases are different. Controls for symbols and cases.
    """
    p0_stripped = re.sub(r'[^A-Za-z0-9 ]+', '', p0).lower()
    for p in p_list:
        p = re.sub(r'[^A-Za-z0-9 ]+', '', p).lower()
        if p == p0_stripped:
            return False
    return True        

def split_text(f_name: str, lang: str="en") -> pd.DataFrame:
    """
    Takes file suffix of text file. 
    Handles different punctuation marks. 
    If there's an end mark inside of a row, it shifts the remainder to the next row.
    Ideally in the end every row represents a single sentence.
    """
    df = pd.read_csv(f"data/aligned/{lang}_{f_name}.txt", header=None, sep="\n|\r\n")
    df_clean = df.apply(lambda x: x.replace(r'^\s*\-*\s*', '', regex=True).replace(r"\.{2,}\s*$", ".", regex=True).replace(r"\.{2,}", '', regex=True))
    df_split = pd.DataFrame(columns=["line"])
    j = 0
    for _, row in df_clean.iterrows():
        line_split = re.split(r'(?<=[\.\!\?])\s', row[0])
        df_split.loc[j, "line"] = line_split[0]
        j += 1
        for l in line_split[1:]:
            df_split.loc[j, "line"] = l
            j += 1
    df_split.to_csv(f"data/formatted/{lang}_{f_name}_fmt.txt", header=None, index=None, mode="a")
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

def filter_phrases(path: str):
    """Looks through .jsonl files in the path folder, removes phrases with more than 70% word matching,
    saves into separate "filtered" folder in the form of "{filename}_filtered.jsonl" file.
    New version is formatted the same, only removes phrases that have >70% matching with a different phrase."""
    data = os.listdir(os.getcwd()+"\\"+path)
    for d in data:
        if len(d.split(".")) == 2: #looking for files, not folders    
            para_list = []
            d_path_name = d.split(".")[0]
            with open("\\".join((path, d)), "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                        line_dict = json.loads(line)
                        p1 = line_dict["phrase"].split()
                        p2 = line_dict["paraphrase"].split()
                        intersect = set(p1).intersection(set(p2))
                        if int(0.7*max(len(p1), len(p2))) >= len(intersect):
                            para_list.append(line_dict)
            with jsonlines.open("/".join((path, "filtered", f"{d_path_name}_f.jsonl")), 'w') as outfile:
                outfile.write_all(para_list)