import code_bert_score
import evaluate
import pandas as pd
from lemma_rouge import custom_rouge

# 10 sample paraphrases
with open("data/sample10_phrases.json", "r", encoding="utf-8") as f: 
    df = pd.read_json(f)

# test set paraphrases
# with open("data/gen_phrases.json", "r", encoding="utf-8") as f: 
#     df = pd.read_json(f)

model_types = [c for c in df.columns if c != "p"]

metric_btscore = evaluate.load("bertscore")

for m in model_types:
    df[f"{m}_rouge"] = df.apply(lambda x: custom_rouge(ref=x["p"], gen=x[m])["rougeL"]["f1"]*100, axis=1)
    df[f"{m}_bertscore"] = metric_btscore.compute(predictions=df[m].to_list(), references=df["p"].to_list(), lang="sl")["f1"]
    df[f"{m}_sloberta"] = code_bert_score.score(cands=df[m].to_list(), refs=df["p"].to_list(), model_type="EMBEDDIA/sloberta")[2]

