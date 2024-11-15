import code_bert_score
import evaluate
import pandas as pd
from lemma_rouge import custom_rouge
import json

metric_btscore = evaluate.load("bertscore")

#with open("data/sample10_phrases.json", "r", encoding="utf-8") as f: 
#    phrases = json.load(f)

with open("data/gen_phrases.json", "r", encoding="utf-8") as f: 
    phrases = json.load(f)

def get_metrics_df(phrases: dict, metrics: list[str]) -> pd.DataFrame:
    
    def get_metric(phrases: list[str], metric: str) -> float:
        if metric == "rouge":
            val = custom_rouge(ref=phrases[0], gen=phrases[1])["rougeL"]["f1"]
        elif metric == "mbertscore":
            val = metric_btscore.compute(predictions=[phrases[1]], references=[phrases[0]], lang="sl")["f1"][0]
        elif metric == "sloberta":
            val = [t.item() for t in code_bert_score.score(cands=[phrases[1]], refs=[phrases[0]], model_type="EMBEDDIA/sloberta")][2]
        return val
    
    models = [k for k in phrases.keys() if "p" != k]
    n_phr = len(phrases["p"])
    df = pd.DataFrame(columns=["model", "ref_fraza", "gen_fraza"]+metrics, index=range(n_phr*len(models)))
    for i in range(n_phr):
        for j, model_type in enumerate(models):
            df.loc[i*3+j, ["model", "ref_fraza", "gen_fraza"]] = [model_type, phrases["p"][i], phrases[model_type][i]]
            for m in metrics:
                df.loc[i*3+j, m] = get_metric([phrases["p"][i], phrases[model_type][i]], m)
    return df

#df_m = get_metrics_df(phrases, ["rouge", "mbertscore", "sloberta"])