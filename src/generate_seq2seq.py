import code_bert_score
import evaluate
import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from utils.lemma_rouge import custom_rouge

df = pd.read_json("data/test_largest.jsonl", lines=True)

metric_btscore = evaluate.load("bertscore")

tokenizer = T5Tokenizer.from_pretrained("aluminij/paragen_sloT5")
model = T5ForConditionalGeneration.from_pretrained("aluminij/paragen_sloT5")
prefix = "generiraj parafrazo: "

gen_text = pd.DataFrame()


def choose_para(sent: str, i_repeat: int) -> str:
    input_ids = tokenizer(
        f"{prefix} {sent}", return_tensors="pt", max_length=128
    ).input_ids

    def generate_para(input_ids, i: int) -> list[str]:
        temp = 0.8 + 0.1 * i
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=2,
            temperature=temp,
            num_return_sequences=3,
        )

        decoded_preds = [
            tokenizer.decode(o.cpu(), skip_special_tokens=True) for o in outputs
        ]
        return decoded_preds

    for i_repeat in range(10):
        decoded_preds = generate_para(input_ids, i_repeat)
        s = pd.Series(decoded_preds).value_counts()
        top_s = [x for x in s.index.to_list() if x != sent]
        if len(top_s) > 0:
            break
    if len(top_s) == 0:
        top_s = [sent]
    return top_s[0]


df["prediction"] = df["phrase"].apply(lambda x: choose_para(x, 10))
df["bertscore"] = metric_btscore.compute(
    predictions=df["prediction"], references=df["phrase"], lang="sl"
)["f1"]
df["sloberta"] = code_bert_score.score(
    cands=df["prediction"],
    refs=df["phrase"],
    model_type="EMBEDDIA/sloberta",
)[2].tolist()
df["rouge"] = df.apply(
    lambda x: custom_rouge(ref=x["phrase"], gen=x["prediction"])["rougeL"]["f1"] * 100,
    axis=1,
)