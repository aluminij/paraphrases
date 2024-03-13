import evaluate
import pandas as pd
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

data_files = {
    "train": "paired_1_datasets/train.jl",
    "val": "paired_1_datasets/eval.jl",
    "test": "paired_1_datasets/test.jl"
}
new_para = load_dataset('json', data_files=data_files)

metric_btscore = evaluate.load("bertscore")
metric_rouge = evaluate.load("rouge")

model_path = "src/paraphrase/model/t5-sl-small-paired1-whole-mbert"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
prefix = "paraphrase: "

gen_text = pd.DataFrame()

def generate_para(orig: str) -> str:
    input_ids = tokenizer(f"paraphrase {orig}", return_tensors="pt", max_length=128, truncation=True).input_ids
    outputs = model.generate(input_ids,
                                max_length=128,
                                num_beams=5,
                                do_sample=True,
                                top_k=5,
                                temperature=0.4
                                )

    decoded_preds = [tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)]
    return decoded_preds


for _i in range(len(new_para["val"])):
    sent = new_para["val"][_i]["phrase"]
    gen_text.loc[_i, "source"] = sent

    decoded_labels = [new_para["val"][_i]["paraphrase"]]
    gen_text.loc[_i, "target"] = decoded_labels
    
    input_ids = tokenizer(f"{prefix} {sent}", return_tensors="pt", max_length=128, truncation=True).input_ids
    outputs = model.generate(input_ids,
                                max_length=128,
                                num_beams=5,
                                do_sample=True,
                                top_k=5,
                                temperature=0.7
                                )

    decoded_preds = [tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)]
    gen_text.loc[_i, "prediction"] = decoded_preds

    result_btsc = metric_btscore.compute(predictions=decoded_preds, references=decoded_labels, lang="sl")
    result_rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    print(f"original: {sent} \npara: {decoded_labels[0]} \npredict: {decoded_preds[0]}" )
    for m in ["precision", "recall", "f1"]:
        gen_text.loc[_i, f"bertscore_{m}"] = result_btsc[m]
    for m2 in ["1","2", "L" ]:
        gen_text.loc[_i, f"rouge{m2}"] = result_rouge[f"rouge{m2}"]
gen_text.to_excel("eval_paras.xlsx")