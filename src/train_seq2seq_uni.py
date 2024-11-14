import os
import code_bert_score
import numpy as np
from datasets import load_dataset
from typing import Literal
import evaluate
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration, MT5Tokenizer


def train_model(model_name: Literal["sloT5", "mT5"], metric_name: Literal["rouge", "sloberta", "mbert"]):
    data_files = {
        "train": f"paired_1_datasets/train_largest.jsonl",
        "test": f"paired_1_datasets/test_largest.jsonl",
        "val": f"paired_1_datasets/val_largest.jsonl",
    }
    exp = f"{model_name}-largest-{metric_name}"
    max_epochs = 64
    batch_size = 8
    learning_rate = 5e-5
    max_seq_length = 64
    validate_every_n_steps = 50
    validate_every_n_batches = (validate_every_n_steps + batch_size - 1) // batch_size

    paras = load_dataset("json", data_files=data_files)

    if metric_name == "rouge":
        metric = evaluate.load("rouge")
        best_metric = "eval_rougeL"
    elif metric_name == "mbert":
        metric = evaluate.load("bertscore")
        best_metric = "f1"
    elif metric_name == "sloberta":
        best_metric = "f1"

    if model_name == "sloT5":
        model_name_hf = "cjvt/t5-sl-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name_hf)
        model = T5ForConditionalGeneration.from_pretrained(model_name_hf)
    elif model_name == "mT5":
        model_name_hf = "google/mt5-small"
        tokenizer = MT5Tokenizer.from_pretrained(model_name_hf)
        model = MT5ForConditionalGeneration.from_pretrained(model_name_hf)

    prefix = "generiraj parafrazo: "

    # Convert data from strings to model-specific format (truncate inputs beyond `max_length` subwords)
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["phrase"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["paraphrase"], max_length=max_seq_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_paras = paras.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred # eval_pred.inputs je None
        # Replace -100 in the labels as we can't decode them (-100 = "ignore label")
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = list(map(lambda _seq: [_seq], tokenizer.batch_decode(labels, skip_special_tokens=True)))

        if metric_name == "rouge":
            r = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, rouge_types=['rouge1', 'rouge2', 'rougeL'])
        elif metric_name == "sloberta": 
            sl_r = code_bert_score.score(cands=decoded_preds, refs=decoded_labels, model_type="EMBEDDIA/sloberta")
            r = dict()
            for i, k in enumerate(["precision", "recall", "f1", "f3"]):
                r[k] = np.mean(sl_r[i].flatten().tolist())
        elif metric_name == "mbert":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="sl")
            r = dict()
            for _, k in enumerate(["precision", "recall", "f1"]):
                r[k] = np.mean(result[k])
        return r

    output_dir = os.path.join("./results", exp)
    save_path = os.path.join("./model", exp)
    log_dir = os.path.join("./logs", exp)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=4,
        num_train_epochs=max_epochs,
        fp16=False,
        save_steps=validate_every_n_batches,
        eval_steps=validate_every_n_batches,
        logging_steps=validate_every_n_batches,
        logging_dir=log_dir,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,
        predict_with_generate=True,
        generation_max_length=max_seq_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_paras["train"],
        eval_dataset=tokenized_paras["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(save_path)

    val_results = trainer.evaluate()
    test_results = trainer.predict(test_dataset=tokenized_paras["test"])

    lines = [f"Evaluating model version {exp}"]
    lines.append(f"Val results: {val_results} | ")
    lines.append(f"Test results: {test_results.metrics} | ")

    with open(f'metrics_{exp}.txt', 'w') as f:
        f.writelines(lines)

#train_model("sloT5", "rouge")