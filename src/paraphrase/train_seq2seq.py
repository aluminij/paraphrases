import os

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration, T5Tokenizer


if __name__ == "__main__":
    data_files = {
        "train": "data/data_paired1/train_paired1.jsonl",
        "val": "data/data_paired1/eval_paired1.jsonl",
        "test": "data/data_paired1/test_1.jsonl"
    }



    exp = "paired1-whole-mbert"  # experiment name
    max_epochs = 64
    batch_size = 8
    learning_rate = 5e-5
    max_seq_length = 64
    validate_every_n_steps = 50
    validate_every_n_batches = (validate_every_n_steps + batch_size - 1) // batch_size

    paras = load_dataset("json", data_files=data_files)

    bertscore = evaluate.load("bertscore")   

    model_name = "cjvt/t5-sl-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    prefix = "paraphrase: "

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

        result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="sl")
        result["f1"] = np.mean(result["f1"])
        result["precision"] = np.mean(result["precision"])
        result["recall"] = np.mean(result["recall"])
        return result

    output_dir = os.path.join("./results", f"t5-sl-small-{exp}")
    save_path = os.path.join("./model", f"t5-sl-small-{exp}")
    log_dir = os.path.join("./logs", f"t5-sl-small-{exp}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # sacrebleu returns the metric value as the key "score" (see docs)
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=3,
        num_train_epochs=max_epochs,
        fp16=False,
        save_steps=validate_every_n_batches,
        eval_steps=validate_every_n_batches,
        logging_steps=validate_every_n_batches,
        logging_dir=log_dir,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,
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
    lines.append(f"Val results: {val_results}")
    lines.append(f"Test results: {test_results.metrics}")

    with open(f'metrics_{exp}.txt', 'w') as f:
        f.writelines(lines)