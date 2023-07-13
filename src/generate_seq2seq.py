import evaluate
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer


if __name__ == "__main__":
    data_files = {
        "train": "train_midi.jsonl",
        "val": "eval_midi.jsonl",
        "test": "test_midi.jsonl"
    }
    new_para = load_dataset('json', data_files=data_files)

    metric = evaluate.load("bertscore")

    model_path = "./model/t5-sl-small-midi-data-bert"

    # Use the model you trained here, or something pre-trained from https://huggingface.co/models
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    prefix = "paraphrase: "

    for _i in range(len(new_para["test"])):
        sent = new_para["test"][_i]["phrase"]
        print("SOURCE:\t", sent)

        decoded_labels = [new_para["test"][_i]["paraphrase"]]
        print("TARGET:\t", decoded_labels)

        input_ids = tokenizer(f"{prefix} {sent}", return_tensors="pt", max_length=128, truncation=True).input_ids
        # Tweak generation parameters to increase/decrease diversity of generated text
        # See https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        outputs = model.generate(input_ids,
                                 max_length=128,
                                 num_beams=5,
                                 do_sample=True,
                                 top_k=5,
                                 temperature=0.7
                                 )

        decoded_preds = [tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)]
        print("PREDICTION:\t", decoded_preds, "\n")

        # Note that this is the example-level metric (as opposed to dataset-level metric)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="sl")

        print(result)
        print("\n")