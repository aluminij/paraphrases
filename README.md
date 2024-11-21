# Slovene Paraphrase Dataset and Generation Models

This repository contains resources for creating and using a Slovene paraphrase dataset, [slopodpara](https://huggingface.co/datasets/aluminij/slopodpara)
along with models for generating Slovene paraphrases, [paragen_sloT5](https://huggingface.co/aluminij/paragen_sloT5)
and [paragen_mT5](https://huggingface.co/aluminij/paragen_mT5).
The models are [*Slot5*](https://huggingface.co/cjvt/t5-sl-small) 
and [*mT5*](https://huggingface.co/google/mt5-small) fine-tuned using the slopodpara dataset specifically for the task of generating paraphrases in Slovene.


## Project Overview

- **Dataset**: the data folder contains a dataset of Slovene paraphrases, prepared for use in training and evaluating paraphrase generation models. 
The files are already split into train, test and validation sets (70-15-15), whereas the [huggingface dataset](https://huggingface.co/datasets/aluminij/slopodpara) contains a single .jsonl file.
It was created using OPUS OpenSubtitlesv2018 for Slovene and English as source material. 
Input text files were somewhat aligned as-is from OPUS but were hand-aligned to ensure complete
line matching during development. We used DeepL API for translation and saved the pairs in JSON format. 
The pairs consist of source Slovene phrases and Slovene translations of English source phrases.

- **Models**: Scripts used for fine-tuning *Slot5* and *mT5* models using the Slovene paraphrase dataset.

## Requirements

Ensure you have the necessary dependencies, which can be installed via:
```bash
pip install -r requirements.txt
```