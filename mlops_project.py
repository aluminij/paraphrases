import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/LaBSE")  # only loads once


def read_aligned(f_name: str, lang: str = "en") -> pd.DataFrame:
    if lang == "en":
        df = pd.read_csv(f"aligned/en_{f_name}.txt", header=None, sep="\n|\r\n")
    elif lang == "sl":
        df = pd.read_csv(
            f"aligned/sl_{f_name}.txt", header=None, sep="\n|\r\n", encoding="utf-8"
        )
    return df


def compare_phrases(phr1: str, phr2: str) -> bool:
    phrases = [phr1, phr2]
    encoded_phrases = model.encode(phrases)
    similarity = util.pytorch_cos_sim(encoded_phrases[0], encoded_phrases[1]).item()
    return similarity > 0.7

print(compare_phrases("Dobro jutro!", "Dobro jutro tudi tebi!"))
pass
