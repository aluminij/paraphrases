import os
import openai
import re

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPEN_AI_KEY")

def get_paraphrase_chatgpt(phrase: str, num_p: int=2, max_tokens: int=100, lang: str="eng"):

    if lang=="eng":
        p = f"Create {num_p} paraphrases of {phrase}"
    elif lang == "si": 
        p = f"Ustvari {num_p} parafraze {phrase}"
    
    # for text-davinci-003
    response = openai.Completion.create(model="text-davinci-003", prompt=p, max_tokens=max_tokens)
    r_clean = re.sub("\\n\d?\.?", "", response.choices[0].text)
    t_list = [x.strip() for x in re.split("(?<=\.)\W", r_clean)]
    return t_list

p1 = "Pozimi pa rožice ne cveto, zato sem doma."
p_list = get_paraphrase_chatgpt(p1, lang="si")


pass