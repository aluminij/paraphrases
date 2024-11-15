import classla
import difflib

classla.download('sl')
nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')

def custom_rouge(ref: str, gen: str, lemma=True) -> dict:
    if lemma:
        ref = lemmatize(ref)
        gen = lemmatize(gen)
    ref_tokens = ref.split(" ")
    gen_tokens = gen.split(" ")

    def rouge1():
        p_n = len(gen_tokens)
        r_n = len(ref_tokens)
        match_tokens = len(set(ref_tokens).intersection(set(gen_tokens)))
        
        p = match_tokens/p_n
        r = match_tokens/r_n

        s = (2*p*r)/(p+r) if p+r > 0 else 0
        return {"rouge1": {"precision": p, "recall": r, "f1": s}}
    
    def rouge2():
        pairs_ref = [(ref_tokens[i], ref_tokens[i+1]) for i in range(len(ref_tokens)-1)]
        pairs_gen = [(gen_tokens[i], gen_tokens[i+1]) for i in range(len(gen_tokens)-1)]
        match_tokens = len(set(pairs_ref).intersection(set(pairs_gen)))

        if len(pairs_gen)*len(pairs_ref) == 0:
            return {"rouge2": {"precision": 0, "recall": 0, "f1": 0}}

        p = match_tokens/len(pairs_gen)
        r = match_tokens/len(pairs_ref)

        s = (2*p*r)/(p+r) if p+r > 0 else 0
        return {"rouge2": {"precision": p, "recall": r, "f1": s}}

    def rougeL():
        # Find matching items in order
        matcher = difflib.SequenceMatcher(None, ref_tokens, gen_tokens)
        matches = matcher.get_matching_blocks()

        # Extract the longest matching subsequence
        lcs = []
        for match in matches:
            if match.size > 0:
                lcs.extend(ref_tokens[match.a: match.a + match.size])

        p = len(lcs)/len(gen_tokens)
        r = len(lcs)/len(ref_tokens)

        s = (2*p*r)/(p+r) if p+r > 0 else 0
        return {"rougeL": {"precision": p, "recall": r, "f1": s}}
    r1 = rouge1()
    r2 = rouge2()
    rL = rougeL()
    return {**r1, **r2, **rL}

def lemmatize(sent: str) -> str:
    doc = nlp(sent)
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words if word.upos != 'PUNCT']
    lemma_sent = " ".join(lemmas)
    return lemma_sent

