"""Transparent lexical abstractions from supplied evidence."""
import re
_WORD=re.compile(r"[a-z0-9_]+",re.I)
def extract_terms(texts, *, min_length=3):
    if type(min_length) is not int or min_length<1: raise ValueError("min_length must be positive")
    counts={}
    for text in texts:
        if not isinstance(text,str): raise TypeError("texts must contain strings")
        for word in _WORD.findall(text.casefold()):
            if len(word)>=min_length: counts[word]=counts.get(word,0)+1
    return tuple(sorted(counts.items(),key=lambda x:(-x[1],x[0])))
