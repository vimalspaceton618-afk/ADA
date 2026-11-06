"""Deterministic lexical attention baseline; not learned attention."""
import re
_WORD=re.compile(r"[a-z0-9_]+",re.I)
def select_focus(query, items, *, limit=5):
    if not isinstance(query,str) or not query.strip(): raise ValueError("query must be non-empty")
    if type(limit) is not int or limit<1: raise ValueError("limit must be positive")
    terms={w.casefold() for w in _WORD.findall(query) if len(w)>1}; ranked=[]
    for i,item in enumerate(items):
        words={w.casefold() for w in _WORD.findall(str(item)) if len(w)>1}
        ranked.append((i,len(terms&words)/max(1,len(terms)),item))
    return tuple(sorted(ranked,key=lambda x:(-x[1],x[0]))[:limit])
