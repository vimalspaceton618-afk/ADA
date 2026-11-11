"""Introspection-based API reference index for importable Python modules."""
from __future__ import annotations
import importlib, inspect
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ApiSymbol:
    module: str
    name: str
    kind: str
    signature: str | None
    doc: str

def index_module(module_name: str):
    module = importlib.import_module(module_name)
    symbols = []
    for name, value in sorted(vars(module).items()):
        if name.startswith("_"): continue
        if inspect.isclass(value) or inspect.isfunction(value):
            try: signature = str(inspect.signature(value))
            except (TypeError, ValueError): signature = None
            kind = "class" if inspect.isclass(value) else "function"
            symbols.append(ApiSymbol(module_name, name, kind, signature, inspect.getdoc(value) or ""))
    return tuple(symbols)

__all__ = ["ApiSymbol", "index_module"]
