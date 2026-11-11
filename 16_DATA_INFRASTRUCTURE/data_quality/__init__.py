"""Schema, text and token-level quality checks for corpus preparation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    record_index: int
    field: str
    code: str
    message: str

@dataclass(frozen=True, slots=True)
class SchemaField:
    name: str
    expected_type: type
    required: bool = True
    def __post_init__(self):
        if not self.name.strip(): raise ValueError("field name required")

class RecordValidator:
    def __init__(self, fields, *, allow_extra: bool = False):
        self.fields = tuple(fields); self.allow_extra = allow_extra
        if any(not isinstance(x, SchemaField) for x in self.fields): raise TypeError("fields must be SchemaField")
        if len({x.name for x in self.fields}) != len(self.fields): raise ValueError("schema field names must be unique")
    def validate(self, records):
        issues = []
        known = {x.name for x in self.fields}
        for i, row in enumerate(records):
            if not isinstance(row, Mapping):
                issues.append(ValidationIssue(i, "", "not_object", "record must be a mapping")); continue
            for field in self.fields:
                if field.name not in row:
                    if field.required: issues.append(ValidationIssue(i, field.name, "missing", "required field missing"))
                elif field.expected_type is int and isinstance(row[field.name], bool):
                    issues.append(ValidationIssue(i, field.name, "wrong_type", "boolean is not an integer"))
                elif not isinstance(row[field.name], field.expected_type):
                    issues.append(ValidationIssue(i, field.name, "wrong_type", f"expected {field.expected_type.__name__}"))
            if not self.allow_extra:
                for name in set(row) - known: issues.append(ValidationIssue(i, name, "unknown", "unexpected field"))
        return tuple(issues)

def validate_text(text: str, *, min_chars: int = 1, max_chars: int = 1_000_000) -> tuple[str, ...]:
    if not isinstance(text, str): return ("not_string",)
    issues = []
    if len(text.strip()) < min_chars: issues.append("too_short")
    if len(text) > max_chars: issues.append("too_long")
    if "\x00" in text: issues.append("nul_byte")
    return tuple(issues)

__all__ = ["ValidationIssue", "SchemaField", "RecordValidator", "validate_text"]
