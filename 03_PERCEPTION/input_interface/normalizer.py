"""Fail-closed validation and normalization at the perception boundary."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..contracts import Modality, PerceptionInput


class InputValidationError(ValueError):
    """Input violates a declared perception-boundary contract."""


class InputNormalizer:
    """Validates modality/content compatibility and enforces size limits."""

    def __init__(
        self,
        *,
        max_text_characters: int = 2_000_000,
        max_binary_bytes: int = 50 * 1024 * 1024,
        max_structured_items: int = 100_000,
    ) -> None:
        for name, value in (
            ("max_text_characters", max_text_characters),
            ("max_binary_bytes", max_binary_bytes),
            ("max_structured_items", max_structured_items),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_text_characters = max_text_characters
        self.max_binary_bytes = max_binary_bytes
        self.max_structured_items = max_structured_items

    def normalize(self, item: PerceptionInput) -> PerceptionInput:
        if not isinstance(item, PerceptionInput):
            raise TypeError("item must be a PerceptionInput")

        content: Any = item.content
        if item.modality is Modality.TEXT:
            if not isinstance(content, str):
                raise InputValidationError("TEXT input requires str content")
            if not content.strip():
                raise InputValidationError("TEXT input cannot be empty or whitespace-only")
            if len(content) > self.max_text_characters:
                raise InputValidationError("TEXT input exceeds configured character limit")
            if "\x00" in content:
                raise InputValidationError("TEXT input contains a null character")
            content = content.replace("\r\n", "\n").replace("\r", "\n")

        elif item.modality in {Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}:
            if not isinstance(content, bytes):
                raise InputValidationError(
                    f"{item.modality.value.upper()} input requires raw bytes; "
                    "provide decoded bytes from a trusted ingestion adapter"
                )
            if not content:
                raise InputValidationError(f"{item.modality.value.upper()} input cannot be empty")
            if len(content) > self.max_binary_bytes:
                raise InputValidationError("binary input exceeds configured byte limit")
            if not item.content_type:
                raise InputValidationError("binary input requires an explicit content_type")

        elif item.modality is Modality.STRUCTURED:
            if not hasattr(content, "items"):
                raise InputValidationError("STRUCTURED input requires a mapping")
            if len(content) > self.max_structured_items:
                raise InputValidationError("structured input exceeds configured item limit")

        return replace(item, content=content)
