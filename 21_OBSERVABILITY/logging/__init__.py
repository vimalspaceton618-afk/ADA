"""Structured JSON logging with deterministic field handling."""
from __future__ import annotations
import json, logging, time
from typing import Any, Mapping

class JsonEventFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": time.time(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "event_fields", None)
        if isinstance(extra, Mapping):
            payload["fields"] = {str(k): v for k, v in sorted(extra.items(), key=lambda x: str(x[0]))}
        if record.exc_info: payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

def configure_json_logger(name: str, *, level: int = logging.INFO, handler=None):
    if not name.strip(): raise ValueError("logger name required")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if handler is None: handler = logging.StreamHandler()
    handler.setFormatter(JsonEventFormatter())
    if not any(getattr(h, "_ada_json_handler", False) for h in logger.handlers):
        handler._ada_json_handler = True
        logger.addHandler(handler)
    return logger

def log_event(logger, level: int, message: str, fields: Mapping[str, Any] | None = None):
    logger.log(level, message, extra={"event_fields": dict(fields or {})})

__all__ = ["JsonEventFormatter", "configure_json_logger", "log_event"]
