"""Structured logging for PropOS transparency / audit compliance."""

import logging, json, sys
from datetime import datetime, timezone
from app.core.config import settings


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            log["trace_id"] = record.trace_id
        for k in ("path", "method", "status", "latency_ms", "agent", "action", "reward"):
            if hasattr(record, k):
                log[k] = getattr(record, k)
        return json.dumps(log)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger("propos")
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.LOG_LEVEL))
