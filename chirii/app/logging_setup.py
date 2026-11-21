"""
Privacy-safe logging. Users' messages, memories and names must never reach a log file.

Code logs events through log_event(), which accepts only the field names in ALLOWED_FIELDS. User ids are replaced by a
short one-way hash, so a log line can be matched to a user only by someone who already knows the id.
"""

import hashlib
import logging
from typing import Any

ALLOWED_FIELDS = {
    "user", "endpoint", "status", "duration_ms", "item_type", "item_count", "result_count", "tokens_in", "tokens_out",
    "model", "error_type", "stage",
}

logger = logging.getLogger("chirii")


def hash_user(user_id: Any) -> str:
    return hashlib.sha256(f"chirii-user:{user_id}".encode("utf-8")).hexdigest()[:12]


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Logs one event. Unknown field names raise, so message content cannot be logged by accident."""
    unknown = set(fields) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Refusing to log fields that may hold personal data: {sorted(unknown)}")
    if not event.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Event names must be short identifiers, not free text")
    if "user" in fields:
        fields["user"] = hash_user(fields["user"])
    details = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
    logger.log(level, f"{event} {details}".strip())


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
