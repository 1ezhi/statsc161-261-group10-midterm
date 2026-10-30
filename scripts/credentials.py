"""Read course data credentials from the environment or a gitignored .env file."""

import os
import re
from pathlib import Path

LITE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = LITE_DIR / ".env"

KEYS = ("JUNE_DB_URL", "JUNE_S3_HOST", "JUNE_S3_USER", "JUNE_S3_PASS")


def load():
    """Return the credential values, preferring real environment variables."""
    values = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            match = re.match(r"(\w+)=(.*)", line.strip())
            if match:
                values[match.group(1)] = match.group(2)
    values.update({key: os.environ[key] for key in KEYS if os.environ.get(key)})

    missing = [key for key in KEYS if not values.get(key)]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in."
        )
    return values
