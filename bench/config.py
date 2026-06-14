"""
Single source of truth for the Postgres connection. Reads PG_URL from the
environment (a `.env` file is loaded if present), falling back to the URL the
bundled docker-compose Postgres listens on. Nothing else in the bench
hardcodes a connection string.
"""
from __future__ import annotations

import os
import pathlib

# docker compose maps the postgres service to localhost:5544 (see
# docker-compose.yml) to avoid clashing with any local 5432/5433 cluster.
DEFAULT_PG_URL = "postgresql+psycopg2://inferah:inferah@localhost:5544/inferah"


def _load_dotenv():
    env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def pg_url() -> str:
    return os.environ.get("PG_URL", DEFAULT_PG_URL)
