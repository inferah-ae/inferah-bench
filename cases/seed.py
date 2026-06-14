"""
Seed all 28 cases into the Postgres at $PG_URL (idempotent: each case_NN
schema is dropped and recreated). Connects to whatever PG_URL points at —
the docker-compose Postgres by default — and needs no engine-repo bootstrap.

    python -m cases.seed
"""
from __future__ import annotations

from sqlalchemy import create_engine

from bench.config import pg_url
from cases.generators import seed_postgres


def main():
    url = pg_url()
    eng = create_engine(url, future=True)
    print(f"seeding 28 cases into {url}")
    labels = seed_postgres(eng)
    print(f"done: {len(labels)} case schemas")


if __name__ == "__main__":
    main()
