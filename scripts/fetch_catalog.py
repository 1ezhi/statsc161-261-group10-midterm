"""Download the two catalog tables that build_features.ipynb reads."""

import shutil
import time
from pathlib import Path

import duckdb
import pandas as pd
import sqlalchemy

import credentials

LITE_DIR = Path(__file__).resolve().parents[1]
CATALOG_DIR = LITE_DIR / "data" / "catalog"
CHUNK = 200_000

PULLS = [
    ("markets", "slug, event_slug, neg_risk, created_at, end_date", "slug"),
    ("resolutions", "market_slug, resolved_at", "market_slug"),
]


def ask(engine, sql, **params):
    """Query the catalog database and return the result as a pandas table."""
    with engine.connect() as connection:
        return pd.read_sql(sqlalchemy.text(sql), connection, params=params or None)


def dump(engine, table, columns, key):
    """Page one table to disk in shards, then merge them into a single file."""
    out_path = CATALOG_DIR / f"{table}.parquet"
    parts = CATALOG_DIR / f"{table}.parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True, exist_ok=True)

    last, total, shard, started = "", 0, 0, time.time()
    while True:
        page = ask(
            engine,
            f"SELECT {columns} FROM {table} "
            f"WHERE {key} > :last ORDER BY {key} LIMIT {CHUNK}",
            last=last,
        )
        got = len(page)
        if got == 0:
            break
        page.to_parquet(parts / f"part{shard:03d}.parquet", index=False)
        last = str(page[key].iloc[-1])
        total += got
        shard += 1
        del page
        print(f"{table}: {total} rows in {time.time() - started:.0f}s")
        if got < CHUNK:
            break

    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{parts}/*.parquet', union_by_name=true)) "
        f"TO '{out_path}' (FORMAT PARQUET)"
    )
    shutil.rmtree(parts)
    print(f"{table}: wrote {total} rows to {out_path}")


def main():
    engine = sqlalchemy.create_engine(
        credentials.load()["JUNE_DB_URL"], connect_args={"sslmode": "require"}
    )
    for table, columns, key in PULLS:
        dump(engine, table, columns, key)


if __name__ == "__main__":
    main()
