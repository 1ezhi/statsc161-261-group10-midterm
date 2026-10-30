"""Aggregate the remote daily trade files into the hourly panel hp."""

import re
import time
from pathlib import Path

import duckdb
import pandas as pd

import credentials

LITE_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = LITE_DIR / "data" / "volume_panel.duckdb"

FIRST_DAY = "2026-05-30"
LAST_DAY = "2026-10-25"
RETRIES = 3

AGGREGATE = """
SELECT market, date_trunc('hour', {ts}) AS h,
       sum(dollars) AS vol, count(*) AS fills,
       count(DISTINCT wallet) AS wallets,
       sum(CASE WHEN side = 'BUY'  THEN dollars ELSE 0 END) AS buy_d,
       sum(CASE WHEN side = 'SELL' THEN dollars ELSE 0 END) AS sell_d,
       max(price) AS p_max, min(price) AS p_min,
       max_by(price, time) AS p_last
FROM '{url}' GROUP BY 1, 2
"""


def percent_encode(text):
    """Encode every character that is not safe inside a URL authority."""
    return re.sub(r"([^\w.-])", lambda match: "%%%02X" % ord(match.group(1)), text)


def trade_url(values, day):
    """Build the authenticated URL of one daily trade file."""
    host = values["JUNE_S3_HOST"]
    authority = f"{percent_encode(values['JUNE_S3_USER'])}:{percent_encode(values['JUNE_S3_PASS'])}"
    return f"{host.replace('://', '://' + authority + '@')}/wallet_trades/trades-{day}.parquet"


def main():
    values = credentials.load()
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DATABASE_PATH))
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql("SET http_timeout = 120")
    con.sql("""
        CREATE TABLE IF NOT EXISTS hp(
            market VARCHAR, h TIMESTAMP, vol DOUBLE, fills BIGINT, wallets BIGINT,
            buy_d DOUBLE, sell_d DOUBLE, p_max DOUBLE, p_min DOUBLE, p_last DOUBLE)
    """)

    # The source column is either naive or timezone-aware depending on the snapshot.
    probe = con.sql(
        f"SELECT typeof(any_value(time)) AS time_type FROM '{trade_url(values, LAST_DAY)}'"
    ).df()
    timestamp = (
        "time AT TIME ZONE 'UTC'"
        if "TIME ZONE" in str(probe.time_type[0]).upper()
        else "time"
    )

    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D").strftime("%Y-%m-%d")
    stored = {
        row[0]
        for row in con.sql("SELECT DISTINCT strftime(h, '%Y-%m-%d') FROM hp").fetchall()
    }
    todo = [day for day in days if day not in stored]
    print(f"{len(days) - len(todo)} days already stored, {len(todo)} to fetch")

    for position, day in enumerate(todo):
        insert = "INSERT INTO hp " + AGGREGATE.format(
            ts=timestamp, url=trade_url(values, day)
        )
        for attempt in range(RETRIES):
            try:
                con.sql(insert)
                break
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(4 * (attempt + 1))
        if (position + 1) % 10 == 0 or position == len(todo) - 1:
            con.sql("CHECKPOINT")
        print(f"{position + 1}/{len(todo)} {day}")

    print(f"hp holds {con.sql('SELECT count(*) FROM hp').fetchone()[0]} rows")
    con.close()


if __name__ == "__main__":
    main()
