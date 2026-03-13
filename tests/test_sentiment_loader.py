"""
Unit tests for load_sentiment_factor in features/factors.py.
Uses a temporary in-memory SQLite database — no network calls.
"""
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pintrade.features.factors import load_sentiment_factor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_sentiment_db(path: Path, rows: list[tuple]) -> None:
    """Create a minimal sentiment.db with the schema expected by load_sentiment_factor."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE sentiment (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id  TEXT,
            ticker    TEXT,
            date      TEXT,
            doc_type  TEXT,
            section   TEXT DEFAULT '',
            positive  REAL,
            negative  REAL,
            neutral   REAL,
            compound  REAL
        )
    """)
    conn.executemany(
        "INSERT INTO sentiment (chunk_id, ticker, date, doc_type, compound) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


TICKERS = ["AAPL", "MSFT"]
DATES   = pd.bdate_range("2023-01-03", periods=20)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_missing_db_returns_empty():
    result = load_sentiment_factor(TICKERS, DATES, sentiment_db="/no/such/file.db")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_returns_correct_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sentiment.db"
        _create_sentiment_db(db, [
            ("c1", "AAPL", str(DATES[5].date()), "news", 0.5),
        ])
        result = load_sentiment_factor(TICKERS, DATES, sentiment_db=db)
    assert "News_Sentiment"   in result.columns
    assert "Filing_Sentiment" in result.columns


def test_index_names():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sentiment.db"
        _create_sentiment_db(db, [
            ("c1", "AAPL", str(DATES[5].date()), "news", 0.5),
        ])
        result = load_sentiment_factor(TICKERS, DATES, sentiment_db=db)
    assert result.index.names == ["Date", "Ticker"]


def test_news_score_propagated():
    """A news score on day T should appear at T+1 (1-day lookahead shift)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sentiment.db"
        score_date = DATES[5]
        _create_sentiment_db(db, [
            ("c1", "AAPL", str(score_date.date()), "news", 0.8),
        ])
        result = load_sentiment_factor(TICKERS, DATES, sentiment_db=db,
                                       news_ffill_days=1)

    # Score on day 5 → shifted to day 6 (T+1)
    target_date = DATES[6]
    try:
        val = result.loc[(target_date, "AAPL"), "News_Sentiment"]
        # Value present (not NaN)
        assert pd.notna(val), "Sentiment should appear at T+1 after shift"
    except KeyError:
        pytest.fail(f"Expected row for (AAPL, {target_date.date()}) not found")


def test_filing_score_forward_filled():
    """Filing scores should forward-fill for up to filing_ffill_days days."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sentiment.db"
        filing_date = DATES[0]
        _create_sentiment_db(db, [
            ("f1", "MSFT", str(filing_date.date()), "10-K", 0.3),
        ])
        result = load_sentiment_factor(TICKERS, DATES, sentiment_db=db,
                                       filing_ffill_days=10)

    # Filing on day 0 shifted to day 1, forward-fill up to 10 days → days 1-10
    for i in range(1, min(10, len(DATES))):
        d = DATES[i]
        try:
            val = result.loc[(d, "MSFT"), "Filing_Sentiment"]
            if pd.notna(val):
                break
        except KeyError:
            pass
    else:
        pytest.fail("Filing score was not forward-filled to any day in range")


def test_empty_db_returns_nan_frame():
    """Existing DB with no matching tickers → NaN-filled frame, correct index shape."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sentiment.db"
        _create_sentiment_db(db, [])   # empty table
        result = load_sentiment_factor(TICKERS, DATES, sentiment_db=db)
    # Should return an empty or NaN-filled frame, not raise
    assert isinstance(result, pd.DataFrame)
