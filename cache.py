"""SQLite-backed result cache and sliding-window rate limiter."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import time
from collections import deque

from models import TensionResult

DEFAULT_CACHE_PATH = pathlib.Path("./data/cache.db")


# ------------------------------------------------------------------
# Rate limiter
# ------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter that sleeps when the budget is exhausted.

    Parameters
    ----------
    max_rpm : maximum API requests allowed per 60-second window.
    """

    def __init__(self, max_rpm: int = 50) -> None:
        self.max_rpm = max_rpm
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        """Block until a request slot is available."""
        now = time.monotonic()
        # Evict timestamps older than 60 s.
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_rpm:
            sleep_until = self._timestamps[0] + 60.0
            delay = sleep_until - now
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()

        self._timestamps.append(time.monotonic())


# ------------------------------------------------------------------
# SQLite result cache
# ------------------------------------------------------------------


class ResultCache:
    """Persistent cache for embeddings and tension-analysis results.

    Keys are content-based SHA-256 hashes so a changed belief text
    automatically invalidates its cached entries.
    """

    def __init__(self, path: str | pathlib.Path = DEFAULT_CACHE_PATH) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash  TEXT NOT NULL,
                model      TEXT NOT NULL,
                embedding  TEXT NOT NULL,
                PRIMARY KEY (text_hash, model)
            );
            CREATE TABLE IF NOT EXISTS tension_results (
                pair_hash   TEXT NOT NULL,
                model       TEXT NOT NULL,
                result_json TEXT NOT NULL,
                PRIMARY KEY (pair_hash, model)
            );
            """
        )
        self._conn.commit()

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _pair_hash(text_a: str, text_b: str) -> str:
        """Order-independent hash for a pair of texts."""
        a, b = sorted([text_a, text_b])
        return hashlib.sha256(f"{a}|||{b}".encode("utf-8")).hexdigest()

    # -- embeddings ---------------------------------------------------

    def get_embedding(self, text: str, model: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ? AND model = ?",
            (self._hash(text), model),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_embedding(self, text: str, model: str, embedding: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding) VALUES (?, ?, ?)",
            (self._hash(text), model, json.dumps(embedding)),
        )
        self._conn.commit()

    # -- tension results ----------------------------------------------

    def get_tension(self, text_a: str, text_b: str, model: str) -> TensionResult | None:
        row = self._conn.execute(
            "SELECT result_json FROM tension_results WHERE pair_hash = ? AND model = ?",
            (self._pair_hash(text_a, text_b), model),
        ).fetchone()
        return TensionResult(**json.loads(row[0])) if row else None

    def put_tension(
        self, text_a: str, text_b: str, model: str, result: TensionResult
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tension_results (pair_hash, model, result_json) VALUES (?, ?, ?)",
            (self._pair_hash(text_a, text_b), model, result.model_dump_json()),
        )
        self._conn.commit()

    # -----------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
