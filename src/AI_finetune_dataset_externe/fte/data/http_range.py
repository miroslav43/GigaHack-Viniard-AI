"""Parallel HTTP Range reader exposed as a sequential, read-only file object.

Zenodo serves ~0.3-0.5 MB/s per connection but scales with parallel Range requests, so large
archives are fetched as fixed-size chunks by a thread pool and consumed strictly in order.
At most ``max_ahead`` chunks are held in memory, so nothing touches the disk.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

log = logging.getLogger(__name__)

DEFAULT_CHUNK = 32 * 1024 * 1024
USER_AGENT = "solemtrix-fte/0.1 (GigaHack 2026 research)"


class RangeError(RuntimeError):
    """A byte range could not be fetched after all retries."""


def remote_size(url: str, timeout: float = 60.0) -> int:
    """Total size in bytes, read from the Content-Range of a 1-byte ranged GET (HEAD lies on Zenodo)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_range = resp.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RangeError(f"server did not return Content-Range for {url!r}")
    return int(content_range.rsplit("/", 1)[1])


def fetch_range(url: str, start: int, end: int, retries: int = 6, timeout: float = 120.0) -> bytes:
    """Fetch bytes [start, end] inclusive, retrying on 429/5xx/short reads with backoff."""
    want = end - start + 1
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 206:
                    raise RangeError(f"expected 206, got {resp.status}")
                data = resp.read()
            if len(data) != want:
                raise RangeError(f"short read {len(data)}/{want} at {start}")
            return data
        except (urllib.error.URLError, RangeError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            wait = 60.0 if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 else 2.0 * (attempt + 1)
            log.warning("range %d-%d failed (%s); retry in %.0fs", start, end, exc, wait)
            time.sleep(wait)
    raise RangeError(f"range {start}-{end} failed after {retries} attempts: {last}")


def chunk_spans(start: int, stop: int, chunk: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) spans covering [start, stop) in steps of ``chunk``."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    return [(s, min(s + chunk, stop) - 1) for s in range(start, stop, chunk)]


class ParallelRangeReader(io.RawIOBase):
    """Sequential reader over [start, stop) of ``url`` backed by parallel ranged GETs."""

    def __init__(
        self,
        url: str,
        stop: int,
        start: int = 0,
        chunk: int = DEFAULT_CHUNK,
        workers: int = 8,
        max_ahead: int = 12,
        fetch=fetch_range,
    ) -> None:
        super().__init__()
        self._spans = deque(chunk_spans(start, stop, chunk))
        self._url = url
        self._fetch = fetch
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._pending: deque[Future] = deque()
        self._max_ahead = max(max_ahead, workers)
        self._buf = memoryview(b"")
        self.bytes_read = 0
        self.total = stop - start
        self._fill()

    def _fill(self) -> None:
        while self._spans and len(self._pending) < self._max_ahead:
            s, e = self._spans.popleft()
            self._pending.append(self._pool.submit(self._fetch, self._url, s, e))

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buf:
            if not self._pending:
                return 0
            self._buf = memoryview(self._pending.popleft().result())
            self._fill()
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self.bytes_read += n
        return n

    def close(self) -> None:
        for fut in self._pending:
            fut.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()
