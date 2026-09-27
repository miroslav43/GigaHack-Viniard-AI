import io

import pytest

from fte.data.http_range import ParallelRangeReader, chunk_spans


def test_chunk_spans_cover_range_inclusive():
    spans = chunk_spans(10, 35, 10)
    assert spans == [(10, 19), (20, 29), (30, 34)]


def test_chunk_spans_rejects_non_positive_chunk():
    with pytest.raises(ValueError):
        chunk_spans(0, 10, 0)


def test_parallel_reader_reassembles_in_order():
    payload = bytes(range(256)) * 40  # 10 240 bytes
    calls = []

    def fake_fetch(url, start, end):
        calls.append((start, end))
        return payload[start:end + 1]

    reader = ParallelRangeReader("u", stop=len(payload), chunk=1000, workers=4, max_ahead=3, fetch=fake_fetch)
    data = io.BufferedReader(reader, 333).read()
    reader.close()
    assert data == payload
    assert reader.bytes_read == len(payload)
    assert len(calls) == 11


def test_parallel_reader_partial_window():
    payload = b"abcdefghij" * 10
    reader = ParallelRangeReader("u", start=5, stop=25, chunk=7, workers=2,
                                 fetch=lambda u, s, e: payload[s:e + 1])
    assert reader.read(100) + reader.read(100) + reader.read(100) == payload[5:25]
    reader.close()
