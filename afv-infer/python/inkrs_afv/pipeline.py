"""The overlapped column pipeline: tile reads on a thread pool, the network and the column finalise on the device,
zarr writes on one writer thread

A serial predictor runs a column as read (one thread, ahead within the column only) -> forward per batch with a host
sync and a 67 MB device-to-host copy -> blend on the host -> finalise on the host -> write, and only then starts the
next column. Here:
- ``TileFeed`` reads every column's tiles in the reference order on ``readers`` threads, running ahead across
  columns. It drops the tiles that do not run and groups the rest into the reference batches: ``batch`` tiles at a
  time
  within a column, with the column's remainder as its last, smaller batch. The batches are the same tile groups at
  the same sizes, so the network sees exactly the inputs it sees in the reference
- each batch is staged in pinned host memory and copied without a host sync, so the host queues the next batch's
  work while the device is still running the last one
- the blend and finalise run on the device (``finalize.ColumnAccumulator``)
- ``ColumnWriter`` writes each finished column, then its progress record, in column order on one thread. Columns
  of 480 do not align with the 128-voxel output chunks, so neighbours share chunks; writing them one at a time keeps
  every read-modify-write of a shared chunk serial, and a progress record never names a column whose data is not
  written yet
"""

from __future__ import annotations

import collections
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import TILE, read_tile


@dataclass
class Batch:
    """Non-air tiles of one column, staged as a (n, 1, 128, 128, 128) uint8 host tensor"""

    col: int
    tiles: list
    x: object


@dataclass
class ColumnEnd:
    """Every tile of column ``col`` has been read; ``air`` of them were all zero"""

    col: int
    air: int


class FeedError(RuntimeError):
    pass


@dataclass
class _Failed:
    err: BaseException


_DONE = object()


class TileFeed:
    """Reads the tiles of ``columns`` ([(col_index, tiles)]) ahead of the device, in order, as Batch and ColumnEnd
    items on a bounded queue

    By default a tile whose input is all zero is skipped as air, as the band reference skips it. With ``runs`` (a
    predicate on the tile's padded origin, ``villa.VillaSelection.tile_runs``), exactly the tiles it accepts run,
    whatever their values, and the others are counted as skipped without being read"""

    def __init__(self, torch, ct_path: str, z0: int, z1: int, columns, batch: int, readers: int, pin: bool,
                 depth: int = 6, runs=None):
        self._torch = torch
        self._ct_path = ct_path
        self._z0, self._z1 = z0, z1
        if any(not tiles for _, tiles in columns):
            raise ValueError("every column has at least one tile")
        self._columns = columns
        self._batch = batch
        self._readers = readers
        self._pin = pin
        self._runs = runs
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._local = threading.local()
        self._thread = threading.Thread(target=self._run, name="inkrs-afv-feed", daemon=True)
        self._thread.start()

    def _ct(self):
        # one zarr array per reader thread, so no reader shares another's chunk-key state
        ct = getattr(self._local, "ct", None)
        if ct is None:
            import zarr

            ct = self._local.ct = zarr.open_array(self._ct_path, mode="r")
        return ct

    def _read(self, tile):
        return read_tile(self._ct(), self._z0, self._z1, *tile)

    def _stage(self, raws: list[np.ndarray]):
        torch = self._torch
        x = torch.empty((len(raws), 1, TILE, TILE, TILE), dtype=torch.uint8, pin_memory=self._pin)
        view = x.numpy()
        for i, raw in enumerate(raws):
            view[i, 0] = raw
        return x

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            with ThreadPoolExecutor(self._readers, thread_name_prefix="inkrs-afv-read") as pool:
                order = ((ci, t) for ci, tiles in self._columns for t in tiles)
                window: collections.deque = collections.deque()
                # tiles of every column in one ordered stream, `lookahead` reads in flight
                lookahead = 4 * self._readers
                ends = {ci: len(tiles) for ci, tiles in self._columns}
                seen = collections.Counter()
                pending: list = []
                air = 0
                exhausted = False
                while window or not exhausted:
                    while not exhausted and len(window) < lookahead:
                        nxt = next(order, None)
                        if nxt is None:
                            exhausted = True
                            break
                        if self._runs is not None and not self._runs(*nxt[1]):
                            window.append((nxt[0], nxt[1], None))
                        else:
                            window.append((nxt[0], nxt[1], pool.submit(self._read, nxt[1])))
                    if not window:
                        break
                    ci, tile, fut = window.popleft()
                    raw = None if fut is None else fut.result()
                    if self._stop.is_set():
                        return
                    seen[ci] += 1
                    if raw is not None and (self._runs is not None or raw.any()):
                        pending.append((tile, raw))
                    else:
                        air += 1
                    if len(pending) == self._batch:
                        if not self._put(Batch(ci, [t for t, _ in pending], self._stage([r for _, r in pending]))):
                            return
                        pending = []
                    if seen[ci] == ends[ci]:
                        if pending and not self._put(
                                Batch(ci, [t for t, _ in pending], self._stage([r for _, r in pending]))):
                            return
                        if not self._put(ColumnEnd(ci, air)):
                            return
                        pending, air = [], 0
            self._put(_DONE)
        except BaseException as e:  # noqa: BLE001 - re-raised on the consumer's thread
            self._put(_Failed(e))

    def get(self):
        item = self._q.get()
        if isinstance(item, _Failed):
            raise FeedError(f"tile feed failed: {item.err!r}") from item.err
        if item is _DONE:
            raise FeedError("tile feed ended before the column did")
        return item

    def close(self) -> None:
        """Stop reading ahead and wait for the reader thread"""
        self._stop.set()
        while self._thread.is_alive():
            try:
                self._q.get(timeout=0.1)
            except queue.Empty:
                pass
        self._thread.join()


class ColumnWriter:
    """Writes finished columns and then their progress records, in submission order, on one thread"""

    def __init__(self, sem, tra, progress_f: Path, depth: int = 2):
        self._sem, self._tra = sem, tra
        self._progress = progress_f
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._err: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="inkrs-afv-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _DONE:
                return
            # after a failure the rest is discarded; the main thread raises it
            if self._err is not None:
                continue
            try:
                self._write(*item)
            except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
                self._err = e

    def _write(self, col, sem, tra, rec: dict, t0: float) -> None:
        y0, y1, x0, x1 = col
        if sem is not None:
            self._sem[:, :, y0:y1, x0:x1] = sem
            self._tra[:, :, y0 // 2:y0 // 2 + (y1 - y0) // 2, x0 // 2:x0 // 2 + (x1 - x0) // 2] = tra
        rec = dict(rec, wall_s=time.time() - t0, t_end=time.time())
        # the band reference's key order, so the records read the same
        rec = {k: rec[k] for k in ("col", "tiles", "wall_s", "attempt", "t_end", "forward_s", "tiles_run",
                                   "tiles_air", "voxels_uncovered")}
        with self._progress.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def _raise(self) -> None:
        if self._err is not None:
            raise RuntimeError("inkrs-afv column writer failed") from self._err

    def submit(self, col, sem, tra, rec: dict, t0: float) -> None:
        self._raise()
        self._q.put((col, sem, tra, rec, t0))

    def drain(self) -> None:
        """Write everything submitted, stop the thread and raise any write failure"""
        if self._thread.is_alive():
            self._q.put(_DONE)
            self._thread.join()
        self._raise()
