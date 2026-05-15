"""Atomic IO for memory/_stock/inventory.json.

Concurrent writers serialised by an exclusive flock on a sibling lockfile
(`inventory.json.lock`) — locking the inventory file itself only works
when every reader/writer opens the same path, and the publish step
swaps the underlying inode via os.replace, which would invalidate any
fd-based lock mid-flight. The dedicated lockfile is opened once per
mutation and held for the full read-modify-write span.

Atomic publish: write-temp-then-rename in the same directory.
See spec §12.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from api.config import get_settings
from api.stock.schemas import (
    ConsumedEvent,
    DonorEntry,
    StockInventory,
)


def _memory_root() -> Path:
    return Path(get_settings().memory_root)


def _stock_root() -> Path:
    return _memory_root() / "_stock"


def _inventory_path() -> Path:
    return _stock_root() / "inventory.json"


def _lock_path() -> Path:
    return _stock_root() / "inventory.json.lock"


@contextlib.contextmanager
def _exclusive_lock():
    """Hold an OS-level exclusive lock on _stock/inventory.json.lock.

    Serialises read-modify-write across processes (uvicorn workers) and
    threads. The lockfile is created on demand; never deleted (avoids
    a delete/create race that would defeat the lock).
    """
    root = _stock_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path()
    with lock_path.open("a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _read_inventory_unlocked() -> StockInventory:
    p = _inventory_path()
    if not p.exists():
        return StockInventory(schema_version="1.0", donors={})
    with p.open("r", encoding="utf-8") as f:
        return StockInventory.model_validate_json(f.read())


def _write_inventory_unlocked(inv: StockInventory) -> None:
    """Atomic publish: write temp file in same dir, fsync, rename."""
    root = _stock_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = inv.model_dump_json(indent=2)
    fd, tmp_path = tempfile.mkstemp(prefix=".inventory.", suffix=".json", dir=str(root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, _inventory_path())
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_inventory() -> StockInventory:
    """Read the inventory under a shared lock (multiple readers OK)."""
    p = _inventory_path()
    if not p.exists():
        return StockInventory(schema_version="1.0", donors={})
    root = _stock_root()
    root.mkdir(parents=True, exist_ok=True)
    with _lock_path().open("a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_SH)
        try:
            return _read_inventory_unlocked()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def save_inventory(inv: StockInventory) -> None:
    """Atomic write under exclusive lock."""
    with _exclusive_lock():
        _write_inventory_unlocked(inv)


def _next_donor_id_unlocked(inv: StockInventory, slug: str) -> str:
    year = datetime.now(UTC).year
    pattern = re.compile(rf"^{re.escape(slug)}-donor-{year}-(\d{{3}})$")
    max_n = 0
    for did in inv.donors:
        m = pattern.match(did)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"{slug}-donor-{year}-{max_n + 1:03d}"


def next_donor_id(slug: str) -> str:
    """Find next NNN counter for slug → '{slug}-donor-{YYYY}-{NNN}'.

    Read-only preview — does not allocate. For atomic allocation, mark_donor
    re-derives the id under the same lock that publishes the new entry.
    """
    return _next_donor_id_unlocked(load_inventory(), slug)


def mark_donor(
    device_slug: str,
    label: str,
    condition: str = "donor_only",
) -> str:
    """Add a donor entry. Returns the generated donor_id.

    Raises FileNotFoundError if the device_slug doesn't exist in memory/.
    The id allocation + write happen under one exclusive lock so concurrent
    callers cannot collide on the same NNN counter.
    """
    if not (_memory_root() / device_slug).exists():
        raise FileNotFoundError(f"device_slug not found in memory/: {device_slug}")

    with _exclusive_lock():
        inv = _read_inventory_unlocked()
        donor_id = _next_donor_id_unlocked(inv, device_slug)
        inv.donors[donor_id] = DonorEntry(
            donor_id=donor_id,
            device_slug=device_slug,
            label=label,
            added_at=datetime.now(UTC),
            condition=condition,  # type: ignore[arg-type]
            consumed={},
        )
        _write_inventory_unlocked(inv)
        return donor_id


def unmark_donor(donor_id: str) -> bool:
    with _exclusive_lock():
        inv = _read_inventory_unlocked()
        if donor_id not in inv.donors:
            return False
        del inv.donors[donor_id]
        _write_inventory_unlocked(inv)
        return True


def consume_part(
    donor_id: str,
    refdes: str,
    repair_id: str | None = None,
    notes: str | None = None,
) -> bool:
    """Mark a refdes consumed on a donor. Idempotent — re-call updates notes."""
    with _exclusive_lock():
        inv = _read_inventory_unlocked()
        if donor_id not in inv.donors:
            return False
        inv.donors[donor_id].consumed[refdes] = ConsumedEvent(
            refdes=refdes,
            consumed_at=datetime.now(UTC),
            repair_id=repair_id,
            notes=notes,
        )
        _write_inventory_unlocked(inv)
        return True


def unconsume_part(donor_id: str, refdes: str) -> bool:
    with _exclusive_lock():
        inv = _read_inventory_unlocked()
        if donor_id not in inv.donors:
            return False
        if refdes not in inv.donors[donor_id].consumed:
            return False
        del inv.donors[donor_id].consumed[refdes]
        _write_inventory_unlocked(inv)
        return True
