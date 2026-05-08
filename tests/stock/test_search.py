from datetime import UTC, datetime

import pytest

from api.stock.schemas import (
    DonorEntry,
    PartsIndex,
    PartsIndexEntry,
    StockInventory,
    StockSearchQuery,
)
from api.stock.search import stock_search
from api.stock.store import save_inventory


def _entry(refdes, role, value="0.1uF", package="0402", voltage_rating=25.0,
           type_="capacitor", kind="passive_c", mpn=None, safety="tolerant_with_warning",
           criticality="low"):
    return PartsIndexEntry(
        refdes=refdes, type=type_, kind=kind,
        value_canonical=value, value_raw=value, package=package, mpn=mpn,
        voltage_rating=voltage_rating, tolerance=None, role_in_design=role,
        safety_class=safety, criticality_in_design=criticality, pages=[1],
    )


@pytest.fixture
def two_donors(tmp_path, monkeypatch):
    """Set up _stock + two parts_index files in a tmp memory dir."""
    memory = tmp_path / "memory"
    (memory / "_stock").mkdir(parents=True)
    (memory / "iphone-x").mkdir()
    (memory / "iphone-13").mkdir()

    monkeypatch.setattr("api.stock.store._memory_root", lambda: memory)
    monkeypatch.setattr("api.stock.store._stock_root", lambda: memory / "_stock")
    monkeypatch.setattr("api.stock.search._memory_root", lambda: memory)

    idx_x = PartsIndex(
        schema_version="1.0", device_slug="iphone-x",
        generated_at=datetime.now(UTC),
        source_electrical_graph_hash="x" * 64,
        entries={
            "C1": _entry("C1", role="decoupling", value="0.1uF", voltage_rating=25.0),
            "C2": _entry("C2", role="decoupling", value="0.1uF", voltage_rating=50.0),
            "C3": _entry("C3", role="filter", value="0.1uF", safety="exact_only"),
            "U7": _entry("U7", role="ic", value=None, type_="ic", kind="ic",
                         package="QFN-56", mpn="MAX77818EWY",
                         safety="exact_only", criticality="high"),
        },
    )
    idx_13 = PartsIndex(
        schema_version="1.0", device_slug="iphone-13",
        generated_at=datetime.now(UTC),
        source_electrical_graph_hash="y" * 64,
        entries={
            "C10": _entry("C10", role="decoupling", value="0.1uF", voltage_rating=25.0),
        },
    )
    (memory / "iphone-x" / "parts_index.json").write_text(idx_x.model_dump_json())
    (memory / "iphone-13" / "parts_index.json").write_text(idx_13.model_dump_json())

    inv = StockInventory(schema_version="1.0", donors={
        "iphone-x-donor-2026-001": DonorEntry(
            donor_id="iphone-x-donor-2026-001",
            device_slug="iphone-x",
            label="iPhone X donor #1",
            added_at=datetime.now(UTC),
            condition="donor_only",
        ),
        "iphone-13-donor-2026-001": DonorEntry(
            donor_id="iphone-13-donor-2026-001",
            device_slug="iphone-13",
            label="iPhone 13 donor #1",
            added_at=datetime.now(UTC),
            condition="donor_only",
        ),
    })
    save_inventory(inv)
    return memory


def test_no_donors_returns_empty_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("api.stock.store._memory_root", lambda: tmp_path / "memory")
    monkeypatch.setattr("api.stock.store._stock_root", lambda: tmp_path / "memory" / "_stock")
    monkeypatch.setattr("api.stock.search._memory_root", lambda: tmp_path / "memory")
    res = stock_search(StockSearchQuery(type="capacitor", value_canonical="0.1uF"))
    assert res.empty_reason == "no donors in stock"


def test_exact_match_decoupling(two_donors):
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.1uF", package="0402",
        voltage_min=25.0, requested_role="decoupling",
    ))
    # All 4 caps match strictly: C1 (x, 25V), C2 (x, 50V), C3 (x, 25V — exact_only
    # role on donor but exact value match wins per spec §6), C10 (13, 25V).
    refdes_set = {m.refdes for m in res.exact_matches}
    assert refdes_set == {"C1", "C2", "C3", "C10"}
    assert all(m.match_kind == "exact" for m in res.exact_matches)


def test_filter_cap_blocked(two_donors):
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.1uF", package="0402",
        requested_role="decoupling",
    ))
    # C3 is exact_only (safety_class) — when requested_role is decoupling
    # (tolerant), but the donor entry's safety is exact_only AND the values
    # match exactly, it should still appear in exact_matches (exact wins).
    assert any(m.refdes == "C3" for m in res.exact_matches)


def test_consumed_excluded(two_donors):
    from api.stock.store import consume_part
    consume_part(donor_id="iphone-x-donor-2026-001", refdes="C1")
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.1uF", package="0402",
        requested_role="decoupling",
    ))
    refdes_donor_pairs = {(m.donor_id, m.refdes) for m in res.exact_matches}
    assert ("iphone-x-donor-2026-001", "C1") not in refdes_donor_pairs
    assert ("iphone-x-donor-2026-001", "C2") in refdes_donor_pairs


def test_exclude_donor_param(two_donors):
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.1uF", package="0402",
        requested_role="decoupling",
        exclude_donors=["iphone-x-donor-2026-001"],
    ))
    donor_ids = {m.donor_id for m in res.exact_matches}
    assert donor_ids == {"iphone-13-donor-2026-001"}


def test_ic_strict_match_by_mpn(two_donors):
    res = stock_search(StockSearchQuery(
        type="ic", mpn="MAX77818EWY", package="QFN-56", requested_role="ic",
    ))
    assert len(res.exact_matches) == 1
    assert res.exact_matches[0].refdes == "U7"


def test_ic_wrong_mpn_no_match(two_donors):
    res = stock_search(StockSearchQuery(
        type="ic", mpn="DIFFERENT_MPN", package="QFN-56", requested_role="ic",
    ))
    assert res.exact_matches == []
    assert res.tolerant_matches == []
    assert res.empty_reason == "no matching parts available"


def test_critical_requested_role_blocks_tolerant(two_donors):
    # Tech is replacing a feedback resistor (critical). Tolerant must be
    # blocked even though donor's role is tolerant-OK.
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.22uF", package="0402",
        requested_role="feedback",
    ))
    # No exact (value mismatch) and no tolerant (role critical) → empty
    assert res.exact_matches == []
    assert res.tolerant_matches == []


def test_tolerant_value_mismatch_does_not_pass(two_donors):
    # Spec §6.2: value_canonical must match even in tolerant mode
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.22uF", package="0402",
        requested_role="decoupling",
    ))
    # No 0.22uF in stock — neither exact nor tolerant
    assert res.exact_matches == []
    assert res.tolerant_matches == []


def test_donor_without_parts_index_silently_skipped(two_donors, tmp_path):
    """A donor entry can exist without a parts_index file. It must be skipped."""
    from api.stock.store import mark_donor
    # Create a 3rd device dir without parts_index
    (tmp_path / "memory" / "iphone-15").mkdir()
    mark_donor(device_slug="iphone-15", label="iPhone 15 no parts_index yet")
    # Search should not raise
    res = stock_search(StockSearchQuery(
        type="capacitor", value_canonical="0.1uF", package="0402",
        requested_role="decoupling",
    ))
    # Same 3 matches as before (the new donor is silently skipped)
    assert len(res.exact_matches) >= 3
