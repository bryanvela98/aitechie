"""Stock search engine — strict-then-tolerant matching with safety filter.

See docs/superpowers/specs/2026-05-08-stock-inventory-design.md §6.
"""

from __future__ import annotations

from pathlib import Path

from api.config import get_settings
from api.stock.safety import classify_safety
from api.stock.schemas import (
    PartsIndex,
    PartsIndexEntry,
    StockSearchMatch,
    StockSearchQuery,
    StockSearchResult,
)
from api.stock.store import load_inventory


def _memory_root() -> Path:
    return Path(get_settings().memory_root)


def _load_parts_index(slug: str) -> PartsIndex | None:
    p = _memory_root() / slug / "parts_index.json"
    if not p.exists():
        return None
    return PartsIndex.model_validate_json(p.read_text(encoding="utf-8"))


def _strict_match(entry: PartsIndexEntry, q: StockSearchQuery) -> bool:
    """Type, package, value_canonical, MPN must match. voltage_rating ≥ requested."""
    if q.type and entry.type != q.type:
        return False
    if q.package and entry.package != q.package:
        return False
    if q.value_canonical and entry.value_canonical != q.value_canonical:
        return False
    if q.mpn and entry.mpn != q.mpn:
        return False
    if q.voltage_min is not None:
        if entry.voltage_rating is None or entry.voltage_rating < q.voltage_min:
            return False
    return True


def _classify_match(entry: PartsIndexEntry, q: StockSearchQuery) -> tuple[str, list[str]]:
    """Return (kind, warnings) where kind ∈ {"exact", "tolerant", "blocked", "no_match"}."""
    if not _strict_match(entry, q):
        return ("no_match", [])

    # Strict succeeded. If voltage_rating > requested, it's still exact-grade
    # (over-spec is fine). The "tolerant" path is reserved for value/tolerance
    # variations — which we do NOT allow on value, so in v1 every strict_match
    # returns exact. The tolerant lane is here for forward-compatibility.

    # Safety filter: if the donor's safety_class is exact_only AND the requested
    # role would be tolerant, the donor's role is by definition critical and
    # we'd refuse substitution — but since we already established a strict
    # value match, this is actually fine: it's not a substitution at all.
    # So strict matches always go through, regardless of safety_class.
    return ("exact", [])


def _block_reason(entry: PartsIndexEntry, q: StockSearchQuery) -> str:
    if entry.safety_class == "exact_only":
        return f"role source critique ({entry.role_in_design or 'inconnu'}) — substitution refusée"
    if q.requested_role and classify_safety(role=q.requested_role, type=q.type or "") == "exact_only":
        return f"role cible critique ({q.requested_role}) — substitution refusée"
    return "fail-safe : rôle non classifiable"


def _rank_key(match: StockSearchMatch) -> tuple:
    """Sort: exact first, low criticality first."""
    crit_order = {"low": 0, "medium": 1, "high": 2}
    return (
        0 if match.match_kind == "exact" else 1,
        crit_order.get(match.criticality_in_donor, 9),
    )


def stock_search(query: StockSearchQuery) -> StockSearchResult:
    inv = load_inventory()
    if not inv.donors:
        return StockSearchResult(empty_reason="no donors in stock")

    exact: list[StockSearchMatch] = []
    tolerant: list[StockSearchMatch] = []
    blocked: list[dict] = []

    for donor in inv.donors.values():
        if donor.donor_id in query.exclude_donors:
            continue
        idx = _load_parts_index(donor.device_slug)
        if idx is None:
            continue  # donor declared but no schematic ingested — skip silently

        for refdes, entry in idx.entries.items():
            if refdes in donor.consumed:
                continue

            kind, warnings = _classify_match(entry, query)
            if kind == "no_match":
                continue
            if kind == "blocked":
                blocked.append({
                    "donor_id": donor.donor_id,
                    "refdes": refdes,
                    "blocked_reason": _block_reason(entry, query),
                })
                continue

            match = StockSearchMatch(
                donor_id=donor.donor_id,
                donor_label=donor.label,
                device_slug=donor.device_slug,
                refdes=refdes,
                value_canonical=entry.value_canonical,
                package=entry.package,
                mpn=entry.mpn,
                voltage_rating=entry.voltage_rating,
                pages=entry.pages,
                criticality_in_donor=entry.criticality_in_design,
                match_kind=kind,  # type: ignore[arg-type]
                substitution_warnings=warnings,
            )
            (exact if kind == "exact" else tolerant).append(match)

    return StockSearchResult(
        exact_matches=sorted(exact, key=_rank_key),
        tolerant_matches=sorted(tolerant, key=_rank_key),
        blocked_substitutes=blocked,
        empty_reason=None if (exact or tolerant) else "no matching parts available",
    )
