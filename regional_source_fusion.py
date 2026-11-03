"""Phase 3B.2 diagnostic regional source fusion.

Fuses calibrated role-specific candidate paths into consensus hypotheses while
preserving unresolved gaps and all alternative candidates. This module never
changes the accepted warp or packaged timing.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence
import math

SCHEMA_VERSION = 1
CLUSTER_TOLERANCE_MS = 70.0
MIN_MULTI_ROLE_CONFIDENCE = 0.52
MIN_SINGLE_ROLE_CONFIDENCE = 0.88
MAX_SINGLE_ROLE_DISPERSION_MS = 55.0
MIN_NEIGHBOUR_SUPPORT = 2
LONG_GAP_MEASURES = 8


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median_absolute_deviation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    centre = median(values)
    return median(abs(value - centre) for value in values)


def _weighted_median(items: Sequence[tuple[float, float]]) -> float:
    ordered = sorted((float(value), max(0.0, float(weight))) for value, weight in items)
    total = sum(weight for _, weight in ordered)
    if total <= 1e-12:
        return median(value for value, _ in ordered)
    target = total / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def role_reliability(instrument_report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Estimate reliability from calibrated paths, without song or role priors."""
    output = {}
    for role, report in (instrument_report.get("roles") or {}).items():
        selected = []
        accepted = []
        for measure in report.get("measures") or []:
            candidate = measure.get("selected_candidate")
            if candidate:
                selected.append(candidate)
                if measure.get("accepted"):
                    accepted.append(candidate)
        residuals = [_number(candidate.get("residual_ms")) for candidate in accepted]
        confidences = [_number(candidate.get("calibrated_confidence")) for candidate in accepted]
        candidate_count = int(report.get("candidate_measure_count", 0) or 0)
        coverage = len(accepted) / candidate_count if candidate_count else 0.0
        dispersion = _median_absolute_deviation(residuals)
        confidence = median(confidences) if confidences else 0.0
        dispersion_term = 1.0 / (1.0 + (dispersion or 250.0) / 75.0)
        reliability = coverage * 0.35 + confidence * 0.40 + dispersion_term * 0.25
        output[role] = {
            "selected_count": len(selected),
            "accepted_count": len(accepted),
            "candidate_measure_count": candidate_count,
            "accepted_coverage": round(coverage, 6),
            "median_confidence": round(confidence, 6),
            "residual_mad_ms": None if dispersion is None else round(dispersion, 3),
            "reliability": round(reliability, 6),
        }
    return output


def _candidate_records(instrument_report: Mapping[str, Any], reliability: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_measure: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for role, report in (instrument_report.get("roles") or {}).items():
        role_weight = _number((reliability.get(role) or {}).get("reliability"), 0.0)
        for measure in report.get("measures") or []:
            playback_measure = int(measure["playback_measure"])
            selected = measure.get("selected_candidate")
            for candidate in measure.get("candidates") or []:
                if candidate.get("corridor_saturated") or not candidate.get("locally_eligible"):
                    continue
                confidence = _number(candidate.get("calibrated_confidence"))
                selected_here = selected is candidate or (
                    selected is not None
                    and candidate.get("residual_ms") == selected.get("residual_ms")
                    and candidate.get("rank") == selected.get("rank")
                )
                by_measure[playback_measure].append({
                    "role": role,
                    "residual_ms": _number(candidate.get("residual_ms")),
                    "confidence": confidence,
                    "role_reliability": role_weight,
                    "fusion_weight": max(0.01, confidence * max(0.10, role_weight)),
                    "rank": candidate.get("rank"),
                    "path_selected": bool(selected_here),
                    "individually_accepted": bool(measure.get("accepted") and selected_here),
                    "neighbour_supported": bool(measure.get("neighbour_supported") and selected_here),
                    "raw": candidate,
                })
    return by_measure


def _clusters(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cluster by residual while allowing only the strongest candidate per role."""
    ordered = sorted(records, key=lambda item: item["residual_ms"])
    clusters: list[list[dict[str, Any]]] = []
    for record in ordered:
        placed = False
        for cluster in clusters:
            centre = median(item["residual_ms"] for item in cluster)
            if abs(record["residual_ms"] - centre) <= CLUSTER_TOLERANCE_MS:
                existing = next((item for item in cluster if item["role"] == record["role"]), None)
                if existing is None:
                    cluster.append(record)
                elif record["fusion_weight"] > existing["fusion_weight"]:
                    cluster.remove(existing)
                    cluster.append(record)
                placed = True
                break
        if not placed:
            clusters.append([record])
    return clusters


def _score_cluster(cluster: Sequence[dict[str, Any]]) -> dict[str, Any]:
    roles = sorted({item["role"] for item in cluster})
    residual = _weighted_median([(item["residual_ms"], item["fusion_weight"]) for item in cluster])
    dispersion = _median_absolute_deviation([item["residual_ms"] for item in cluster]) or 0.0
    confidence = _weighted_median([(item["confidence"], item["fusion_weight"]) for item in cluster])
    selected_count = sum(bool(item["path_selected"]) for item in cluster)
    accepted_count = sum(bool(item["individually_accepted"]) for item in cluster)
    role_bonus = min(0.30, 0.12 * max(0, len(roles) - 1))
    path_bonus = min(0.15, selected_count * 0.05)
    dispersion_penalty = min(0.35, dispersion / 200.0)
    score = max(0.0, min(1.0, confidence + role_bonus + path_bonus - dispersion_penalty))
    return {
        "roles": roles,
        "role_count": len(roles),
        "residual_ms": round(residual, 3),
        "dispersion_ms": round(dispersion, 3),
        "confidence": round(confidence, 6),
        "fusion_score": round(score, 6),
        "path_selected_count": selected_count,
        "individually_accepted_count": accepted_count,
        "contributors": list(cluster),
    }


def _best_hypothesis(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [_score_cluster(cluster) for cluster in _clusters(records)]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["fusion_score"], item["role_count"], item["path_selected_count"]), reverse=True)
    best = candidates[0]
    multi_role = best["role_count"] >= 2 and best["fusion_score"] >= MIN_MULTI_ROLE_CONFIDENCE
    single_role = (
        best["role_count"] == 1
        and best["fusion_score"] >= MIN_SINGLE_ROLE_CONFIDENCE
        and best["dispersion_ms"] <= MAX_SINGLE_ROLE_DISPERSION_MS
        and best["path_selected_count"] >= 1
    )
    best["accepted_hypothesis"] = bool(multi_role or single_role)
    best["acceptance_route"] = "multi_role_consensus" if multi_role else "exceptional_single_role" if single_role else "unresolved"
    best["alternative_hypotheses"] = candidates[1:]
    return best


def _add_neighbour_support(hypotheses: list[dict[str, Any]]) -> None:
    accepted_indices = [index for index, item in enumerate(hypotheses) if item.get("accepted_hypothesis")]
    for index in accepted_indices:
        item = hypotheses[index]
        support = 0
        for other_index in accepted_indices:
            if index == other_index:
                continue
            other = hypotheses[other_index]
            distance = abs(int(item["playback_measure"]) - int(other["playback_measure"]))
            if distance <= 2 and abs(float(item["residual_ms"]) - float(other["residual_ms"])) <= 100.0:
                support += 1
        item["neighbour_hypothesis_support"] = support
        if item["acceptance_route"] == "exceptional_single_role" and support < MIN_NEIGHBOUR_SUPPORT:
            item["accepted_hypothesis"] = False
            item["acceptance_route"] = "unresolved_single_role_without_neighbours"


def _coverage_gaps(measure_numbers: Sequence[int], accepted: set[int]) -> list[dict[str, Any]]:
    gaps = []
    start = None
    previous = None
    for measure in measure_numbers:
        if measure not in accepted:
            if start is None:
                start = measure
            previous = measure
        elif start is not None:
            gaps.append({"start_playback_measure": start, "end_playback_measure": previous,
                         "measure_count": previous - start + 1})
            start = previous = None
    if start is not None:
        gaps.append({"start_playback_measure": start, "end_playback_measure": previous,
                     "measure_count": previous - start + 1})
    for gap in gaps:
        gap["long_gap"] = gap["measure_count"] >= LONG_GAP_MEASURES
        gap["recommended_action"] = "wider_structural_search_or_manual_anchor" if gap["long_gap"] else "retain_unresolved"
    return gaps


def analyse(instrument_report: Mapping[str, Any], beatmap: Mapping[str, Any]) -> dict[str, Any]:
    reliability = role_reliability(instrument_report)
    records = _candidate_records(instrument_report, reliability)
    downbeats = [point for point in beatmap.get("beats") or [] if point.get("structural_role") == "downbeat"]
    measure_numbers = [int(point["identity"]["playback_measure"]) for point in downbeats]
    hypotheses = []
    for measure in measure_numbers:
        best = _best_hypothesis(records.get(measure, []))
        if best is None:
            hypotheses.append({"playback_measure": measure, "accepted_hypothesis": False,
                               "acceptance_route": "no_eligible_candidates", "roles": [],
                               "alternative_hypotheses": []})
        else:
            hypotheses.append({"playback_measure": measure, **best})
    _add_neighbour_support(hypotheses)
    accepted = {int(item["playback_measure"]) for item in hypotheses if item.get("accepted_hypothesis")}
    rescued = [item for item in hypotheses if item.get("accepted_hypothesis")
               and item.get("individually_accepted_count", 0) == 0]
    gaps = _coverage_gaps(measure_numbers, accepted)
    return {
        "schema": "feedpak.regional_source_fusion",
        "schema_version": SCHEMA_VERSION,
        "phase": "3B.2",
        "timing_changes_applied": False,
        "policy": {
            "cluster_tolerance_ms": CLUSTER_TOLERANCE_MS,
            "multi_role_consensus_required_by_default": True,
            "single_role_route": "exceptional confidence plus path and neighbourhood support",
            "all_alternative_hypotheses_retained": True,
            "note": "Fusion increases correspondence density only through independent agreement; unresolved gaps remain visible.",
        },
        "role_reliability": reliability,
        "hypotheses": hypotheses,
        "coverage_gaps": gaps,
        "summary": {
            "playback_measure_count": len(measure_numbers),
            "measures_with_eligible_candidates": sum(bool(records.get(measure)) for measure in measure_numbers),
            "accepted_hypothesis_count": len(accepted),
            "rescued_by_fusion_count": len(rescued),
            "long_gap_count": sum(bool(gap["long_gap"]) for gap in gaps),
            "largest_gap_measures": max([gap["measure_count"] for gap in gaps] or [0]),
        },
    }
