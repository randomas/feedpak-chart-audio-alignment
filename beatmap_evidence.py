"""Phase 3.1: instrument-aware evidence interpretation and strategy diagnostics.

Production timing is never changed here. The module normalizes existing
alignment diagnostics, annotates playback-expanded downbeats, measures regional
residual behaviour, distinguishes plausible failure mechanisms, and recommends
an alignment strategy for a later proposal stage.

All heuristics are song-independent. Instrument/source roles and feature
families remain explicit so future adapters can use optimized evidence.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from statistics import median
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 3
SOURCE_POLICY = "instrument_aware_adapters_with_shared_schema"

FAMILY_RHYTHMIC = "rhythmic_onset"
FAMILY_HARMONIC = "harmonic_context"
FAMILY_STRUCTURAL = "structural_boundary"
FAMILY_PATH = "path_hypothesis"
FAMILY_MANUAL = "manual_reference"

# Diagnostic interpretation thresholds. These do not authorize timing changes.
MIN_TREND_OBSERVATIONS = 4
MIN_TREND_SPAN_SECONDS = 30.0
MIN_DRIFT_SLOPE_MS_PER_SECOND = 1.0
MIN_ACCUMULATED_DRIFT_MS = 75.0
MIN_OFFSET_MS = 80.0
HIGH_DISPERSION_MS = 180.0
SOURCE_SCALE_CONFLICT = 0.01


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _labels(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(value) for value in (values or []) if value is not None})


def _residual_seconds(item: Mapping[str, Any]) -> float | None:
    for key in ("residual_seconds", "median_correction_seconds", "combined_residual_seconds"):
        value = _number(item.get(key))
        if value is not None:
            return value
    value = _number(item.get("combined_residual_ms"))
    return None if value is None else value / 1000.0


def _measure_from_score_positions(item: Mapping[str, Any]) -> int | None:
    positions = item.get("score_positions") or []
    if positions:
        value = positions[0].get("playback_bar")
        return None if value is None else int(value)
    value = item.get("playback_bar", item.get("measure"))
    return None if value is None else int(value)


def _details(item: Mapping[str, Any], allowed: Iterable[str]) -> dict[str, Any]:
    return {key: deepcopy(item[key]) for key in allowed if key in item}


def _observation(family: str, method: str, measure: int, item: Mapping[str, Any], *,
                 accepted: bool | None = None, source_roles: Iterable[Any] | None = None,
                 confidence: float | None = None, subtype: str | None = None,
                 details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    residual = _residual_seconds(item)
    return {
        "evidence_family": family,
        "method": method,
        "subtype": subtype,
        "playback_measure": int(measure),
        "accepted": bool(item.get("accepted")) if accepted is None else bool(accepted),
        "residual_ms": None if residual is None else round(residual * 1000.0, 3),
        "source_roles": _labels(source_roles),
        "confidence": None if confidence is None else round(float(confidence), 6),
        "warnings": list(item.get("warnings") or []),
        "details": dict(details or {}),
    }


def _periodic_observations(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = decision.get("checkpoint_diagnostics") or {}
    diagnostic_roles = diagnostics.get("source_roles") or diagnostics.get("sources") or []
    output = []
    for item in diagnostics.get("checkpoints") or []:
        if item.get("measure") is None:
            continue
        output.append(_observation(
            FAMILY_RHYTHMIC, "periodic_event_checkpoint", int(item["measure"]), item,
            source_roles=item.get("source_roles") or item.get("sources") or diagnostic_roles,
            details=_details(item, ("matched_events", "mad_seconds", "predicted_time", "nominal_time")),
        ))
    return output


def _structural_observations(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    diagnostics = decision.get("silence_checkpoint_diagnostics") or {}
    for item in diagnostics.get("checkpoints") or []:
        if item.get("measure") is None:
            continue
        confirmed = item.get("audio_confirmed_stems") or []
        supporting = item.get("supporting_stems") or []
        entering = item.get("boundary_entering_stems_120ms") or []
        roles = confirmed or supporting
        recognizable = bool(item.get("boundary_recognizable"))
        subtype = ("multi_instrument_entrance" if len(_labels(entering or roles)) >= 2
                   else "single_instrument_entrance") if recognizable else "source_inactivity_interval"
        confirmations = item.get("audio_confirmation") or {}
        confidence = max([_number(value.get("confidence")) or 0.0 for value in confirmations.values()] or [0.0])
        output.append(_observation(
            FAMILY_STRUCTURAL,
            "score_audio_activity_boundary" if recognizable else "score_audio_inactivity_interval",
            int(item["measure"]), item, source_roles=roles, confidence=confidence, subtype=subtype,
            details={
                **_details(item, ("score_start", "score_end", "predicted_audio_start", "predicted_audio_end",
                                  "duration_seconds", "duration_beats", "rule", "post_silence_clusters_750ms",
                                  "boundary_recognizable")),
                "supporting_source_roles": _labels(supporting),
                "confirmed_source_roles": _labels(confirmed),
                "boundary_source_roles": _labels(entering),
                "source_confirmation": deepcopy(confirmations),
                "full_mix_confirmation": deepcopy(item.get("full_mix_confirmation")),
            },
        ))
    return output


def _harmonic_observations(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    diagnostics = decision.get("chroma_checkpoint_diagnostics") or {}
    for item in diagnostics.get("checkpoints") or []:
        if item.get("measure") is None:
            continue
        sources = deepcopy(item.get("sources") or {})
        available = [role for role, value in sources.items() if value.get("available")]
        accepted_roles = item.get("strong_isolated_sources") or []
        output.append(_observation(
            FAMILY_HARMONIC, "windowed_chroma_checkpoint", int(item["measure"]), item,
            source_roles=accepted_roles or available,
            details={
                **_details(item, ("linear_prediction", "window_nominal_start", "window_nominal_end",
                                  "cross_stem_disagreement_ms", "onset_residual_ms",
                                  "chroma_onset_disagreement_ms", "combined_candidate_time",
                                  "combined_residual_ms")),
                "source_evidence": sources,
                "accepted_source_roles": _labels(accepted_roles),
            },
        ))
    return output


def _path_observations(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    preview = decision.get("continuous_anchor_path_preview") or {}
    for item in preview.get("selected_anchors") or []:
        measure = _measure_from_score_positions(item)
        if measure is None:
            continue
        roles = item.get("sources") or item.get("supporting_score_sources") or []
        output.append(_observation(
            FAMILY_PATH, "selected_continuous_anchor_hypothesis", measure, item,
            accepted=True, source_roles=roles, confidence=item.get("confidence"),
            subtype="cross_source_consensus" if len(_labels(roles)) >= 2 else "single_source_hypothesis",
            details={
                **_details(item, ("audio_time", "score_time", "cluster_index", "emission_score",
                                  "supporting_source_count", "single_stem_sufficient", "score_event_ids",
                                  "score_positions")),
                "observations": deepcopy(item.get("observations") or []),
            },
        ))
    return output


def _manual_observations(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in (decision.get("manual_anchor_quality") or {}).get("anchors") or []:
        measure = _measure_from_score_positions(item)
        if measure is not None:
            output.append(_observation(
                FAMILY_MANUAL, "manual_anchor", measure, item, accepted=True,
                source_roles=["manual"], confidence=1.0, subtype="reviewed_reference",
                details=deepcopy(item),
            ))
    return output


def normalize_evidence(alignment_report: Mapping[str, Any],
                       instrument_evidence_report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    decision = alignment_report.get("decision") or {}
    observations = (_periodic_observations(decision) + _structural_observations(decision)
                    + _harmonic_observations(decision) + _path_observations(decision)
                    + _manual_observations(decision))
    if instrument_evidence_report:
        import instrument_evidence as ie
        observations.extend(ie.to_shared_observations(instrument_evidence_report))
    return observations


def source_scale_diagnostics(alignment_report: Mapping[str, Any]) -> dict[str, Any]:
    """Expose accepted instrument scale disagreement without selecting a winner."""
    decision = alignment_report.get("decision") or {}
    instrument_scale = decision.get("instrument_scale") or {}
    candidates = []
    for item in instrument_scale.get("candidates") or []:
        scale = _number(item.get("scale", item.get("measured_scale")))
        source = item.get("source")
        accepted = bool(item.get("accepted"))
        if scale is not None and source is not None:
            candidates.append({"source_role": str(source), "scale": scale, "accepted": accepted,
                               "warnings": list(item.get("warnings") or []),
                               "coverage": item.get("coverage", item.get("score_span_coverage")),
                               "event_count": item.get("events", item.get("event_count"))})
    accepted_values = [item["scale"] for item in candidates if item["accepted"]]
    spread = max(accepted_values) - min(accepted_values) if len(accepted_values) >= 2 else None
    return {
        "candidates": candidates,
        "accepted_source_count": len(accepted_values),
        "accepted_scale_spread": None if spread is None else round(spread, 9),
        "source_weighting_conflict": bool(spread is not None and spread >= SOURCE_SCALE_CONFLICT),
    }


def _theil_sen(points: list[tuple[float, float]]) -> float | None:
    slopes = []
    for index, (x1, y1) in enumerate(points):
        for x2, y2 in points[index + 1:]:
            if x2 > x1:
                slopes.append((y2 - y1) / (x2 - x1))
    return median(slopes) if slopes else None


def _trend(samples: list[dict[str, Any]]) -> dict[str, Any]:
    points = sorted((float(sample["audio_time"]), float(sample["residual_ms"])) for sample in samples)
    residuals = [value for _, value in points]
    centre = median(residuals) if residuals else None
    mad = median(abs(value - centre) for value in residuals) if residuals else None
    slope = _theil_sen(points)
    span = points[-1][0] - points[0][0] if len(points) >= 2 else 0.0
    accumulated = slope * span if slope is not None else None
    return {
        "observation_count": len(samples), "audio_span_seconds": round(span, 6),
        "median_residual_ms": None if centre is None else round(centre, 3),
        "residual_mad_ms": None if mad is None else round(mad, 3),
        "robust_slope_ms_per_second": None if slope is None else round(slope, 6),
        "diagnostic_accumulated_drift_ms": None if accumulated is None else round(accumulated, 3),
    }


def _sample(point: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "beat_id": point["identity"]["beat_id"],
        "playback_measure": point["identity"]["playback_measure"],
        "audio_time": point["timing"]["audio_time"], "residual_ms": observation["residual_ms"],
        "evidence_family": observation["evidence_family"], "method": observation["method"],
        "subtype": observation.get("subtype"), "source_roles": list(observation.get("source_roles") or []),
    }


def _group_trends(samples: list[dict[str, Any]], key: str, values: Iterable[str]) -> dict[str, Any]:
    return {value: _trend([sample for sample in samples
                           if value in (sample.get(key) if isinstance(sample.get(key), list) else [sample.get(key)])])
            for value in sorted(set(values))}


def _eligible(trend: Mapping[str, Any]) -> bool:
    return (trend["observation_count"] >= MIN_TREND_OBSERVATIONS
            and trend["audio_span_seconds"] >= MIN_TREND_SPAN_SECONDS)


def _significant_drift(trend: Mapping[str, Any]) -> bool:
    slope = trend.get("robust_slope_ms_per_second")
    accumulated = trend.get("diagnostic_accumulated_drift_ms")
    return bool(_eligible(trend) and slope is not None and accumulated is not None
                and abs(slope) >= MIN_DRIFT_SLOPE_MS_PER_SECOND
                and abs(accumulated) >= MIN_ACCUMULATED_DRIFT_MS)


def _shape_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 6:
        return {"direction_reversal": False, "first_half": _trend([]), "second_half": _trend([])}
    ordered = sorted(samples, key=lambda sample: sample["audio_time"])
    middle = len(ordered) // 2
    first = _trend(ordered[:middle])
    second = _trend(ordered[middle:])
    first_slope = first.get("robust_slope_ms_per_second")
    second_slope = second.get("robust_slope_ms_per_second")
    reversal = bool(_significant_drift(first) and _significant_drift(second)
                    and first_slope is not None and second_slope is not None
                    and first_slope * second_slope < 0)
    return {"direction_reversal": reversal, "first_half": first, "second_half": second}


def _interpret(family_trends: Mapping[str, Any], source_trends: Mapping[str, Any],
               all_trend: Mapping[str, Any], shape: Mapping[str, Any],
               scale_diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    eligible_families = {name: trend for name, trend in family_trends.items() if _eligible(trend)}
    significant_families = {name: trend for name, trend in eligible_families.items() if _significant_drift(trend)}
    significant_sources = {name: trend for name, trend in source_trends.items() if _significant_drift(trend)}
    warnings = []
    basis = []
    if all_trend.get("residual_mad_ms") is not None and all_trend["residual_mad_ms"] > HIGH_DISPERSION_MS:
        warnings.append("high_residual_dispersion")
    family_slopes = [trend["robust_slope_ms_per_second"] for trend in eligible_families.values()
                     if trend.get("robust_slope_ms_per_second") is not None]
    if len(family_slopes) >= 2 and min(family_slopes) < -MIN_DRIFT_SLOPE_MS_PER_SECOND < max(family_slopes):
        warnings.append("evidence_family_disagreement")
    source_slopes = [trend["robust_slope_ms_per_second"] for name, trend in source_trends.items()
                     if _eligible(trend) and trend.get("robust_slope_ms_per_second") is not None]
    if len(source_slopes) >= 2 and min(source_slopes) < -MIN_DRIFT_SLOPE_MS_PER_SECOND < max(source_slopes):
        warnings.append("source_role_disagreement")
    if scale_diagnostics.get("source_weighting_conflict"):
        warnings.append("source_scale_disagreement")
        basis.append("accepted_source_scales_diverge")
    if shape.get("direction_reversal"):
        basis.append("residual_direction_reversal")
    if _significant_drift(all_trend):
        basis.extend(["long_evidence_span", "accumulated_drift_above_threshold"])
    elif (all_trend.get("robust_slope_ms_per_second") is not None
          and abs(all_trend["robust_slope_ms_per_second"]) >= MIN_DRIFT_SLOPE_MS_PER_SECOND):
        basis.append("slope_present_but_accumulated_change_minor")
    median_residual = all_trend.get("median_residual_ms")
    if median_residual is not None and abs(median_residual) >= MIN_OFFSET_MS:
        basis.append("median_offset_above_threshold")

    if not eligible_families:
        short_but_observed = (all_trend.get("observation_count", 0) >= MIN_TREND_OBSERVATIONS
                              and all_trend.get("diagnostic_accumulated_drift_ms") is not None
                              and abs(all_trend["diagnostic_accumulated_drift_ms"]) < MIN_ACCUMULATED_DRIFT_MS)
        classification = "minor_local_variation" if short_but_observed else "insufficient_evidence"
        clock_behavior = "stable_clock" if short_but_observed else "uncertain"
    elif shape.get("direction_reversal"):
        classification = "possible_nonlinear_timing"
        clock_behavior = "nonlinear_expressive_or_structural_timing"
    elif warnings and any(warning in warnings for warning in ("evidence_family_disagreement", "source_role_disagreement")):
        classification = "conflicting_evidence"
        clock_behavior = "uncertain"
    elif significant_families or _significant_drift(all_trend):
        classification = "possible_drift"
        clock_behavior = "regional_scale_mismatch"
    elif median_residual is not None and abs(median_residual) >= MIN_OFFSET_MS:
        classification = "possible_offset"
        clock_behavior = "constant_offset"
    elif (all_trend.get("diagnostic_accumulated_drift_ms") is not None
          and abs(all_trend["diagnostic_accumulated_drift_ms"]) < MIN_ACCUMULATED_DRIFT_MS):
        classification = "minor_local_variation"
        clock_behavior = "stable_clock"
    else:
        classification = "stable"
        clock_behavior = "stable_clock"

    if scale_diagnostics.get("source_weighting_conflict"):
        evidence_behavior = "source_weighting_conflict"
        strategy = "test_regional_source_fusion"
    elif shape.get("direction_reversal"):
        evidence_behavior = "residual_direction_reversal"
        strategy = "test_structural_jump_then_regional_alignment"
    elif classification == "possible_drift":
        evidence_behavior = "family_supported_residual_trend"
        strategy = "test_piecewise_regional_warp"
    elif classification in ("conflicting_evidence", "insufficient_evidence"):
        evidence_behavior = "sparse_or_conflicting_evidence"
        strategy = "manual_anchor_assisted"
    else:
        evidence_behavior = "consistent_evidence"
        strategy = "retain_global_map"

    agreement_count = len(significant_families)
    confidence = ("high" if agreement_count >= 2 and not warnings
                  else "moderate" if classification not in ("insufficient_evidence", "conflicting_evidence")
                  else "low")
    return {
        "classification": classification,
        "confidence": confidence,
        "warnings": sorted(set(warnings)),
        "basis": sorted(set(basis)),
        "structural_correspondence": "possible_insertion_or_deletion" if shape.get("direction_reversal") else "continuous_or_undetermined",
        "clock_behavior": clock_behavior,
        "evidence_behavior": evidence_behavior,
        "recommended_alignment_strategy": strategy,
    }


def analyze_regions(beatmap: Mapping[str, Any], observations: list[dict[str, Any]],
                    scale_diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    downbeats = {int(point["identity"]["playback_measure"]): point for point in beatmap.get("beats") or []
                 if point.get("structural_role") == "downbeat"}
    segment_measures: dict[int, list[int]] = defaultdict(list)
    for measure, point in downbeats.items():
        segment_measures[int(point["identity"]["playback_segment"])].append(measure)
    accepted = [observation for observation in observations if observation.get("accepted")
                and observation.get("residual_ms") is not None
                and int(observation["playback_measure"]) in downbeats]
    boundaries = {int(observation["playback_measure"]) for observation in observations
                  if observation.get("accepted") and observation.get("evidence_family") == FAMILY_STRUCTURAL
                  and observation.get("subtype") in ("multi_instrument_entrance", "single_instrument_entrance")}
    results = []
    for segment, measures in sorted(segment_measures.items()):
        measures = sorted(measures)
        cuts = sorted(measure for measure in boundaries if measure in measures and measure != measures[0])
        starts = [measures[0]] + cuts
        ends = [measure - 1 for measure in cuts] + [measures[-1]]
        for region_index, (start, end) in enumerate(zip(starts, ends)):
            samples = []
            for observation in accepted:
                measure = int(observation["playback_measure"])
                point = downbeats[measure]
                if int(point["identity"]["playback_segment"]) == segment and start <= measure <= end:
                    samples.append(_sample(point, observation))
            families = [sample["evidence_family"] for sample in samples]
            roles = [role for sample in samples for role in sample["source_roles"]]
            all_trend = _trend(samples)
            family_trends = _group_trends(samples, "evidence_family", families)
            source_trends = _group_trends(samples, "source_roles", roles)
            consensus_trend = _trend([sample for sample in samples if len(sample["source_roles"]) >= 2])
            shape = _shape_diagnostics(samples)
            interpretation = _interpret(family_trends, source_trends, all_trend, shape, scale_diagnostics)
            results.append({
                "region_id": f"seg{segment:03d}.r{region_index:03d}", "playback_segment": segment,
                "start_playback_measure": start, "end_playback_measure": end,
                "left_boundary": "activity_boundary" if start in boundaries else "playback_segment_start",
                "all_evidence_trend": all_trend, "family_trends": family_trends,
                "source_role_trends": source_trends, "cross_source_consensus_trend": consensus_trend,
                "residual_shape": shape, "diagnostic_interpretation": interpretation,
                # Compatibility alias for Phase 3A report readers.
                "combined_interpretation": interpretation,
                "samples": samples,
            })
    _annotate_adjacent_reversals(results)
    return results


def _annotate_adjacent_reversals(regions: list[dict[str, Any]]) -> None:
    """Flag sign reversals across neighbouring substantial regions."""
    by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for region in regions:
        by_segment[int(region["playback_segment"])].append(region)
    for segment_regions in by_segment.values():
        segment_regions.sort(key=lambda region: region["start_playback_measure"])
        for left, right in zip(segment_regions, segment_regions[1:]):
            lt = left["all_evidence_trend"]
            rt = right["all_evidence_trend"]
            ls = lt.get("robust_slope_ms_per_second")
            rs = rt.get("robust_slope_ms_per_second")
            if not (_significant_drift(lt) and _significant_drift(rt)
                    and ls is not None and rs is not None and ls * rs < 0):
                continue
            for region in (left, right):
                interpretation = region["diagnostic_interpretation"]
                interpretation["structural_correspondence"] = "possible_insertion_or_deletion"
                interpretation["evidence_behavior"] = "adjacent_region_direction_reversal"
                interpretation["recommended_alignment_strategy"] = "test_structural_jump_then_regional_alignment"
                interpretation["basis"] = sorted(set(interpretation.get("basis", []) + ["adjacent_region_direction_reversal"]))


def annotate_beatmap(beatmap: Mapping[str, Any], alignment_report: Mapping[str, Any],
                     instrument_evidence_report: Mapping[str, Any] | None = None):
    annotated = deepcopy(beatmap)
    observations = normalize_evidence(alignment_report, instrument_evidence_report)
    scale_diagnostics = source_scale_diagnostics(alignment_report)
    by_measure: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_measure[int(observation["playback_measure"])].append(observation)
    observed = accepted_count = 0
    for point in annotated.get("beats") or []:
        if point.get("structural_role") != "downbeat":
            continue
        items = by_measure.get(int(point["identity"]["playback_measure"]), [])
        observed += bool(items)
        accepted_count += any(item.get("accepted") for item in items)
        point["evidence"] = {
            "status": "supported" if any(item.get("accepted") for item in items)
                      else "observed_unaccepted" if items else "not_observed",
            "observations": items,
        }
    regions = analyze_regions(annotated, observations, scale_diagnostics)
    strategy_counts: dict[str, int] = defaultdict(int)
    for region in regions:
        strategy_counts[region["diagnostic_interpretation"]["recommended_alignment_strategy"]] += 1
    report = {
        "schema": "feedpak.beatmap.evidence.report", "schema_version": SCHEMA_VERSION,
        "phase": "3B", "timing_changes_applied": False, "source_policy": SOURCE_POLICY,
        "instrument_adapter_summary": dict((instrument_evidence_report or {}).get("summary") or {}),
        "thresholds": {
            "minimum_trend_observations": MIN_TREND_OBSERVATIONS,
            "minimum_trend_span_seconds": MIN_TREND_SPAN_SECONDS,
            "minimum_drift_slope_ms_per_second": MIN_DRIFT_SLOPE_MS_PER_SECOND,
            "minimum_accumulated_drift_ms": MIN_ACCUMULATED_DRIFT_MS,
            "minimum_offset_ms": MIN_OFFSET_MS,
            "source_scale_conflict": SOURCE_SCALE_CONFLICT,
        },
        "source_scale_diagnostics": scale_diagnostics,
        "summary": {
            "normalized_observation_count": len(observations),
            "downbeats_with_observations": observed,
            "downbeats_with_accepted_evidence": accepted_count,
            "region_count": len(regions),
            "recommended_strategy_counts": dict(sorted(strategy_counts.items())),
        },
        "regions": regions,
        "notes": [
            "Diagnostic only: no beat or tempo times were changed.",
            "Measured facts are separated from diagnostic hypotheses and strategy recommendations.",
            "Structural, clock, and source-weighting interpretations are provisional.",
            "No song names or song-specific measure rules are used.",
            "Instrument adapters use role-optimized features but share one observation schema.",
        ],
    }
    annotated.setdefault("metadata", {})["evidence_annotation"] = "phase_3b_instrument_adapters_diagnostic_only"
    return annotated, report
