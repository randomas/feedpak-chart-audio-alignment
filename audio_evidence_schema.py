"""v16.7R2 common schema and read-only adapters for existing audio evidence.

This module does not decode audio, compute features, score candidates, choose
paths, create anchors, or alter timing.  Adapters deep-copy values from existing
reports into a stable JSON-serializable evidence envelope with explicit clocks
and provenance.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence
import math

VERSION = "v16.7R2-audio-evidence-schema"
AUTHORITY_DIAGNOSTIC = "diagnostic"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _copy_json(value: Any) -> Any:
    """Deep-copy JSON-like input without coercing missing data to zero."""
    return deepcopy(value)


def _clock(**values: Any) -> dict[str, float]:
    allowed = ("source_audio_time", "padded_audio_time", "chart_time", "mapped_audio_time")
    out: dict[str, float] = {}
    for key in allowed:
        parsed = _number(values.get(key))
        if parsed is not None:
            out[key] = parsed
    return out


def _strings(values: Iterable[Any] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values if value is not None and str(value)]


def make_evidence(*, evidence_id: str, evidence_type: str, detector_module: str,
                  detector_method: str, detector_version: Any = 1,
                  source: str | None = None, nominal_role: str | None = None,
                  observed_signal_type: str | None = None,
                  clocks: Mapping[str, Any] | None = None,
                  residual_seconds: Any = None, confidence: Any = None,
                  positive_factors: Iterable[Any] | None = None,
                  limitations: Iterable[Any] | None = None,
                  authority: str = AUTHORITY_DIAGNOSTIC,
                  provenance: Mapping[str, Any] | None = None,
                  native_quality: Mapping[str, Any] | None = None,
                  adapter_status: str = "complete",
                  original: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build one immutable diagnostic evidence envelope."""
    if not evidence_id or not evidence_type or not detector_module or not detector_method:
        raise ValueError("evidence_id, evidence_type, detector_module, and detector_method are required")
    result: dict[str, Any] = {
        "schema_version": VERSION,
        "evidence_id": str(evidence_id),
        "evidence_type": str(evidence_type),
        "detector": {"module": str(detector_module), "method": str(detector_method),
                     "version": _copy_json(detector_version)},
        "clocks": _clock(**dict(clocks or {})),
        "positive_factors": _strings(positive_factors),
        "limitations": _strings(limitations),
        "authority": str(authority),
        "diagnostic_only": True,
        "timing_changes_applied": False,
        "midi_authority_changed": False,
        "provenance": _copy_json(dict(provenance or {})),
        "adapter_status": str(adapter_status),
        "native_quality": _copy_json(dict(native_quality or {})),
    }
    for key, value in (("source", source), ("nominal_role", nominal_role),
                       ("observed_signal_type", observed_signal_type)):
        if value is not None:
            result[key] = str(value)
    residual = _number(residual_seconds)
    if residual is not None:
        result["residual_seconds"] = residual
    conf = _number(confidence)
    if conf is not None:
        result["confidence"] = max(0.0, min(1.0, conf))
    if original is not None:
        result["original_evidence"] = _copy_json(dict(original))
    return result


def _opening_confidence(row: Mapping[str, Any]) -> float | None:
    for key in ("confidence", "score", "probability"):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def adapt_opening_report(report: Mapping[str, Any] | None, *, padding_added: float = 0.0) -> list[dict[str, Any]]:
    """Adapt the exact `decision.opening_anchor` report shape."""
    if not report:
        return []
    score = _number(report.get("score_anchor")); audio = _number(report.get("audio_anchor"))
    status = str(report.get("status", "status_only"))
    limitations = list(report.get("warnings") or [])
    if score is None or audio is None:
        return [make_evidence(evidence_id="opening-status-1", evidence_type="score_informed_opening_status",
            detector_module="process_gp_alignment", detector_method="choose_opening_anchor",
            observed_signal_type="opening_selection_status", limitations=limitations+[status],
            provenance={"origin_report":"decision.opening_anchor"}, adapter_status="status_only", original=report)]
    selected=list(report.get("selected_sources") or [])
    return [make_evidence(evidence_id="opening-selected-1", evidence_type="score_informed_opening_anchor",
        detector_module="process_gp_alignment", detector_method="choose_opening_anchor",
        source="+".join(selected) if selected else None, observed_signal_type="selected_opening_anchor",
        clocks={"source_audio_time":audio-float(padding_added),"padded_audio_time":audio,"chart_time":score,"mapped_audio_time":audio},
        residual_seconds=audio-score, positive_factors=[status], limitations=limitations,
        provenance={"origin_report":"decision.opening_anchor","selected_sources":selected},
        native_quality={"source_count":len(selected)}, adapter_status="complete", original=report)]

def adapt_opening_candidates(candidates: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
                             *, padding_added: float = 0.0,
                             detector_version: Any = 1) -> list[dict[str, Any]]:
    """Adapt existing score-informed opening candidates without re-ranking them."""
    if isinstance(candidates, Mapping):
        rows = candidates.get("candidates") or []
    else:
        rows = candidates or []
    output = []
    for index, row in enumerate(rows):
        source = row.get("source") or row.get("role")
        chart = _number(row.get("score_time", row.get("score_start")))
        padded = _number(row.get("audio_time", row.get("observed_time", row.get("opening_time"))))
        source_time = None if padded is None else padded - float(padding_added)
        expected = _number(row.get("expected_time", row.get("mapped_audio_time")))
        residual = _number(row.get("residual_seconds"))
        if residual is None and padded is not None and expected is not None:
            residual = padded - expected
        limitations = list(row.get("warnings") or [])
        if source and source != "full":
            limitations.append("nominal_role_not_verified")
        if not row.get("accepted", row.get("available", True)):
            limitations.append("not_selected_by_origin_detector")
        output.append(make_evidence(
            evidence_id=f"opening-{index+1}", evidence_type="score_informed_opening_activity",
            detector_module="process_gp_alignment", detector_method="build_opening_anchor_candidates",
            detector_version=detector_version, source=source, nominal_role=source,
            observed_signal_type="sustained_activity_or_onset",
            clocks={"source_audio_time": source_time, "padded_audio_time": padded,
                    "chart_time": chart, "mapped_audio_time": expected},
            residual_seconds=residual, confidence=_opening_confidence(row),
            positive_factors=row.get("positive_factors") or [], limitations=limitations,
            provenance={"origin_report": "opening_anchor_candidates", "origin_index": index},
            original=row))
    return output


def adapt_coverage_report(report: Mapping[str, Any] | None, *, source: str | None = None,
                          padding_added: float = 0.0) -> list[dict[str, Any]]:
    """Adapt an existing track-audio coverage report into start/end observations."""
    if not report or report.get("available") is False:
        return []
    role = source or report.get("source") or report.get("role")
    output = []
    pairs = (
        ("audio_activity_start", "source_activity_start", "first_sustained_activity"),
        ("audio_activity_end", "source_activity_end", "last_sustained_activity"),
    )
    for key, evidence_type, signal in pairs:
        value = _number(report.get(key))
        if value is None:
            # Compatibility with older coverage-report field names.
            aliases = {"audio_activity_start": ("audio_start", "activity_start", "audio_sustained_activity_begin"),
                       "audio_activity_end": ("audio_end", "activity_end", "audio_sustained_activity_end")}
            value = next((_number(report.get(name)) for name in aliases[key]
                          if _number(report.get(name)) is not None), None)
        if value is None:
            continue
        output.append(make_evidence(
            evidence_id=f"coverage-{role or 'source'}-{len(output)+1}", evidence_type=evidence_type,
            detector_module="process_gp_alignment", detector_method="diagnose_track_audio_coverage",
            source=role, nominal_role=role, observed_signal_type=signal,
            clocks={"source_audio_time": value - float(padding_added), "padded_audio_time": value},
            confidence=report.get("confidence"),
            limitations=["nominal_role_not_verified"] if role and role != "full" else [],
            provenance={"origin_report": "track_audio_coverage", "origin_field": key},
            original=report))
    return output


def adapt_terminal_report(content_end: Any, report: Mapping[str, Any] | None,
                          *, padding_added: float = 0.0) -> list[dict[str, Any]]:
    """Adapt detect_audio_content_end output without recomputing the endpoint."""
    report = report or {}
    value = _number(content_end if content_end is not None else report.get("content_end"))
    if value is None or report.get("available") is False:
        return []
    return [make_evidence(
        evidence_id="terminal-full-mix-1", evidence_type="full_mix_terminal_activity",
        detector_module="process_gp_alignment", detector_method="detect_audio_content_end",
        source="full", nominal_role="full", observed_signal_type="last_sustained_full_mix_activity",
        clocks={"source_audio_time": value - float(padding_added), "padded_audio_time": value},
        positive_factors=["sustained_activity_requirement"],
        limitations=[] if report.get("file_duration") is not None else ["file_duration_unavailable"],
        provenance={"origin_report": "audio_content_end"}, original=report)]


def adapt_three_clock_incipit(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Expose existing incipit reconciliation as evidence; never reclassify it."""
    if not report:
        return []
    role = report.get("role")
    confidence = 1.0 if report.get("status") == "validated" else None
    limitations = [] if report.get("status") == "validated" else [str(report.get("status", "unvalidated"))]
    return [make_evidence(
        evidence_id="three-clock-incipit-1", evidence_type="three_clock_incipit_status",
        detector_module="incipit_three_clock", detector_method="reconcile",
        source=role, nominal_role=role, observed_signal_type="incipit_clock_reconciliation",
        clocks={"chart_time": report.get("score_time"), "padded_audio_time": report.get("audio_time")},
        residual_seconds=report.get("root_offset_seconds"), confidence=confidence,
        positive_factors=["independent_candidate_consensus"] if report.get("status") == "validated" else [],
        limitations=limitations, provenance={"origin_report": "three_clock_incipit"}, original=report)]


def adapt_continuous_anchor_report(report: Mapping[str, Any] | None, *, padding_added: float = 0.0) -> list[dict[str, Any]]:
    """Adapt only exact retained audio hypotheses nested under `matches[].candidates`."""
    if not report:
        return []
    output=[]
    for match_index, match in enumerate(report.get("matches") or []):
        chart=_number(match.get("score_time")); predicted=_number(match.get("baseline_prediction"))
        for candidate_index, candidate in enumerate(match.get("candidates") or []):
            audio=_number(candidate.get("time")); residual=_number(candidate.get("residual_seconds"))
            if residual is None and audio is not None and predicted is not None: residual=audio-predicted
            source=candidate.get("evidence_source") or match.get("source")
            limitations=[]
            if candidate.get("corresponding_stem") and source!=candidate.get("corresponding_stem"):limitations.append("full_mix_fallback")
            output.append(make_evidence(
                evidence_id=f"continuous-anchor-{match_index+1}-{candidate_index+1}",
                evidence_type="retained_local_audio_hypothesis", detector_module="continuous_anchors",
                detector_method="diagnose_continuous_anchors", detector_version=report.get("version",1),
                source=source, nominal_role=match.get("source"), observed_signal_type=match.get("kind","chart_guided_local_audio_match"),
                clocks={"chart_time":chart,"source_audio_time":None if audio is None else audio-float(padding_added),
                    "padded_audio_time":audio,"mapped_audio_time":predicted},
                residual_seconds=residual, confidence=candidate.get("confidence"),
                positive_factors=[x for x,ok in (("single_stem_sufficient",candidate.get("single_stem_sufficient")),
                    ("joint_rhythm_chroma",candidate.get("joint_pass"))) if ok], limitations=limitations,
                provenance={"origin_report":"continuous_anchor_diagnostics","score_event_id":match.get("score_event_id"),
                    "match_index":match_index,"candidate_index":candidate_index},
                native_quality={k:candidate.get(k) for k in ("raw_audio_confidence","score_confidence","quiet_before","activity_rise","onset_support","rhythm_similarity","chroma_similarity","temporal_prior","combined_probability") if candidate.get(k) is not None},
                adapter_status="complete" if audio is not None else "partial", original=candidate))
    return output

def adapt_structural_search_report(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Adapt retained structural clusters, not broad raw hypotheses."""
    if not report:
        return []
    output=[]
    for index, cluster in enumerate(report.get("clusters") or []):
        members=cluster.get("members") or []; best=members[0] if members else {}
        source_start=best.get("source_start_measure",cluster.get("source_start_measure"))
        target_start=best.get("target_start_measure",cluster.get("target_start_measure"))
        limitations=[]
        if cluster.get("occurrence_uniqueness")!="locally_distinctive":limitations.append("occurrence_not_locally_distinctive")
        validation=cluster.get("sequence_validation") or {}
        output.append(make_evidence(
            evidence_id=f"structural-cluster-{index+1}", evidence_type="retained_structural_occurrence_cluster",
            detector_module="structural_search", detector_method="analyse", detector_version=report.get("schema_version",1),
            observed_signal_type="lag_tolerant_structural_similarity", confidence=cluster.get("best_score"),
            positive_factors=[x for x in (cluster.get("persistence_status"),cluster.get("occurrence_uniqueness")) if x],
            limitations=limitations, provenance={"origin_report":"structural_search.clusters","cluster_index":index,
                "cluster_id":cluster.get("cluster_id"),"source_start_measure":source_start,"target_start_measure":target_start,"member_count":len(members)},
            native_quality={"best_score":cluster.get("best_score"),"median_baseline_improvement":cluster.get("median_baseline_improvement"),
                "window_uniqueness_margin":best.get("window_uniqueness_margin"),"audio_shift_seconds":best.get("audio_shift_seconds"),
                "sequence_validation":validation}, adapter_status="complete" if members else "partial", original=cluster))
    return output

def adapt_instrument_evidence_report(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Adapt accepted per-measure consensus from the exact `roles[].measures[]` shape."""
    if not report:
        return []
    by_measure={}
    for role, role_report in (report.get("roles") or {}).items():
        for measure in role_report.get("measures") or []:
            candidate=measure.get("selected_candidate")
            if not candidate or not measure.get("accepted"):continue
            by_measure.setdefault(int(measure["playback_measure"]),[]).append((role,role_report,measure,candidate))
    output=[]
    for playback_measure, contributors in sorted(by_measure.items()):
        residuals=sorted(float(x[3]["residual_ms"]) for x in contributors)
        confidences=sorted(float(x[3]["calibrated_confidence"]) for x in contributors)
        mid=len(residuals)//2
        residual=residuals[mid] if len(residuals)%2 else (residuals[mid-1]+residuals[mid])/2
        conf=confidences[mid] if len(confidences)%2 else (confidences[mid-1]+confidences[mid])/2
        roles=sorted(x[0] for x in contributors)
        output.append(make_evidence(
            evidence_id=f"instrument-consensus-m{playback_measure}", evidence_type="accepted_instrument_pattern_consensus",
            detector_module="instrument_evidence", detector_method="to_shared_observations",
            detector_version=report.get("schema_version",1), source="+".join(roles), nominal_role="+".join(roles),
            observed_signal_type="cross_role_consensus" if len(roles)>=2 else contributors[0][1].get("feature_family"),
            residual_seconds=residual/1000.0, confidence=conf,
            positive_factors=[x for x,ok in (("neighbour_supported",any(c[2].get("neighbour_supported") for c in contributors)),
                ("cross_role_supported",any(c[2].get("cross_role_supported") for c in contributors))) if ok],
            limitations=[] if len(roles)>=2 else ["single_role_only","nominal_role_not_verified"],
            provenance={"origin_report":"instrument_evidence.roles.measures","playback_measure":playback_measure,"source_roles":roles},
            native_quality={"residual_ms":round(residual,3),"contributor_count":len(contributors)}, original={"contributors":[c[2] for c in contributors]}))
    return output

def build_read_only_evidence_bundle(*, opening_report=None, opening_candidates=None, coverage_reports=None,
                                    terminal_result=None, three_clock_incipit=None,
                                    continuous_anchor_report=None,
                                    structural_search_report=None,
                                    instrument_evidence_report=None,
                                    padding_added: float = 0.0) -> dict[str, Any]:
    """Aggregate adapters only; does not call any detector or modify source reports."""
    evidence: list[dict[str, Any]] = []
    evidence.extend(adapt_opening_report(opening_report, padding_added=padding_added))
    evidence.extend(adapt_opening_candidates(opening_candidates, padding_added=padding_added))
    for source, report in dict(coverage_reports or {}).items():
        evidence.extend(adapt_coverage_report(report, source=source, padding_added=padding_added))
    if terminal_result:
        if isinstance(terminal_result, (tuple, list)) and len(terminal_result) == 2:
            evidence.extend(adapt_terminal_report(terminal_result[0], terminal_result[1], padding_added=padding_added))
        elif isinstance(terminal_result, Mapping):
            evidence.extend(adapt_terminal_report(terminal_result.get("content_end"), terminal_result,
                                                  padding_added=padding_added))
    evidence.extend(adapt_three_clock_incipit(three_clock_incipit))
    evidence.extend(adapt_continuous_anchor_report(continuous_anchor_report, padding_added=padding_added))
    evidence.extend(adapt_structural_search_report(structural_search_report))
    evidence.extend(adapt_instrument_evidence_report(instrument_evidence_report))
    counts: dict[str, int] = {}
    for row in evidence:
        counts[row["evidence_type"]] = counts.get(row["evidence_type"], 0) + 1
    return {"schema_version": VERSION, "diagnostic_only": True,
            "timing_changes_applied": False, "midi_authority_changed": False,
            "detectors_executed": False, "evidence": evidence,
            "summary": {"evidence_count": len(evidence), "evidence_type_counts": counts},
            "notes": ["Read-only adapters preserve origin reports and do not execute detectors.",
                      "Missing audio evidence is neutral and cannot lower MIDI authority."]}
