"""v16.7R4 diagnostic qualification of existing audio-evidence observations.

This module is deliberately read-only.  It does not decode audio, call a
detector, create an anchor, select a path, alter timing, or change MIDI
authority.  It classifies already-adapted observations and answers strict
queries with explicit insufficiency where appropriate.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Sequence
import math

VERSION="v16.7B1.2A.2-qualification-schema-normalization-v1"


def _num(value):
    if value is None or isinstance(value,bool):return None
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def _physical_source(row: Mapping[str,Any]) -> str | None:
    provenance=row.get("provenance") or {};native=row.get("native_quality") or {}
    for key in ("physical_audio_source","audio_source_id","audio_path_sha256","audio_path"):
        value=provenance.get(key) if provenance.get(key) is not None else native.get(key)
        if value is not None:return str(value)
    source=row.get("source")
    return None if source is None else str(source)


def _clock_complete(row):
    clocks=row.get("clocks") or {}
    return all(_num(clocks.get(x)) is not None for x in ("chart_time","padded_audio_time","mapped_audio_time"))


def normalize_qualification_components(*, strict_eligible: bool, adapter_status: str,
                                       clock_complete: bool, physical_source_id: str | None,
                                       confidence: float | None, limitations: Sequence[str]) -> dict[str,Any]:
    """Normalize R4 component states without inventing false failures.

    R4 strict eligibility explicitly evaluates adapter completeness, three-clock
    completeness, confidence, and detector limitations. Physical-source identity
    is exported separately. Component states covered by the strict predicate may
    therefore be explicit or inherited from that predicate, but never silently
    converted from a missing field to False.
    """
    detector_explicit=(adapter_status=="complete" and not limitations)
    physical_explicit=physical_source_id is not None
    detector_status="explicit_complete" if detector_explicit else (
        "inherited_from_strict_qualification" if strict_eligible else "explicit_incomplete")
    physical_status="explicit_complete" if physical_explicit else (
        "inherited_from_strict_qualification" if strict_eligible else "missing")
    return {"detector_identity_complete":detector_explicit or strict_eligible,
        "detector_identity_status":detector_status,
        "physical_source_identity_complete":physical_explicit or strict_eligible,
        "physical_source_identity_status":physical_status,
        "clock_complete":bool(clock_complete),
        "confidence_complete":confidence is not None,
        "limitations_clear":not bool(limitations),
        "completeness_inherited_from_strict":strict_eligible and (
            not detector_explicit or not physical_explicit)}

def qualify_observation(row: Mapping[str,Any]) -> dict[str,Any]:
    """Return a diagnostic qualification without mutating the observation."""
    evidence=deepcopy(dict(row));etype=str(evidence.get("evidence_type") or "unknown")
    residual=_num(evidence.get("residual_seconds"));confidence=_num(evidence.get("confidence"))
    limitations=[str(x) for x in evidence.get("limitations") or []]
    positive=[str(x) for x in evidence.get("positive_factors") or []]
    adapter=evidence.get("adapter_status") or "unknown"
    clock_complete=_clock_complete(evidence)
    physical=_physical_source(evidence)

    if etype=="three_clock_incipit_status":
        category="status_only"
    elif etype=="retained_structural_occurrence_cluster":
        category="structural_recurrence_support"
    elif etype=="accepted_instrument_pattern_consensus":
        category="instrument_pattern_corroboration"
    elif residual is None:
        category="insufficient_strict_audio_evidence"
    elif abs(residual)<=0.10:
        category="near_baseline_support"
    elif abs(residual)<=0.25:
        category="compatible_displaced_candidate"
    elif abs(residual)<=0.50:
        category="ambiguous_alternative"
    else:
        category="credible_displaced_candidate" if confidence is not None and confidence>=0.80 else "ambiguous_alternative"

    strict_eligible=(category in {"near_baseline_support","compatible_displaced_candidate"} and
        adapter=="complete" and clock_complete and confidence is not None and confidence>=0.70 and not limitations)
    components=normalize_qualification_components(strict_eligible=strict_eligible,
        adapter_status=adapter,clock_complete=clock_complete,physical_source_id=physical,
        confidence=confidence,limitations=limitations)
    reasons=[]
    if adapter!="complete":reasons.append("adapter_incomplete")
    if not clock_complete and etype=="retained_local_audio_hypothesis":reasons.append("three_clock_mapping_incomplete")
    if confidence is None:reasons.append("confidence_missing")
    elif confidence<0.70:reasons.append("confidence_below_strict_threshold")
    if limitations:reasons.append("detector_limitations_present")
    if category not in {"near_baseline_support","compatible_displaced_candidate"}:reasons.append("qualification_not_strict")
    return {"evidence_id":evidence.get("evidence_id"),"evidence_type":etype,"source":evidence.get("source"),
        "physical_source_id":physical,"qualification":category,"strict_eligible":strict_eligible,
        "strict_ineligibility_reasons":reasons,"residual_seconds":residual,"confidence":confidence,
        "clock_complete":clock_complete,"detector_identity_complete":components["detector_identity_complete"],
        "detector_identity_status":components["detector_identity_status"],
        "physical_source_identity_complete":components["physical_source_identity_complete"],
        "physical_source_identity_status":components["physical_source_identity_status"],
        "completeness_inherited_from_strict":components["completeness_inherited_from_strict"],
        "clocks":deepcopy(evidence.get("clocks") or {}),
        "adapter_status":adapter,"positive_factors":positive,
        "limitations":limitations,"authority":"diagnostic","timing_changes_applied":False,
        "midi_authority_changed":False}


def qualify_bundle(bundle: Mapping[str,Any]) -> dict[str,Any]:
    rows=[qualify_observation(x) for x in bundle.get("evidence") or []]
    category_counts={};strict=0;physical_groups={}
    for row in rows:
        category_counts[row["qualification"]]=category_counts.get(row["qualification"],0)+1
        strict+=int(row["strict_eligible"])
        if row.get("physical_source_id"):
            physical_groups.setdefault(row["physical_source_id"],[]).append(row["evidence_id"])
    dependent={key:value for key,value in physical_groups.items() if len(value)>1}
    return {"version":VERSION,"diagnostic_only":True,"detectors_executed":False,
        "timing_changes_applied":False,"midi_authority_changed":False,
        "source_schema_version":bundle.get("schema_version"),"qualifications":rows,
        "physical_source_groups":physical_groups,"dependent_physical_sources":dependent,
        "summary":{"observation_count":len(rows),"strict_eligible_count":strict,
            "qualification_counts":category_counts,"physical_source_count":len(physical_groups),
            "dependent_physical_source_count":len(dependent)},
        "notes":["Qualification reuses existing evidence only.",
                 "Correlated observations sharing one physical source are not independent votes.",
                 "Missing or insufficient strict evidence is neutral and cannot lower MIDI authority."]}


def answer_strict_query(qualification_report: Mapping[str,Any], query_type: str,
                        chart_start: float | None=None, chart_end: float | None=None,
                        sources: Sequence[str] | None=None) -> dict[str,Any]:
    """Answer a diagnostic query conservatively from qualified observations."""
    allowed={"opening_root_uncertain","competing_occurrence_boundary","persistent_phase_transition",
        "materialization_gap_boundary","rejected_region_boundary","terminal_mismatch"}
    if query_type not in allowed:raise ValueError(f"unsupported strict audio query: {query_type}")
    wanted=set(str(x) for x in (sources or []));matches=[]
    for row in qualification_report.get("qualifications") or []:
        if not row.get("strict_eligible"):continue
        if wanted and str(row.get("source")) not in wanted:continue
        chart_time=_num((row.get("clocks") or {}).get("chart_time"))
        if chart_start is not None and chart_time is not None and chart_time<float(chart_start)-1e-9:continue
        if chart_end is not None and chart_time is not None and chart_time>float(chart_end)+1e-9:continue
        # An observation without chart time cannot answer a bounded regional query.
        if (chart_start is not None or chart_end is not None) and chart_time is None:continue
        matches.append(deepcopy(row))
    physical={}
    for row in matches:
        key=row.get("physical_source_id") or row.get("evidence_id")
        physical.setdefault(key,[]).append(row)
    independent=[max(group,key=lambda x:((x.get("confidence") or 0),-abs(x.get("residual_seconds") or 0)))
                 for group in physical.values()]
    status="strict_audio_support_available" if independent else "insufficient_strict_audio_evidence"
    return {"version":VERSION,"query_type":query_type,"chart_start":chart_start,"chart_end":chart_end,
        "status":status,"diagnostic_only":True,"timing_changes_applied":False,"midi_authority_changed":False,
        "independent_physical_source_count":len(independent),"selected_support":independent,
        "alternative_count":len(matches),"notes":["One representative observation is retained per physical source."]}
