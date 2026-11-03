#!/usr/bin/env python3
"""Measure-aligned overlapping chart-to-performance-MIDI section candidates.

Step 1 is diagnostic-only. It creates duration-aware playback-measure windows,
searches fixed performance-MIDI onset groups for ordered symbolic matches, fits
the chart clock onto MIDI time, and retains distinct candidate occurrences.
It never changes FeedPak timing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from bisect import bisect_left, bisect_right
import math
import statistics
import time
import hashlib
import json
import os
import tempfile
import solo_role_authority as sra
import incipit_three_clock as itc
import structural_repeat_diagnostics as srd
import section_evidence_quality as seq
import rooted_phrase_support as rps
import phase_migration_qualification as pmq
from typing import Any, Callable, Iterable, Mapping, Sequence


SUPPORTED_ALIGNMENT_METHODS = frozenset({
    "dtw-checkpoint", "checkpoint-linear", "checkpoint-dtw-diagnostic",
    "checkpoint-dtw-selective", "continuous-anchor-linear",
    "continuous-anchor-dtw", "continuous-anchor-structural",
    "bar-landmark-regularized",
})
EXCLUDED_ALIGNMENT_METHODS = frozenset({"nominal", "offset", "linear", "dtw"})

def validate_alignment_method(method: str) -> None:
    if method not in SUPPORTED_ALIGNMENT_METHODS:
        raise ValueError("midi-section-fit is unavailable for nominal, offset, linear, and unguarded dtw; "
                         f"received {method!r}")


@dataclass(frozen=True)
class SectionFitConfig:
    target_measures: int = 4
    minimum_measures: int = 3
    maximum_measures: int = 6
    overlap_measures: int = 1
    context_measures: int = 8
    preferred_min_seconds: float = 6.0
    preferred_max_seconds: float = 16.0
    search_radius_seconds: float = 8.0
    stretch_min: float = 0.90
    stretch_max: float = 1.10
    stretch_step: float = 0.01
    translation_step_seconds: float = 0.10
    onset_group_tolerance_seconds: float = 0.04
    minimum_chart_groups: int = 5
    minimum_midi_groups: int = 5
    minimum_matched_groups: int = 5
    minimum_chart_coverage: float = 0.50
    maximum_median_residual_seconds: float = 0.12
    maximum_residual_seconds: float = 0.35
    missing_chart_penalty: float = 0.22
    extra_midi_penalty: float = 0.025
    exact_pitch_reward: float = 1.0
    octave_pitch_reward: float = 0.72
    pitch_class_reward: float = 0.45
    candidate_limit: int = 5
    distinct_occurrence_seconds: float = 0.75
    context_weight: float = 0.12


@dataclass
class MeasureWindow:
    window_id: str
    role: str
    playback_start_index: int
    playback_end_index: int
    playback_measure_numbers: list[int]
    written_measure_numbers: list[int]
    nominal_start: float
    nominal_end: float
    nominal_duration: float
    overlap_measures: int
    groups: list[dict[str, Any]] = field(default_factory=list)
    context_groups: list[dict[str, Any]] = field(default_factory=list)
    observability: str = "unobservable"
    observability_reason: str = "no_chart_groups"


@dataclass
class SectionCandidate:
    window_id: str
    role: str
    chart_start: float
    chart_end: float
    midi_start: float
    midi_end: float
    chart_stretch: float
    translation_seconds: float
    matched_chart_groups: int
    chart_group_count: int
    midi_group_count: int
    chart_coverage: float
    median_residual_seconds: float
    maximum_residual_seconds: float
    sequence_score: float
    context_score: float
    combined_score: float
    matched_landmarks: list[dict[str, float]]
    status: str
    rejection_reasons: list[str]
    search_hypothesis: str = "requested_method_baseline"


def _time(row: Mapping[str, Any]) -> float:
    for key in ("time", "t_gp", "score_time", "audio_time"):
        if row.get(key) is not None:
            return float(row[key])
    raise KeyError("event has no supported time field")


DRUM_FAMILY_IDS={"kick":1000,"snare":1001,"hihat":1002,"hi_hat":1002,"hh_closed":1002,
                 "hh_open":1002,"cymbal":1003,"crash":1003,"crash_r":1003,"ride":1003,
                 "tom":1004,"tom_hi":1004,"tom_mid":1004,"tom_floor":1004}

def _symbol_id(value: Any) -> int:
    try:return int(value)
    except (TypeError,ValueError):
        key=str(value).strip().lower().replace("-","_").replace(" ","_")
        for token,identifier in DRUM_FAMILY_IDS.items():
            if token in key:return identifier
        return 1999

GM_DRUM_FAMILY_IDS={
    35:1000,36:1000,
    37:1001,38:1001,39:1001,40:1001,
    42:1002,44:1002,46:1002,
    41:1004,43:1004,45:1004,47:1004,48:1004,50:1004,
    49:1003,51:1003,52:1003,53:1003,55:1003,57:1003,59:1003,
}

def _pitches(row: Mapping[str, Any]) -> set[int]:
    canonical=row.get("canonical_drum_piece")
    if canonical is not None:return {_symbol_id(canonical)}
    raw = row.get("pitches")
    if raw is not None:return {_symbol_id(x) for x in raw}
    raw = row.get("pieces")
    if raw is not None:return {_symbol_id(x) for x in raw}
    if row.get("p") is not None:return {_symbol_id(row.get("p"))}
    for key in ("midi", "pitch", "note"):
        if row.get(key) is not None:
            value=_symbol_id(row[key])
            if row.get("source")=="drums" or row.get("event_type")=="drum_hit":
                return {GM_DRUM_FAMILY_IDS.get(value,1999)}
            return {value}
    return set()


def group_onsets(events: Iterable[Mapping[str, Any]], tolerance_seconds: float = 0.04) -> list[dict[str, Any]]:
    """Group near-simultaneous notes into ordered pitch sets."""
    rows = sorted(((_time(x), _pitches(x)) for x in events), key=lambda x: x[0])
    groups: list[dict[str, Any]] = []
    for time, pitches in rows:
        if not pitches:
            continue
        if not groups or time - groups[-1]["time"] > tolerance_seconds:
            groups.append({"time": time, "pitches": set(pitches)})
        else:
            groups[-1]["pitches"].update(pitches)
    return [{"time": float(x["time"]), "pitches": sorted(x["pitches"])} for x in groups]


def _normalise_measure(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    start = row.get("start_time", row.get("start", row.get("time")))
    end = row.get("end_time", row.get("end"))
    if start is None or end is None:
        raise ValueError(f"measure {index} requires start/end time")
    return {
        "playback_index": int(row.get("playback_index", index)),
        "playback_measure_number": int(row.get("playback_measure_number", index + 1)),
        "written_measure_number": int(row.get("written_measure_number", row.get("measure_number", index + 1))),
        "start": float(start),
        "end": float(end),
    }


def _choose_window_end(measures: Sequence[dict[str, Any]], start: int, cfg: SectionFitConfig) -> int:
    remaining = len(measures) - start
    minimum = min(cfg.minimum_measures, remaining)
    maximum = min(cfg.maximum_measures, remaining)
    if minimum <= 0:
        return start
    choices = []
    for count in range(minimum, maximum + 1):
        duration = measures[start + count - 1]["end"] - measures[start]["start"]
        outside = max(0.0, cfg.preferred_min_seconds - duration) + max(0.0, duration - cfg.preferred_max_seconds)
        target_delta = abs(count - cfg.target_measures)
        midpoint = (cfg.preferred_min_seconds + cfg.preferred_max_seconds) / 2
        choices.append((outside, target_delta, abs(duration - midpoint), count))
    return start + min(choices)[-1]


def build_measure_aligned_windows(
    playback_measures: Sequence[Mapping[str, Any]],
    chart_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    cfg: SectionFitConfig | None = None,
) -> dict[str, list[MeasureWindow]]:
    """Construct overlapping, duration-aware windows on expanded playback measures."""
    cfg = cfg or SectionFitConfig()
    measures = [_normalise_measure(x, i) for i, x in enumerate(playback_measures)]
    if any(m["end"] <= m["start"] for m in measures):
        raise ValueError("measure end must follow measure start")
    if any(b["start"] < a["start"] for a, b in zip(measures, measures[1:])):
        raise ValueError("playback measures must be monotonic")
    grouped = {role: group_onsets(events, cfg.onset_group_tolerance_seconds) for role, events in chart_events_by_role.items()}
    result: dict[str, list[MeasureWindow]] = {role: [] for role in chart_events_by_role}
    for role, all_groups in grouped.items():
        start = 0
        ordinal = 0
        while start < len(measures):
            end = _choose_window_end(measures, start, cfg)
            if end <= start:
                break
            selected = measures[start:end]
            nominal_start = selected[0]["start"]
            nominal_end = selected[-1]["end"]
            groups = [x for x in all_groups if nominal_start <= x["time"] < nominal_end]
            context_end = min(len(measures), start + cfg.context_measures)
            context_start = max(0, context_end - cfg.context_measures)
            context_a = measures[context_start]["start"]
            context_b = measures[context_end - 1]["end"]
            context_groups = [x for x in all_groups if context_a <= x["time"] < context_b]
            observable = len(groups) >= cfg.minimum_chart_groups
            result[role].append(MeasureWindow(
                window_id=f"{role}-section-{ordinal:04d}", role=role,
                playback_start_index=start, playback_end_index=end - 1,
                playback_measure_numbers=[x["playback_measure_number"] for x in selected],
                written_measure_numbers=[x["written_measure_number"] for x in selected],
                nominal_start=nominal_start, nominal_end=nominal_end,
                nominal_duration=nominal_end - nominal_start,
                overlap_measures=cfg.overlap_measures,
                groups=groups, context_groups=context_groups,
                observability="observable" if observable else "unobservable",
                observability_reason="sufficient_chart_groups" if observable else "insufficient_chart_groups",
            ))
            ordinal += 1
            count = end - start
            step = max(1, count - cfg.overlap_measures)
            start += step
    return result


def _set_similarity(chart: set[int], midi: set[int], cfg: SectionFitConfig) -> float:
    if not chart:
        return 0.0
    rewards = []
    for pitch in chart:
        if pitch in midi:
            rewards.append(cfg.exact_pitch_reward)
        elif any(abs(pitch - other) % 12 == 0 for other in midi):
            rewards.append(cfg.octave_pitch_reward)
        elif any(pitch % 12 == other % 12 for other in midi):
            rewards.append(cfg.pitch_class_reward)
        else:
            rewards.append(0.0)
    return sum(rewards) / len(rewards)


def _similarity_matrix(chart: Sequence[dict[str, Any]], midi: Sequence[dict[str, Any]],
                       cfg: SectionFitConfig) -> list[list[float]]:
    """Compute each symbolic group similarity once for DP and backtrace."""
    chart_sets=[x["pitches"] if isinstance(x["pitches"],set) else set(x["pitches"]) for x in chart]
    midi_sets=[x["pitches"] if isinstance(x["pitches"],set) else set(x["pitches"]) for x in midi]
    return [[_set_similarity(c,m,cfg) for m in midi_sets] for c in chart_sets]


def _align_ordered_groups(chart: Sequence[dict[str, Any]], midi: Sequence[dict[str, Any]], cfg: SectionFitConfig):
    n, m = len(chart), len(midi)
    similarities=_similarity_matrix(chart,midi,cfg)
    neg = -1e12
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - cfg.missing_chart_penalty
        back[i][0] = (i - 1, 0, "skip_chart")
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - cfg.extra_midi_penalty
        back[0][j] = (0, j - 1, "skip_midi")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = similarities[i-1][j-1]
            options = [
                (dp[i - 1][j] - cfg.missing_chart_penalty, i - 1, j, "skip_chart"),
                (dp[i][j - 1] - cfg.extra_midi_penalty, i, j - 1, "skip_midi"),
            ]
            if sim > 0:
                options.append((dp[i - 1][j - 1] + sim, i - 1, j - 1, "match"))
            dp[i][j], pi, pj, op = max(options, key=lambda x: x[0])
            back[i][j] = (pi, pj, op)
    pairs = []
    i, j = n, m
    while i or j:
        pi, pj, op = back[i][j]
        if op == "match":
            pairs.append((i - 1, j - 1, similarities[i-1][j-1]))
        i, j = pi, pj
    pairs.reverse()
    scale=max(1,n)
    return dp[n][m]/scale,pairs


def _context_fingerprint(groups: Sequence[dict[str, Any]]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    roots = [min(x["pitches"]) for x in groups if x.get("pitches")]
    intervals = tuple(b - a for a, b in zip(roots, roots[1:]))
    spacing = tuple(round(b["time"] - a["time"], 2) for a, b in zip(groups, groups[1:]))
    return intervals, spacing


def _context_similarity(chart: Sequence[dict[str, Any]], midi: Sequence[dict[str, Any]]) -> float:
    ci, _ = _context_fingerprint(chart)
    mi, _ = _context_fingerprint(midi)
    if not ci or not mi:
        return 0.0
    count = min(len(ci), len(mi))
    return sum(1.0 if ci[i] == mi[i] else 0.5 if ci[i] % 12 == mi[i] % 12 else 0.0 for i in range(count)) / count


def prepare_grouped_midi(events: Sequence[Mapping[str,Any]], cfg: SectionFitConfig):
    groups=group_onsets(events,cfg.onset_group_tolerance_seconds)
    return {"groups":groups,"times":[float(x["time"]) for x in groups]}


def _evaluate_transform(window: MeasureWindow, prepared_midi: Mapping[str,Any], stretch: float,
                        translation: float, cfg: SectionFitConfig,
                        search_hypothesis: str="requested_method_baseline",
                        alignment_cache: dict | None=None, counters: dict | None=None):
    expected_start = translation + stretch * window.nominal_start
    expected_end = translation + stretch * window.nominal_end
    margin = max(0.5, cfg.maximum_residual_seconds * 2)
    groups=prepared_midi["groups"];times=prepared_midi["times"]
    left=bisect_left(times,expected_start-margin)
    right=bisect_right(times,expected_end+margin)
    if counters is not None:counters["midi_slice_lookups"]+=1
    if right-left < cfg.minimum_midi_groups:return None
    local=groups[left:right];key=(left,right)
    cached=alignment_cache.get(key) if alignment_cache is not None else None
    if cached is None:
        sequence_score,pairs=_align_ordered_groups(window.groups,local,cfg)
        context_score=_context_similarity(window.context_groups,local)
        cached=(sequence_score,pairs,context_score)
        if alignment_cache is not None:alignment_cache[key]=cached
        if counters is not None:counters["dp_evaluations"]+=1
    else:
        if counters is not None:counters["alignment_cache_hits"]+=1
    sequence_score,pairs,context_score=cached
    if len(pairs)<cfg.minimum_matched_groups:return None
    landmarks=[];residuals=[]
    for ci,mi,similarity in pairs:
        ct=float(window.groups[ci]["time"]);mt=float(local[mi]["time"])
        predicted=translation+stretch*ct
        residuals.append(abs(mt-predicted))
        landmarks.append({"chart_time":ct,"midi_time":mt,"similarity":float(similarity)})
    coverage=len({x[0] for x in pairs})/max(1,len(window.groups))
    median=statistics.median(residuals);maximum=max(residuals)
    combined=0.58*sequence_score+0.20*coverage+0.10*math.exp(-median/0.12)+cfg.context_weight*context_score
    reasons=[]
    if coverage<cfg.minimum_chart_coverage:reasons.append("insufficient_chart_coverage")
    if median>cfg.maximum_median_residual_seconds:reasons.append("median_residual_too_large")
    if maximum>cfg.maximum_residual_seconds:reasons.append("maximum_residual_too_large")
    return SectionCandidate(window_id=window.window_id,role=window.role,
        chart_start=window.nominal_start,chart_end=window.nominal_end,
        midi_start=expected_start,midi_end=expected_end,chart_stretch=float(stretch),
        translation_seconds=float(translation),matched_chart_groups=len(pairs),
        chart_group_count=len(window.groups),midi_group_count=len(local),chart_coverage=coverage,
        median_residual_seconds=median,maximum_residual_seconds=maximum,
        sequence_score=sequence_score,context_score=context_score,combined_score=combined,
        matched_landmarks=landmarks,status="valid" if not reasons else "rejected",
        rejection_reasons=reasons,search_hypothesis=search_hypothesis)


def search_section_candidates(window: MeasureWindow, midi_events, baseline=None,
                              cfg: SectionFitConfig | None=None,
                              search_hypothesis: str="requested_method_baseline",
                              alignment_cache: dict | None=None,counters: dict | None=None):
    """Exact exhaustive search with prepared MIDI, indexed slices, and DP caching."""
    cfg=cfg or SectionFitConfig()
    if window.observability!="observable":return []
    prepared=midi_events if isinstance(midi_events,dict) and "groups" in midi_events else prepare_grouped_midi(midi_events,cfg)
    if len(prepared["groups"])<cfg.minimum_midi_groups:return []
    baseline=baseline or (lambda t:t);expected=float(baseline(window.nominal_start));candidates=[]
    cache=alignment_cache if alignment_cache is not None else {}
    stretch_steps=max(1,int(round((cfg.stretch_max-cfg.stretch_min)/cfg.stretch_step)))
    translation_steps=max(1,int(round(2*cfg.search_radius_seconds/cfg.translation_step_seconds)))
    for si in range(stretch_steps+1):
        stretch=cfg.stretch_min+si*cfg.stretch_step
        for ti in range(translation_steps+1):
            if counters is not None:counters["transform_trials"]+=1
            start_target=expected-cfg.search_radius_seconds+ti*cfg.translation_step_seconds
            translation=start_target-stretch*window.nominal_start
            candidate=_evaluate_transform(window,prepared,stretch,translation,cfg,search_hypothesis,cache,counters)
            if candidate and candidate.status=="valid":candidates.append(candidate)
    candidates.sort(key=lambda x:(x.combined_score,x.chart_coverage,-x.median_residual_seconds),reverse=True)
    distinct=[]
    for candidate in candidates:
        if any(abs(candidate.midi_start-other.midi_start)<cfg.distinct_occurrence_seconds for other in distinct):continue
        distinct.append(candidate)
        if len(distinct)>=cfg.candidate_limit:break
    return distinct


def _phase_distinctiveness(candidate_rows: Sequence[SectionCandidate]):
    """Return diagnostic-only temporal-occurrence competition for one window.

    Candidate rows are already clustered into distinct MIDI occurrences by the
    exhaustive search. These annotations do not change sorting, selection, or
    authority in v16.6A.
    """
    rows=list(candidate_rows)
    diagnostics=[]
    near_margin=0.04
    near_ratio=1.04
    for rank,candidate in enumerate(rows,1):
        alternatives=[x for x in rows if x is not candidate]
        runner=max(alternatives,key=lambda x:x.combined_score) if alternatives else None
        score=float(candidate.combined_score)
        runner_score=float(runner.combined_score) if runner is not None else None
        margin=None if runner is None else score-runner_score
        ratio=None if runner is None or runner_score<=0 else score/runner_score
        midpoint=(float(candidate.midi_start)+float(candidate.midi_end))/2
        runner_midpoint=None if runner is None else (float(runner.midi_start)+float(runner.midi_end))/2
        separation=None if runner_midpoint is None else abs(midpoint-runner_midpoint)
        near_equivalent=sum(1 for other in alternatives
            if score-float(other.combined_score)<=near_margin or
               (float(other.combined_score)>0 and score/float(other.combined_score)<=near_ratio))
        if runner is None:classification="unique_only_occurrence"
        elif rank==1 and margin is not None and margin>=0.08 and ratio is not None and ratio>=1.08:
            classification="dominant"
        elif rank==1:classification="ambiguous_best"
        else:classification="alternative_occurrence"
        diagnostics.append({"occurrence_rank":rank,"selected_score":score,
            "runner_up_score":runner_score,"score_margin":margin,"score_ratio":ratio,
            "runner_up_temporal_separation_seconds":separation,
            "near_equivalent_occurrence_count":near_equivalent,
            "distinct_temporal_occurrence_count":len(rows),
            "phase_distinctiveness":classification})
    return diagnostics


def annotate_candidate_phase_diagnostics(candidate_report: Mapping[str,Any]):
    """Add v16.6A diagnostics without changing candidate values or ordering."""
    annotated=0;ambiguous=0;multiple=0
    for role_report in (candidate_report.get("roles") or {}).values():
        for row in role_report.get("windows") or []:
            candidates=row.get("candidates") or []
            objects=[x if isinstance(x,SectionCandidate) else _candidate_from_dict(x) for x in candidates]
            diagnostics=_phase_distinctiveness(objects)
            serialized=[]
            for candidate,diagnostic in zip(candidates,diagnostics):
                item=asdict(candidate) if isinstance(candidate,SectionCandidate) else dict(candidate)
                item["phase_diagnostics"]=diagnostic
                serialized.append(item)
                annotated+=1
                if diagnostic["phase_distinctiveness"]=="ambiguous_best":ambiguous+=1
            row["candidates"]=serialized
            if len(serialized)>1:multiple+=1
            row["phase_diagnostics"]={"candidate_count":len(serialized),
                "distinct_temporal_occurrence_count":len(serialized),
                "best_phase_distinctiveness":diagnostics[0]["phase_distinctiveness"] if diagnostics else "unavailable",
                "best_score_margin":diagnostics[0]["score_margin"] if diagnostics else None,
                "best_score_ratio":diagnostics[0]["score_ratio"] if diagnostics else None,
                "near_equivalent_occurrence_count":diagnostics[0]["near_equivalent_occurrence_count"] if diagnostics else 0}
    candidate_report["phase_diagnostics_summary"]={"version":"v16.6A-diagnostic-only",
        "candidate_count":annotated,"windows_with_multiple_occurrences":multiple,
        "ambiguous_best_window_count":ambiguous,"selection_behavior_changed":False}
    return candidate_report


def _window_timing_lookup(candidate_report):
    lookup={}
    for role_report in (candidate_report.get("roles") or {}).values():
        for row in role_report.get("windows") or []:
            window=row.get("window") or {}
            if window.get("window_id"):
                count=max(1,len(window.get("playback_measure_numbers") or []))
                duration=float(window.get("nominal_duration") or
                    (float(window.get("nominal_end",0))-float(window.get("nominal_start",0))))
                measure=duration/count if duration>0 else None
                lookup[window["window_id"]]={"estimated_measure_duration_seconds":measure,
                    "estimated_beat_duration_seconds":measure/4 if measure else None}
    return lookup


def build_selected_phase_diagnostics(path_report: Mapping[str,Any],candidate_report: Mapping[str,Any]):
    """Describe offset jumps and cumulative drift in selected paths only."""
    lookup=_window_timing_lookup(candidate_report);roles={};jump_count=0;beat_like=0;measure_like=0
    for role,report in (path_report.get("roles") or {}).items():
        streak_rows=[]
        for streak in report.get("streaks") or []:
            candidates=streak.get("candidates") or []
            transitions=[]
            for previous,current in zip(candidates,candidates[1:]):
                previous_offset=float(previous["midi_start"])-float(previous["chart_start"])
                current_offset=float(current["midi_start"])-float(current["chart_start"])
                jump=current_offset-previous_offset
                timing=lookup.get(current.get("window_id"),{})
                beat=timing.get("estimated_beat_duration_seconds");measure=timing.get("estimated_measure_duration_seconds")
                beat_error=None if not beat else abs(abs(jump)-beat)
                measure_error=None if not measure else abs(abs(jump)-measure)
                beat_match=bool(beat and abs(jump)>=0.20 and beat_error<=max(0.08,beat*.18))
                measure_match=bool(measure and abs(jump)>=0.40 and measure_error<=max(0.12,measure*.15))
                jump_count+=1;beat_like+=int(beat_match);measure_like+=int(measure_match)
                transitions.append({"from_window_id":previous.get("window_id"),"to_window_id":current.get("window_id"),
                    "chart_time":float(current["chart_start"]),"previous_start_offset_seconds":previous_offset,
                    "current_start_offset_seconds":current_offset,"offset_jump_seconds":jump,
                    "estimated_beat_duration_seconds":beat,"estimated_measure_duration_seconds":measure,
                    "beat_sized_phase_jump":beat_match,"measure_sized_phase_jump":measure_match,
                    "diagnostic_only":True})
            if candidates:
                entry=float(candidates[0]["midi_start"])-float(candidates[0]["chart_start"])
                exit_=float(candidates[-1]["midi_end"])-float(candidates[-1]["chart_end"])
            else:entry=exit_=None
            streak_rows.append({"streak_id":streak.get("streak_id"),"chart_start":streak.get("chart_start"),
                "chart_end":streak.get("chart_end"),"entry_offset_seconds":entry,"exit_offset_seconds":exit_,
                "cumulative_offset_trend_seconds":None if entry is None else exit_-entry,
                "transition_count":len(transitions),"transitions":transitions})
        roles[role]={"streak_count":len(streak_rows),"streaks":streak_rows}
    return {"version":"v16.6A-diagnostic-only","selection_behavior_changed":False,
        "transition_count":jump_count,"beat_sized_phase_jump_count":beat_like,
        "measure_sized_phase_jump_count":measure_like,"roles":roles}


def build_section_candidate_report(
    playback_measures: Sequence[Mapping[str, Any]],
    chart_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    midi_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Callable[[float], float] | None = None,
    cfg: SectionFitConfig | None = None,
    baseline_hypotheses: Mapping[str, Callable[[float], float]] | None = None,
) -> dict[str, Any]:
    """Build exact Step-1 diagnostics with reusable MIDI preparation and caches."""
    started=time.perf_counter();cfg=cfg or SectionFitConfig()
    hypotheses=dict(baseline_hypotheses or {})
    if not hypotheses:hypotheses["requested_method_baseline"]=baseline or (lambda t:float(t))
    hypotheses.setdefault("prepared_chart_clock",lambda t:float(t))
    windows=build_measure_aligned_windows(playback_measures,chart_events_by_role,cfg)
    prep_started=time.perf_counter()
    prepared_by_role={role:prepare_grouped_midi(midi_events_by_role.get(role) or [],cfg) for role in windows}
    grouping_seconds=time.perf_counter()-prep_started
    role_reports={};totals={"window_count":0,"observable_window_count":0,"candidate_count":0}
    global_perf={"transform_trials":0,"midi_slice_lookups":0,"dp_evaluations":0,"alignment_cache_hits":0,
                 "unique_midi_slices":0,"midi_grouping_seconds":grouping_seconds}
    for role,role_windows in windows.items():
        role_started=time.perf_counter();rows=[];prepared=prepared_by_role[role]
        counters={"transform_trials":0,"midi_slice_lookups":0,"dp_evaluations":0,"alignment_cache_hits":0}
        for window in role_windows:
            pooled=[];alignment_cache={}
            for hypothesis_name,hypothesis_baseline in hypotheses.items():
                pooled.extend(search_section_candidates(window,prepared,hypothesis_baseline,cfg,hypothesis_name,
                                                        alignment_cache,counters))
            pooled.sort(key=lambda x:(x.combined_score,x.chart_coverage,-x.median_residual_seconds),reverse=True)
            candidates=[]
            for item in pooled:
                if any(abs(item.midi_start-other.midi_start)<cfg.distinct_occurrence_seconds for other in candidates):continue
                candidates.append(item)
                if len(candidates)>=cfg.candidate_limit:break
            totals["window_count"]+=1;totals["observable_window_count"]+=int(window.observability=="observable")
            totals["candidate_count"]+=len(candidates)
            window_data=asdict(window)
            window_data["groups"]=[{"time":x["time"],"pitches":list(x["pitches"])} for x in window.groups]
            window_data["context_groups"]=[]
            rows.append({"window":window_data,"candidates":[asdict(x) for x in candidates],
                         "performance":{"unique_midi_slices":len(alignment_cache)}})
            global_perf["unique_midi_slices"]+=len(alignment_cache)
        role_perf={**counters,"elapsed_seconds":time.perf_counter()-role_started,
                   "prepared_midi_group_count":len(prepared["groups"]),
                   "window_count":len(role_windows)}
        for key in ("transform_trials","midi_slice_lookups","dp_evaluations","alignment_cache_hits"):
            global_perf[key]+=counters[key]
        role_reports[role]={"window_count":len(role_windows),"windows":rows,"performance":role_perf}
    global_perf["candidate_search_seconds"]=time.perf_counter()-started
    global_perf["cache_hit_rate"]=(global_perf["alignment_cache_hits"]/
        max(1,global_perf["alignment_cache_hits"]+global_perf["dp_evaluations"]))
    report={"version":"1.2-v16.6A-phase-diagnostics","method":"midi-section-fit","diagnostic_only":True,
        "timing_changes_applied":False,"monotonic_path_selected":False,"config":asdict(cfg),
        "search_hypotheses":list(hypotheses),"summary":totals,"performance":global_perf,"roles":role_reports}
    return annotate_candidate_phase_diagnostics(report)


# ---------------------------------------------------------------------------
# Step 2: globally monotonic overlapping per-role paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionPathConfig:
    beam_width: int = 12
    overlap_strong_seconds: float = 0.08
    overlap_acceptable_seconds: float = 0.18
    overlap_reject_seconds: float = 0.35
    stretch_change_penalty: float = 0.80
    overlap_disagreement_penalty: float = 1.50
    translation_change_penalty: float = 0.08
    start_new_streak_penalty: float = 0.25
    minimum_streak_candidates: int = 1


@dataclass
class CandidateTransition:
    valid: bool
    reason: str | None
    overlap_start: float | None
    overlap_end: float | None
    overlap_samples: list[dict[str, float]]
    maximum_overlap_disagreement_seconds: float | None
    mean_overlap_disagreement_seconds: float | None
    stretch_change: float
    translation_change: float
    score_adjustment: float


@dataclass
class SelectedSectionStreak:
    streak_id: str
    role: str
    candidates: list[SectionCandidate]
    score: float
    start_window_index: int
    end_window_index: int
    chart_start: float
    chart_end: float
    midi_start: float
    midi_end: float
    largest_overlap_disagreement_seconds: float
    monotonic: bool


def _map_candidate(candidate: SectionCandidate, chart_time: float) -> float:
    return candidate.translation_seconds + candidate.chart_stretch * float(chart_time)


def evaluate_candidate_transition(
    previous: SectionCandidate,
    current: SectionCandidate,
    cfg: SectionPathConfig | None = None,
) -> CandidateTransition:
    """Validate a transition without ever repairing temporal reversal.

    Adjacent measure windows overlap in chart time, so current.midi_start may be
    earlier than previous.midi_end. Monotonicity is defined at like-for-like
    chart positions: both candidate starts and ends must advance, every affine
    map must have positive slope, and their predictions in the shared chart
    interval must agree without crossing by more than the hard tolerance.
    """
    cfg = cfg or SectionPathConfig()
    reasons = []
    if current.chart_start <= previous.chart_start:
        reasons.append("chart_window_did_not_advance")
    if current.chart_end <= previous.chart_end:
        reasons.append("chart_window_end_did_not_advance")
    if current.chart_stretch <= 0 or previous.chart_stretch <= 0:
        reasons.append("non_positive_chart_stretch")
    if current.midi_start <= previous.midi_start:
        reasons.append("midi_window_start_did_not_advance")
    if current.midi_end <= previous.midi_end:
        reasons.append("midi_window_end_did_not_advance")

    overlap_start = max(previous.chart_start, current.chart_start)
    overlap_end = min(previous.chart_end, current.chart_end)
    samples = []
    max_delta = None
    mean_delta = None
    if overlap_end > overlap_start:
        points = [overlap_start + (overlap_end - overlap_start) * fraction
                  for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
        for chart_time in points:
            a = _map_candidate(previous, chart_time)
            b = _map_candidate(current, chart_time)
            samples.append({"chart_time": chart_time, "previous_midi_time": a,
                            "current_midi_time": b, "absolute_delta_seconds": abs(a - b)})
        deltas = [x["absolute_delta_seconds"] for x in samples]
        max_delta = max(deltas)
        mean_delta = sum(deltas) / len(deltas)
        if max_delta > cfg.overlap_reject_seconds:
            reasons.append("overlap_mapping_disagreement")

        # Each mapping is positive by construction. This union-boundary check
        # proves that selecting the two maps cannot make the shared interval run
        # backward when ownership changes from one window to the next.
        switch = (overlap_start + overlap_end) / 2.0
        before = _map_candidate(previous, switch)
        after = _map_candidate(current, switch)
        epsilon = min(0.01, max(1e-6, (overlap_end - overlap_start) / 1000.0))
        left = _map_candidate(previous, max(overlap_start, switch - epsilon))
        right = _map_candidate(current, min(overlap_end, switch + epsilon))
        if right <= left:
            reasons.append("crossing_or_reversing_overlap_maps")
    else:
        # Non-overlapping windows can only continue when the latter begins after
        # the former ends on the fixed MIDI clock.
        if current.midi_start <= previous.midi_end:
            reasons.append("non_overlapping_sections_reverse_or_overlap_midi_time")

    stretch_change = abs(current.chart_stretch - previous.chart_stretch)
    translation_change = abs(current.translation_seconds - previous.translation_seconds)
    overlap_penalty = 0.0 if mean_delta is None else cfg.overlap_disagreement_penalty * mean_delta
    adjustment = -(cfg.stretch_change_penalty * stretch_change +
                   cfg.translation_change_penalty * translation_change + overlap_penalty)
    return CandidateTransition(
        valid=not reasons, reason=None if not reasons else reasons[0],
        overlap_start=overlap_start if overlap_end > overlap_start else None,
        overlap_end=overlap_end if overlap_end > overlap_start else None,
        overlap_samples=samples,
        maximum_overlap_disagreement_seconds=max_delta,
        mean_overlap_disagreement_seconds=mean_delta,
        stretch_change=stretch_change,
        translation_change=translation_change,
        score_adjustment=adjustment,
    )


def _candidate_from_dict(row: Mapping[str, Any]) -> SectionCandidate:
    allowed = SectionCandidate.__dataclass_fields__.keys()
    values={key:row[key] for key in allowed if key in row}
    values.setdefault("search_hypothesis","requested_method_baseline")
    return SectionCandidate(**values)


def _window_candidates(row: Mapping[str, Any]) -> list[SectionCandidate]:
    return [x if isinstance(x, SectionCandidate) else _candidate_from_dict(x)
            for x in (row.get("candidates") or [])]


def _validate_selected_candidate_sequence(candidates: Sequence[SectionCandidate], cfg: SectionPathConfig):
    transitions = []
    for previous, current in zip(candidates, candidates[1:]):
        transition = evaluate_candidate_transition(previous, current, cfg)
        transitions.append(transition)
        if not transition.valid:
            return False, transitions, transition.reason
    if any(b.chart_start <= a.chart_start or b.midi_start <= a.midi_start
           for a, b in zip(candidates, candidates[1:])):
        return False, transitions, "global_start_monotonicity_failure"
    if any(b.chart_end <= a.chart_end or b.midi_end <= a.midi_end
           for a, b in zip(candidates, candidates[1:])):
        return False, transitions, "global_end_monotonicity_failure"
    return True, transitions, None


def _select_contiguous_block(
    role: str,
    indexed_rows: Sequence[tuple[int, Mapping[str, Any]]],
    cfg: SectionPathConfig,
    streak_ordinal: int,
) -> tuple[SelectedSectionStreak | None, dict[str, Any]]:
    """Beam-select one candidate per window in a contiguous observable block."""
    beam = []
    diagnostics = {"window_candidate_counts": [], "rejected_transition_counts": {},
                   "beam_width": cfg.beam_width}
    for layer, (window_index, row) in enumerate(indexed_rows):
        candidates = _window_candidates(row)
        diagnostics["window_candidate_counts"].append({"window_index": window_index,
                                                        "candidate_count": len(candidates)})
        if not candidates:
            return None, diagnostics
        next_beam = []
        if layer == 0:
            for candidate in candidates:
                next_beam.append((candidate.combined_score, [candidate], []))
        else:
            for score, path, transitions in beam:
                for candidate in candidates:
                    transition = evaluate_candidate_transition(path[-1], candidate, cfg)
                    if not transition.valid:
                        key = transition.reason or "invalid_transition"
                        diagnostics["rejected_transition_counts"][key] = diagnostics["rejected_transition_counts"].get(key, 0) + 1
                        continue
                    next_beam.append((score + candidate.combined_score + transition.score_adjustment,
                                      path + [candidate], transitions + [transition]))
        if not next_beam:
            return None, diagnostics
        next_beam.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        beam = next_beam[:cfg.beam_width]
    score, selected, transitions = max(beam, key=lambda x: x[0])
    valid, validation_transitions, reason = _validate_selected_candidate_sequence(selected, cfg)
    if not valid:
        diagnostics["validation_failure"] = reason
        return None, diagnostics
    maximum_overlap = max([x.maximum_overlap_disagreement_seconds or 0.0
                           for x in validation_transitions] or [0.0])
    first_index = indexed_rows[0][0]
    last_index = indexed_rows[-1][0]
    streak = SelectedSectionStreak(
        streak_id=f"{role}-section-streak-{streak_ordinal}", role=role,
        candidates=selected, score=score,
        start_window_index=first_index, end_window_index=last_index,
        chart_start=selected[0].chart_start, chart_end=selected[-1].chart_end,
        midi_start=selected[0].midi_start, midi_end=selected[-1].midi_end,
        largest_overlap_disagreement_seconds=maximum_overlap,
        monotonic=True,
    )
    diagnostics["selected_transition_count"] = len(validation_transitions)
    diagnostics["selected_transitions"] = [asdict(x) for x in validation_transitions]
    return streak, diagnostics


def select_monotonic_section_paths(
    candidate_report: Mapping[str, Any],
    cfg: SectionPathConfig | None = None,
) -> dict[str, Any]:
    """Select independent, globally monotonic paths for every role.

    Empty/unobservable windows split authority. A later supported block starts a
    new streak; Step 2 never interpolates evidence across the gap.
    """
    cfg = cfg or SectionPathConfig()
    role_reports = {}
    total_streaks = 0
    total_selected = 0
    for role, role_report in (candidate_report.get("roles") or {}).items():
        windows = role_report.get("windows") or []
        blocks = []
        current = []
        gap_windows = []
        for index, row in enumerate(windows):
            observable = (row.get("window") or {}).get("observability") == "observable"
            candidates = row.get("candidates") or []
            if observable and candidates:
                current.append((index, row))
            else:
                if current:
                    blocks.append(current)
                    current = []
                gap_windows.append({"window_index": index,
                                    "window_id": (row.get("window") or {}).get("window_id"),
                                    "reason": "unobservable" if not observable else "no_valid_candidates"})
        if current:
            blocks.append(current)

        streaks = []
        block_diagnostics = []
        ordinal = 1
        # A block may still contain an impossible layer-to-layer transition. If
        # so, split at that boundary and retry smaller contiguous sub-blocks.
        pending = list(blocks)
        while pending:
            block = pending.pop(0)
            streak, diagnostic = _select_contiguous_block(role, block, cfg, ordinal)
            if streak is not None:
                streaks.append(streak)
                block_diagnostics.append(diagnostic)
                ordinal += 1
                continue
            if len(block) == 1:
                gap_windows.append({"window_index": block[0][0],
                                    "window_id": (block[0][1].get("window") or {}).get("window_id"),
                                    "reason": "no_monotonic_candidate_path"})
                block_diagnostics.append(diagnostic)
                continue
            # Find the longest valid prefix. This is deterministic and preserves
            # chart order; the suffix becomes an independent evidence streak.
            split = 1
            for end in range(2, len(block) + 1):
                prefix, _ = _select_contiguous_block(role, block[:end], cfg, ordinal)
                if prefix is None:
                    break
                split = end
            if split >= len(block):
                # Defensive fallback; should be unreachable when the full block failed.
                split = len(block) - 1
            pending.insert(0, block[split:])
            pending.insert(0, block[:split])

        selected_streaks = []
        for streak in streaks:
            valid, transitions, reason = _validate_selected_candidate_sequence(streak.candidates, cfg)
            if not valid:
                raise ValueError(f"non-monotonic selected path escaped validation: {reason}")
            selected_streaks.append({
                **asdict(streak),
                "candidates": [asdict(x) for x in streak.candidates],
                "transition_count": len(transitions),
            })
        total_streaks += len(selected_streaks)
        total_selected += sum(len(x["candidates"]) for x in selected_streaks)
        role_reports[role] = {
            "status": "available" if selected_streaks else "no_monotonic_path",
            "streak_count": len(selected_streaks),
            "selected_candidate_count": sum(len(x["candidates"]) for x in selected_streaks),
            "streaks": selected_streaks,
            "unsupported_windows": sorted(gap_windows, key=lambda x: x["window_index"]),
            "block_diagnostics": block_diagnostics,
            "monotonicity_validated": all(x["monotonic"] for x in selected_streaks),
        }
    return {
        "version": "1.0-step2",
        "method": "midi-section-fit",
        "diagnostic_only": True,
        "timing_changes_applied": False,
        "global_fusion_applied": False,
        "path_config": asdict(cfg),
        "summary": {"role_count": len(role_reports), "streak_count": total_streaks,
                    "selected_candidate_count": total_selected},
        "roles": role_reports,
    }


def build_section_path_report(
    playback_measures: Sequence[Mapping[str, Any]],
    chart_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    midi_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Callable[[float], float] | None = None,
    candidate_config: SectionFitConfig | None = None,
    path_config: SectionPathConfig | None = None,
    baseline_hypotheses: Mapping[str, Callable[[float], float]] | None = None,
) -> dict[str, Any]:
    """Run Steps 1 and 2 while remaining diagnostic-only."""
    candidates = build_section_candidate_report(
        playback_measures, chart_events_by_role, midi_events_by_role,
        baseline=baseline, cfg=candidate_config, baseline_hypotheses=baseline_hypotheses)
    paths = select_monotonic_section_paths(candidates, path_config)
    return {"version": "1.0-step2", "method": "midi-section-fit",
            "diagnostic_only": True, "timing_changes_applied": False,
            "candidate_report": candidates, "path_report": paths}

# ---------------------------------------------------------------------------
# Step 3: compatible multi-role fusion and diagnostic timing-map proposals
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionFusionConfig:
    role_agreement_seconds: float = 0.25
    strong_role_agreement_seconds: float = 0.10
    chart_time_merge_seconds: float = 0.035
    minimum_control_spacing_seconds: float = 1.0
    maximum_supported_control_gap_seconds: float = 18.0
    edge_stretch_min: float = 0.90
    edge_stretch_max: float = 1.10
    beam_width: int = 16
    multi_role_bonus: float = 0.30
    single_role_penalty: float = 0.10
    disagreement_penalty: float = 1.20
    edge_stretch_penalty: float = 0.75
    catch_up_threshold: float = 0.06


@dataclass
class FusedControlCandidate:
    chart_time: float
    midi_time: float
    roles: list[str]
    sources: list[dict[str, Any]]
    role_count: int
    agreement_spread_seconds: float
    confidence: float
    classification: str


def _candidate_support_weight(candidate: Mapping[str, Any]) -> float:
    score = float(candidate.get("combined_score", 0.0))
    coverage = float(candidate.get("chart_coverage", 0.0))
    residual = float(candidate.get("median_residual_seconds", 1.0))
    return max(0.05, 0.55 * score + 0.30 * coverage + 0.15 * math.exp(-residual / 0.12))


def _selected_role_candidates(path_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for role, role_report in (path_report.get("roles") or {}).items():
        for streak in role_report.get("streaks") or []:
            for raw in streak.get("candidates") or []:
                candidate = raw if isinstance(raw, SectionCandidate) else _candidate_from_dict(raw)
                rows.append({"role": role, "streak_id": streak.get("streak_id"),
                             "streak_candidate_count":len(streak.get("candidates") or []),
                             "candidate": candidate})
    return rows


def _support_knots(selected: Sequence[dict[str, Any]]) -> list[float]:
    points = []
    for row in selected:
        candidate = row["candidate"]
        points.extend((candidate.chart_start,
                       (candidate.chart_start + candidate.chart_end) / 2.0,
                       candidate.chart_end))
    points.sort()
    knots = []
    for point in points:
        if not knots or point - knots[-1] > 1e-6:
            knots.append(point)
    return knots


def _cluster_role_predictions(predictions: Sequence[dict[str, Any]], cfg: SectionFusionConfig):
    """Cluster role predictions without averaging incompatible evidence."""
    clusters = []
    for prediction in sorted(predictions, key=lambda x: x["midi_time"]):
        if not clusters or prediction["midi_time"] - clusters[-1][-1]["midi_time"] > cfg.role_agreement_seconds:
            clusters.append([prediction])
        else:
            clusters[-1].append(prediction)
    fused = []
    for cluster in clusters:
        # Keep at most the strongest source from each role at one chart time.
        by_role = {}
        for row in cluster:
            role = row["role"]
            if role not in by_role or row["weight"] > by_role[role]["weight"]:
                by_role[role] = row
        rows = list(by_role.values())
        total = sum(x["weight"] for x in rows)
        midi_time = sum(x["midi_time"] * x["weight"] for x in rows) / total
        times = [x["midi_time"] for x in rows]
        spread = max(times) - min(times) if times else math.inf
        role_count = len(rows)
        strong_single=(role_count==1 and int(rows[0].get("streak_candidate_count",1))>=2 and
                       rows[0]["candidate"].chart_coverage>=.60 and
                       rows[0]["candidate"].median_residual_seconds<=.10)
        classification = ("multi_role_corroborated" if role_count >= 2 and spread <= cfg.strong_role_agreement_seconds else
                          "multi_role_compatible" if role_count >= 2 else
                          "single_role_strong" if strong_single else "single_role_provisional")
        confidence = sum(x["weight"] for x in rows) / role_count
        confidence += cfg.multi_role_bonus * max(0, role_count - 1)
        confidence -= cfg.single_role_penalty if role_count == 1 else 0.0
        confidence -= cfg.disagreement_penalty * spread
        fused.append(FusedControlCandidate(
            chart_time=float(rows[0]["chart_time"]), midi_time=float(midi_time),
            roles=sorted(by_role), role_count=role_count,
            agreement_spread_seconds=float(spread), confidence=float(confidence),
            classification=classification,
            sources=[{"role": x["role"], "streak_id": x["streak_id"],
                      "window_id": x["candidate"].window_id,
                      "streak_candidate_count":x.get("streak_candidate_count",1),
                      "predicted_midi_time": x["midi_time"],
                      "weight": x["weight"]} for x in rows],
        ))
    return fused


def build_fused_control_candidates(path_report: Mapping[str, Any],
                                   cfg: SectionFusionConfig | None = None):
    cfg = cfg or SectionFusionConfig()
    selected = _selected_role_candidates(path_report)
    knots = _support_knots(selected)
    layers = []
    unsupported = []
    role_conflicts = []
    for chart_time in knots:
        predictions = []
        for row in selected:
            candidate = row["candidate"]
            if candidate.chart_start - 1e-9 <= chart_time <= candidate.chart_end + 1e-9:
                predictions.append({"chart_time": chart_time, "midi_time": _map_candidate(candidate, chart_time),
                                    "role": row["role"], "streak_id": row["streak_id"],
                                    "streak_candidate_count":row.get("streak_candidate_count",1),
                                    "candidate": candidate, "weight": _candidate_support_weight(asdict(candidate))})
        if not predictions:
            unsupported.append({"chart_time": chart_time, "reason": "no_role_support"})
            continue
        clusters = _cluster_role_predictions(predictions, cfg)
        if len(clusters) > 1:
            role_conflicts.append({"chart_time": chart_time,
                                   "cluster_count": len(clusters),
                                   "clusters": [asdict(x) for x in clusters]})
        layers.append({"chart_time": chart_time, "candidates": clusters})
    return layers, unsupported, role_conflicts


def _fused_transition(previous: FusedControlCandidate,
                      current: FusedControlCandidate,
                      cfg: SectionFusionConfig):
    chart_delta = current.chart_time - previous.chart_time
    midi_delta = current.midi_time - previous.midi_time
    if chart_delta <= 0:
        return False, "chart_control_did_not_advance", None, -math.inf
    if midi_delta <= 0:
        return False, "midi_control_did_not_advance", None, -math.inf
    edge_stretch = midi_delta / chart_delta
    absolute_correction=abs(midi_delta-chart_delta)
    normal=cfg.edge_stretch_min<=edge_stretch<=cfg.edge_stretch_max
    # Candidate fusion retains short bounded corrections for later mode-specific
    # assessment. Materialization still applies conservative/permissive/unsafe
    # policy and may reject them.
    short_delta_exception=(chart_delta<=16.0 and absolute_correction<=.50)
    if not (normal or short_delta_exception):
        return False, "unsafe_fused_edge_geometry", edge_stretch, -math.inf
    adjustment = -cfg.edge_stretch_penalty * abs(edge_stretch - 1.0)
    return True, None, edge_stretch, adjustment


def _select_fused_monotonic_controls(layers: Sequence[Mapping[str, Any]],
                                     cfg: SectionFusionConfig):
    if not layers:
        return [], [], {"status": "no_supported_controls"}
    beam = []
    transition_rejections = {}
    for index, layer in enumerate(layers):
        next_beam = []
        candidates = layer.get("candidates") or []
        if index == 0:
            for candidate in candidates:
                next_beam.append((candidate.confidence, [candidate], []))
        else:
            for score, path, edges in beam:
                for candidate in candidates:
                    valid, reason, edge_stretch, adjustment = _fused_transition(path[-1], candidate, cfg)
                    if not valid:
                        transition_rejections[reason] = transition_rejections.get(reason, 0) + 1
                        continue
                    next_beam.append((score + candidate.confidence + adjustment,
                                      path + [candidate], edges + [edge_stretch]))
        if not next_beam:
            return [], [], {"status": "no_global_monotonic_fused_path",
                            "failed_layer_index": index,
                            "transition_rejections": transition_rejections}
        next_beam.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        beam = next_beam[:cfg.beam_width]
    score, path, edges = max(beam, key=lambda x: x[0])
    if any(b.chart_time <= a.chart_time or b.midi_time <= a.midi_time for a, b in zip(path, path[1:])):
        raise ValueError("non-monotonic fused controls escaped selection")
    return path, edges, {"status": "available", "path_score": score,
                         "transition_rejections": transition_rejections}


def _split_control_regions(path: Sequence[FusedControlCandidate],
                           edges: Sequence[float], cfg: SectionFusionConfig):
    if not path:
        return []
    regions = []
    current = [path[0]]
    current_edges = []
    for index, candidate in enumerate(path[1:]):
        previous = path[index]
        chart_gap = candidate.chart_time - previous.chart_time
        if chart_gap > cfg.maximum_supported_control_gap_seconds:
            regions.append((current, current_edges, "large_unsupported_gap"))
            current = [candidate]
            current_edges = []
        else:
            current.append(candidate)
            current_edges.append(edges[index])
    regions.append((current, current_edges, None))
    return regions


def _sparsify_controls(path: Sequence[FusedControlCandidate], cfg: SectionFusionConfig):
    if not path:
        return []
    selected = [path[0]]
    for item in path[1:-1]:
        if item.chart_time - selected[-1].chart_time >= cfg.minimum_control_spacing_seconds:
            selected.append(item)
    if path[-1] is not selected[-1]:
        selected.append(path[-1])
    return selected


def _catch_up_events(edges: Sequence[float], cfg: SectionFusionConfig):
    events = []
    for index, (a, b) in enumerate(zip(edges, edges[1:])):
        first = a - 1.0
        second = b - 1.0
        if first * second < 0 and abs(first) >= cfg.catch_up_threshold and abs(second) >= cfg.catch_up_threshold:
            events.append({"first_edge_index": index, "second_edge_index": index + 1,
                           "first_stretch": a, "second_stretch": b,
                           "kind": "correction_and_catch_up"})
    return events


def fuse_monotonic_role_paths(path_report: Mapping[str, Any],
                              cfg: SectionFusionConfig | None = None) -> dict[str, Any]:
    """Fuse compatible role paths into sparse diagnostic timing-map proposals.

    Incompatible role clusters are never averaged. Global control monotonicity
    and edge stretch are enforced during selection. Large support gaps split the
    result into independent regions; this function never bridges those gaps.
    """
    cfg = cfg or SectionFusionConfig()
    layers, unsupported, role_conflicts = build_fused_control_candidates(path_report, cfg)

    # Split at chart-time support gaps before selection so a later region cannot
    # invalidate a strong earlier path or imply authority across the gap.
    blocks = []
    current = []
    for layer in layers:
        if current and layer["chart_time"] - current[-1]["chart_time"] > cfg.maximum_supported_control_gap_seconds:
            blocks.append(current)
            current = []
        current.append(layer)
    if current:
        blocks.append(current)

    regions = []
    rejected_blocks = []
    for block_index, block in enumerate(blocks):
        cursor=0
        while cursor<len(block):
            best=None;best_end=None;failure=None
            for end in range(cursor+1,len(block)+1):
                path,edges,selection=_select_fused_monotonic_controls(block[cursor:end],cfg)
                if not path:
                    failure=selection;break
                best=(path,edges,selection);best_end=end
            if best is None:
                rejected_blocks.append({"block_index":block_index,"layer_index":cursor,
                    "status":"isolated_incompatible_layer",**(failure or {})})
                cursor+=1;continue
            path,edges,selection=best
            sparse=_sparsify_controls(path,cfg)
            controls=[{"score_time":x.chart_time,"audio_time":x.midi_time,
                "roles":x.roles,"role_count":x.role_count,"classification":x.classification,
                "agreement_spread_seconds":x.agreement_spread_seconds,"confidence":x.confidence,
                "provenance":x.sources} for x in sparse]
            monotonic=all(b["score_time"]>a["score_time"] and b["audio_time"]>a["audio_time"]
                          for a,b in zip(controls,controls[1:]))
            if not monotonic:raise ValueError("non-monotonic sparse timing proposal")
            strong=sum(x["classification"]!="single_role_provisional" for x in controls)
            regions.append({"region_id":f"midi-section-region-{len(regions)+1}",
                "chart_start":controls[0]["score_time"],"chart_end":controls[-1]["score_time"],
                "midi_start":controls[0]["audio_time"],"midi_end":controls[-1]["audio_time"],
                "control_count":len(controls),"strong_control_count":strong,"controls":controls,
                "edge_stretches":edges,"minimum_edge_stretch":min(edges) if edges else None,
                "maximum_edge_stretch":max(edges) if edges else None,
                "catch_up_events":_catch_up_events(edges,cfg),"monotonic":monotonic,
                "split_reason_before_next_region":None if best_end==len(block) else
                    (failure or {}).get("status","incompatible_fusion_layer"),
                "diagnostic_only":True})
            if best_end==len(block):break
            rejected_blocks.append({"block_index":block_index,"layer_index":best_end,
                "status":"fusion_split_after_valid_prefix",**(failure or {})})
            cursor=best_end+1

    # Keep only globally forward, non-overlapping regions. A contradictory
    # isolated region is evidence we report, not a reason to erase earlier or
    # later valid regions.
    ordered_regions=[]
    for region in sorted(regions,key=lambda x:x["chart_start"]):
        if ordered_regions and (region["chart_start"]<=ordered_regions[-1]["chart_end"] or
                                region["midi_start"]<=ordered_regions[-1]["midi_end"]):
            rejected_blocks.append({"status":"region_not_globally_ordered",
                "region_id":region["region_id"],"chart_start":region["chart_start"],
                "midi_start":region["midi_start"]})
            continue
        ordered_regions.append(region)
    regions=ordered_regions
    all_controls = [control for region in regions for control in region["controls"]]
    globally_ordered = all(b["score_time"] > a["score_time"] and b["audio_time"] > a["audio_time"]
                           for a, b in zip(all_controls, all_controls[1:]))
    return {
        "version": "1.0-step3",
        "method": "midi-section-fit",
        "diagnostic_only": True,
        "timing_changes_applied": False,
        "normal_production_limits": {"minimum_edge_stretch": cfg.edge_stretch_min,
                                     "maximum_edge_stretch": cfg.edge_stretch_max},
        "monotonicity_certificate": {
            "strict_chart_time_increase": globally_ordered,
            "strict_midi_time_increase": globally_ordered,
            "regions_independent_across_gaps": True,
            "unsupported_gaps_interpolated": False,
        },
        "summary": {"input_layer_count": len(layers), "region_count": len(regions),
                    "control_count": len(all_controls), "role_conflict_count": len(role_conflicts),
                    "unsupported_point_count": len(unsupported),
                    "rejected_block_count": len(rejected_blocks)},
        "regions": regions,
        "role_conflicts": role_conflicts,
        "unsupported_points": unsupported,
        "rejected_blocks": rejected_blocks,
    }


MATERIALIZATION_MODES={
    "conservative":{"tiers":{"A"},"transition_min":.90,"transition_max":1.10,
        "short_delta_seconds":.12,"short_duration_seconds":6.0,"catch_up_episode_interval_measures":24,
        "episode_growth_start_measures":24,"episode_delta_growth_per_measure":.0025,"episode_delta_max_seconds":.25,
        "episode_stretch_min":.80,"episode_stretch_max":1.20},
    "permissive":{"tiers":{"A","B"},"transition_min":.85,"transition_max":1.15,
        "short_delta_seconds":.25,"short_duration_seconds":10.0,"catch_up_episode_interval_measures":16,
        "episode_growth_start_measures":16,"episode_delta_growth_per_measure":.005,"episode_delta_max_seconds":.50,
        "episode_stretch_min":.70,"episode_stretch_max":1.30},
    "diagnostic-unsafe":{"tiers":{"A","B","C"},"transition_min":.70,"transition_max":1.30,
        "short_delta_seconds":.50,"short_duration_seconds":16.0,"catch_up_episode_interval_measures":8,
        "episode_growth_start_measures":8,"episode_delta_growth_per_measure":.01,"episode_delta_max_seconds":.75,
        "episode_stretch_min":.50,"episode_stretch_max":1.50},
}


def _catch_up_episodes(events):
    ordered=sorted((dict(x) for x in (events or [])),key=lambda x:(int(x.get("first_edge_index",-1)),int(x.get("second_edge_index",-1))))
    episodes=[]
    for event in ordered:
        first=int(event.get("first_edge_index",-1));second=int(event.get("second_edge_index",first+1))
        if episodes and first<=episodes[-1]["last_edge_index"]:
            episodes[-1]["last_edge_index"]=max(episodes[-1]["last_edge_index"],second)
            episodes[-1]["pair_count"]+=1;episodes[-1]["events"].append(event)
        else:episodes.append({"episode_index":len(episodes),"first_edge_index":first,"last_edge_index":second,"pair_count":1,"events":[event],"kind":"local_correction_episode"})
    return episodes


def _region_evidence_tier(region,policy=None):
    controls=region.get("controls") or [];count=len(controls)
    strong=int(region.get("strong_control_count") or sum(x.get("classification")!="single_role_provisional" for x in controls))
    ratio=strong/max(1,count);events=region.get("catch_up_events") or [];episodes=_catch_up_episodes(events)
    measures=max(1,int(region.get("supported_playback_measure_count") or region.get("playback_measure_count") or 1))
    interval=max(1,int((policy or {}).get("catch_up_episode_interval_measures",16)));allowed=max(1,measures//interval)
    roles=set(role for item in controls for role in (item.get("roles") or []))
    base="A" if count>=3 and strong>=3 and ratio>=.75 else "B" if count>=3 and strong>=2 and ratio>=.40 else "C"
    tier="C" if base=="C" or len(episodes)>allowed else "B" if base=="A" and episodes else base
    return tier,{"control_count":count,"strong_control_count":strong,"strong_ratio":ratio,
        "catch_up_event_count":len(events),"catch_up_episode_count":len(episodes),"catch_up_episodes":episodes,
        "supported_playback_measure_count":measures,"catch_up_episode_interval_measures":interval,
        "allowed_catch_up_episode_count":allowed,"catch_up_density_per_measure":len(episodes)/measures,"roles":sorted(roles)}


def _transition_stretches(baseline, start, end, start_offset, end_offset, samples=12):
    if end<=start:return []
    points=[start+(end-start)*i/samples for i in range(samples+1)]
    mapped=[]
    for t in points:
        f=(t-start)/(end-start)
        mapped.append(float(baseline(t))+(1-f)*start_offset+f*end_offset)
    return [(mapped[i+1]-mapped[i])/(points[i+1]-points[i]) for i in range(samples)]


def _geometry_acceptance(chart_delta: float, audio_delta: float, policy: Mapping[str,Any]):
    if chart_delta<=0 or audio_delta<=0:
        return {"accepted":False,"acceptance_basis":"catastrophic_non_positive_geometry",
            "stretch":None,"absolute_correction_seconds":None}
    stretch=audio_delta/chart_delta
    correction=abs(audio_delta-chart_delta)
    normal=policy["transition_min"]<=stretch<=policy["transition_max"]
    short=(chart_delta<=policy["short_duration_seconds"] and
           correction<=policy["short_delta_seconds"])
    basis=("normal_stretch_range" if normal else
           "short_region_delta_exception" if short else
           "geometry_out_of_policy")
    return {"accepted":bool(normal or short),"acceptance_basis":basis,
        "stretch":stretch,"absolute_correction_seconds":correction,
        "duration_seconds":chart_delta,
        "short_delta_limit_seconds":policy["short_delta_seconds"],
        "short_duration_limit_seconds":policy["short_duration_seconds"]}


def _classify_catch_up(first: Mapping[str,Any], second: Mapping[str,Any]):
    if not first.get("accepted") or not second.get("accepted"):
        return "probable_artificial_catch_up"
    if (first.get("acceptance_basis")=="short_region_delta_exception" or
        second.get("acceptance_basis")=="short_region_delta_exception"):
        return "possible_local_chart_timing_defect"
    return "baseline_return_catch_up"


def _materialization_edge_assessments(region, policy):
    """Assess every internal control edge under the selected production policy."""
    controls=region.get("controls") or []
    _,quality=_region_evidence_tier(region,policy)
    assessments=[]
    for index,(a,b) in enumerate(zip(controls,controls[1:])):
        ds=float(b["score_time"])-float(a["score_time"])
        da=float(b["audio_time"])-float(a["audio_time"])
        assessment=_geometry_acceptance(ds,da,policy)
        episodes=quality.get("catch_up_episodes") or []
        belongs=any(int(ep["first_edge_index"])<=index<=int(ep["last_edge_index"]) for ep in episodes)
        measures=int(quality.get("supported_playback_measure_count") or 1)
        allowance=min(float(policy["episode_delta_max_seconds"]),
            float(policy["short_delta_seconds"])+max(0,measures-int(policy["episode_growth_start_measures"]))*float(policy["episode_delta_growth_per_measure"]))
        episode_pass=(not assessment["accepted"] and belongs and ds<=policy["short_duration_seconds"] and
            allowance>=float(assessment.get("absolute_correction_seconds") or 1e9) and
            policy["episode_stretch_min"]<=float(assessment.get("stretch") or 0)<=policy["episode_stretch_max"])
        if episode_pass:
            assessment["accepted"]=True
            assessment["acceptance_basis"]="length_adjusted_episode_delta_exception"
        assessment.update({"edge_belongs_to_catch_up_episode":belongs,
            "episode_delta_allowance_seconds":allowance,
            "region_supported_playback_measure_count":measures,
            "episode_stretch_min":policy["episode_stretch_min"],"episode_stretch_max":policy["episode_stretch_max"],
            "edge_index":index,"chart_start":float(a["score_time"]),"chart_end":float(b["score_time"])})
        assessments.append(assessment)
    return assessments


def _materialization_fragments(region, policy):
    """Split one fused region at edges that the selected production mode rejects."""
    controls=list(region.get("controls") or [])
    assessments=_materialization_edge_assessments(region,policy)
    failed={int(x["edge_index"]):x for x in assessments if not x["accepted"]}
    if not failed:
        complete=dict(region,edge_assessments=assessments,source_region_id=region.get("region_id"),
                      fragment_index=0,fragment_count=1,materialization_recovered=False)
        if controls:
            complete.setdefault("chart_start",float(controls[0]["score_time"]))
            complete.setdefault("chart_end",float(controls[-1]["score_time"]))
            complete.setdefault("midi_start",float(controls[0]["audio_time"]))
            complete.setdefault("midi_end",float(controls[-1]["audio_time"]))
        complete.setdefault("control_count",len(controls))
        return [complete],[]
    spans=[];begin=0
    for edge_index in sorted(failed):
        spans.append((begin,edge_index+1))
        begin=edge_index+1
    spans.append((begin,len(controls)))
    fragments=[];discarded=[]
    source_id=region.get("region_id")
    valid_spans=[span for span in spans if span[1]-span[0]>=2]
    for fragment_index,(a,b) in enumerate(valid_spans):
        part=controls[a:b]
        fragment=dict(region)
        fragment.update({"region_id":f"{source_id}-materialized-{fragment_index+1}",
            "source_region_id":source_id,"fragment_index":fragment_index,
            "fragment_count":len(valid_spans),"materialization_recovered":True,
            "controls":part,"chart_start":float(part[0]["score_time"]),"chart_end":float(part[-1]["score_time"]),
            "midi_start":float(part[0]["audio_time"]),"midi_end":float(part[-1]["audio_time"]),
            "control_count":len(part),
            "strong_control_count":sum(x.get("classification")!="single_role_provisional" for x in part),
            "monotonic":all(float(y["score_time"])>float(x["score_time"]) and float(y["audio_time"])>float(x["audio_time"])
                            for x,y in zip(part,part[1:])),
            "edge_assessments":[dict(x,edge_index=int(x["edge_index"])-a) for x in assessments[a:max(a,b-1)]],
            "split_failed_edge_before":failed.get(a-1),"split_failed_edge_after":failed.get(b-1)})
        fragments.append(fragment)
    for a,b in spans:
        if b-a<2:
            adjacent=[failed.get(a-1),failed.get(b-1)]
            catastrophic=any(x and x.get("acceptance_basis")=="catastrophic_non_positive_geometry" for x in adjacent)
            discarded.append({"region_id":source_id,"reason":"materialization_fragment_too_short",
                "control_count":b-a,"chart_start":float(controls[a]["score_time"]) if b>a else None,
                "chart_end":float(controls[b-1]["score_time"]) if b>a else None,
                "severity":"catastrophic" if catastrophic else "tier_1_close"})
    return fragments,discarded


def build_section_materialized_warp(fusion_report: Mapping[str,Any], baseline: Callable[[float],float],
                                    mode: str="conservative"):
    """Materialize policy-safe fragments while preserving evidence around bad edges.

    Unsupported spans retain the baseline's local shape. A fused region is first
    subdivided at edges rejected by the selected materialization mode. Later
    fragments are reconsidered independently and may trim leading controls until
    a safe baseline-shaped handoff is available; no failed edge is represented as
    MIDI-validated timing.
    """
    materialization_started=time.perf_counter()
    if mode not in MATERIALIZATION_MODES:
        raise ValueError(f"invalid MIDI section materialization mode: {mode}")
    policy=MATERIALIZATION_MODES[mode];eligible=[];rejected=[];split_edges=[]
    baseline,baseline_sanity=sra.safe_materialization_baseline(fusion_report,baseline)
    if not baseline_sanity.get("valid"):
        return baseline,{"status":"materialization_blocked","mode":mode,"reason":baseline_sanity.get("blocked_reason"),"baseline_sanity":baseline_sanity,"applied_regions":[],"rejected_regions":[],"transitions":[],"timing_changes_applied":False,"global_monotonicity_validated":False,"materialization_seconds":time.perf_counter()-materialization_started}

    for source in sorted(fusion_report.get("regions") or [],key=lambda x:float(x.get("chart_start",0))):
        controls=source.get("controls") or []
        if len(controls)<2 or not source.get("monotonic",False):
            tier,quality=_region_evidence_tier(source,policy)
            rejected.append({"region_id":source.get("region_id"),"source_region_id":source.get("region_id"),
                "evidence_tier":tier,"quality":quality,"chart_start":source.get("chart_start"),"chart_end":source.get("chart_end"),
                "reason":"structurally_invalid_region","severity":"catastrophic"})
            continue
        fragments,discarded=_materialization_fragments(source,policy)
        rejected.extend(discarded)
        source_assessments=_materialization_edge_assessments(source,policy)
        split_edges.extend({"source_region_id":source.get("region_id"),**x} for x in source_assessments if not x["accepted"])
        for fragment in fragments:
            tier,quality=_region_evidence_tier(fragment,policy)
            row={**fragment,"evidence_tier":tier,"quality":quality}
            if tier not in policy["tiers"]:
                rejected.append({**row,"reason":"evidence_tier_not_enabled",
                    "severity":"tier_1_close" if tier=="B" else "tier_2_high_risk"})
                continue
            row["entry_offset_seconds"]=float(row["controls"][0]["audio_time"])-float(baseline(row["chart_start"]))
            row["exit_offset_seconds"]=float(row["controls"][-1]["audio_time"])-float(baseline(row["chart_end"]))
            eligible.append(row)

    eligible.sort(key=lambda x:float(x["chart_start"]))
    applied=[];transitions=[]
    for original in eligible:
        region=original
        if not applied:
            applied.append(region);continue
        previous=applied[-1]
        accepted=False
        # A split suffix may discard leading controls until it has enough runway
        # for a safe baseline-shaped re-entry. This preserves later evidence
        # without pretending that the failed edge itself is validated.
        for trim in range(0,max(1,len(region["controls"])-1)):
            if trim:
                part=region["controls"][trim:]
                if len(part)<2:break
                candidate=dict(region)
                candidate.update({"region_id":f"{original['region_id']}-trim{trim}","controls":part,
                    "chart_start":float(part[0]["score_time"]),"chart_end":float(part[-1]["score_time"]),
                    "midi_start":float(part[0]["audio_time"]),"midi_end":float(part[-1]["audio_time"]),
                    "control_count":len(part),"trimmed_leading_control_count":trim,
                    "strong_control_count":sum(x.get("classification")!="single_role_provisional" for x in part)})
                tier,quality=_region_evidence_tier(candidate,policy)
                if tier not in policy["tiers"]:continue
                candidate.update(evidence_tier=tier,quality=quality,
                    entry_offset_seconds=float(part[0]["audio_time"])-float(baseline(candidate["chart_start"])),
                    exit_offset_seconds=float(part[-1]["audio_time"])-float(baseline(candidate["chart_end"])))
                region=candidate
            gap_start=float(previous["chart_end"]);gap_end=float(region["chart_start"])
            if gap_end<=gap_start:
                continue
            stretches=_transition_stretches(baseline,gap_start,gap_end,
                float(previous["exit_offset_seconds"]),float(region["entry_offset_seconds"]))
            minimum=min(stretches);maximum=max(stretches)
            start_audio=float(baseline(gap_start))+float(previous["exit_offset_seconds"])
            end_audio=float(baseline(gap_end))+float(region["entry_offset_seconds"])
            geometry=_geometry_acceptance(gap_end-gap_start,end_audio-start_audio,policy)
            transition={"from_region":previous["region_id"],"to_region":region["region_id"],
                "chart_start":gap_start,"chart_end":gap_end,"duration_seconds":gap_end-gap_start,
                "start_offset_seconds":previous["exit_offset_seconds"],"end_offset_seconds":region["entry_offset_seconds"],
                "minimum_stretch":minimum,"maximum_stretch":maximum,"mode":mode,
                "classification":"baseline_shaped_offset_transition","midi_validated":False,
                "accepted":geometry["accepted"],**geometry}
            transitions.append(transition)
            if geometry["accepted"]:
                applied.append(region);accepted=True;break
        if not accepted:
            last=transitions[-1] if transitions else None
            failed=dict(original)
            failed.update(reason="transition_stretch_out_of_policy",severity="tier_2_high_risk")
            if last:
                failed.update(required_minimum_stretch=last.get("minimum_stretch"),
                              required_maximum_stretch=last.get("maximum_stretch"))
            rejected.append(failed)

    def interpolation(controls,t):
        for a,b in zip(controls,controls[1:]):
            if t<=float(b["score_time"]):
                f=(t-float(a["score_time"]))/(float(b["score_time"])-float(a["score_time"]))
                return float(a["audio_time"])+f*(float(b["audio_time"])-float(a["audio_time"]))
        return float(controls[-1]["audio_time"])

    def warp(t):
        t=float(t)
        if not applied:return float(baseline(t))
        if t<applied[0]["chart_start"]:
            return float(baseline(t))+float(applied[0]["entry_offset_seconds"])
        for index,region in enumerate(applied):
            if region["chart_start"]<=t<=region["chart_end"]:
                return interpolation(region["controls"],t)
            if index+1<len(applied):
                nxt=applied[index+1]
                if region["chart_end"]<t<nxt["chart_start"]:
                    f=(t-region["chart_end"])/(nxt["chart_start"]-region["chart_end"])
                    offset=(1-f)*region["exit_offset_seconds"]+f*nxt["entry_offset_seconds"]
                    return float(baseline(t))+offset
        return float(baseline(t))+float(applied[-1]["exit_offset_seconds"])

    samples=[]
    for region in applied:
        controls=region["controls"]
        samples.extend(float(x["score_time"]) for x in controls)
        samples.extend((float(region["chart_start"])-.001,float(region["chart_end"])+.001))
    samples=sorted(set(samples));globally_monotonic=all(warp(b)>warp(a) for a,b in zip(samples,samples[1:]))
    if not globally_monotonic:
        return baseline,{"status":"materialization_blocked","mode":mode,"reason":"global_monotonicity_failure",
            "applied_regions":[],"rejected_regions":rejected,"transitions":transitions,
            "materialization_split_edges":split_edges,"timing_changes_applied":False,
            "global_monotonicity_validated":False,"materialization_seconds":time.perf_counter()-materialization_started}
    recovered=any(x.get("materialization_recovered") for x in applied)
    status="available_recovered" if applied and recovered else "available" if applied else "no_eligible_regions"
    return warp,{"status":status,"mode":mode,"authority":"midi_section_fit_materialization",
        "timing_changes_applied":bool(applied),"applied_regions":applied,"rejected_regions":rejected,
        "transitions":transitions,"materialization_split_edges":split_edges,
        "materialization_recovery_applied":recovered,
        "unsupported_gap_policy":"baseline_shape_with_carried_or_interpolated_offset",
        "unsupported_gaps_midi_validated":False,"global_monotonicity_validated":True,
        "raw_endpoint_delta_used_as_rejection":False,
        "geometry_policy":{**dict(policy),"tiers":sorted(policy["tiers"])},
        "baseline_sanity":baseline_sanity,
        "production_phase_diagnostics":build_materialization_phase_diagnostics(fusion_report,applied,rejected,transitions),
        "materialization_seconds":time.perf_counter()-materialization_started}


def build_forced_section_warp(fusion_report: Mapping[str,Any], baseline: Callable[[float],float],
                              maximum_handoff_delta_seconds: float=.35):
    """Backward-compatible alias for permissive materialization."""
    return build_section_materialized_warp(fusion_report,baseline,"permissive")


def _candidate_cache_json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): _candidate_cache_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_candidate_cache_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_candidate_cache_json_safe(v) for v in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _candidate_cache_payload(playback_measures, chart_events_by_role, midi_events_by_role,
                             baseline, candidate_config, baseline_hypotheses):
    points=set()
    for index,row in enumerate(playback_measures):
        measure=_normalise_measure(row,index)
        points.add(float(measure["start"]));points.add(float(measure["end"]))
    for events in chart_events_by_role.values():
        for row in events:
            for key in ("t_gp","time","score_time","start_time","t"):
                if isinstance(row,Mapping) and row.get(key) is not None:
                    points.add(float(row[key]));break
    points=sorted(points)
    hypotheses=dict(baseline_hypotheses or {})
    if baseline is not None and "baseline" not in hypotheses:
        hypotheses["baseline"]=baseline
    samples={name:[[x,float(fn(x))] for x in points] for name,fn in sorted(hypotheses.items())}
    return _candidate_cache_json_safe({
        "schema_version":1,
        "implementation_version":"midi-section-candidates-v16-phase1",
        "playback_measures":playback_measures,
        "chart_events_by_role":chart_events_by_role,
        "midi_events_by_role":midi_events_by_role,
        "candidate_config":asdict(candidate_config or SectionFitConfig()),
        "baseline_samples":samples,
    })


def _candidate_cache_fingerprint(payload):
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_candidate_cache(path,fingerprint,disabled,rebuild):
    report={"enabled":not disabled,"hit":False,"reason":"cache_disabled" if disabled else
            "cache_rebuild_requested" if rebuild else "cache_file_missing",
            "path":path,"fingerprint":fingerprint,"load_seconds":0.0}
    if disabled or rebuild or not path:
        return None,report
    started=time.perf_counter()
    if not os.path.isfile(path):
        report["load_seconds"]=time.perf_counter()-started;return None,report
    try:
        with open(path,"r",encoding="utf-8") as f:data=json.load(f)
    except Exception:
        report.update(reason="cache_load_error",load_seconds=time.perf_counter()-started);return None,report
    if int(data.get("schema_version",-1))!=1:
        report.update(reason="cache_schema_changed",load_seconds=time.perf_counter()-started);return None,report
    if data.get("fingerprint")!=fingerprint:
        report.update(reason="input_fingerprint_changed",load_seconds=time.perf_counter()-started);return None,report
    candidate_report=data.get("candidate_report")
    if not isinstance(candidate_report,dict):
        report.update(reason="cache_payload_invalid",load_seconds=time.perf_counter()-started);return None,report
    report.update(hit=True,reason="exact_cache_hit",load_seconds=time.perf_counter()-started)
    return candidate_report,report


def _write_candidate_cache(path,fingerprint,payload,candidate_report):
    if not path:return
    parent=os.path.dirname(os.path.abspath(path));os.makedirs(parent,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".midi-section-cache-",suffix=".tmp",dir=parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as f:
            json.dump({"schema_version":1,"fingerprint":fingerprint,"payload":payload,
                       "candidate_report":_candidate_cache_json_safe(candidate_report)},
                      f,sort_keys=True,separators=(",",":"),allow_nan=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.remove(tmp)


def _phase_measure_at(measures, chart_time):
    for row in measures:
        if float(row["start"])<=float(chart_time)<=float(row["end"])+1e-9:
            duration=float(row["end"])-float(row["start"])
            beats=int(row.get("numerator") or row.get("time_signature_numerator") or 4)
            return {"measure_duration_seconds":duration,
                "beat_duration_seconds":duration/max(1,beats),"beat_count":beats,
                "playback_measure_number":int(row.get("playback_measure_number") or 0)}
    return {"measure_duration_seconds":None,"beat_duration_seconds":None,
        "beat_count":None,"playback_measure_number":None}


def _phase_units(seconds, timing):
    beat=timing.get("beat_duration_seconds");measure=timing.get("measure_duration_seconds")
    return {"seconds":seconds,"beats":None if not beat else seconds/beat,
        "measures":None if not measure else seconds/measure}


def _phase_persistence(offsets, index):
    previous=offsets[index];current=offsets[index+1];delta=current-previous
    following=offsets[index+2:index+4]
    if not following:return "unresolved_end_of_region"
    remains=sum(abs(x-current)<=abs(x-previous) for x in following)
    returns=sum(abs(x-previous)<abs(x-current) for x in following)
    if returns==len(following):return "temporary_correction"
    if remains==len(following):return "persistent_offset_state"
    if delta and all((x-current)*delta>=-.08 for x in following):return "gradual_cumulative_drift"
    return "mixed_or_noisy"


def build_fused_production_phase_diagnostics(fusion_report, playback_measures):
    """Diagnostic-only production-control, region, opening, and terminal analysis."""
    measures=[_normalise_measure(x,i) for i,x in enumerate(playback_measures)]
    region_reports=[];review=[]
    for region in fusion_report.get("regions") or []:
        controls=region.get("controls") or []
        offsets=[float(x["audio_time"])-float(x["score_time"]) for x in controls]
        edges=[]
        for index,(a,b) in enumerate(zip(controls,controls[1:])):
            delta=offsets[index+1]-offsets[index]
            timing=_phase_measure_at(measures,float(b["score_time"]));units=_phase_units(delta,timing)
            beat_abs=abs(units["beats"]) if units["beats"] is not None else None
            measure_abs=abs(units["measures"]) if units["measures"] is not None else None
            beat_like=bool(beat_abs is not None and abs(beat_abs-1)<=.22)
            measure_like=bool(measure_abs is not None and abs(measure_abs-1)<=.18)
            persistence=_phase_persistence(offsets,index)
            common_roles=sorted(set(a.get("roles") or []) & set(b.get("roles") or []))
            row={"edge_index":index,"chart_start":float(a["score_time"]),
                "chart_end":float(b["score_time"]),"previous_offset_seconds":offsets[index],
                "current_offset_seconds":offsets[index+1],"offset_change":units,
                **timing,"beat_sized_phase_change":beat_like,
                "measure_sized_phase_change":measure_like,"persistence":persistence,
                "roles_before":a.get("roles") or [],"roles_after":b.get("roles") or [],
                "persistent_supporting_roles":common_roles,"diagnostic_only":True}
            edges.append(row)
            severity=0
            if persistence=="persistent_offset_state":severity+=2
            if persistence=="gradual_cumulative_drift":severity+=1
            if beat_like:severity+=2
            if measure_like:severity+=3
            if severity>=2:
                review.append({"kind":"persistent_phase_change" if persistence=="persistent_offset_state" else "phase_change",
                    "severity_score":severity,"chart_time":float(b["score_time"]),
                    "region_id":region.get("region_id"),"roles":sorted(set(row["roles_before"]+row["roles_after"])),
                    "offset_change":units,"persistence":persistence,
                    "reason":"production fused-control phase movement","diagnostic_only":True})
        if controls:
            entry=offsets[0];exit_=offsets[-1];trend=exit_-entry
            timing=_phase_measure_at(measures,float(controls[-1]["score_time"]));trend_units=_phase_units(trend,timing)
            maximum=max(abs(x-entry) for x in offsets)
            returns=abs(exit_-entry)<=max(.12,maximum*.35) if maximum else True
        else:
            entry=exit_=maximum=None;trend_units=_phase_units(None,{"beat_duration_seconds":None,"measure_duration_seconds":None})
            returns=False
        region_row={"region_id":region.get("region_id"),"chart_start":region.get("chart_start"),
            "chart_end":region.get("chart_end"),"control_count":len(controls),
            "entry_offset_seconds":entry,"exit_offset_seconds":exit_,
            "cumulative_offset_trend":trend_units,"maximum_departure_from_entry_seconds":maximum,
            "returns_toward_entry_phase":returns,"edge_count":len(edges),"edges":edges,
            "diagnostic_only":True}
        region_reports.append(region_row)
        beats=trend_units.get("beats")
        if beats is not None and abs(beats)>=1.25 and not returns:
            review.append({"kind":"gradual_region_drift","severity_score":min(5,1+int(abs(beats))),
                "chart_start":region.get("chart_start"),"chart_end":region.get("chart_end"),
                "region_id":region.get("region_id"),"cumulative_offset_trend":trend_units,
                "reason":"region exits materially displaced from its entry phase","diagnostic_only":True})
    first_onset=None
    for region in fusion_report.get("regions") or []:
        if region.get("controls"):
            value=float(region["controls"][0]["score_time"])
            first_onset=value if first_onset is None else min(first_onset,value)
    root=next((x for x in fusion_report.get("regions") or []
        if (x.get("acquisition_root") or {}).get("validated")),None)
    opening={"first_fused_control_chart_time":first_onset,
        "validated_acquisition_root":bool(root),
        "acquisition_root_region_id":root.get("region_id") if root else None,
        "status":"rooted" if root else "unrooted_opening","diagnostic_only":True}
    if not root:
        review.append({"kind":"unrooted_opening","severity_score":4,"chart_time":first_onset,
            "reason":"no validated acquisition root precedes production-region authority","diagnostic_only":True})
    terminal_chart=float(measures[-1]["end"]) if measures else None
    last_control=None
    for region in fusion_report.get("regions") or []:
        for control in region.get("controls") or []:
            if last_control is None or float(control["score_time"])>float(last_control["score_time"]):last_control=control
    terminal={"chart_terminal_time":terminal_chart,
        "last_fused_control_chart_time":float(last_control["score_time"]) if last_control else None,
        "last_fused_control_offset_seconds":float(last_control["audio_time"])-float(last_control["score_time"]) if last_control else None,
        "unsupported_terminal_tail_seconds":None if terminal_chart is None or last_control is None else terminal_chart-float(last_control["score_time"]),
        "diagnostic_only":True}
    review.sort(key=lambda x:(-int(x.get("severity_score",0)),float(x.get("chart_time",x.get("chart_start",1e18)) or 0)))
    return {"version":"v16.6A.1-production-phase-diagnostics","selection_behavior_changed":False,
        "opening":opening,"terminal":terminal,"regions":region_reports,
        "review_points":review[:24]}


def build_materialization_phase_diagnostics(fusion_report, applied, rejected, transitions):
    source=(fusion_report.get("production_phase_diagnostics") or {})
    region_lookup={x.get("region_id"):x for x in source.get("regions") or []}
    fragments=[];review=[]
    for region in applied:
        controls=region.get("controls") or []
        if not controls:continue
        entry=float(controls[0]["audio_time"])-float(controls[0]["score_time"])
        exit_=float(controls[-1]["audio_time"])-float(controls[-1]["score_time"])
        fragments.append({"region_id":region.get("region_id"),"source_region_id":region.get("source_region_id") or region.get("region_id"),
            "chart_start":region.get("chart_start"),"chart_end":region.get("chart_end"),
            "entry_offset_seconds":entry,"exit_offset_seconds":exit_,
            "offset_trend_seconds":exit_-entry,"trimmed_leading_control_count":region.get("trimmed_leading_control_count",0),
            "diagnostic_only":True})
    gap_rows=[]
    for transition in transitions:
        start=float(transition.get("start_offset_seconds") or 0);end=float(transition.get("end_offset_seconds") or 0)
        change=end-start
        row={"from_region":transition.get("from_region"),"to_region":transition.get("to_region"),
            "chart_start":transition.get("chart_start"),"chart_end":transition.get("chart_end"),
            "start_offset_seconds":start,"end_offset_seconds":end,"offset_change_seconds":change,
            "accepted":transition.get("accepted"),"midi_validated":False,"diagnostic_only":True}
        gap_rows.append(row)
        if abs(change)>=.35:
            review.append({"kind":"unsupported_gap_phase_change","severity_score":3,
                "chart_start":row["chart_start"],"chart_end":row["chart_end"],
                "offset_change_seconds":change,"reason":"baseline-shaped gap changes carried phase",
                "diagnostic_only":True})
    terminal=dict(source.get("terminal") or {})
    if fragments:
        terminal["final_applied_region_id"]=fragments[-1]["region_id"]
        terminal["final_carried_offset_seconds"]=fragments[-1]["exit_offset_seconds"]
    opening=dict(source.get("opening") or {})
    opening["first_applied_region_chart_time"]=fragments[0]["chart_start"] if fragments else None
    if opening.get("status")=="unrooted_opening" and fragments:
        opening["unrooted_leading_span_seconds"]=float(fragments[0]["chart_start"] or 0)
    inherited=list(source.get("review_points") or [])
    combined=sorted(inherited+review,key=lambda x:(-int(x.get("severity_score",0)),float(x.get("chart_time",x.get("chart_start",1e18)) or 0)))
    return {"version":"v16.6A.1-production-phase-diagnostics","selection_behavior_changed":False,
        "opening":opening,"terminal":terminal,"applied_fragments":fragments,
        "unsupported_gap_phase_accounting":gap_rows,
        "rejected_region_count":len(rejected),"review_points":combined[:24]}


def build_section_fusion_report(
    playback_measures: Sequence[Mapping[str, Any]],
    chart_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    midi_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Callable[[float], float] | None = None,
    candidate_config: SectionFitConfig | None = None,
    path_config: SectionPathConfig | None = None,
    fusion_config: SectionFusionConfig | None = None,
    baseline_hypotheses: Mapping[str, Callable[[float], float]] | None = None,
    role_audio_starts: Mapping[str, float] | None = None,
    candidate_cache_path: str | None = None,
    candidate_cache_rebuild: bool = False,
    candidate_cache_disabled: bool = False,
) -> dict[str, Any]:
    """Run Steps 1-3, optionally reusing only the exact candidate report."""
    total_started=time.perf_counter()
    payload=_candidate_cache_payload(playback_measures,chart_events_by_role,midi_events_by_role,
                                     baseline,candidate_config,baseline_hypotheses)
    fingerprint=_candidate_cache_fingerprint(payload)
    candidate_report,cache_report=_load_candidate_cache(
        candidate_cache_path,fingerprint,bool(candidate_cache_disabled),bool(candidate_cache_rebuild))
    candidate_started=time.perf_counter()
    if candidate_report is None:
        candidate_report = build_section_candidate_report(
            playback_measures, chart_events_by_role, midi_events_by_role,
            baseline=baseline, cfg=candidate_config, baseline_hypotheses=baseline_hypotheses)
        if not candidate_cache_disabled and candidate_cache_path:
            _write_candidate_cache(candidate_cache_path,fingerprint,payload,candidate_report)
    candidate_seconds=time.perf_counter()-candidate_started
    candidate_report=annotate_candidate_phase_diagnostics(candidate_report)
    path_started=time.perf_counter()
    path_report = select_monotonic_section_paths(candidate_report, path_config)
    path_report["phase_diagnostics"]=build_selected_phase_diagnostics(path_report,candidate_report)
    path_seconds=time.perf_counter()-path_started
    fusion_started=time.perf_counter()
    fusion_report = fuse_monotonic_role_paths(path_report, fusion_config)
    sra.annotate_fusion_report(fusion_report,chart_events_by_role)
    itc.reconcile_opening(fusion_report,chart_events_by_role,role_audio_starts or {})
    normalised_measures=[_normalise_measure(x,i) for i,x in enumerate(playback_measures)]
    for region in fusion_report.get("regions") or []:
        covered=[x for x in normalised_measures if float(x["start"])<float(region["chart_end"])+1e-9 and float(x["end"])>float(region["chart_start"])-1e-9]
        numbers=sorted({int(x["playback_measure_number"]) for x in covered})
        region["supported_playback_measure_count"]=len(numbers);region["playback_measure_numbers"]=numbers
        region["catch_up_episodes"]=_catch_up_episodes(region.get("catch_up_events") or [])
    fusion_report["production_phase_diagnostics"]=build_fused_production_phase_diagnostics(fusion_report,normalised_measures)
    structural_repeat_report=srd.build(playback_measures,chart_events_by_role,midi_events_by_role)
    fusion_report["structural_repeat_diagnostics"]=structural_repeat_report
    # v16.6D is a guarded second selection pass. Phrase evidence never creates
    # timing controls and never removes candidates; it only rescales existing
    # section candidates when an already validated opening root agrees with an
    # exact, bounded phrase-ordinal run.
    supported_candidate_report,phrase_authority=rps.apply(candidate_report,structural_repeat_report,fusion_report)
    initial_summary={"region_count":len(fusion_report.get("regions") or []),
                     "control_count":sum(len(x.get("controls") or []) for x in fusion_report.get("regions") or [])}
    if phrase_authority.get("adjusted_candidate_count"):
        candidate_report=supported_candidate_report
        path_report=select_monotonic_section_paths(candidate_report,path_config)
        path_report["phase_diagnostics"]=build_selected_phase_diagnostics(path_report,candidate_report)
        fusion_report=fuse_monotonic_role_paths(path_report,fusion_config)
        sra.annotate_fusion_report(fusion_report,chart_events_by_role)
        itc.reconcile_opening(fusion_report,chart_events_by_role,role_audio_starts or {})
        for region in fusion_report.get("regions") or []:
            covered=[x for x in normalised_measures if float(x["start"])<float(region["chart_end"])+1e-9 and float(x["end"])>float(region["chart_start"])-1e-9]
            numbers=sorted({int(x["playback_measure_number"]) for x in covered})
            region["supported_playback_measure_count"]=len(numbers);region["playback_measure_numbers"]=numbers
            region["catch_up_episodes"]=_catch_up_episodes(region.get("catch_up_events") or [])
        fusion_report["production_phase_diagnostics"]=build_fused_production_phase_diagnostics(fusion_report,normalised_measures)
        fusion_report["structural_repeat_diagnostics"]=structural_repeat_report
    phrase_authority["initial_fusion_summary"]=initial_summary
    phrase_authority["final_fusion_summary"]={"region_count":len(fusion_report.get("regions") or []),
        "control_count":sum(len(x.get("controls") or []) for x in fusion_report.get("regions") or [])}
    fusion_report["rooted_phrase_ordinal_support"]=phrase_authority
    fusion_seconds=time.perf_counter()-fusion_started
    performance={"candidate_generation_seconds":candidate_seconds,
        "candidate_cache":cache_report,
        "path_selection_seconds":path_seconds,"fusion_seconds":fusion_seconds,
        "total_section_fit_seconds":time.perf_counter()-total_started,
        "candidate_counters":candidate_report.get("performance") or {}}
    report={"version": "2.2-v16.6E.3-final-phase-diagnostic-cleanup", "method": "midi-section-fit",
            "diagnostic_only": True, "timing_changes_applied": False,
            "performance":performance,"candidate_report": candidate_report,
            "path_report": path_report,"fusion_report": fusion_report,
            "structural_repeat_report":structural_repeat_report,
            "rooted_phrase_ordinal_support":phrase_authority}
    report["section_evidence_quality"]=seq.build(report)
    fusion_report["section_evidence_quality"]=report["section_evidence_quality"]
    report["phase_migration_qualification"]=pmq.build(fusion_report,normalised_measures,report["section_evidence_quality"])
    fusion_report["phase_migration_qualification"]=report["phase_migration_qualification"]
    return report

