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
    return {"version":"1.1-exact-optimized","method":"midi-section-fit","diagnostic_only":True,
        "timing_changes_applied":False,"monotonic_path_selected":False,"config":asdict(cfg),
        "search_hypotheses":list(hypotheses),"summary":totals,"performance":global_perf,"roles":role_reports}


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
        "short_delta_seconds":.12,"short_duration_seconds":6.0,
        "catch_up_episode_interval_measures":24,"episode_growth_start_measures":16,
        "episode_delta_growth_per_measure":.003,"episode_delta_max_seconds":.25,
        "episode_stretch_min":.80,"episode_stretch_max":1.20},
    "permissive":{"tiers":{"A","B"},"transition_min":.85,"transition_max":1.15,
        "short_delta_seconds":.25,"short_duration_seconds":10.0,
        "catch_up_episode_interval_measures":16,"episode_growth_start_measures":16,
        "episode_delta_growth_per_measure":.005,"episode_delta_max_seconds":.50,
        "episode_stretch_min":.70,"episode_stretch_max":1.30},
    "diagnostic-unsafe":{"tiers":{"A","B","C"},"transition_min":.70,"transition_max":1.30,
        "short_delta_seconds":.50,"short_duration_seconds":16.0,
        "catch_up_episode_interval_measures":8,"episode_growth_start_measures":16,
        "episode_delta_growth_per_measure":.008,"episode_delta_max_seconds":.80,
        "episode_stretch_min":.50,"episode_stretch_max":1.50},
}


def _catch_up_episodes(events: Sequence[Mapping[str,Any]]):
    """Collapse overlapping adjacent-pair warnings into local correction episodes."""
    ordered=sorted((dict(x) for x in (events or [])),
                   key=lambda x:(int(x.get("first_edge_index",-1)),int(x.get("second_edge_index",-1))))
    episodes=[]
    for event in ordered:
        first=int(event.get("first_edge_index",-1));second=int(event.get("second_edge_index",first+1))
        if episodes and first<=episodes[-1]["last_edge_index"]:
            episode=episodes[-1]
            episode["last_edge_index"]=max(episode["last_edge_index"],second)
            episode["pair_count"]+=1;episode["events"].append(event)
        else:
            episodes.append({"episode_index":len(episodes),"first_edge_index":first,
                "last_edge_index":second,"pair_count":1,"events":[event],
                "kind":"local_correction_episode"})
    return episodes


def _region_evidence_tier(region: Mapping[str,Any], policy: Mapping[str,Any] | None=None):
    controls=region.get("controls") or [];count=len(controls)
    strong=int(region.get("strong_control_count") or
               sum(x.get("classification")!="single_role_provisional" for x in controls))
    ratio=strong/max(1,count);events=region.get("catch_up_events") or []
    episodes=_catch_up_episodes(events)
    measures=max(1,int(region.get("supported_playback_measure_count") or
                       region.get("playback_measure_count") or 1))
    interval=max(1,int((policy or {}).get("catch_up_episode_interval_measures",16)))
    allowed=max(1,measures//interval)
    roles=set(role for item in controls for role in (item.get("roles") or []))
    base="A" if count>=3 and strong>=3 and ratio>=.75 else "B" if count>=3 and strong>=2 and ratio>=.40 else "C"
    if base=="C" or len(episodes)>allowed:tier="C"
    elif base=="A" and episodes:tier="B"
    else:tier=base
    return tier,{"control_count":count,"strong_control_count":strong,"strong_ratio":ratio,
        "catch_up_event_count":len(events),"catch_up_episode_count":len(episodes),
        "catch_up_episodes":episodes,"supported_playback_measure_count":measures,
        "catch_up_episode_interval_measures":interval,"allowed_catch_up_episode_count":allowed,
        "catch_up_density_per_measure":len(episodes)/measures,"roles":sorted(roles)}


def _episode_delta_allowance(measures: int, policy: Mapping[str,Any]) -> float:
    extra=max(0,int(measures)-int(policy["episode_growth_start_measures"]))
    return min(float(policy["episode_delta_max_seconds"]),
               float(policy["short_delta_seconds"])+extra*float(policy["episode_delta_growth_per_measure"]))


def _episode_edge_indices(episodes: Sequence[Mapping[str,Any]]) -> set[int]:
    indices=set()
    for episode in episodes or []:
        first=int(episode.get("first_edge_index",-1));last=int(episode.get("last_edge_index",first))
        indices.update(range(first,last+1))
    return indices


def _apply_episode_exception(assessment: dict[str,Any], edge_index: int, quality: Mapping[str,Any],
                             policy: Mapping[str,Any]) -> dict[str,Any]:
    if assessment.get("accepted") or edge_index not in _episode_edge_indices(quality.get("catch_up_episodes") or []):
        return assessment
    measures=int(quality["supported_playback_measure_count"])
    allowance=_episode_delta_allowance(measures,policy)
    stretch=assessment.get("stretch");correction=assessment.get("absolute_correction_seconds")
    accepted=(stretch is not None and correction is not None and
              float(policy["episode_stretch_min"])<=stretch<=float(policy["episode_stretch_max"]) and
              correction<=allowance)
    return {**assessment,"accepted":accepted,
        "acceptance_basis":"length_adjusted_episode_delta_exception" if accepted else assessment["acceptance_basis"],
        "edge_belongs_to_catch_up_episode":True,"episode_delta_allowance_seconds":allowance,
        "region_supported_playback_measure_count":measures,
        "episode_stretch_min":policy["episode_stretch_min"],
        "episode_stretch_max":policy["episode_stretch_max"]}


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


def build_section_materialized_warp(fusion_report: Mapping[str,Any], baseline: Callable[[float],float],
                                    mode: str="conservative"):
    """Materialize MIDI regions by evidence tier and transition warp, never raw delta.

    Unsupported spans retain the baseline's local shape. Before the first and
    after the last region, the nearest trusted displacement is carried. Between
    regions, displacement is interpolated only when the resulting transition
    stretch is permitted by the selected mode.
    """
    materialization_started=time.perf_counter()
    if mode not in MATERIALIZATION_MODES:
        raise ValueError(f"invalid MIDI section materialization mode: {mode}")
    policy=MATERIALIZATION_MODES[mode]; eligible=[]; rejected=[]
    for region in sorted(fusion_report.get("regions") or [],key=lambda x:float(x.get("chart_start",0))):
        tier,quality=_region_evidence_tier(region,policy)
        controls=region.get("controls") or []
        chart_start=region.get("chart_start") if region.get("chart_start") is not None else (controls[0].get("score_time") if controls else None)
        chart_end=region.get("chart_end") if region.get("chart_end") is not None else (controls[-1].get("score_time") if controls else None)
        row={"region_id":region.get("region_id"),"evidence_tier":tier,"quality":quality,
             "chart_start":chart_start,"chart_end":chart_end}
        if tier not in policy["tiers"]:
            rejected.append({**row,"reason":"evidence_tier_not_enabled","severity":"tier_1_close" if tier=="B" else "tier_2_high_risk"});continue
        if len(controls)<2 or not region.get("monotonic",False):
            rejected.append({**row,"reason":"structurally_invalid_region","severity":"catastrophic"});continue
        edge_assessments=[]
        for index,(a,b) in enumerate(zip(controls,controls[1:])):
            ds=float(b["score_time"])-float(a["score_time"])
            da=float(b["audio_time"])-float(a["audio_time"])
            assessment=_geometry_acceptance(ds,da,policy)
            assessment=_apply_episode_exception(assessment,index,quality,policy)
            assessment.update({"edge_index":index,"chart_start":float(a["score_time"]),
                "chart_end":float(b["score_time"])})
            edge_assessments.append(assessment)
        catastrophic=next((x for x in edge_assessments if x["acceptance_basis"]=="catastrophic_non_positive_geometry"),None)
        if catastrophic:
            rejected.append({**row,"reason":"non_monotonic_region_controls","severity":"catastrophic",
                "failed_edge":catastrophic});continue
        failed=[x for x in edge_assessments if not x["accepted"]]
        if failed:
            worst=max(failed,key=lambda x:x.get("absolute_correction_seconds") or 0)
            rejected.append({**row,"reason":"region_edge_geometry_out_of_policy",
                "severity":"tier_1_close" if tier=="A" else "tier_2_high_risk",
                "failed_edge":worst,"edge_assessments":edge_assessments});continue
        catch_up_classifications=[]
        for first,second in zip(edge_assessments,edge_assessments[1:]):
            x=(first["stretch"] or 1)-1;y=(second["stretch"] or 1)-1
            if x*y<0 and abs(x)>=.06 and abs(y)>=.06:
                catch_up_classifications.append({"first_edge_index":first["edge_index"],
                    "second_edge_index":second["edge_index"],"kind":_classify_catch_up(first,second)})
        local_stretches=[x["stretch"] for x in edge_assessments]
        eligible.append({**row,"controls":[dict(x) for x in controls],
            "entry_offset_seconds":float(controls[0]["audio_time"])-float(baseline(float(controls[0]["score_time"]))),
            "exit_offset_seconds":float(controls[-1]["audio_time"])-float(baseline(float(controls[-1]["score_time"]))),
            "minimum_region_stretch":min(local_stretches),"maximum_region_stretch":max(local_stretches),
            "edge_assessments":edge_assessments,
            "catch_up_classifications":catch_up_classifications})

    applied=[];transitions=[]
    for region in eligible:
        if not applied:
            applied.append(region);continue
        previous=applied[-1]
        gap_start=float(previous["chart_end"]);gap_end=float(region["chart_start"])
        if gap_end<=gap_start:
            rejected.append({**region,"reason":"overlapping_or_reversed_region","severity":"catastrophic"});continue
        stretches=_transition_stretches(baseline,gap_start,gap_end,
            float(previous["exit_offset_seconds"]),float(region["entry_offset_seconds"]))
        minimum=min(stretches);maximum=max(stretches)
        start_audio=float(baseline(gap_start))+float(previous["exit_offset_seconds"])
        end_audio=float(baseline(gap_end))+float(region["entry_offset_seconds"])
        geometry=_geometry_acceptance(gap_end-gap_start,end_audio-start_audio,policy)
        valid=geometry["accepted"]
        transition={"from_region":previous["region_id"],"to_region":region["region_id"],
            "chart_start":gap_start,"chart_end":gap_end,"duration_seconds":gap_end-gap_start,
            "start_offset_seconds":previous["exit_offset_seconds"],"end_offset_seconds":region["entry_offset_seconds"],
            "minimum_stretch":minimum,"maximum_stretch":maximum,"mode":mode,
            "classification":"baseline_shaped_offset_transition","midi_validated":False,
            "accepted":valid,**geometry}
        transitions.append(transition)
        if not valid:
            distance=max(policy["transition_min"]-minimum,maximum-policy["transition_max"],0)
            severity="tier_1_close" if distance<=.05 else "tier_2_high_risk" if minimum>0 and maximum<=1.5 else "catastrophic"
            rejected.append({**region,"reason":"transition_stretch_out_of_policy","severity":severity,
                "required_minimum_stretch":minimum,"required_maximum_stretch":maximum});continue
        applied.append(region)

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
    samples=sorted(set(samples)); globally_monotonic=all(warp(b)>warp(a) for a,b in zip(samples,samples[1:]))
    if not globally_monotonic:
        return baseline,{"status":"materialization_blocked","mode":mode,"reason":"global_monotonicity_failure",
            "applied_regions":[],"rejected_regions":rejected,"transitions":transitions,
            "timing_changes_applied":False,"global_monotonicity_validated":False,
            "materialization_seconds":time.perf_counter()-materialization_started}
    return warp,{"status":"available" if applied else "no_eligible_regions","mode":mode,
        "authority":"midi_section_fit_materialization","timing_changes_applied":bool(applied),
        "applied_regions":applied,"rejected_regions":rejected,"transitions":transitions,
        "unsupported_gap_policy":"baseline_shape_with_carried_or_interpolated_offset",
        "unsupported_gaps_midi_validated":False,"global_monotonicity_validated":True,
        "raw_endpoint_delta_used_as_rejection":False,
        "geometry_policy":{**dict(policy),"tiers":sorted(policy["tiers"])},
        "materialization_seconds":time.perf_counter()-materialization_started}


def build_forced_section_warp(fusion_report: Mapping[str,Any], baseline: Callable[[float],float],
                              maximum_handoff_delta_seconds: float=.35):
    """Backward-compatible alias for permissive materialization."""
    return build_section_materialized_warp(fusion_report,baseline,"permissive")


def build_section_fusion_report(
    playback_measures: Sequence[Mapping[str, Any]],
    chart_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    midi_events_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Callable[[float], float] | None = None,
    candidate_config: SectionFitConfig | None = None,
    path_config: SectionPathConfig | None = None,
    fusion_config: SectionFusionConfig | None = None,
    baseline_hypotheses: Mapping[str, Callable[[float], float]] | None = None,
) -> dict[str, Any]:
    """Run Steps 1-3 without modifying timing, with stage instrumentation."""
    total_started=time.perf_counter()
    candidate_started=time.perf_counter()
    candidate_report = build_section_candidate_report(
        playback_measures, chart_events_by_role, midi_events_by_role,
        baseline=baseline, cfg=candidate_config, baseline_hypotheses=baseline_hypotheses)
    candidate_seconds=time.perf_counter()-candidate_started
    path_started=time.perf_counter()
    path_report = select_monotonic_section_paths(candidate_report, path_config)
    path_seconds=time.perf_counter()-path_started
    fusion_started=time.perf_counter()
    fusion_report = fuse_monotonic_role_paths(path_report, fusion_config)
    normalised_measures=[_normalise_measure(x,i) for i,x in enumerate(playback_measures)]
    for region in fusion_report.get("regions") or []:
        covered=[x for x in normalised_measures
                 if float(x["start"])<float(region["chart_end"])+1e-9 and
                    float(x["end"])>float(region["chart_start"])-1e-9]
        numbers=sorted({int(x["playback_measure_number"]) for x in covered})
        region["supported_playback_measure_count"]=len(numbers)
        region["playback_measure_numbers"]=numbers
        region["catch_up_episodes"]=_catch_up_episodes(region.get("catch_up_events") or [])
    fusion_seconds=time.perf_counter()-fusion_started
    performance={"candidate_generation_seconds":candidate_seconds,
        "path_selection_seconds":path_seconds,"fusion_seconds":fusion_seconds,
        "total_section_fit_seconds":time.perf_counter()-total_started,
        "candidate_counters":candidate_report.get("performance") or {}}
    return {"version": "1.1-step3-exact-optimized", "method": "midi-section-fit",
            "diagnostic_only": True, "timing_changes_applied": False,
            "performance":performance,"candidate_report": candidate_report,
            "path_report": path_report,"fusion_report": fusion_report}
