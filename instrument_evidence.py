"""Phase 3B.1 calibrated instrument-pattern evidence.

Diagnostic only. Role-optimized candidates are retained, then filtered through
soft temporal path selection, corridor-edge rejection, neighbourhood support,
and cross-role agreement. No accepted warp or packaged timing is modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
import math
import os

import numpy as np

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

SCHEMA_VERSION = 2
EDGE_FRACTION = 0.85
CANDIDATES_PER_MEASURE = 5
NEIGHBOUR_TOLERANCE_MS = 90.0
CROSS_ROLE_TOLERANCE_MS = 70.0
HIGH_CONFIDENCE = 0.78
SKIP_COST = 0.42
TRANSITION_SCALE_MS = 120.0
ACCELERATION_SCALE_MS = 180.0


@dataclass(frozen=True)
class AdapterProfile:
    role: str
    feature_family: str
    fmin: float
    fmax: float | None
    onset_weight: float
    spectral_flux_weight: float
    harmonic_change_weight: float
    search_radius_seconds: float
    minimum_score_events: int
    minimum_peak_margin: float


PROFILES = {
    "drums": AdapterProfile("drums", "multiband_transient_pattern", 30.0, None, .70, .30, .00, .30, 2, .08),
    "bass": AdapterProfile("bass", "low_frequency_attack_pattern", 30.0, 1200.0, .65, .25, .10, .35, 2, .07),
    "guitar": AdapterProfile("guitar", "rhythmic_spectral_flux_pattern", 70.0, 6500.0, .45, .45, .10, .35, 2, .07),
    "piano": AdapterProfile("piano", "onset_harmonic_change_pattern", 40.0, 7000.0, .45, .20, .35, .35, 2, .07),
    "full": AdapterProfile("full", "broadband_structure_pattern", 30.0, None, .45, .35, .20, .35, 3, .10),
}
DEFAULT_PROFILE = AdapterProfile("unknown", "generic_novelty_pattern", 30.0, None, .60, .30, .10, .35, 2, .08)


def profile_for(role: str) -> AdapterProfile:
    return PROFILES.get(str(role).lower(), DEFAULT_PROFILE)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return values
    values = values - np.nanmin(values)
    peak = float(np.nanmax(values))
    return values / peak if peak > 1e-12 else np.zeros_like(values)


def _harmonic_change(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    if librosa is None or not len(y):
        return np.zeros(0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    if chroma.shape[1] < 2:
        return np.zeros(chroma.shape[1])
    return np.pad(np.sqrt(np.sum(np.diff(chroma, axis=1) ** 2, axis=0)), (1, 0))


def extract_role_feature(audio_path: str | None, role: str, sr: int = 22050,
                         hop_length: int = 256) -> dict[str, Any]:
    if librosa is None:
        return {"status": "unavailable", "reason": "librosa_not_installed", "role": role}
    if not audio_path or not os.path.isfile(audio_path):
        return {"status": "unavailable", "reason": "missing_audio", "role": role}
    p = profile_for(role)
    y, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    if not len(y):
        return {"status": "unavailable", "reason": "empty_audio", "role": role}
    nyquist = actual_sr / 2
    fmax = min(p.fmax or nyquist, nyquist)
    onset = librosa.onset.onset_strength(y=y, sr=actual_sr, hop_length=hop_length,
                                         fmin=max(0, p.fmin), fmax=fmax)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=actual_sr, n_fft=2048)
    selected = stft[(frequencies >= p.fmin) & (frequencies <= fmax)]
    if not selected.size:
        selected = stft
    flux = np.pad(np.sqrt(np.sum(np.maximum(np.diff(selected, axis=1), 0) ** 2, axis=0)), (1, 0))
    harmonic = _harmonic_change(y, actual_sr, hop_length)
    length = min(len(onset), len(flux), len(harmonic))
    onset, flux, harmonic = (_normalize(onset[:length]), _normalize(flux[:length]), _normalize(harmonic[:length]))
    novelty = _normalize(p.onset_weight * onset + p.spectral_flux_weight * flux + p.harmonic_change_weight * harmonic)
    return {"status": "ok", "role": role, "feature_family": p.feature_family,
            "sample_rate": actual_sr, "hop_length": hop_length,
            "times": librosa.frames_to_time(np.arange(length), sr=actual_sr, hop_length=hop_length),
            "novelty": novelty, "components": {"onset": onset, "spectral_flux": flux, "harmonic_change": harmonic},
            "profile": asdict(p)}


def _event_times(events: Iterable[Mapping[str, Any] | float]) -> list[float]:
    output = []
    for event in events or []:
        if isinstance(event, (int, float)):
            output.append(float(event)); continue
        for key in ("t_gp", "time", "t"):
            if key in event:
                output.append(float(event[key])); break
    return sorted(set(output))


def _template(times: np.ndarray, events: Sequence[float], sigma: float = .025) -> np.ndarray:
    value = np.zeros(len(times))
    for event in events:
        value += np.exp(-.5 * ((times - event) / sigma) ** 2)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else value


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    divisor = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / divisor) if divisor > 1e-12 else 0.0


def _calibrated_confidence(score: float, margin: float, residual: float, radius: float,
                           event_count: int) -> float:
    score_term = min(1.0, max(0.0, score / .75))
    margin_term = min(1.0, max(0.0, margin / .18))
    edge_term = max(0.0, 1.0 - abs(residual) / max(radius, 1e-9))
    count_term = min(1.0, event_count / 6.0)
    return round(.35 * score_term + .30 * margin_term + .25 * edge_term + .10 * count_term, 6)


def measure_candidates(role: str, nominal_start: float, nominal_end: float,
                       score_events: Sequence[float], warp_fn: Callable[[float], float],
                       feature: Mapping[str, Any], playback_measure: int) -> dict[str, Any]:
    p = profile_for(role)
    events = [x for x in score_events if nominal_start <= x < nominal_end]
    base = {"playback_measure": int(playback_measure), "source_role": role,
            "feature_family": p.feature_family, "score_event_count": len(events)}
    if len(events) < p.minimum_score_events:
        return {**base, "status": "unavailable", "reason": "insufficient_score_events", "candidates": []}
    if feature.get("status") != "ok":
        return {**base, "status": "unavailable", "reason": feature.get("reason", "feature_unavailable"), "candidates": []}
    times, novelty = np.asarray(feature["times"]), np.asarray(feature["novelty"])
    predicted_start, predicted_end = float(warp_fn(nominal_start)), float(warp_fn(nominal_end))
    predicted_events = [float(warp_fn(x)) for x in events]
    radius = p.search_radius_seconds
    mask = (times >= predicted_start - radius) & (times <= predicted_end + radius)
    local_times, local_audio = times[mask], novelty[mask]
    if len(local_times) < 8:
        return {**base, "status": "unavailable", "reason": "insufficient_audio_frames", "candidates": []}
    step = float(np.median(np.diff(local_times)))
    lag_frames = max(1, int(round(radius / step)))
    lags = np.arange(-lag_frames, lag_frames + 1)
    scores = np.array([_cosine(_template(local_times, [x + lag * step for x in predicted_events]), local_audio) for lag in lags])
    order = list(np.argsort(scores)[::-1])
    exclusion = max(1, int(round(.05 / step)))
    chosen = []
    for index in order:
        if all(abs(int(index) - old) > exclusion for old in chosen):
            chosen.append(int(index))
        if len(chosen) >= CANDIDATES_PER_MEASURE:
            break
    candidates = []
    for rank, index in enumerate(chosen):
        residual = float(lags[index] * step)
        alternatives = [float(scores[j]) for j in chosen if j != index]
        runner = max(alternatives) if alternatives else 0.0
        margin = float(scores[index] - runner)
        edge = abs(residual) >= EDGE_FRACTION * radius
        confidence = _calibrated_confidence(float(scores[index]), margin, residual, radius, len(events))
        candidates.append({"rank": rank + 1, "residual_seconds": round(residual, 6),
                           "residual_ms": round(residual * 1000, 3), "score": round(float(scores[index]), 6),
                           "runner_up_margin": round(margin, 6), "calibrated_confidence": confidence,
                           "corridor_saturated": edge,
                           "locally_eligible": bool(not edge and float(scores[index]) > .05 and margin >= p.minimum_peak_margin)})
    return {**base, "status": "ok", "reason": None, "candidates": candidates,
            "nominal_start": round(nominal_start, 6), "nominal_end": round(nominal_end, 6),
            "predicted_audio_start": round(predicted_start, 6), "predicted_audio_end": round(predicted_end, 6),
            "search_radius_seconds": radius, "frame_step_seconds": round(step, 9)}




def match_measure_pattern(role: str, nominal_start: float, nominal_end: float,
                          score_events: Sequence[float], warp_fn: Callable[[float], float],
                          feature: Mapping[str, Any], playback_measure: int) -> dict[str, Any]:
    """Compatibility wrapper for the Phase 3B single-measure diagnostic API.

    Phase 3B.1 production uses measure_candidates followed by soft path and
    neighbourhood calibration. This wrapper is not used by analyze_role().
    """
    result = measure_candidates(role, nominal_start, nominal_end, score_events,
                                warp_fn, feature, playback_measure)
    candidates = list(result.get("candidates") or [])
    if not candidates:
        return {**result, "accepted": False,
                "reason": result.get("reason") or "no_pattern_candidate"}
    best = candidates[0]
    accepted = bool(best.get("locally_eligible"))
    return {
        **result,
        "accepted": accepted,
        # Ambiguity is the primary failure when the local peak is not
        # eligible. Corridor saturation is reported only for an otherwise
        # distinctive candidate at the search boundary.
        "reason": None if accepted else (
            "ambiguous_pattern_match" if not best.get("locally_eligible")
            and float(best.get("runner_up_margin", 0.0))
                < profile_for(role).minimum_peak_margin
            else "search_corridor_saturated" if best.get("corridor_saturated")
            else "ambiguous_pattern_match"
        ),
        "residual_seconds": best.get("residual_seconds"),
        "residual_ms": best.get("residual_ms"),
        "best_score": best.get("score"),
        "second_score": round(float(best.get("score", 0.0))
                              - float(best.get("runner_up_margin", 0.0)), 6),
        "runner_up_margin": best.get("runner_up_margin"),
        "calibrated_confidence": best.get("calibrated_confidence"),
        "corridor_saturated": best.get("corridor_saturated"),
    }


def _transition_cost(previous: Mapping[str, Any] | None, current: Mapping[str, Any] | None,
                     older: Mapping[str, Any] | None) -> float:
    if current is None:
        return SKIP_COST
    emission = 1.0 - float(current["calibrated_confidence"])
    if current.get("corridor_saturated"):
        emission += 2.0
    if previous is None:
        return emission
    delta_ms = abs(float(current["residual_ms"]) - float(previous["residual_ms"]))
    cost = emission + min(2.0, delta_ms / TRANSITION_SCALE_MS)
    if older is not None:
        old_delta = float(previous["residual_ms"]) - float(older["residual_ms"])
        new_delta = float(current["residual_ms"]) - float(previous["residual_ms"])
        cost += min(1.5, abs(new_delta - old_delta) / ACCELERATION_SCALE_MS)
    return cost


def select_role_path(measures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Soft Viterbi-style selection. Skips are allowed; local drifts are penalized, not forbidden."""
    states: list[list[Mapping[str, Any] | None]] = []
    for measure in measures:
        eligible = [c for c in measure.get("candidates", []) if not c.get("corridor_saturated")]
        states.append([None] + eligible)
    if not states:
        return []
    dp = [{i: (SKIP_COST if state is None else 1 - state["calibrated_confidence"], None)
           for i, state in enumerate(states[0])}]
    for position in range(1, len(states)):
        current_costs = {}
        for ci, current in enumerate(states[position]):
            best = (math.inf, None)
            for pi, previous in enumerate(states[position - 1]):
                older = None
                back = dp[position - 1][pi][1]
                if position >= 2 and back is not None:
                    older = states[position - 2][back]
                cost = dp[position - 1][pi][0] + _transition_cost(previous, current, older)
                if cost < best[0]: best = (cost, pi)
            current_costs[ci] = best
        dp.append(current_costs)
    index = min(dp[-1], key=lambda x: dp[-1][x][0])
    path = [None] * len(states)
    for position in range(len(states) - 1, -1, -1):
        path[position] = states[position][index]
        index = dp[position][index][1] if position else 0
    return [{"playback_measure": int(measures[i]["playback_measure"]), "candidate": candidate}
            for i, candidate in enumerate(path)]


def _neighbour_supported(path: Sequence[Mapping[str, Any]], index: int) -> bool:
    current = path[index].get("candidate")
    if current is None: return False
    for other_index in (index - 1, index + 1):
        if 0 <= other_index < len(path):
            other = path[other_index].get("candidate")
            if other and abs(float(current["residual_ms"]) - float(other["residual_ms"])) <= NEIGHBOUR_TOLERANCE_MS:
                return True
    return False


def analyze_role(role: str, score_events: Iterable[Mapping[str, Any] | float], audio_path: str | None,
                 warp_fn: Callable[[float], float], nominal_downbeats: Sequence[Mapping[str, Any]],
                 sr: int = 22050) -> dict[str, Any]:
    events = _event_times(score_events)
    feature = extract_role_feature(audio_path, role, sr=sr)
    measures = []
    for i, downbeat in enumerate(nominal_downbeats):
        start = float(downbeat["t_gp"])
        end = (float(nominal_downbeats[i + 1]["t_gp"]) if i + 1 < len(nominal_downbeats)
               else start + (start - float(nominal_downbeats[i - 1]["t_gp"])) if i else start)
        measures.append(measure_candidates(role, start, end, events, warp_fn, feature,
                                           int(downbeat.get("measure", i + 1))))
    path = select_role_path(measures)
    selected = {item["playback_measure"]: item["candidate"] for item in path}
    for i, measure in enumerate(measures):
        candidate = selected.get(measure["playback_measure"])
        measure["selected_candidate"] = candidate
        measure["path_selected"] = candidate is not None
        measure["neighbour_supported"] = _neighbour_supported(path, i)
        measure["cross_role_supported"] = False
        measure["accepted"] = False
        measure["acceptance_reason"] = "pending_cross_role_calibration" if candidate else "path_skip"
    return {"version": SCHEMA_VERSION, "status": "ok" if feature.get("status") == "ok" else "unavailable",
            "role": role, "feature_family": profile_for(role).feature_family, "audio_path": str(audio_path) if audio_path else None,
            "score_event_count": len(events), "measure_count": len(measures), "accepted_measure_count": 0,
            "candidate_measure_count": sum(bool(m.get("candidates")) for m in measures), "measures": measures,
            "path": path, "timing_changes_applied": False}


def _apply_cross_role_support(roles: Mapping[str, dict[str, Any]]) -> None:
    by_measure: dict[int, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    for role, report in roles.items():
        for measure in report["measures"]:
            candidate = measure.get("selected_candidate")
            if candidate:
                by_measure.setdefault(int(measure["playback_measure"]), []).append((role, measure, candidate))
    for observations in by_measure.values():
        for role, measure, candidate in observations:
            agrees = any(other_role != role and abs(float(candidate["residual_ms"]) - float(other["residual_ms"])) <= CROSS_ROLE_TOLERANCE_MS
                         for other_role, _, other in observations)
            measure["cross_role_supported"] = agrees
            high = float(candidate["calibrated_confidence"]) >= HIGH_CONFIDENCE
            accepted = bool(candidate.get("locally_eligible") and (measure["neighbour_supported"] or agrees or high))
            measure["accepted"] = accepted
            if accepted:
                measure["acceptance_reason"] = ("cross_role_support" if agrees else "neighbour_support" if measure["neighbour_supported"] else "high_confidence")
            elif candidate.get("corridor_saturated"):
                measure["acceptance_reason"] = "search_corridor_saturated"
            else:
                measure["acceptance_reason"] = "insufficient_neighbourhood_support"
    for report in roles.values():
        report["accepted_measure_count"] = sum(bool(m["accepted"]) for m in report["measures"])


def analyze_all(role_inputs: Mapping[str, Mapping[str, Any]], warp_fn: Callable[[float], float],
                nominal_downbeats: Sequence[Mapping[str, Any]], sr: int = 22050) -> dict[str, Any]:
    roles = {role: analyze_role(role, values.get("events") or [], values.get("audio_path"), warp_fn, nominal_downbeats, sr)
             for role, values in role_inputs.items() if values}
    _apply_cross_role_support(roles)
    return {"schema": "feedpak.instrument_evidence", "schema_version": SCHEMA_VERSION,
            "phase": "3B.1", "timing_changes_applied": False,
            "selection_policy": {"edge_fraction": EDGE_FRACTION, "soft_path_penalties": True,
                                 "neighbour_tolerance_ms": NEIGHBOUR_TOLERANCE_MS,
                                 "cross_role_tolerance_ms": CROSS_ROLE_TOLERANCE_MS,
                                 "note": "Penalties are soft so genuine local drift remains representable; all candidates are retained."},
            "roles": roles,
            "summary": {"role_count": len(roles), "available_role_count": sum(r["status"] == "ok" for r in roles.values()),
                        "candidate_measure_count": sum(r["candidate_measure_count"] for r in roles.values()),
                        "accepted_measure_count": sum(r["accepted_measure_count"] for r in roles.values())}}




def _selected_candidate_for_export(measure: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Read Phase 3B.1 records and legacy Phase 3B synthetic test records."""
    selected = measure.get("selected_candidate")
    if selected is not None:
        return selected
    if measure.get("accepted") and measure.get("residual_ms") is not None:
        score = float(measure.get("best_score", measure.get("confidence", 0.0)) or 0.0)
        return {
            "residual_ms": float(measure["residual_ms"]),
            "residual_seconds": float(measure["residual_ms"]) / 1000.0,
            "score": score,
            "calibrated_confidence": score,
            "runner_up_margin": measure.get("runner_up_margin"),
            "corridor_saturated": False,
            "locally_eligible": True,
            "legacy_test_record": True,
        }
    return None


def to_shared_observations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Emit at most one calibrated instrument-family vote per playback measure.

    Raw per-role candidates and paths remain in the instrument report. The
    beatmap interpretation receives a bounded consensus observation, preventing
    a dense adapter family from numerically overwhelming structural evidence.
    """
    by_measure: dict[int, list[dict[str, Any]]] = {}
    for role, role_report in (report.get("roles") or {}).items():
        for measure in role_report.get("measures") or []:
            candidate = _selected_candidate_for_export(measure)
            if not candidate or not measure.get("accepted"):
                continue
            by_measure.setdefault(int(measure["playback_measure"]), []).append({
                "role": role, "feature_family": role_report.get("feature_family"),
                "candidate": candidate, "acceptance_reason": measure.get("acceptance_reason"),
                "neighbour_supported": measure.get("neighbour_supported"),
                "cross_role_supported": measure.get("cross_role_supported")})
    output = []
    for playback_measure, contributors in sorted(by_measure.items()):
        residuals = sorted(float(item["candidate"]["residual_ms"]) for item in contributors)
        confidences = sorted(float(item["candidate"]["calibrated_confidence"]) for item in contributors)
        middle = len(residuals) // 2
        residual = residuals[middle] if len(residuals) % 2 else (residuals[middle - 1] + residuals[middle]) / 2
        confidence = confidences[middle] if len(confidences) % 2 else (confidences[middle - 1] + confidences[middle]) / 2
        roles = sorted(item["role"] for item in contributors)
        output.append({"evidence_family": "instrument_pattern", "method": "role_optimized_calibrated_consensus",
                       "subtype": "cross_role_consensus" if len(roles) >= 2 else contributors[0]["feature_family"],
                       "playback_measure": playback_measure, "accepted": True,
                       "residual_ms": round(residual, 3), "source_roles": roles,
                       "confidence": round(confidence, 6), "warnings": [],
                       "details": {"contributor_count": len(contributors), "contributors": contributors,
                                   "aggregation": "median_residual_and_confidence",
                                   "family_vote_limit": "one_per_playback_measure"}})
    return output

