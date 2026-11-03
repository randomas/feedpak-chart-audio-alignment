"""v16.6E.2 calibrated diagnostic-only phase-migration qualification."""
from __future__ import annotations
from typing import Mapping, Sequence
import statistics

VERSION="v16.6E.3-final-phase-diagnostic-cleanup"


def _f(v,d=0.0):
    try:return float(v)
    except (TypeError,ValueError):return d


def _measure_at(measures,t):
    for m in measures:
        a=_f(m.get("start",m.get("start_time")));b=_f(m.get("end",m.get("end_time")))
        if a-1e-9<=t<=b+1e-9:
            duration=max(1e-6,b-a);beats=int(m.get("beat_count",m.get("numerator",4)) or 4)
            return duration,duration/max(1,beats)
    return None,None


def _root(fusion):
    incipit=fusion.get("three_clock_incipit") or {};solo=fusion.get("solo_role_authority") or {}
    valid=incipit.get("status")=="validated" or bool(solo.get("validated_root_region_ids"))
    controls=[c for r in fusion.get("regions") or [] for c in (r.get("controls") or [])]
    controls.sort(key=lambda x:_f(x.get("score_time")))
    if not valid or not controls:return False,None
    return True,_f(controls[0].get("audio_time"))-_f(controls[0].get("score_time"))


def _steps(region,measures):
    controls=region.get("controls") or [];rows=[]
    for i,(a,b) in enumerate(zip(controls,controls[1:])):
        ta=_f(a.get("score_time"));tb=_f(b.get("score_time"));oa=_f(a.get("audio_time"))-ta;ob=_f(b.get("audio_time"))-tb
        measure,beat=_measure_at(measures,tb);delta=ob-oa
        rows.append({"index":i,"chart_start":ta,"chart_end":tb,"offset_start":oa,"offset_end":ob,
          "delta_seconds":delta,"delta_beats":None if not beat else delta/beat,"beat_duration_seconds":beat,
          "roles_before":sorted(a.get("roles") or []),"roles_after":sorted(b.get("roles") or [])})
    return rows


def _direction(delta,threshold=.045):return 1 if delta>threshold else -1 if delta<-threshold else 0


def _phrase_context(fusion,start,end):
    structural=fusion.get("structural_repeat_diagnostics") or {};rows=[]
    for row in structural.get("phrase_run_correspondences") or []:
        span=row.get("chart_span") or [0,0]
        if max(start,_f(span[0]))<min(end,_f(span[1]))+1e-9:rows.append(row)
    risky={"multiple_possible_run_instances","additional_midi_occurrences","phrase_count_mismatch",
           "possible_extra_chart_or_missing_midi_phrase","probable_complete_run"}
    exact=any(x.get("classification")=="matching_phrase_run" and _f(x.get("boundary_context_score"))>=.45 for x in rows)
    ambiguous=any(x.get("classification") in risky for x in rows)
    return exact,ambiguous,[x.get("run_id") for x in rows]


def _gap_spans(fusion,application_report=None):
    phase=fusion.get("production_phase_diagnostics") or {}
    spans=[]
    for gap in phase.get("unsupported_gap_phase_accounting") or []:
        spans.append((_f(gap.get("chart_start")),_f(gap.get("chart_end")),"phase_gap"))
    app=application_report or {}
    for row in app.get("transitions") or []:
        start=row.get("chart_start",row.get("from_score_time"));end=row.get("chart_end",row.get("to_score_time"))
        if start is not None and end is not None:spans.append((_f(start),_f(end),"materialization_transition"))
    for row in app.get("rejected_regions") or []:
        start=row.get("chart_start");end=row.get("chart_end")
        if start is not None and end is not None:spans.append((_f(start),_f(end),"rejected_region"))
    applied=sorted(((_f(x.get("chart_start")),_f(x.get("chart_end"))) for x in app.get("applied_regions") or []),key=lambda x:x[0])
    for left,right in zip(applied,applied[1:]):
        if right[0]>left[1]+1e-6:spans.append((left[1],right[0],"between_applied_regions"))
    return spans


def _near_gap(fusion,start,end,application_report=None,limit=15.0):
    nearest=None;nearest_kind=None;nearest_span=None
    for a,b,kind in _gap_spans(fusion,application_report):
        distance=0.0 if max(start,a)<min(end,b)+1e-9 else min(abs(start-a),abs(start-b),abs(end-a),abs(end-b))
        if nearest is None or distance<nearest:nearest=distance;nearest_kind=kind;nearest_span=[a,b]
    return nearest is not None and nearest<=limit,nearest,nearest_kind,nearest_span


def _episodes(region,measures,root_valid,root_offset,fusion,application_report=None):
    steps=_steps(region,measures);groups=[];current=[];direction=0
    for step in steps:
        d=_direction(step["delta_seconds"])
        if not current:current=[step];direction=d
        elif d==0 or direction==0 or d==direction:
            current.append(step)
            if direction==0:direction=d
        else:groups.append(current);current=[step];direction=d
    if current:groups.append(current)
    rows=[]
    for ordinal,group in enumerate(groups,1):
        total=sum(x["delta_seconds"] for x in group);entry=group[0]["offset_start"];exit_=group[-1]["offset_end"]
        beat_values=[x["delta_beats"] for x in group if x["delta_beats"] is not None]
        total_beats=sum(beat_values) if beat_values else None;duration=group[-1]["chart_end"]-group[0]["chart_start"]
        following=steps[group[-1]["index"]+1:group[-1]["index"]+5];follow_offsets=[x["offset_end"] for x in following]
        returns=bool(follow_offsets and min(abs(x-entry) for x in follow_offsets)<abs(exit_-entry)*.45)
        persists=bool((follow_offsets and statistics.median(abs(x-exit_) for x in follow_offsets)<=max(.12,abs(exit_-entry)*.35))
                      or (not follow_offsets and len(group)>=3 and abs(exit_-entry)>=.30))
        roles=sorted({r for x in group for r in x["roles_after"]});role_count=len(roles);abs_beats=abs(total_beats or 0)
        max_step=max(abs(x["delta_seconds"]) for x in group);median_step=statistics.median(abs(x["delta_seconds"]) for x in group)
        rate=total/max(duration,1e-9);root_departure=None if root_offset is None else exit_-root_offset
        beat=group[-1].get("beat_duration_seconds");root_beats=None if not beat or root_departure is None else root_departure/beat
        nearest_beat=None if root_beats is None else round(root_beats)
        plateau_error=None if root_beats is None else abs(root_beats-nearest_beat)
        metrical_plateau=bool(nearest_beat is not None and abs(nearest_beat)>=1 and plateau_error<=.16 and persists)
        exact_phrase,ambiguous_phrase,phrase_ids=_phrase_context(fusion,group[0]["chart_start"],group[-1]["chart_end"])
        near_gap,gap_distance,gap_kind,gap_span=_near_gap(fusion,group[0]["chart_start"],group[-1]["chart_end"],application_report)
        smooth=bool(len(group)>=3 and max_step<=.22 and median_step<=.12 and abs(rate)<=.055)
        abrupt=bool(max_step>=.32 or (beat and max_step/beat>=.70))
        multi_role=role_count>=2
        reasons=[]
        if smooth:reasons.append("smooth_directional_evolution")
        if multi_role:reasons.append("multi_role_support")
        if abrupt:reasons.append("abrupt_phase_step")
        if metrical_plateau:reasons.append("persistent_metrical_plateau")
        if ambiguous_phrase:reasons.append("repeated_structure_ambiguity")
        if exact_phrase:reasons.append("exact_phrase_ordinal_support")
        if near_gap:reasons.append("near_unsupported_or_recovery_gap")
        single_role_large=bool(not multi_role and abs_beats>=.70)
        structural_contradiction=bool(metrical_plateau or ambiguous_phrase or near_gap)
        positive_error_qualifier=bool(abrupt or structural_contradiction or single_role_large)
        movement_confidence=min(1.0,.30+.09*min(len(group),5)+(.20 if multi_role else 0)+(.10 if smooth else 0))
        phase_error_confidence=.08
        phase_error_confidence+=(.30 if abrupt else 0)+(.28 if metrical_plateau else 0)+(.22 if ambiguous_phrase else 0)
        phase_error_confidence+=(.24 if near_gap else 0)+(.18 if single_role_large else 0)+(.12 if persists else 0)
        if exact_phrase and not ambiguous_phrase:phase_error_confidence-=.12
        phase_error_confidence=max(0.0,min(1.0,phase_error_confidence))
        if abs(total)<.12:classification="stable"
        elif returns:classification="correction_and_catch_up"
        elif duration<=8 and not persists:classification="temporary_correction"
        elif not root_valid and near_gap and abs_beats>=.70 and persists:classification="probable_occurrence_switch"
        elif not root_valid and abs_beats>=.70:classification="structurally_unrooted"
        elif smooth and multi_role and not ambiguous_phrase and not metrical_plateau and not abrupt:
            classification="supported_tempo_evolution"
        elif exact_phrase and smooth and not metrical_plateau and not abrupt:
            classification="supported_tempo_evolution"
        elif persists and positive_error_qualifier:
            classification="persistent_migration"
        elif duration>=20 and smooth:classification="supported_tempo_evolution"
        else:classification="provisional_phase_change"
        severity=5 if classification=="persistent_migration" and (abs_beats>=1.5 or metrical_plateau) else 4 if classification in ("persistent_migration","structurally_unrooted","probable_occurrence_switch") else 3 if classification in ("correction_and_catch_up","provisional_phase_change") else 1
        rows.append({"episode_id":f"{region.get('region_id','region')}-phase-{ordinal}","region_id":region.get("region_id"),
          "chart_start":group[0]["chart_start"],"chart_end":group[-1]["chart_end"],"duration_seconds":duration,
          "entry_offset_seconds":entry,"exit_offset_seconds":exit_,"total_movement_seconds":total,"total_movement_beats":total_beats,
          "movement_rate_seconds_per_chart_second":rate,"classification":classification,"severity_score":severity,
          "root_validated":root_valid,"root_offset_seconds":root_offset,"exit_root_departure_seconds":root_departure,
          "exit_root_departure_beats":root_beats,"metrical_plateau":metrical_plateau,"metrical_plateau_error_beats":plateau_error,
          "returns_toward_entry":returns,"persists_after_episode":persists,"supporting_roles":roles,"role_count":role_count,
          "step_count":len(group),"maximum_step_seconds":max_step,"median_step_seconds":median_step,"smooth_evolution":smooth,
          "abrupt_phase_step":abrupt,"exact_phrase_support":exact_phrase,"ambiguous_phrase_context":ambiguous_phrase,
          "overlapping_phrase_run_ids":phrase_ids,"near_gap":near_gap,"nearest_gap_distance_seconds":gap_distance,
          "nearest_gap_kind":gap_kind,"nearest_gap_span":gap_span,"movement_confidence":round(movement_confidence,3),
          "phase_error_confidence":round(phase_error_confidence,3),"positive_error_qualifier":positive_error_qualifier,
          "classification_reasons":reasons,"suggested_review_time":group[0]["chart_start"],"diagnostic_only":True})
    return rows


def _section_state(start,end,episodes,opening):
    rows=[x for x in episodes if max(start,x["chart_start"])<min(end,x["chart_end"])+1e-9]
    if opening.get("status")=="unrooted_opening" and start<=_f(opening.get("first_fused_control_chart_time"),0)<end:return "structurally_unrooted"
    priority=["probable_occurrence_switch","persistent_migration","structurally_unrooted","correction_and_catch_up",
              "temporary_correction","provisional_phase_change","supported_tempo_evolution","stable"]
    for state in priority:
        if any(x["classification"]==state for x in rows):return state
    return "not_assessed"


def build(fusion:Mapping,playback_measures:Sequence[Mapping],section_summary:Mapping|None=None,application_report:Mapping|None=None):
    root_valid,root_offset=_root(fusion);episodes=[]
    for region in fusion.get("regions") or []:episodes.extend(_episodes(region,playback_measures,root_valid,root_offset,fusion,application_report))
    counts={k:sum(x["classification"]==k for x in episodes) for k in sorted({x["classification"] for x in episodes})}
    high=[x for x in episodes if x["severity_score"]>=4];section_states=[]
    opening=(fusion.get("production_phase_diagnostics") or {}).get("opening") or {}
    absolute_root_state="rooted_and_consistent" if root_valid else "audio_root_required" if opening.get("status")=="unrooted_opening" else "unrooted"
    for section in (section_summary or {}).get("sections") or []:
        state=_section_state(_f(section.get("section_start")),_f(section.get("section_end")),episodes,opening)
        section_states.append({"section_index":section.get("section_index"),"section_start":section.get("section_start"),
          "section_end":section.get("section_end"),"evidence_support":section.get("quality"),"phase_stability":state,
          "production_status":section.get("semantic_type"),"diagnostic_only":True})
    return {"version":VERSION,"diagnostic_only":True,"selection_behavior_changed":False,"timing_changes_applied":False,
      "root_validated":root_valid,"root_offset_seconds":root_offset,"absolute_root_state":absolute_root_state,
      "episode_count":len(episodes),"classification_counts":counts,"high_risk_episode_count":len(high),"episodes":episodes,
      "section_phase_states":section_states,"suggested_review_points":list(dict.fromkeys(round(x["suggested_review_time"],6) for x in sorted(high,key=lambda y:(-y["severity_score"],y["chart_start"]))))[:8],
      "notes":["Persistent migration requires a positive error qualifier; multi-role support alone is not one.",
               "Movement confidence and phase-error confidence are reported separately.",
               "Materialization transitions and rejected regions contribute gap/restart context.",
               "All candidates, paths, controls, regions, and timing are preserved."]}
