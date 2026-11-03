import numpy as np
import structural_search as ss


def downbeats(n=20):return [{"t_gp":float(i*2),"measure":i+1} for i in range(n)]

def beatmap(n=20):return {"beats":[{"structural_role":"downbeat","identity":{"beat_id":f"b{i}","playback_measure":i},"timing":{"audio_time":i*2.0}} for i in range(1,n+1)]}

def candidate(score=.8,residual=0):return {"calibrated_confidence":score,"residual_ms":residual,"rank":1,"locally_eligible":True,"corridor_saturated":False}


def test_combine_roles_accepts_independent_improvement():
    evals=[{"role":"drums","valid_measure_count":4,"window_score":.8},{"role":"bass","valid_measure_count":4,"window_score":.75}]
    bases={"drums":{"window_score":.5},"bass":{"window_score":.5}}
    result=ss._combine_roles(evals,bases)
    assert result["accepted"] is True and result["role_count"]==2


def test_single_role_does_not_create_structural_hypothesis():
    result=ss._combine_roles([{"role":"guitar","valid_measure_count":4,"window_score":.9}],{"guitar":{"window_score":.4}})
    assert result["accepted"] is False


def test_window_starts_target_long_gap():
    rows=ss._measure_rows(downbeats(24))
    starts=ss._window_starts({"start_playback_measure":5,"end_playback_measure":20},rows,4)
    assert starts and min(starts)>=5 and max(starts)<=17 and len(starts)<=4


def test_append_structural_proposal_never_recommends_it():
    report={"proposals":[{"proposal_id":"global.baseline","method":"retain_baseline","comparison_score":{"total":0}}],"notes":[]}
    structural={"summary":{"accepted_hypothesis_count":1},"hypotheses":[{
      "hypothesis_id":"h1","source_start_measure":5,"source_end_measure":8,"measure_shift":2,"audio_shift_seconds":4,
      "median_improvement_over_local_baseline":.2,"combined_score":.8,"role_evaluations":[{"valid_measure_count":4}],"roles":["bass","drums"]}]}
    out=ss.append_to_alignment_proposals(report,structural,beatmap())
    assert out["phase"]=="3C.4"
    assert out["recommendation"]=="retain_baseline"
    p=next(x for x in out["proposals"] if x["method"]=="structural_measure_shift")
    assert p["timing_changes_applied"] is False
    assert p["song_level_eligibility"]=="structural_search_required"


def test_no_long_gaps_means_no_attempts(monkeypatch):
    monkeypatch.setattr(ss.ie,"extract_role_feature",lambda *a,**k:{"status":"unavailable"})
    result=ss.analyse({},lambda x:x,downbeats(),{"coverage_gaps":[]})
    assert result["attempt_count"]==0
    assert result["hypotheses"]==[]


def test_report_is_diagnostic_and_song_independent():
    import inspect
    source=inspect.getsource(ss).lower()
    for name in ("spellbound","supremacy","all that i bleed","don't stop me now"):
        assert name not in source
    report=ss.analyse({},lambda x:x,downbeats(),{"coverage_gaps":[]})
    assert report["timing_changes_applied"] is False
