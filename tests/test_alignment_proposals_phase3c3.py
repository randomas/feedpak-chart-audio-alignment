import alignment_proposals as ap


def beatmap(n=16,step=2.0):
    return {"beats":[{"structural_role":"downbeat","identity":{"beat_id":f"b{i}","playback_measure":i},
                      "timing":{"audio_time":i*step}} for i in range(1,n+1)]}


def region(residuals,start=3):
    return {"region_id":"seg000.r000","start_playback_measure":start,"end_playback_measure":start+len(residuals)-1,
            "samples":[{"audio_time":2*(start+i),"playback_measure":start+i,"residual_ms":r,
                        "evidence_family":"instrument_pattern","source_roles":["drums"]} for i,r in enumerate(residuals)]}


def fusion(measures):
    return {"summary":{"playback_measure_count":16,"accepted_hypothesis_count":len(measures),"long_gap_count":0},
            "hypotheses":[{"playback_measure":m,"accepted_hypothesis":True,"residual_ms":r,
                           "roles":["drums","bass"],"fusion_score":.85} for m,r in measures]}


def test_regional_proposal_maps_complete_timeline():
    p=ap.offset_proposal(beatmap(),region([100]*8))
    assert len(p["mapped_downbeats"])==16
    assert p["mapped_downbeats"][0]["correction_ms"]==0
    assert p["mapped_downbeats"][-1]["correction_ms"]==0


def test_taper_exposes_and_reduces_boundary_transition():
    p=ap.offset_proposal(beatmap(),region([120]*8))
    corrections=[x["correction_ms"] for x in p["mapped_downbeats"]]
    assert max(corrections)==120
    assert 0<corrections[2]<120
    v=ap.validate_proposal(p)
    assert v["boundary_intervals"]
    assert v["maximum_adjacent_correction_jump_ms"]<120


def test_dispersion_worsening_is_penalized():
    proposal={"evidence_fit":{"median_absolute_improvement_ms":30,"baseline_residual_mad_ms":10,"proposal_residual_mad_ms":40},
              "validation":{"warning_count":0,"error_count":0,"maximum_adjacent_correction_jump_ms":0},
              "complexity":{"control_count":0}}
    score=ap._proposal_score(proposal)
    assert score["dispersion_penalty"]==30
    assert score["total"]==0


def test_piecewise_proposal_uses_fused_controls_and_returns_to_baseline():
    r=region([0,20,40,60,80,60,40,20],3)
    f=fusion([(3,0),(5,50),(7,90),(9,30)])
    p=ap.piecewise_proposal(beatmap(),r,f)
    assert p is not None
    assert p["method"]=="source_fused_piecewise_linear"
    assert len(p["controls"])>=3
    assert p["mapped_downbeats"][0]["correction_ms"]==0
    assert p["mapped_downbeats"][-1]["correction_ms"]==0


def test_piecewise_requires_three_controls():
    assert ap.piecewise_proposal(beatmap(),region([0,20,40,60]),fusion([(3,0),(5,50)])) is None


def test_large_adjacent_jump_is_error():
    p=ap.baseline_proposal(beatmap(4))
    p["method"]="synthetic"
    p["mapped_downbeats"][1]["correction_ms"]=250
    p["mapped_downbeats"][1]["proposed_audio_time"]+=.25
    v=ap.validate_proposal(p)
    assert v["status"]=="error"
    assert any(e["kind"]=="abrupt_correction_jump" for e in v["errors"])


def test_sparse_song_marks_local_improvement_only():
    r=region([100]*8)
    report=ap.generate(beatmap(),{"regions":[r]},
                       {"summary":{"playback_measure_count":100,"accepted_hypothesis_count":5,"long_gap_count":4},"hypotheses":[]})
    alt=next(p for p in report["proposals"] if p["method"]!="retain_baseline")
    assert alt["song_level_eligibility"]=="partial_local_improvement_only"
    assert report["recommendation"]=="retain_baseline"


def test_control_song_can_carry_safe_piecewise_candidate():
    r=region([0,20,40,60,80,60,40,20],3)
    f=fusion([(3,0),(5,50),(7,90),(9,30)])
    report=ap.generate(beatmap(),{"regions":[r]},f)
    assert report["phase"]=="3C.3"
    assert any(p["method"]=="source_fused_piecewise_linear" for p in report["proposals"])
    assert report["timing_changes_applied"] is False
