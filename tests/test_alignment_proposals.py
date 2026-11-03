import alignment_proposals as ap


def beatmap(n=12,step=2.0):
    return {"beats":[{"structural_role":"downbeat","identity":{"beat_id":f"b{i}","playback_measure":i},
                      "timing":{"audio_time":i*step}} for i in range(1,n+1)]}


def region(residuals,start=1):
    return {"region_id":"seg000.r000","start_playback_measure":start,
            "end_playback_measure":start+len(residuals)-1,
            "samples":[{"audio_time":2*(start+i),"residual_ms":r,
                        "evidence_family":"rhythmic_onset","source_roles":["drums"]}
                       for i,r in enumerate(residuals)]}


def test_baseline_is_always_present_and_identical():
    report=ap.generate(beatmap(),{"regions":[]},{})
    assert report["baseline_proposal_id"]=="global.baseline"
    p=report["proposals"][0]
    assert p["method"]=="retain_baseline"
    assert all(x["baseline_audio_time"]==x["proposed_audio_time"] for x in p["mapped_downbeats"])


def test_constant_offset_proposal_is_generated():
    r=region([100,110,90,105,95,100])
    p=ap.offset_proposal(beatmap(),r)
    assert p is not None
    assert p["method"]=="regional_constant_offset"
    assert 90 <= p["controls"][0]["offset_ms"] <= 110


def test_linear_scale_proposal_recovers_trend():
    r=region([0,10,20,30,40,50,60,70])
    p=ap.scale_proposal(beatmap(),r)
    assert p is not None
    assert p["diagnostic_residual_slope_ms_per_second"]>0
    assert p["implied_scale"]>1


def test_validation_rejects_non_monotonic_mapping():
    p={"mapped_downbeats":[{"playback_measure":1,"baseline_audio_time":1,"proposed_audio_time":2},
                           {"playback_measure":2,"baseline_audio_time":2,"proposed_audio_time":1}]}
    result=ap.validate_proposal(p)
    assert result["status"]=="error"
    assert result["errors"][0]["kind"]=="non_monotonic_proposal"


def test_safe_stretch_is_reported():
    r=region([0,10,20,30,40,50,60,70])
    p=ap.scale_proposal(beatmap(),r)
    result=ap.validate_proposal(p)
    assert result["safe_stretch_range"]==[0.9,1.1]
    assert result["minimum_local_stretch"]>0


def test_insufficient_data_does_not_generate_alternative():
    r=region([100,110,90])
    assert ap.offset_proposal(beatmap(),r) is None
    assert ap.scale_proposal(beatmap(),r) is None


def test_diagnostic_only_and_song_independent():
    import inspect
    report=ap.generate(beatmap(),{"regions":[region([100,100,100,100])]},None)
    assert report["timing_changes_applied"] is False
    assert all(p["timing_changes_applied"] is False for p in report["proposals"])
    source=inspect.getsource(ap).lower()
    for name in ("spellbound","supremacy","all that i bleed","don't stop me now"):
        assert name not in source
