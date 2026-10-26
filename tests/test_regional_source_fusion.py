import regional_source_fusion as rsf


def cand(residual, confidence=.75, rank=1, selected=False, accepted=False):
    c={"residual_ms":residual,"calibrated_confidence":confidence,"rank":rank,
       "corridor_saturated":False,"locally_eligible":True}
    m={"playback_measure":1,"candidates":[c],"selected_candidate":c if selected else None,
       "accepted":accepted,"neighbour_supported":selected}
    return c,m


def beatmap(n=8):
    return {"beats":[{"structural_role":"downbeat","identity":{"playback_measure":i}} for i in range(1,n+1)]}


def test_two_independent_roles_rescue_unaccepted_candidates():
    _,dm=cand(30,.72,selected=True,accepted=False)
    _,bm=cand(55,.70,selected=True,accepted=False)
    report={"roles":{"drums":{"candidate_measure_count":1,"measures":[dm]},
                     "bass":{"candidate_measure_count":1,"measures":[bm]}}}
    result=rsf.analyse(report,beatmap(1))
    h=result["hypotheses"][0]
    assert h["accepted_hypothesis"] is True
    assert h["acceptance_route"]=="multi_role_consensus"
    assert result["summary"]["rescued_by_fusion_count"]==1


def test_disagreeing_roles_do_not_create_false_consensus():
    _,dm=cand(-180,.8,selected=True,accepted=True)
    _,bm=cand(160,.8,selected=True,accepted=True)
    report={"roles":{"drums":{"candidate_measure_count":1,"measures":[dm]},
                     "bass":{"candidate_measure_count":1,"measures":[bm]}}}
    result=rsf.analyse(report,beatmap(1))
    assert result["hypotheses"][0]["accepted_hypothesis"] is False


def test_alternative_candidates_are_retained():
    c1,m=cand(20,.8,selected=True,accepted=True)
    c2={"residual_ms":190,"calibrated_confidence":.7,"rank":2,"corridor_saturated":False,"locally_eligible":True}
    m["candidates"].append(c2)
    _,b=cand(40,.8,selected=True,accepted=True)
    report={"roles":{"drums":{"candidate_measure_count":1,"measures":[m]},
                     "bass":{"candidate_measure_count":1,"measures":[b]}}}
    result=rsf.analyse(report,beatmap(1))
    assert result["hypotheses"][0]["alternative_hypotheses"]


def test_long_gaps_remain_visible():
    result=rsf.analyse({"roles":{}},beatmap(12))
    assert result["summary"]["accepted_hypothesis_count"]==0
    assert result["coverage_gaps"][0]["long_gap"] is True
    assert result["coverage_gaps"][0]["recommended_action"]=="wider_structural_search_or_manual_anchor"


def test_single_role_needs_exceptional_confidence_and_neighbours():
    roles={"piano":{"candidate_measure_count":3,"measures":[]}}
    for i,residual in enumerate([10,20,25],1):
        c={"residual_ms":residual,"calibrated_confidence":.95,"rank":1,"corridor_saturated":False,"locally_eligible":True}
        roles["piano"]["measures"].append({"playback_measure":i,"candidates":[c],"selected_candidate":c,"accepted":True,"neighbour_supported":True})
    result=rsf.analyse({"roles":roles},beatmap(3))
    assert sum(h["accepted_hypothesis"] for h in result["hypotheses"])>=1


def test_diagnostic_only_and_song_independent():
    import inspect
    result=rsf.analyse({"roles":{}},beatmap(1))
    assert result["timing_changes_applied"] is False
    source=inspect.getsource(rsf).lower()
    for name in ("spellbound","supremacy","all that i bleed","don't stop me now"):
        assert name not in source
