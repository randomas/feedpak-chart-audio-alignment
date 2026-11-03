import numpy as np
import instrument_evidence as ie


def candidate(residual, confidence=.8, eligible=True, edge=False):
    return {"residual_ms":float(residual),"residual_seconds":float(residual)/1000,
            "calibrated_confidence":confidence,"locally_eligible":eligible,
            "corridor_saturated":edge,"score":.7,"runner_up_margin":.12}


def test_corridor_edge_candidate_is_flagged_and_not_path_eligible():
    events=[.4,1.2,2.2]
    times=np.arange(0,4.01,.01)
    novelty=sum(np.exp(-.5*((times-(x+.31))/.025)**2) for x in events)
    result=ie.measure_candidates("bass",0,4,events,lambda x:x,{"status":"ok","times":times,"novelty":novelty},1)
    assert any(c["corridor_saturated"] for c in result["candidates"])
    assert all(not c["locally_eligible"] for c in result["candidates"] if c["corridor_saturated"])


def test_soft_path_can_follow_gradual_local_drift():
    measures=[]
    for measure,residual in enumerate([0,25,50,75,100,125],1):
        measures.append({"playback_measure":measure,"candidates":[candidate(residual,.92)]})
    path=ie.select_role_path(measures)
    selected=[p["candidate"]["residual_ms"] for p in path if p["candidate"]]
    assert len(selected)>=4
    assert selected==sorted(selected)
    assert selected[-1]-selected[0]>=75


def test_path_skips_isolated_peak_hop():
    measures=[{"playback_measure":1,"candidates":[candidate(0,.9)]},
              {"playback_measure":2,"candidates":[candidate(20,.9)]},
              {"playback_measure":3,"candidates":[candidate(330,.65)]},
              {"playback_measure":4,"candidates":[candidate(35,.9)]}]
    path=ie.select_role_path(measures)
    assert path[2]["candidate"] is None or abs(path[2]["candidate"]["residual_ms"]-330)>1


def test_cross_role_or_neighbour_support_required():
    roles={
      "drums":{"measures":[{"playback_measure":1,"selected_candidate":candidate(30,.7),"neighbour_supported":False}],"accepted_measure_count":0},
      "bass":{"measures":[{"playback_measure":1,"selected_candidate":candidate(55,.7),"neighbour_supported":False}],"accepted_measure_count":0}}
    ie._apply_cross_role_support(roles)
    assert roles["drums"]["measures"][0]["accepted"] is True
    assert roles["bass"]["measures"][0]["accepted"] is True
    assert roles["drums"]["measures"][0]["acceptance_reason"]=="cross_role_support"


def test_all_candidates_remain_in_report_even_when_rejected():
    measure={"playback_measure":1,"source_role":"guitar","feature_family":"rhythmic_spectral_flux_pattern",
             "candidates":[candidate(320,.5,False,True)],"selected_candidate":None,"accepted":False,
             "acceptance_reason":"path_skip"}
    report={"roles":{"guitar":{"feature_family":"rhythmic_spectral_flux_pattern","measures":[measure]}}}
    assert len(report["roles"]["guitar"]["measures"][0]["candidates"])==1
    assert ie.to_shared_observations(report)==[]


def test_report_note_explicitly_preserves_local_drift(monkeypatch):
    monkeypatch.setattr(ie,"analyze_role",lambda role,*a,**k:{"status":"unavailable","role":role,"feature_family":"x","measures":[],"candidate_measure_count":0,"accepted_measure_count":0})
    report=ie.analyze_all({"drums":{"events":[],"audio_path":None}},lambda x:x,[])
    assert report["timing_changes_applied"] is False
    assert "local drift" in report["selection_policy"]["note"]

def test_dense_family_is_bounded_to_one_vote_per_measure():
    report={"roles":{
      "drums":{"feature_family":"a","measures":[{"playback_measure":4,"accepted":True,"selected_candidate":candidate(20,.8),"acceptance_reason":"x"}]},
      "bass":{"feature_family":"b","measures":[{"playback_measure":4,"accepted":True,"selected_candidate":candidate(40,.7),"acceptance_reason":"x"}]}}}
    observations=ie.to_shared_observations(report)
    assert len(observations)==1
    assert observations[0]["source_roles"]==["bass","drums"]
    assert observations[0]["residual_ms"]==30.0
    assert observations[0]["details"]["family_vote_limit"]=="one_per_playback_measure"
