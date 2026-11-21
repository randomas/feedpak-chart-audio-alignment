import targeted_boundary_evidence as tbe


def occurrence():
    return {"targeted_boundary_queries":[
        {"boundary_query_id":"q-entry","window_id":"w1","role":"guitar","boundary":"entry",
         "candidate_occurrence_audio_times":[{"occurrence_rank":1,"audio_time":100.0},{"occurrence_rank":2,"audio_time":130.0}]},
        {"boundary_query_id":"q-exit","window_id":"w1","role":"guitar","boundary":"exit",
         "candidate_occurrence_audio_times":[{"occurrence_rank":1,"audio_time":110.0},{"occurrence_rank":2,"audio_time":140.0}]}]}


def evidence(eid,audio,source='stem:a',family='structural_search',role='guitar'):
    return {"evidence_id":eid,"source":source,"physical_source_id":source,"detector_module":family,
        "nominal_role":role,"confidence":.95,"limitations":[],
        "clocks":{"chart_time":audio,"mapped_audio_time":audio}}


def quals(*ids):
    return {"qualifications":[{"evidence_id":x,"strict_eligible":True} for x in ids]}


def test_request_planner_creates_candidate_specific_entry_and_exit_requests():
    rows=tbe.plan_requests(occurrence())
    assert len(rows)==4
    assert {x['boundary'] for x in rows}=={'entry','exit'}
    assert {x['occurrence_rank'] for x in rows}=={1,2}
    assert all(len(x['cache_key'])==64 for x in rows)


def test_targeted_scan_finds_selected_boundary_pair():
    bundle={"evidence":[evidence('entry',100.02),evidence('exit',110.03,source='stem:b')]}
    out=tbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('entry','exit'))
    pair=next(x for x in out['pair_evaluations'] if x['occurrence_rank']==1)
    assert pair['targeted_pair_classification']=='targeted_boundary_pair_supported'
    assert out['summary']['targeted_boundary_pair_count']==1
    assert out['timing_changes_applied'] is False and out['path_reselected'] is False


def test_correlated_evidence_is_deduplicated():
    bundle={"evidence":[evidence('a',100.02),evidence('b',100.03)]}
    out=tbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('a','b'))
    row=next(x for x in out['evaluations'] if x['occurrence_rank']==1 and x['boundary']=='entry')
    assert row['raw_strict_observation_count']==2
    assert row['independent_support_count']==1


def test_context_only_evidence_does_not_form_pair():
    bundle={"evidence":[evidence('a',103.0)]}
    out=tbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('a'))
    row=next(x for x in out['evaluations'] if x['occurrence_rank']==1 and x['boundary']=='entry')
    assert row['acquisition_status']=='targeted_evidence_still_insufficient'


def test_overlapping_requests_merge_without_losing_links():
    occ={"targeted_boundary_queries":[{"boundary_query_id":"q","window_id":"w","role":"drums","boundary":"entry",
        "candidate_occurrence_audio_times":[{"occurrence_rank":1,"audio_time":10.},{"occurrence_rank":2,"audio_time":14.}]}]}
    requests=tbe.plan_requests(occ);merged=tbe.merge_requests(requests)
    assert len(merged)==1 and len(merged[0]['request_ids'])==2
