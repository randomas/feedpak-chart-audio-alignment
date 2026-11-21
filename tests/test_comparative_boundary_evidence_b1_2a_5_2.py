import comparative_boundary_evidence as cbe


def occurrence():
    return {"targeted_boundary_queries":[
        {"boundary_query_id":"entry","window_id":"w1","role":"guitar","boundary":"entry",
         "candidate_occurrence_audio_times":[{"occurrence_rank":1,"audio_time":100.0},{"occurrence_rank":2,"audio_time":103.0}]},
        {"boundary_query_id":"exit","window_id":"w1","role":"guitar","boundary":"exit",
         "candidate_occurrence_audio_times":[{"occurrence_rank":1,"audio_time":110.0},{"occurrence_rank":2,"audio_time":113.0}]}]}


def evidence(eid,audio,confidence=.62,source='stem:a',family='structural_search',limitations=None,signal='activity'):
    return {"evidence_id":eid,"source":source,"physical_source_id":source,"detector_module":family,
        "nominal_role":"guitar","observed_signal_type":signal,"confidence":confidence,"limitations":limitations or [],
        "clocks":{"chart_time":audio,"mapped_audio_time":audio}}


def quals(*ids,strict=False):
    return {"qualifications":[{"evidence_id":x,"strict_eligible":strict,"clock_complete":True,
        "strict_ineligibility_reasons":[] if strict else ["confidence_below_strict_threshold"]} for x in ids]}


def test_non_strict_comparative_evidence_can_favor_occurrence_without_timing_authority():
    bundle={"evidence":[evidence('entry',100.05),evidence('exit',110.04,source='stem:b')]}
    out=cbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('entry','exit'))
    assert out['summary']['evidence_level_counts']['comparative']==2
    assert out['summary']['comparative_boundary_pair_count']==1
    assert out['pair_evaluations'][0]['entry_favored_occurrence_rank']==1
    assert out['pair_evaluations'][0]['path_reselection_authorized'] is False
    assert out['timing_changes_applied'] is False


def test_below_floor_is_contextual_not_comparative():
    bundle={"evidence":[evidence('weak',100.02,confidence=.30)]}
    out=cbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('weak'))
    assert out['summary']['evidence_level_counts']['contextual']==1
    assert out['summary']['comparative_boundary_pair_count']==0


def test_hard_limitation_excludes_observation():
    bundle={"evidence":[evidence('bad',100.02,limitations=['invalid_clock'])]}
    out=cbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('bad'))
    assert out['summary']['evidence_level_counts']['contextual']==1
    assert out['evidence_inventory'][0]['comparative_eligible'] is False


def test_correlated_observations_have_capped_bonus():
    bundle={"evidence":[evidence('a',100.02),evidence('b',100.03)]}
    out=cbe.build_report(occurrence_report=occurrence(),audio_evidence_bundle=bundle,
        audio_evidence_qualification=quals('a','b'))
    entry=out['boundary_evaluations'][0]
    best=entry['candidate_scores'][0]
    assert best['independent_group_count']==1
    assert best['raw_evidence_count']==2
    assert best['comparative_score']<=.62+.15+1e-9


def test_shared_candidate_support_is_triaged_for_detector_execution():
    occ={"targeted_boundary_queries":[{"boundary_query_id":"q","window_id":"w","role":"drums","boundary":"entry",
        "candidate_occurrence_audio_times":[{"occurrence_rank":1,"audio_time":100.0},{"occurrence_rank":2,"audio_time":100.1}]}]}
    bundle={"evidence":[evidence('a',100.05,signal='structural')]}
    out=cbe.build_report(occurrence_report=occ,audio_evidence_bundle=bundle,audio_evidence_qualification=quals('a'))
    assert out['boundary_evaluations'][0]['comparative_classification']=='comparative_support_shared'
    assert out['boundary_evaluations'][0]['detector_execution_recommended'] is True
