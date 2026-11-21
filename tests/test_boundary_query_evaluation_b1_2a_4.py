import occurrence_audio_qualification as oaq


def candidate(rank,start):
    return {"window_id":"w1","chart_start":10.,"chart_end":20.,"midi_start":start,"midi_end":start+10,
        "chart_stretch":1.,"translation_seconds":start-10.,"combined_score":.9,
        "phase_diagnostics":{"occurrence_rank":rank,"phase_distinctiveness":"ambiguous_best" if rank==1 else "alternative_occurrence"}}


def report():
    c1,c2=candidate(1,100.),candidate(2,130.)
    return {"candidate_report":{"roles":{"guitar":{"windows":[{"window_id":"w1","nominal_start":10.,"nominal_end":20.,"candidates":[c1,c2]}]}}},
        "path_report":{"roles":{"guitar":{"streaks":[{"selected_candidates":[c1]}]}}}}


def clock(eid,chart,audio,source='stem:a'):
    return {"event_id":eid,"chart_time":chart,"mapped_audio_time":audio,"confidence":.95,"source":source,
        "physical_source_id":source,"source_independence_group":source,"detector_module":"structural_search","nominal_role":"guitar",
        "strict_eligible":True,"clock_complete":True,"detector_identity_complete":True,
        "physical_source_identity_complete":True,"limitations":[]}


def ledger(*items):return {"events":[{"event_id":"cluster","status":"exact_cross_modal_onset","chart_time":10.,"audio_clocks":list(items)}]}


def test_selected_entry_and_exit_boundary_pair():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('entry',10.,100.03),clock('exit',20.,110.04,source='stem:b')))
    pair=out['boundary_pair_evaluations'][0]
    assert pair['boundary_pair_classification']=='selected_occurrence_boundary_pair_supported'
    assert pair['entry_independent_source_count']==1 and pair['exit_independent_source_count']==1
    assert out['summary']['discriminative_boundary_pair_count']==1


def test_alternative_entry_and_exit_boundary_pair():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('entry',10.,130.02),clock('exit',20.,140.02,source='stem:b')))
    assert out['boundary_pair_evaluations'][0]['boundary_pair_classification']=='alternative_occurrence_boundary_pair_supported'
    assert out['path_reselected'] is False and out['timing_changes_applied'] is False


def test_shared_boundary_support_stays_non_discriminative():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('a',10.,100.03),clock('b',10.,130.05,source='stem:b')))
    entry=next(x for x in out['boundary_query_evaluations'] if x['boundary']=='entry')
    assert entry['boundary_classification']=='shared_boundary_support'


def test_conflicting_entry_and_exit_are_reported():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('entry',10.,100.02),clock('exit',20.,140.03,source='stem:b')))
    assert out['boundary_pair_evaluations'][0]['boundary_pair_classification']=='conflicting_boundary_support'


def test_no_evidence_remains_insufficient():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger={"events":[]})
    assert out['boundary_pair_evaluations'][0]['boundary_pair_classification']=='insufficient_boundary_evidence'
