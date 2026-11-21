import occurrence_audio_qualification as oaq


def cand(wid,rank,midi_start,chart_start=10.,chart_end=20.):
    return {"window_id":wid,"chart_start":chart_start,"chart_end":chart_end,"midi_start":midi_start,"midi_end":midi_start+10,
        "chart_stretch":1.,"translation_seconds":midi_start-chart_start,"combined_score":.9,
        "phase_diagnostics":{"occurrence_rank":rank,"phase_distinctiveness":"ambiguous_best" if rank==1 else "alternative_occurrence"}}


def report(two_windows=False):
    c1,c2=cand('w1',1,100.),cand('w1',2,130.)
    windows=[{"window_id":"w1","nominal_start":10.,"nominal_end":20.,"candidates":[c1,c2]}]
    selected=[c1]
    if two_windows:
        d1,d2=cand('w2',1,108.,18.,28.),cand('w2',2,138.,18.,28.)
        windows.append({"window_id":"w2","nominal_start":18.,"nominal_end":28.,"candidates":[d1,d2]});selected.append(d1)
    return {"candidate_report":{"roles":{"guitar":{"windows":windows}}},"path_report":{"roles":{"guitar":{"streaks":[{"selected_candidates":selected}]}}}}


def clock(eid,audio,source='stem:a',family='continuous_anchor',chart=15.,role='guitar'):
    return {"event_id":eid,"chart_time":chart,"mapped_audio_time":audio,"confidence":.95,"source":source,
        "physical_source_id":source,"source_independence_group":source,"detector_module":family,"nominal_role":role,
        "strict_eligible":True,"clock_complete":True,"detector_identity_complete":True,
        "physical_source_identity_complete":True,"limitations":[]}


def ledger(*clocks):return {"events":[{"event_id":"cluster","status":"exact_cross_modal_onset","chart_time":15.,"audio_clocks":list(clocks)}]}


def test_correlated_events_collapse_by_physical_source_and_detector_family():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('a1',105.02),clock('a2',105.04)))
    selected=out['qualifications'][0]['candidate_occurrences'][0]
    assert selected['raw_strict_observation_count']==2
    assert selected['strict_support_count']==1
    assert selected['independent_physical_source_count']==1
    assert selected['independent_detector_family_count']==1


def test_distinct_source_or_family_counts_independently():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('a1',105.02),clock('a2',105.04,source='stem:b'),clock('a3',105.03,family='structural_search')))
    selected=out['qualifications'][0]['candidate_occurrences'][0]
    assert selected['raw_strict_observation_count']==3
    assert selected['strict_support_count']==3
    assert selected['independent_physical_source_count']==2
    assert selected['independent_detector_family_count']==2


def test_same_and_cross_role_are_separate():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(
        clock('a1',105.02,role='guitar'),clock('a2',105.03,source='stem:b',role='drums')))
    selected=out['qualifications'][0]['candidate_occurrences'][0]
    assert selected['same_role_support_count']==1
    assert selected['cross_role_support_count']==1


def test_unresolved_window_emits_entry_and_exit_queries():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger={"events":[]})
    assert len(out['targeted_boundary_queries'])==2
    assert {x['boundary'] for x in out['targeted_boundary_queries']}=={'entry','exit'}
    assert all(x['detectors_executed'] is False for x in out['targeted_boundary_queries'])


def test_neighbor_supported_windows_consolidate():
    clocks=[clock('a1',105.02,chart=15.),clock('a2',113.02,chart=23.,source='stem:b')]
    out=oaq.build_report(midi_section_fit_report=report(True),evidence_contribution_ledger=ledger(*clocks))
    assert out['summary']['neighbor_consistent_region_count']==1
    assert out['regional_consolidation'][0]['regional_support_class']=='neighbor_consistent_support'
    assert out['path_reselected'] is False and out['timing_changes_applied'] is False
