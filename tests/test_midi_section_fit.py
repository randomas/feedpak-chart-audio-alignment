import math
import midi_section_fit as msf


def measures(count=12, seconds=2.5):
    return [{"playback_index":i,"playback_measure_number":i+1,"written_measure_number":i+1,
             "start_time":i*seconds,"end_time":(i+1)*seconds} for i in range(count)]


def melody(start, count, step=.6, pitches=(60,62,64,65,67)):
    return [{"time":start+i*step,"midi":pitches[i%len(pitches)]} for i in range(count)]


def test_measure_windows_are_aligned_duration_aware_and_overlapping():
    events={"piano":melody(0,50)}
    windows=msf.build_measure_aligned_windows(measures(),events)["piano"]
    assert windows[0].playback_measure_numbers==[1,2,3,4]
    assert windows[1].playback_measure_numbers==[4,5,6,7]
    assert windows[0].nominal_start==0
    assert windows[0].nominal_end==10
    assert windows[0].observability=="observable"


def test_variable_measure_duration_adjusts_count():
    rows=measures(10,1.0)
    windows=msf.build_measure_aligned_windows(rows,{"guitar":melody(0,30,.3)})["guitar"]
    assert len(windows[0].playback_measure_numbers) in (5,6)


def test_unobservable_window_is_neutral():
    windows=msf.build_measure_aligned_windows(measures(4),{"bass":[]})["bass"]
    assert windows[0].observability=="unobservable"
    assert msf.search_section_candidates(windows[0],[])==[]


def test_asymmetric_matching_tolerates_extra_midi_notes():
    chart=melody(0,16,.5)
    midi=[]
    for e in chart:
        midi.append({"time":1.2+.98*e["time"],"midi":e["midi"]})
        midi.append({"time":1.2+.98*e["time"]+.05,"midi":72})
    window=msf.build_measure_aligned_windows(measures(4,2.0),{"piano":chart})["piano"][0]
    cfg=msf.SectionFitConfig(search_radius_seconds=3,translation_step_seconds=.1,stretch_step=.01)
    candidates=msf.search_section_candidates(window,midi,cfg=cfg)
    assert candidates
    best=candidates[0]
    assert best.chart_coverage>=.8
    assert .9<=best.chart_stretch<=1.1


def test_search_retains_distinct_repeated_occurrences():
    chart=melody(0,14,.5)
    phrase=[{"time":4+.97*e["time"],"midi":e["midi"]} for e in chart]
    repeated=phrase+[{"time":x["time"]+5.0,"midi":x["midi"]} for x in phrase]
    window=msf.build_measure_aligned_windows(measures(4,2.0),{"guitar":chart})["guitar"][0]
    cfg=msf.SectionFitConfig(search_radius_seconds=10,candidate_limit=4)
    candidates=msf.search_section_candidates(window,repeated,cfg=cfg)
    assert len(candidates)>=2
    assert abs(candidates[0].midi_start-candidates[1].midi_start)>=cfg.distinct_occurrence_seconds


def test_report_is_diagnostic_only():
    report=msf.build_section_candidate_report(measures(4,2.0),{"piano":melody(0,14,.5)},
        {"piano":melody(1,14,.5)})
    assert report["diagnostic_only"] is True
    assert report["timing_changes_applied"] is False
    assert report["monotonic_path_selected"] is False


def candidate(window_id, chart_start, chart_end, midi_start, midi_end, score=1.0,
              stretch=None, role="piano"):
    stretch = stretch if stretch is not None else (midi_end-midi_start)/(chart_end-chart_start)
    translation = midi_start-stretch*chart_start
    return {
        "window_id":window_id,"role":role,"chart_start":chart_start,"chart_end":chart_end,
        "midi_start":midi_start,"midi_end":midi_end,"chart_stretch":stretch,
        "translation_seconds":translation,"matched_chart_groups":10,"chart_group_count":10,
        "midi_group_count":12,"chart_coverage":1.0,"median_residual_seconds":.02,
        "maximum_residual_seconds":.05,"sequence_score":score,"context_score":score,
        "combined_score":score,"matched_landmarks":[],"status":"valid","rejection_reasons":[]}


def path_report(rows):
    return {"roles":{"piano":{"windows":rows}}}


def window_row(index, candidates, observable=True):
    return {"window":{"window_id":f"piano-section-{index:04d}",
                      "observability":"observable" if observable else "unobservable"},
            "candidates":candidates}


def test_backward_midi_transition_is_hard_rejected():
    a=msf._candidate_from_dict(candidate("a",0,10,2,12))
    b=msf._candidate_from_dict(candidate("b",7.5,17.5,1,11))
    result=msf.evaluate_candidate_transition(a,b)
    assert result.valid is False
    assert result.reason=="midi_window_start_did_not_advance"


def test_crossing_or_disagreeing_overlap_maps_are_rejected():
    a=msf._candidate_from_dict(candidate("a",0,10,0,10))
    b=msf._candidate_from_dict(candidate("b",7.5,17.5,9,19))
    result=msf.evaluate_candidate_transition(a,b)
    assert result.valid is False
    assert result.reason in {"overlap_mapping_disagreement","crossing_or_reversing_overlap_maps"}


def test_global_selector_prefers_coherent_repeated_occurrence():
    rows=[
      window_row(0,[candidate("w0",0,10,1,11,.8),candidate("w0",0,10,21,31,.95)]),
      window_row(1,[candidate("w1",7.5,17.5,8.5,18.5,.8),candidate("w1",7.5,17.5,28.5,38.5,.95)]),
      window_row(2,[candidate("w2",15,25,16,26,.8),candidate("w2",15,25,36,46,.95)]),
    ]
    report=msf.select_monotonic_section_paths(path_report(rows))
    selected=report["roles"]["piano"]["streaks"][0]["candidates"]
    assert [round(x["midi_start"],1) for x in selected]==[21,28.5,36]
    assert report["roles"]["piano"]["monotonicity_validated"] is True


def test_unsupported_window_splits_authority_into_streaks():
    rows=[
      window_row(0,[candidate("w0",0,10,1,11)]),
      window_row(1,[],observable=False),
      window_row(2,[candidate("w2",15,25,16,26)]),
    ]
    report=msf.select_monotonic_section_paths(path_report(rows))
    role=report["roles"]["piano"]
    assert role["streak_count"]==2
    assert len(role["unsupported_windows"])==1
    assert role["unsupported_windows"][0]["reason"]=="unobservable"


def test_incompatible_adjacent_layers_split_instead_of_forcing_path():
    rows=[
      window_row(0,[candidate("w0",0,10,1,11)]),
      window_row(1,[candidate("w1",7.5,17.5,30,40)]),
      window_row(2,[candidate("w2",15,25,37.5,47.5)]),
    ]
    report=msf.select_monotonic_section_paths(path_report(rows))
    role=report["roles"]["piano"]
    assert role["streak_count"]==2
    assert all(x["monotonic"] for x in role["streaks"])


def test_step2_remains_diagnostic_only():
    rows=[window_row(0,[candidate("w0",0,10,1,11)])]
    report=msf.select_monotonic_section_paths(path_report(rows))
    assert report["diagnostic_only"] is True
    assert report["timing_changes_applied"] is False
    assert report["global_fusion_applied"] is False


def multi_role_path_report(role_candidates):
    roles={}
    for role,candidates in role_candidates.items():
        roles[role]={"streaks":[{"streak_id":f"{role}-s1","candidates":candidates}],
                     "status":"available","monotonicity_validated":True}
    return {"roles":roles}


def test_compatible_roles_are_corroborated_not_blindly_averaged():
    piano=[candidate("p0",0,10,1.00,11.00,role="piano"),
           candidate("p1",7.5,17.5,8.50,18.50,role="piano")]
    guitar=[candidate("g0",0,10,1.05,11.05,role="guitar"),
            candidate("g1",7.5,17.5,8.55,18.55,role="guitar")]
    report=msf.fuse_monotonic_role_paths(multi_role_path_report({"piano":piano,"guitar":guitar}))
    assert report["summary"]["region_count"]==1
    controls=report["regions"][0]["controls"]
    assert any(x["role_count"]==2 for x in controls)
    assert all(x["classification"].startswith("multi_role") for x in controls)
    assert report["monotonicity_certificate"]["unsupported_gaps_interpolated"] is False


def test_incompatible_roles_remain_conflicts():
    piano=[candidate("p0",0,10,1,11,role="piano")]
    guitar=[candidate("g0",0,10,4,14,role="guitar")]
    report=msf.fuse_monotonic_role_paths(multi_role_path_report({"piano":piano,"guitar":guitar}))
    assert report["summary"]["role_conflict_count"]>0
    for control in report["regions"][0]["controls"]:
        assert control["role_count"]==1


def test_fused_path_rejects_backward_or_unsafe_edge():
    piano=[candidate("p0",0,10,1,11,role="piano"),
           candidate("p1",7.5,17.5,30,40,role="piano")]
    report=msf.fuse_monotonic_role_paths(multi_role_path_report({"piano":piano}))
    assert report["summary"]["rejected_block_count"]>=1 or report["summary"]["region_count"]==0


def test_large_gap_creates_independent_regions():
    piano=[candidate("p0",0,10,1,11,role="piano"),
           candidate("p1",40,50,41,51,role="piano")]
    cfg=msf.SectionFusionConfig(maximum_supported_control_gap_seconds=15)
    report=msf.fuse_monotonic_role_paths(multi_role_path_report({"piano":piano}),cfg)
    assert report["summary"]["region_count"]==2
    assert report["monotonicity_certificate"]["regions_independent_across_gaps"] is True


def test_correction_and_catch_up_is_reported():
    cfg=msf.SectionFusionConfig(catch_up_threshold=.06)
    events=msf._catch_up_events([.92,1.08],cfg)
    assert events
    assert events[0]["kind"]=="correction_and_catch_up"


def test_step3_still_never_materializes_timing():
    piano=[candidate("p0",0,10,1,11,role="piano")]
    report=msf.fuse_monotonic_role_paths(multi_role_path_report({"piano":piano}))
    assert report["diagnostic_only"] is True
    assert report["timing_changes_applied"] is False
