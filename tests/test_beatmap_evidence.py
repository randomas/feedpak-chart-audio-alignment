import beatmap_evidence as be


def _beatmap():
    beats=[]
    for measure in range(1,10):
        beats.append({
            "identity":{"beat_id":f"seg000.m{measure:04d}.o01.b00","playback_segment":0,
                        "playback_measure":measure,"source_measure":measure,"source_occurrence":1,"beat_index":0},
            "timing":{"score_time":measure*4.0,"audio_time":measure*4.0,
                      "nominal_bpm":60.0,"mapped_bpm":60.0,"local_stretch":1.0},
            "meter":[4,4],"structural_role":"downbeat","provenance":{"flags":[]}})
    return {"schema":"feedpak.beatmap","schema_version":1,"metadata":{},"summary":{},
            "validation":{"status":"ok"},"beats":beats}


def _report():
    return {"decision":{
        "selected_source":["drums","bass"],
        "checkpoint_diagnostics":{"source_roles":["drums"],"checkpoints":[
            {"measure":2,"accepted":True,"median_correction_seconds":0.01,"matched_events":8,"mad_seconds":0.01,"source_roles":["drums"]},
            {"measure":5,"accepted":True,"median_correction_seconds":0.10,"matched_events":7,"mad_seconds":0.02},
            {"measure":6,"accepted":True,"median_correction_seconds":0.13,"matched_events":7,"mad_seconds":0.02},
            {"measure":7,"accepted":True,"median_correction_seconds":0.16,"matched_events":7,"mad_seconds":0.02},
            {"measure":8,"accepted":True,"median_correction_seconds":0.19,"matched_events":7,"mad_seconds":0.02}]},
        "silence_checkpoint_diagnostics":{"checkpoints":[
            {"measure":5,"accepted":True,"boundary_recognizable":True,
             "supporting_stems":["guitar","drums"],"audio_confirmed_stems":["drums","guitar"],
             "boundary_entering_stems_120ms":["drums","guitar"],
             "audio_confirmation":{"drums":{"confidence":0.9},"guitar":{"confidence":0.8}}}]},
        "chroma_checkpoint_diagnostics":{"checkpoints":[
            {"measure":6,"accepted":False,"sources":{"guitar":{"available":True,"best_score":0.7}},
             "strong_isolated_sources":[],"warnings":["low_margin"]}]},
        "continuous_anchor_path_preview":{"selected_anchors":[
            {"score_positions":[{"playback_bar":5}],"sources":["drums","guitar"],
             "residual_seconds":0.1,"confidence":0.93,"audio_time":20.1,"score_time":20.0,
             "observations":[{"source":"drums"},{"source":"guitar"}]}]},
        "manual_anchor_quality":{"anchors":[]}}}


def test_annotation_is_diagnostic_and_instrument_aware():
    original=_beatmap(); annotated,report=be.annotate_beatmap(original,_report())
    assert [point["timing"] for point in annotated["beats"]]==[point["timing"] for point in original["beats"]]
    assert report["timing_changes_applied"] is False
    assert report["source_policy"]=="instrument_aware_adapters_with_shared_schema"
    at_five=annotated["beats"][4]["evidence"]
    boundary=next(item for item in at_five["observations"] if item["evidence_family"]=="structural_boundary")
    assert boundary["subtype"]=="multi_instrument_entrance"
    assert boundary["source_roles"]==["drums","guitar"]
    assert boundary["details"]["source_confirmation"]["drums"]["confidence"]==0.9


def test_method_specific_details_are_preserved():
    observations=be.normalize_evidence(_report())
    rhythmic=next(item for item in observations if item["method"]=="periodic_event_checkpoint")
    harmonic=next(item for item in observations if item["method"]=="windowed_chroma_checkpoint")
    path=next(item for item in observations if item["method"]=="selected_continuous_anchor_hypothesis")
    assert rhythmic["source_roles"]==["drums"]
    assert rhythmic["details"]["matched_events"]==8
    assert harmonic["source_roles"]==["guitar"]
    assert harmonic["details"]["source_evidence"]["guitar"]["best_score"]==0.7
    assert path["subtype"]=="cross_source_consensus"
    assert len(path["details"]["observations"])==2


def test_boundary_splits_regions_and_keeps_separate_trends():
    _,report=be.annotate_beatmap(_beatmap(),_report())
    region=next(item for item in report["regions"] if item["start_playback_measure"]==5)
    assert region["left_boundary"]=="activity_boundary"
    assert region["family_trends"]["rhythmic_onset"]["observation_count"]==4
    assert region["source_role_trends"]["drums"]["observation_count"]>=4
    assert region["cross_source_consensus_trend"]["observation_count"]>=1
    assert region["combined_interpretation"]["classification"]=="insufficient_evidence"
    assert region["all_evidence_trend"]["diagnostic_accumulated_drift_ms"] is not None


def test_no_song_names_or_measure_specific_rules_in_module():
    import inspect
    source=inspect.getsource(be).lower()
    assert "all that i bleed" not in source
    assert "spellbound" not in source
    assert "against you" not in source

