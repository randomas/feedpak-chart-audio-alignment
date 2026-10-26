from pathlib import Path
import json
import pytest
import transcription_investigator as ti


def test_supported_method_policy():
    for method in ti.SUPPORTED_METHODS:
        ti.validate_method(method)
    for method in ("nominal","offset","linear","dtw"):
        with pytest.raises(ValueError):
            ti.validate_method(method)


def test_prepared_file_discovery_is_case_insensitive(tmp_path):
    (tmp_path/"pIaNo.MuScRiPtOr.MiD").write_bytes(b"midi")
    found=ti.discover_prepared_files(tmp_path)
    assert found["piano"].name=="pIaNo.MuScRiPtOr.MiD"
    assert found["bass"] is None


def test_asymmetric_bidirectional_phrase_match_ignores_extras():
    cfg=ti.InvestigatorConfig(minimum_matched_events=3)
    score=[{"time":0.0,"pitches":[60],"pieces":[]},{"time":1.0,"pitches":[64],"pieces":[]},{"time":2.0,"pitches":[67],"pieces":[]}]
    audio=[{"time":0.0,"pitches":[60],"pieces":[]},{"time":0.4,"pitches":[73],"pieces":[]},
           {"time":1.0,"pitches":[64],"pieces":[]},{"time":1.4,"pitches":[70],"pieces":[]},{"time":2.0,"pitches":[67],"pieces":[]}]
    forward=ti._align_groups(score,audio,"piano",cfg,False)
    backward=ti._align_groups(score,audio,"piano",cfg,True)
    assert forward["matched"]==3
    assert backward["matched"]==3
    assert len(forward["matched_pairs"])==3


def test_drum_notes_collapse_to_chart_canonical_pieces():
    cfg=ti.InvestigatorConfig(minimum_matched_events=3)
    score=[{"time":0.0,"pitches":[],"pieces":["kick"]},
           {"time":0.5,"pitches":[],"pieces":["hh_closed"]},
           {"time":1.0,"pitches":[],"pieces":["snare","hh_closed"]}]
    audio=[{"time":0.0,"pitches":[],"pieces":[ti.fc.gm_drum_to_piece(36)]},
           {"time":0.5,"pitches":[],"pieces":[ti.fc.gm_drum_to_piece(42)]},
           {"time":1.0,"pitches":[],"pieces":[ti.fc.gm_drum_to_piece(38),ti.fc.gm_drum_to_piece(46)]}]
    result=ti._align_groups(score,audio,"drums",cfg)
    assert result["matched"]==3


def test_distinct_location_clustering_removes_overlapping_duplicates():
    cfg=ti.InvestigatorConfig(distinct_location_seconds=.75)
    rows=[{"audio_time":10.0,"score":.8},{"audio_time":10.2,"score":.9},{"audio_time":15.0,"score":.7}]
    distinct=ti._cluster_distinct_locations(rows,cfg)
    assert len(distinct)==2
    assert distinct[0]["audio_time"]==10.2


@pytest.mark.skipif(ti.mido is None,reason="mido not installed")
def test_midi_ingestion_applies_padding_without_rewriting(tmp_path):
    midi=ti.mido.MidiFile(ticks_per_beat=480); track=ti.mido.MidiTrack(); midi.tracks.append(track)
    track.append(ti.mido.Message("note_on",note=60,velocity=80,time=480))
    track.append(ti.mido.Message("note_off",note=60,velocity=0,time=480))
    path=tmp_path/"Piano.muscriptor.mid"; midi.save(path)
    before=path.read_bytes()
    result=ti.parse_muscriptor_midi(path,"piano",padding_added=2.5)
    assert result["status"]=="available"
    assert result["events"][0]["source_time"]==pytest.approx(0.5)
    assert result["events"][0]["audio_time"]==pytest.approx(3.0)
    assert path.read_bytes()==before


def test_missing_prepared_files_are_nonfatal(tmp_path):
    report=ti.investigate({"piano":[]},tmp_path,lambda x:x,padding_added=2.0,
                          alignment_mode="continuous-anchor-structural")
    assert report["status"]=="partial"
    assert report["summary"]["prepared_role_count"]==0
    assert all(x["status"]=="not_prepared" for x in report["prepared_files"])


def test_empty_affine_fit_is_strict_json_safe():
    fit=ti._robust_affine_fit([],[],[])
    assert fit["median_residual"] is None
    assert fit["valid"] is False
    import json
    json.dumps(fit,allow_nan=False)


def test_opening_scale_prior_is_method_independent():
    prior,report=ti.make_nominal_opening_prior(3.0,3.2,.95)
    assert prior(13.0)==pytest.approx(12.7)
    assert report["kind"]=="opening_scale"


def test_global_path_rejects_implausible_edge_stretch():
    cfg=ti.InvestigatorConfig()
    base={"diagnostic_status":"valid_unique_phrase","local_phrase_confidence":.9}
    rows=[{**base,"score_time":10.,"audio_time":10.},
          {**base,"score_time":20.,"audio_time":50.},
          {**base,"score_time":30.,"audio_time":30.}]
    out=ti.select_globally_constrained_path(rows,lambda t:t,cfg)
    assert [x["audio_time"] for x in out["selected"]]==[10.,30.]


def test_repeated_motif_can_be_promoted_by_valid_chain():
    cfg=ti.InvestigatorConfig()
    rows=[]
    for n in range(3):
        rows.append({"diagnostic_status":"valid_repeated_motif_hypothesis",
                     "local_phrase_confidence":.9,"score_time":10.+10*n,"audio_time":10.2+10*n})
    out=ti.select_globally_constrained_path(rows,lambda t:t,cfg)
    assert len(out["selected"])==3
    assert all(x["diagnostic_status"]=="valid_chain_supported_motif" for x in out["selected"])


def test_transcription_prior_interpolates_selected_path():
    report={"global_monotonic_path_preview":{"selected":[
        {"score_time":10.,"audio_time":11.},{"score_time":20.,"audio_time":22.}]}}
    prior,meta=ti.build_transcription_prior(report,lambda t:t)
    assert meta["status"]=="available"
    assert prior(15.)==pytest.approx(16.5)


def _trusted_rows(count=12, role="guitar"):
    return [{"score_time":float(i),"audio_time":float(i)+.2,"source":role,
             "local_phrase_confidence":.98,"global_placement_confidence":.95}
            for i in range(1,count+1)]


def test_trusted_midi_segments_reduce_overlapping_observations():
    report={"global_monotonic_path_preview":{"selected":_trusted_rows()}}
    hierarchy=ti.build_trusted_midi_segments(report,representative_spacing=4.0)
    assert hierarchy["trusted_segments"]
    assert 2 <= hierarchy["trusted_control_count"] < 12
    assert hierarchy["trusted_segments"][0]["authority"]=="trusted_midi_segment"


def test_midi_authority_prior_becomes_primary_when_path_is_good():
    report={"global_monotonic_path_preview":{"selected":_trusted_rows(40)}}
    prior,meta=ti.build_midi_authority_prior(report,lambda t:t)
    assert meta["status"]=="available"
    assert meta["authority"]=="trusted_midi_primary"
    assert prior(20.0)==pytest.approx(20.2)


def test_audio_anchor_compatibility_does_not_move_midi():
    report={"matches":[{"score_time":10.0,"candidates":[{"time":10.05},{"time":10.3},{"time":12.0}]}]}
    out=ti.classify_audio_anchor_compatibility(report,lambda t:t)
    classes=[x["midi_compatibility"] for x in out["matches"][0]["candidates"]]
    assert classes==["strong_corroboration","uncertain","conflict"]
    assert out["matches"][0]["candidates"][2]["production_eligible_against_midi"] is False


def test_late_role_is_observed_but_not_incipit_authority_eligible():
    profiles=ti.build_role_activity_profiles(
        {"piano":[0.0,1.0,2.0],"guitar":[120.0,121.0,122.0]},
        [{"source":"piano","status":"available"},{"source":"guitar","status":"available"}])
    assert profiles["piano"]["incipit_authority_eligible"] is True
    assert profiles["guitar"]["incipit_authority_eligible"] is False
    assert profiles["guitar"]["score_event_count"]==3


def test_role_paths_ignore_single_event_style_candidates():
    cfg=ti.InvestigatorConfig()
    weak={"source":"guitar","score_time":10.0,"audio_time":10.1,
          "local_phrase_confidence":.99,"global_placement_confidence":.99,
          "diagnostic_status":"valid_unique_phrase","sequence_authority_eligible":False}
    paths,fused=ti.build_role_paths([weak],lambda t:t,cfg)
    assert paths["guitar"]["selected_count"]==0
    assert fused["selected_count"]==0


def test_offset_incipit_can_win_soft_consensus_over_expected_origin():
    cfg=ti.InvestigatorConfig()
    candidate={"source":"piano","score_time":2.0,"audio_time":5.0,
        "sequence_first_score_time":0.0,"local_phrase_confidence":.95,
        "phrase_distinctiveness":.8,"uniqueness_margin":.12}
    profiles={"piano":{"first_score_time":0.0,"incipit_authority_eligible":True}}
    out=ti.build_incipit_consensus([candidate],lambda t:t,lambda t:t+3.0,profiles,cfg)
    assert out["selected"]["offset_incipit_delta_seconds"]==pytest.approx(0.0)
    assert out["selected"]["expected_origin_delta_seconds"]==pytest.approx(3.0)


def test_opening_role_classifier_finds_solo_piano():
    cfg=ti.InvestigatorConfig()
    score={"piano":[0, .5, 1, 1.5, 2, 2.5, 3],
           "bass":[12, 13, 14],"guitar":[30,31,32],"drums":[]}
    out=ti.classify_opening_role(score,cfg)
    assert out["classification"]=="solo"
    assert out["role"]=="piano"


def test_opening_role_classifier_rejects_ensemble_opening():
    cfg=ti.InvestigatorConfig()
    score={"piano":[0,.5,1,1.5,2,2.5],"guitar":[0,.5,1,1.5,2,2.5],
           "bass":[],"drums":[]}
    out=ti.classify_opening_role(score,cfg)
    assert out["classification"]=="ensemble"
    assert out["role"] is None


def test_confirmed_incipit_streak_is_interval_limited():
    report={"incipit_role_tracking":{"status":"confirmed","representative_controls":[
        {"score_time":2.,"audio_time":2.2,"source":"piano"},
        {"score_time":6.,"audio_time":6.2,"source":"piano"},
        {"score_time":10.,"audio_time":10.2,"source":"piano"}]},
        "global_monotonic_path_preview":{"selected":[]}}
    prior,meta=ti.build_midi_authority_prior(report,lambda t:t)
    assert meta["authority"]=="incipit_role_streak"
    assert prior(5.)==pytest.approx(5.2)
    assert prior(20.)==pytest.approx(20.0)
    assert prior(0.)==pytest.approx(0.0)


def test_chart_to_midi_streak_fit_recovers_chart_transform():
    cfg=ti.InvestigatorConfig()
    rows=[{"score_time":float(i),"midi_time":.25+1.02*i,"similarity":1.0} for i in range(12)]
    rows.append({"score_time":5.5,"midi_time":20.0,"similarity":.5})
    fit=ti.fit_chart_to_midi_streak(rows,cfg)
    assert fit["valid"] is True
    assert fit["chart_stretch"]==pytest.approx(1.02,abs=1e-6)
    assert fit["offset"]==pytest.approx(.25,abs=1e-6)
    assert fit["rejected_landmark_count"]>=1
    assert fit["fixed_timeline"]=="prepared_midi_audio_clock"
    assert fit["transformed_timeline"]=="chart_clock"


def test_chart_to_midi_streak_fit_rejects_unsafe_chart_stretch():
    cfg=ti.InvestigatorConfig()
    rows=[{"score_time":float(i),"midi_time":.2+1.2*i} for i in range(12)]
    fit=ti.fit_chart_to_midi_streak(rows,cfg)
    assert fit["valid"] is False
    assert fit["reason"]=="unsafe_or_incoherent_chart_to_midi_fit"


def test_streak_authority_uses_true_first_landmark_control():
    report={"incipit_role_tracking":{"status":"confirmed","representative_controls":[
        {"score_time":2.,"audio_time":2.25,"source":"piano","kind":"incipit_acquisition_onset"},
        {"score_time":6.,"audio_time":6.33,"source":"piano","kind":"chart_to_midi_streak_fit"},
        {"score_time":10.,"audio_time":10.41,"source":"piano","kind":"chart_to_midi_streak_fit"}]},
        "global_monotonic_path_preview":{"selected":[]}}
    prior,meta=ti.build_midi_authority_prior(report,lambda t:t)
    assert meta["authority"]=="incipit_role_streak"
    assert prior(2.)==pytest.approx(2.25)
    assert prior(0.)==pytest.approx(0.)
    assert prior(20.)==pytest.approx(20.)
