import process_gp_alignment as pga

def test_rebase_tempo_events_uses_one_clock():
    assert pga.rebase_tempo_events([(0,90),(16320,95),(43200,97)],960)==[(0,90.0),(15360,95.0),(42240,97.0)]

def test_rebase_tempo_event_at_origin_replaces_initial():
    assert pga.rebase_tempo_events([(0,90),(960,95),(2000,100)],960)==[(0,95.0),(1040,100.0)]

def test_chart_offset_avoids_double_count_in():
    assert abs(pga.choose_chart_offset(2.6666667,2.6666667))<1e-7
    assert pga.choose_chart_offset(2.6666667,0.0)==2.6666667

def test_trustworthy_excludes_onset_only():
    assert pga.trustworthy_checkpoint_count({"summary":{"strong":0}},{"measured_boundaries":0})==0




def test_opening_candidates_use_earliest_scored_instrument(monkeypatch):
    detected={"guitar.wav":3.02,"drums.wav":4.15,"full.wav":3.01}
    def fake(path, expected_time, **kwargs):
        return detected.get(path), {"available": path in detected}
    monkeypatch.setattr(pga,"_detect_lenient_opening_time",fake)
    candidates=pga.build_opening_anchor_candidates(
        [("drums",[4.125,5.0,6.0,7.0],"drums.wav"),
         ("guitar",[3.0],"guitar.wav")],
        score_start=3.0,audio_content_start=3.0,full_audio_path="full.wav")
    guitar=next(x for x in candidates if x["source"]=="guitar")
    drums=next(x for x in candidates if x["source"]=="drums")
    assert guitar["accepted"] is True
    assert guitar["measured_audio_first"] == 3.02
    assert drums["accepted"] is False
    assert "later_than_earliest_configured_score_entrance" in drums["warnings"]
    score,audio,report=pga.choose_opening_anchor(candidates,3.0,3.0)
    assert score == 3.0
    assert audio == 3.02
    assert report["selected_sources"] == "guitar"


def test_opening_fallback_preserves_earliest_symbolic_note():
    candidates=[{"source":"guitar","symbolic_first":2.5,"accepted":False,"warnings":["missing_audio"]},
                {"source":"drums","symbolic_first":4.0,"accepted":False,"warnings":["later"]}]
    score,audio,report=pga.choose_opening_anchor(candidates,2.0,3.0)
    assert score == 2.5
    assert audio == 3.0
    assert report["status"] == "fallback_content_start_for_earliest_configured_instrument"
