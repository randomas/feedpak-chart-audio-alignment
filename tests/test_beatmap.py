import json
import pytest
import beatmap as bm
from process_gp_alignment import build_explicit_feedpak_beats, compute_dense_measure_tempos


def data():
    timeline={"time_signatures":[{"t_gp":0,"ts":[4,4]},{"t_gp":8,"ts":[3,4]},{"t_gp":11,"ts":[4,4]}],"beats":[
        {"t_gp":0,"measure":1,"source_bar":1,"occurrence":1,"playback_segment":0,"ts_num":4},
        {"t_gp":4,"measure":2,"source_bar":2,"occurrence":1,"playback_segment":0,"ts_num":4},
        {"t_gp":8,"measure":3,"source_bar":1,"occurrence":2,"playback_segment":1,"ts_num":3},
        {"t_gp":11,"measure":4,"source_bar":3,"occurrence":1,"playback_segment":1,"ts_num":4}]}
    warped=[{"time":1,"measure":1,"ts_num":4},{"time":5,"measure":2,"ts_num":4},
            {"time":9.2,"measure":3,"ts_num":3},{"time":12.2,"measure":4,"ts_num":4}]
    return timeline,warped


def test_identity_and_meter_change():
    timeline,warped=data(); result=bm.build_beatmap(timeline,warped)
    assert result["summary"]["playback_segment_count"]==2
    repeat=next(x for x in result["beats"] if x["identity"]["playback_measure"]==3)
    assert repeat["identity"]["beat_id"]=="seg001.m0003.o02.b00"
    assert repeat["meter"]==[3,4]
    json.dumps(result)


def test_phase2_projection_matches_legacy():
    timeline,warped=data(); result=bm.build_beatmap(timeline,warped)
    lb=build_explicit_feedpak_beats(warped); lt=compute_dense_measure_tempos(warped,60)
    pb=bm.beatmap_to_explicit_beats(result); pt=bm.beatmap_to_dense_tempos(result,60)
    c=bm.compare_timeline_outputs(lb,pb,lt,pt)
    assert c["status"]=="equivalent" and lb==pb and lt==pt


def test_equivalence_gate_rejects_change():
    timeline,warped=data(); result=bm.build_beatmap(timeline,warped)
    beats=bm.beatmap_to_explicit_beats(result); changed=[dict(x) for x in beats]; changed[1]["time"]+=.01
    tempos=bm.beatmap_to_dense_tempos(result,60); c=bm.compare_timeline_outputs(beats,changed,tempos,tempos)
    with pytest.raises(RuntimeError): bm.require_timeline_equivalence(c)


def test_report_includes_equivalence():
    timeline,warped=data(); result=bm.build_beatmap(timeline,warped)
    beats=bm.beatmap_to_explicit_beats(result); tempos=bm.beatmap_to_dense_tempos(result,60)
    c=bm.compare_timeline_outputs(beats,beats,tempos,tempos)
    assert bm.build_beatmap_report(result,c)["timeline_equivalence"]["status"]=="equivalent"
