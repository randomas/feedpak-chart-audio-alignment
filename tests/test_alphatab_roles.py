import json

import alphatab_score as ats


def _data():
    return {
        "tracks": [
            {"index": 0, "name": "Lead Guitar"},
            {"index": 1, "name": "Rhythm Guitar"},
            {"index": 2, "name": "Bass Guitar"},
            {"index": 3, "name": "Drums"},
            {"index": 4, "name": "Piano LH"},
            {"index": 5, "name": "Piano RH"},
            {"index": 6, "name": "Guide"},
        ]
    }


def test_resolved_roles_accepts_project_track_schema(tmp_path):
    metadata = {
        "feedpak_project": {
            "tracks": {
                "lead_vocal": None,
                "drums": "Drums",
                "piano": {"left": "Piano LH", "right": "Piano RH", "combined": None},
                "guitars": ["Lead Guitar", "Rhythm Guitar"],
                "bass": ["Bass Guitar"],
                "ignored_tracks": ["Guide"],
                "unused_tracks": [],
                "overrides": {},
            }
        }
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    roles = ats.resolved_roles(_data(), path)
    assert roles["Lead Guitar"] == "guitar"
    assert roles["Rhythm Guitar"] == "guitar"
    assert roles["Bass Guitar"] == "bass"
    assert roles["Drums"] == "drums"
    assert roles["Piano LH"] == "piano_left"
    assert roles["Piano RH"] == "piano_right"
    assert roles["Guide"] == "ignored"


def test_resolved_roles_ignores_unknown_configured_names(tmp_path):
    metadata = {"feedpak_project": {"tracks": {
        "piano": {"left": None, "right": None, "combined": None},
        "guitars": ["Missing Guitar"], "bass": [],
        "ignored_tracks": [], "unused_tracks": [],
        "overrides": {"Missing Track": "guitar"},
    }}}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    roles = ats.resolved_roles(_data(), path)
    assert "Missing Guitar" not in roles
    assert "Missing Track" not in roles




def test_explicit_null_lead_vocal_disables_automatic_role(tmp_path):
    data={"tracks":[{"index":0,"name":"Vocals"},{"index":1,"name":"Lead Guitar"}]}
    metadata={"feedpak_project":{"tracks":{"lead_vocal":None}}}
    path=tmp_path/"metadata.json"
    path.write_text(json.dumps(metadata),encoding="utf-8")
    roles=ats.resolved_roles(data,path)
    assert roles["Vocals"] == "ignored"
    assert roles["Lead Guitar"] == "guitar"


def test_products_populates_fretted_alignment_events():
    data={
        "source":{"ppq":960},
        "metadata":{"tempo":120},
        "tracks":[{
            "index":0,"name":"Lead Guitar",
            "staves":[{"properties":{"tuning":[64,59,55,50,45,40],"capo":0},
                       "bars":[{"voices":[{"beats":[{"id":10,"state":{},"rhythm":{},
                           "notes":[{"id":20,"techniques":{},"links":{}}]}]}]}]}]
        }],
        "master_bars":[{"meter":{"numerator":4,"denominator":4},"key":{"signature":0,"type":0}}],
        "playback":{
            "available":True,
            "tempo_events":[{"tick":0,"tempo":120}],
            "master_bar_visits":[{"playback_index":0,"source_master_bar_index":0,
                "start_tick":0,"end_tick":3840,"duration_ticks":3840,"occurrence":1}],
            "beat_occurrences":[],
            "playback_notes":[{"track_index":0,"staff_index":0,"voice_index":0,
                "note_id":20,"beat_id":10,"absolute_start_tick":0,"duration_ticks":480,
                "playback_master_bar_index":0,"string":1,"fret":3,"pitch_midi":67}]
        }
    }
    out=ats.products(data)
    assert out["score_content_time"] == 0.0
    assert out["guitar_alignment_events"] == [{"t_gp":0.0}]
    assert out["guitar_pitch_events"] == [{"t_gp":0.0,"midi":67}]
