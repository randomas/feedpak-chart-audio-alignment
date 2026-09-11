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
