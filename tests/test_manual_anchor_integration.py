import json
import build_feedpak as builder
import process_gp_alignment as pga

BARS = [
    {"measure": 1, "source_bar": 1, "occurrence": 1, "t_gp": 0.0, "ts_num": 4},
    {"measure": 2, "source_bar": 2, "occurrence": 1, "t_gp": 4.0, "ts_num": 4},
    {"measure": 3, "source_bar": 3, "occurrence": 1, "t_gp": 8.0, "ts_num": 4},
    {"measure": 4, "source_bar": 4, "occurrence": 1, "t_gp": 12.0, "ts_num": 4},
]


def write(tmp_path, payload):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_metadata_is_automatically_selected(tmp_path):
    p = write(tmp_path, {"alignment": {"manual_anchors": [{"score": {"playback_bar": 2, "beat": 1}, "audio": {"time": 5.0}}]}})
    assert builder.resolve_manual_anchor_path(json.loads(p.read_text()), str(p)) == str(p)


def test_no_metadata_anchor_is_timing_neutral(tmp_path):
    p = write(tmp_path, {"title": "No anchors"})
    base = lambda t: t + 2.0
    warp, report = pga.apply_manual_anchor_layer(base, str(p), BARS, padding_added=1.0)
    assert report["status"] == "not_configured"
    assert report["timing_changes_applied"] is False
    assert warp(7.5) == base(7.5)


def test_exact_metadata_anchor_applies_after_baseline(tmp_path):
    p = write(tmp_path, {"alignment": {"manual_anchor_time_basis": "original", "manual_anchors": [
        {"id": "entry", "score": {"playback_bar": 2, "beat": 1}, "audio": {"time": 5.0}, "mode": "exact"}
    ]}})
    warp, report = pga.apply_manual_anchor_layer(lambda t: t, str(p), BARS, padding_added=1.0)
    assert report["status"] == "applied"
    assert report["timing_changes_applied"] is True
    assert warp(4.0) == 6.0
    assert report["quality"]["anchors"][0]["time_basis"] == "original"


def test_search_anchor_reports_but_does_not_move_timing(tmp_path):
    p = write(tmp_path, {"alignment": {"manual_anchors": [
        {"id": "hint", "score": {"playback_bar": 2, "beat": 1}, "audio": {"time": 8.02}, "mode": "search"}
    ]}})
    base = lambda t: t
    warp, report = pga.apply_manual_anchor_layer(base, str(p), BARS)
    assert report["status"] == "diagnostic_only"
    assert warp(4.0) == 4.0
    assert "nearby_bar_fits_better" in report["quality"]["anchors"][0]["warnings"]


def test_events_after_last_downbeat_keep_baseline_extrapolation(tmp_path):
    p = write(tmp_path, {"alignment": {"manual_anchors": [
        {"id": "entry", "score": {"playback_bar": 2, "beat": 1}, "audio": {"time": 5.0}, "mode": "exact"}
    ]}})
    base = lambda t: t
    warp, _ = pga.apply_manual_anchor_layer(base, str(p), BARS)
    assert warp(13.5) == 13.5


def test_final_bar_exact_anchor_applies(tmp_path):
    p = write(tmp_path, {"alignment": {"manual_anchors": [
        {"id": "final", "score": {"playback_bar": 4, "beat": 1}, "audio": {"time": 13.0}, "mode": "exact"}
    ]}})
    base = lambda t: t
    warp, report = pga.apply_manual_anchor_layer(base, str(p), BARS)
    assert report["status"] == "applied"
    assert warp(12.0) == 13.0
    # Beyond the final known downbeat, preserve baseline extrapolation.
    assert warp(13.5) == 13.5
