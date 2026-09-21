from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "process_gp_alignment.py"


def test_timing_origin_is_initialized_before_report_use():
    source = SOURCE.read_text(encoding="utf-8")

    report = source.index('decision["timing_origin"]')

    alphatab_audio = source.index(
        "audio_content_start=float(count_in_offset)"
    )
    alphatab_chart = source.index(
        "chart_offset=choose_chart_offset(",
        alphatab_audio,
    )

    gp5_audio = source.index(
        "audio_content_start = float(count_in_offset)"
    )
    gp5_chart = source.index(
        "chart_offset=choose_chart_offset(",
        gp5_audio,
    )

    assert alphatab_audio < alphatab_chart < report
    assert gp5_audio < gp5_chart < report

def test_audio_content_start_and_chart_offset_are_not_reused_as_one_variable():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"chart_offset":round(chart_offset,6)' in source
    assert '"chart_offset":round(count_in_offset,6)' not in source
    assert "count_in_offset=choose_chart_offset" not in source


def test_all_downstream_score_shifts_use_chart_offset():
    source = SOURCE.read_text(encoding="utf-8")
    required = [
        'shift_nominal_times(drum_hits_gp, chart_offset)',
        'shift_nominal_times(song_timeline_gp["beats"], chart_offset)',
        'shift_and_warp_notation(notation, chart_offset, warp_fn)',
        'warp_gp_vocals(gp_vocals,chart_offset,warp_fn)',
    ]
    for expression in required:
        assert expression in source



