from pathlib import Path


def test_post_materialization_phase_qualification_uses_in_scope_section_measures():
    source=Path(__file__).resolve().parents[1].joinpath('process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'enumerate(section_measures)' in source
    assert 'enumerate(playback_measures)' not in source


def test_final_phase_qualification_receives_materialization_report():
    source=Path(__file__).resolve().parents[1].joinpath('process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'midi_section_fit_report["section_evidence_quality"],section_application)' in source
