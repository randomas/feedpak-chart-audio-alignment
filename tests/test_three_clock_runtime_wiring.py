from pathlib import Path
import inspect
import midi_section_fit as msf


def test_fusion_api_accepts_role_audio_starts():
    assert "role_audio_starts" in inspect.signature(msf.build_section_fusion_report).parameters


def test_runtime_calls_three_clock_reconciliation():
    text = Path(msf.__file__).read_text(encoding="utf-8")
    assert "import incipit_three_clock as itc" in text
    assert "itc.reconcile_opening(fusion_report,chart_events_by_role,role_audio_starts or {})" in text


def test_process_supplies_real_role_audio_starts():
    import process_gp_alignment as pga
    text = Path(pga.__file__).read_text(encoding="utf-8")
    assert "role_audio_starts={role:fc.detect_leading_silence_seconds(path)" in text
    assert 'if role in ("piano","guitar","bass","drums") and path' in text


def test_light_diagnostics_expose_three_clock_result():
    import collect_run_diagnostics as crd
    text = Path(crd.__file__).read_text(encoding="utf-8")
    assert '"three_clock_incipit":fusion.get("three_clock_incipit")' in text
