import json
from pathlib import Path
import method_orchestration as mo


def verification():
    return {"manual_anchor_count":0,"manual_anchor_status":"not_supplied",
        "maximum_manual_anchor_error_ms":0.0,"manual_anchor_conflicts":[],"verification_tolerance_ms":1.0}


def test_smoke_measure_stretch_warnings_are_grouped_and_status_warns():
    warnings=[]
    for measure in range(81,99):warnings.append({"level":"warning","kind":"measure_stretch","measure":measure,"ratio":1.1562})
    for measure in range(100,115):warnings.append({"level":"warning","kind":"measure_stretch","measure":measure,"ratio":.7985})
    out=mo.build_result_envelope(alignment_mode="continuous-anchor-dtw",decision={"timing_output":"continuous-anchor-dtw-v1"},
        prepared_midi_mode="off",manual_anchor_verification=verification(),alignment_report={"warnings":warnings})
    assert out["status"]=="complete_with_warnings"
    assert out["schema_version"]==3
    assert len(out["measure_stretch_warning_ranges"])==2
    assert out["measure_stretch_warning_ranges"][0]["start_measure"]==81
    assert out["measure_stretch_warning_ranges"][0]["end_measure"]==98
    assert out["measure_stretch_warning_ranges"][1]["start_measure"]==100
    assert out["measure_stretch_warning_ranges"][1]["end_measure"]==114
    assert {x["type"] for x in out["phase_warnings"]}=={"excessive_local_stretch"}
    assert len(out["listening_review_points"])==2


def test_non_stretch_alignment_warning_is_preserved():
    out=mo.build_result_envelope(alignment_mode="continuous-anchor-dtw",decision={"timing_output":"continuous-anchor-dtw-v1"},
        prepared_midi_mode="off",manual_anchor_verification=verification(),
        alignment_report={"warnings":[{"level":"warning","kind":"terminal_mismatch","message":"ending differs","chart_time":200}]})
    assert out["status"]=="complete_with_warnings"
    assert out["alignment_warnings"][0]["kind"]=="terminal_mismatch"
    assert out["listening_review_points"][0]["chart_time"]==200


def test_builder_populates_feedpak_path_before_export():
    source=Path("build_feedpak.py").read_text()
    assert 'method_result_envelope["feedpak_path"]=os.path.abspath(output_path)' in source
    assert 'fc.write_json(method_path,method_result_envelope)' in source


def test_runner_summary_records_envelope_fields_without_powershell7_only_operator():
    source=Path("final_comparison_runner/final_runner/run_v16_8_final_comparison.ps1").read_text()
    assert "envelope_status=$envelopeStatus" in source
    assert "warning_count=$warningCount" in source
    assert "review_point_count=$reviewPointCount" in source
    assert "??" not in source
