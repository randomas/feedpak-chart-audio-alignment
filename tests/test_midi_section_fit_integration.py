from pathlib import Path
import json
import collect_run_diagnostics as crd

ROOT=Path(__file__).resolve().parents[1]


def test_cli_and_runner_wiring():
    process=(ROOT/'process_gp_alignment.py').read_text()
    builder=(ROOT/'build_feedpak.py').read_text()
    runner=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    assert 'parser.add_argument("--midi-section-fit"' in process
    assert 'parser.add_argument("--midi-section-fit"' in builder
    assert '[switch]$MidiSectionFit' in runner
    assert 'if ($MidiSectionFit) { $arguments += "--midi-section-fit" }' in runner
    assert 'midi_section_fit_report=msf.build_section_fusion_report' in process
    assert '"midi_section_fit_report": midi_section_fit_report' in process
    assert 'output_path+".midi_section_fit.json"' in builder


def test_light_skeleton_preserves_safety_and_regions():
    source={"version":"1.0-step3","method":"midi-section-fit","diagnostic_only":True,
        "timing_changes_applied":False,"integration":{"playback_measure_count":12},
        "candidate_report":{"summary":{"window_count":8}},
        "path_report":{"summary":{"streak_count":2},"roles":{"piano":{"status":"available",
            "streak_count":1,"selected_candidate_count":3,"unsupported_windows":[],
            "monotonicity_validated":True,"streaks":[{"streak_id":"piano-s1","chart_start":0,
            "chart_end":20,"midi_start":1,"midi_end":21,"monotonic":True}]}}},
        "fusion_report":{"summary":{"region_count":1,"control_count":3},
            "normal_production_limits":{"minimum_edge_stretch":.9,"maximum_edge_stretch":1.1},
            "monotonicity_certificate":{"strict_chart_time_increase":True,
                "strict_midi_time_increase":True,"unsupported_gaps_interpolated":False},
            "regions":[{"region_id":"r1","chart_start":0,"chart_end":20,"midi_start":1,
                "midi_end":21,"control_count":3,"monotonic":True}],"role_conflicts":[],
            "rejected_blocks":[]}}
    out=crd.skeleton_midi_section_fit(source)
    assert out['timing_changes_applied'] is False
    assert out['monotonicity_certificate']['unsupported_gaps_interpolated'] is False
    assert out['regions'][0]['monotonic'] is True


def test_drum_family_symbols_are_normalized():
    import midi_section_fit as msf
    assert msf._pitches({'p':'kick'})=={1000}
    assert msf._pitches({'pieces':['snare','ride']})=={1001,1003}
