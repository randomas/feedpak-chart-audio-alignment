import json
from pathlib import Path

import collect_run_diagnostics as crd
import hybrid_midi_audio_control as hmac
from regenerate_hybrid_midi_audio_control import regenerate


def _midi_report():
    return {"fusion_report":{"regions":[{"region_id":"r1","chart_start":10.0,"chart_end":14.0,
        "controls":[{"chart_time":10.0,"audio_time":10.0},{"chart_time":14.0,"audio_time":14.0}]}]},
        "application_report":{"applied_regions":[{"chart_start":10.0,"chart_end":14.0}]}}


def _ledger():
    return {"events":[{"event_id":"e1","status":"exact_cross_modal_onset","chart_time":13.0,
        "audio_clocks":[{"event_id":"a1","chart_time":13.0,"mapped_audio_time":13.2,
            "confidence":0.95,"limitations":[],"source":"mix"}],
        "midi_clocks":[{"event_id":"m1","chart_time":13.0,"mapped_audio_time":13.0}]}]}


def test_runtime_uses_canonical_key_and_preserves_alias():
    source=Path("process_gp_alignment.py").read_text(encoding="utf-8")
    assert '"hybrid_midi_audio_control":hybrid_midi_audio_control_proposal' in source
    assert '"hybrid_midi_audio_control_proposal":hybrid_midi_audio_control_proposal' in source
    build=Path("build_feedpak.py").read_text(encoding="utf-8")
    assert 'gp_data.get("hybrid_midi_audio_control")' in build
    assert 'gp_data.get("hybrid_midi_audio_control_proposal")' in build
    assert 'output_path+".hybrid_midi_audio_control.json"' in build
    assert "Hybrid MIDI-audio proposal inputs exist but the proposal report was not generated" in build


def test_retrospective_regeneration_writes_valid_sidecar(tmp_path):
    ledger=tmp_path/"song.feedpak.evidence_contribution_ledger.json"
    midi=tmp_path/"song.feedpak.midi_section_fit.json"
    output=tmp_path/"song.feedpak.hybrid_midi_audio_control.json"
    ledger.write_text(json.dumps(_ledger()),encoding="utf-8")
    midi.write_text(json.dumps(_midi_report()),encoding="utf-8")
    report=regenerate(ledger,midi,output)
    loaded=json.loads(output.read_text(encoding="utf-8"))
    assert loaded==report
    assert loaded["diagnostic_only"] is True
    assert loaded["timing_changes_applied"] is False
    assert loaded["summary"]["proposal_count"]==1


def test_light_skeleton_retains_hybrid_decisions(tmp_path):
    report=hmac.build_proposal(evidence_contribution_ledger=_ledger(),midi_section_fit_report=_midi_report())
    source=tmp_path/"song.feedpak.hybrid_midi_audio_control.json"
    destination=tmp_path/"song.feedpak.hybrid_midi_audio_control.light.json"
    source.write_text(json.dumps(report),encoding="utf-8")
    crd.write_light_json(source,destination)
    light=json.loads(destination.read_text(encoding="utf-8"))
    assert light["diagnostic_profile"]=="light"
    assert light["summary"]["proposal_count"]==1
    assert light["proposals"][0]["classification"]=="audio_corrected_midi_geometry"
    assert light["safety"]["timing_changes_applied"] is False


def test_light_collector_recognizes_canonical_sidecar_name():
    source=Path("collect_run_diagnostics.py").read_text(encoding="utf-8")
    assert "*hybrid_midi_audio_control.json" in source
