from copy import deepcopy
from pathlib import Path
import json
import audio_evidence_schema as aes
import collect_run_diagnostics as crd


def test_runtime_wires_bundle_after_existing_reports():
    root=Path(__file__).resolve().parents[1]
    text=(root/'process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'import audio_evidence_schema as aes' in text
    assert 'audio_evidence_bundle=aes.build_read_only_evidence_bundle' in text
    position=text.index('audio_evidence_bundle=aes.build_read_only_evidence_bundle')
    assert text.index('instrument_evidence_report=ie.analyze_all') < position
    assert text.index('structural_search_report=structural_search_module.analyse') < position
    assert text.index('production_timing=finalize_production_timing') < position


def test_builder_writes_dedicated_read_only_sidecar():
    root=Path(__file__).resolve().parents[1]
    text=(root/'build_feedpak.py').read_text(encoding='utf-8')
    assert '.audio_evidence_bundle.json' in text
    assert 'Wrote read-only audio evidence bundle' in text


def test_light_skeleton_preserves_safety_and_provenance():
    bundle=aes.build_read_only_evidence_bundle(opening_candidates=[{
        'source':'guitar','score_start':1.0,'expected_time':2.0,'audio_time':2.1,'confidence':.7}])
    original=deepcopy(bundle)
    light=crd.skeleton_audio_evidence(bundle)
    assert light['diagnostic_only'] is True
    assert light['timing_changes_applied'] is False
    assert light['midi_authority_changed'] is False
    assert light['detectors_executed'] is False
    assert light['evidence'][0]['detector']['module']=='process_gp_alignment'
    assert light['evidence'][0]['provenance']['origin_report']=='opening_anchor_candidates'
    assert bundle==original
    json.dumps(light)


def test_collector_discovers_audio_evidence_sidecar():
    root=Path(__file__).resolve().parents[1]
    text=(root/'collect_run_diagnostics.py').read_text(encoding='utf-8')
    assert "'*audio_evidence_bundle.json'" in text
    assert "if 'audio_evidence_bundle' in name" in text
