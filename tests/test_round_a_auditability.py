import csv,json
from pathlib import Path
import pytest
import audit_reports as ar
import pipeline_cache as pc


def test_runtime_contract_fails_closed():
    c=ar.build_runtime_contract(requested={'vocals_mode':'required'},resolved={},delivered={'vocals':False},
                                failures=[{'reason':'required_vocals_unavailable'}])
    assert c['status']=='failed'
    with pytest.raises(RuntimeError):ar.require_runtime_contract(c)


def test_runtime_contract_satisfied():
    c=ar.build_runtime_contract(requested={'vocals_mode':'off'},resolved={},delivered={'vocals':False})
    ar.require_runtime_contract(c)
    assert c['status']=='satisfied'


def test_cache_dependency_classification_and_anchor_fingerprints():
    out=pc.build_cache_compatibility([
        {'layer':'prepared_midi','status':'exact_cache_hit'},
        {'layer':'section_materialization','status':'cache_file_missing'}],
        anchor_fingerprint='a',baseline_fingerprint='b')
    assert out['layers'][0]['dependency']=='anchor_independent'
    assert 'anchor_fingerprint' not in out['layers'][0]
    assert out['layers'][1]['dependency']=='anchor_dependent'
    assert out['layers'][1]['anchor_fingerprint']=='a'


def test_compact_audit_bundle(tmp_path):
    folder=tmp_path/'01_song';folder.mkdir()
    (folder/'x.feedpak').write_bytes(b'x')
    (folder/'x.feedpak.method_result_envelope.json').write_text(json.dumps({
        'method_id':'continuous-midi-landmark','status':'complete_with_warnings','warning_count':2,
        'listening_review_points':[{'severity':'manual_review','message':'check'}]}))
    (folder/'x.feedpak.beatmap_report.json').write_text(json.dumps({'manual_anchor_verification':{
        'exported_manual_anchor_status':'satisfied','maximum_exported_manual_anchor_error_ms':0}}))
    (folder/'x.feedpak.material_compatibility.json').write_text(json.dumps({'status':'verified','material_set_sha256':'abc'}))
    (folder/'x.feedpak.runtime_contract.json').write_text(json.dumps({'requested':{'vocals_mode':'off','anchors':'m.json'},'delivered':{'vocals':False}}))
    out=ar.write_comparison_audit(tmp_path)
    assert (out/'method_scorecard.csv').is_file()
    rows=list(csv.DictReader((out/'method_scorecard.csv').open(encoding='utf-8-sig')))
    assert rows[0]['feedpak_produced']=='True'
    assert rows[0]['anchors_finally_satisfied']=='satisfied'
    assert (out/'final_anchor_verification.json').is_file()
