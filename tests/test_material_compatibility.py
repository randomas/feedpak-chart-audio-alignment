import json
import pipeline_cache as pc


def test_material_report_hashes_inputs_and_marks_legacy_midi_unverified(tmp_path):
    chart=tmp_path/'song.gp5'; chart.write_bytes(b'chart')
    piano=tmp_path/'piano.wav'; piano.write_bytes(b'audio')
    midi=tmp_path/'Piano.muscriptor.mid'; midi.write_bytes(b'midi')
    anchor=tmp_path/'metadata.json'; anchor.write_text('{}')
    out=pc.build_material_compatibility_report(chart_path=chart,
        audio_paths={'piano':piano},prepared_midi_files={'piano':midi},
        anchors_path=anchor,padding_added=2.0)
    assert out['chart']['sha256']
    assert out['audio']['piano']['sha256']
    assert out['prepared_midi']['piano']['provenance_status']=='legacy_unverified'
    assert out['status']=='legacy_unverified'
    assert len(out['material_set_sha256'])==64


def test_material_report_is_content_sensitive(tmp_path):
    chart=tmp_path/'song.gp5'; chart.write_bytes(b'chart')
    a=pc.build_material_compatibility_report(chart_path=chart)
    chart.write_bytes(b'changed')
    b=pc.build_material_compatibility_report(chart_path=chart)
    assert a['material_set_sha256']!=b['material_set_sha256']


def test_material_report_verifies_midi_sidecar_against_source_stem(tmp_path):
    chart=tmp_path/'song.gp5'; chart.write_bytes(b'chart')
    piano=tmp_path/'piano.wav'; piano.write_bytes(b'audio')
    midi=tmp_path/'Piano.muscriptor.mid'; midi.write_bytes(b'midi')
    source_hash=pc.file_identity(piano)['sha256']
    (tmp_path/'Piano.muscriptor.mid.provenance.json').write_text(json.dumps({
        'source_stem_sha256':source_hash,'provider':'muscriptor','provider_version':'test'}))
    out=pc.build_material_compatibility_report(chart_path=chart,
        audio_paths={'piano':piano},prepared_midi_files={'piano':midi})
    assert out['prepared_midi']['piano']['provenance_status']=='verified'
    assert out['status']=='verified'


def test_material_report_rejects_mismatched_midi_sidecar(tmp_path):
    chart=tmp_path/'song.gp5'; chart.write_bytes(b'chart')
    piano=tmp_path/'piano.wav'; piano.write_bytes(b'audio')
    midi=tmp_path/'Piano.muscriptor.mid'; midi.write_bytes(b'midi')
    (tmp_path/'Piano.muscriptor.mid.provenance.json').write_text(json.dumps({
        'source_stem_sha256':'0'*64}))
    out=pc.build_material_compatibility_report(chart_path=chart,
        audio_paths={'piano':piano},prepared_midi_files={'piano':midi})
    assert out['status']=='mismatch'
    assert 'prepared_midi_source_stem_mismatch' in out['warnings']
