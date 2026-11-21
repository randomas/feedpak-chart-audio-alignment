import json
from pathlib import Path
import keys_metadata as km
import transcription_investigator as ti

FIX=Path(__file__).parent/'fixtures'

def load():return json.loads((FIX/'metadata_piano_and_synth.json').read_text(encoding='utf-8'))

def test_configured_synth_audio_and_midi_resolve_case_insensitively(tmp_path):
    midi=tmp_path/'KEYS.MUSCRIPTOR.MID';midi.write_bytes(b'x')
    result=km.resolve_evidence_sources(load(),{'piano':'PaddedPiano.ogg','keys':'PaddedKeys.ogg'},tmp_path)
    assert result['piano']['physical_audio_source']=='piano'
    assert result['piano']['audio_path']=='PaddedPiano.ogg'
    assert result['synth']['physical_audio_source']=='other'
    assert result['synth']['audio_path'] is None
    # Fixture config uses synth.mid; an unrelated keys.muscriptor.mid must not be guessed.
    assert result['synth']['midi_path'] is None

def test_caught_in_web_metadata_routes_keys_stem_and_configured_midi(tmp_path):
    data=load();stream=data['feedpak_project']['tracks']['keys']['evidence_streams']['synth']
    stream['audio_source']='keys';stream['performance_midi']='keys.muscriptor.mid'
    midi=tmp_path/'KEYS.MUSCRIPTOR.MID';midi.write_bytes(b'x')
    result=km.resolve_evidence_sources(data,{'keys':'PreparedKeys.ogg'},tmp_path)['synth']
    assert result['logical_role']=='synth'
    assert result['physical_audio_source']=='keys'
    assert result['audio_path']=='PreparedKeys.ogg'
    assert Path(result['midi_path']).name=='KEYS.MUSCRIPTOR.MID'
    assert result['nominal_identity_verified'] is False
    assert result['diagnostic_only'] is True

def test_missing_sources_are_neutral_records(tmp_path):
    result=km.resolve_evidence_sources(load(),{},tmp_path)['synth']
    assert result['audio_available'] is False and result['midi_available'] is False
    assert result['audio_path'] is None and result['midi_path'] is None

def test_transcription_discovery_adds_configured_stream_without_changing_legacy(tmp_path):
    for name in ('Piano.muscriptor.mid','Bass.muscriptor.mid','synth.mid'):(tmp_path/name).write_bytes(b'x')
    configured={'synth':{'performance_midi':'synth.mid'}}
    found=ti.discover_prepared_files(tmp_path,configured)
    assert found['piano'].name=='Piano.muscriptor.mid'
    assert found['bass'].name=='Bass.muscriptor.mid'
    assert found['synth'].name=='synth.mid'
    assert found['guitar'] is None and found['drums'] is None

def test_k3_does_not_feed_synth_into_alignment_role_maps():
    root=Path(__file__).resolve().parents[1]
    process=(root/'process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'continuous_score_events["synth"]' not in process
    assert 'continuous_audio_paths["synth"]' not in process
    assert 'section_midi["synth"]' not in process
    assert 'keys_evidence_sources' not in process
    builder=(root/'build_feedpak.py').read_text(encoding='utf-8')
    assert '.keys_evidence_sources.json' in builder
