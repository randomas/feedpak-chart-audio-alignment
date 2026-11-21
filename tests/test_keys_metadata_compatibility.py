import copy,json
from pathlib import Path
from types import SimpleNamespace as N
import keys_metadata as km
import project_config as pc

FIX=Path(__file__).parent/'fixtures'

def load(name):return json.loads((FIX/name).read_text(encoding='utf-8'))
def inv(*names):return [{'name':name,'automatic_role':'unsupported'} for name in names]

def test_legacy_piano_normalizes_without_changing_roles_or_input():
    data=load('metadata_legacy_piano_hands.json');original=copy.deepcopy(data)
    normalized=km.normalize_keys_configuration(data)
    assert data==original
    assert km.legacy_roles_unchanged(data)=={'Piano LH':'piano_left','Piano RH':'piano_right'}
    assert normalized['legacy_compatibility_applied'] is True
    assert normalized['arrangements'][0]['arrangement_id']=='piano'
    assert normalized['arrangements'][0]['staff_mode']=='grand'
    assert normalized['evidence_streams']['piano']['candidate_audio_source']=='piano'

def test_new_piano_and_synth_preserves_two_synth_arrangements_one_stream():
    data=load('metadata_piano_and_synth.json');n=km.normalize_keys_configuration(data)
    assert [x['arrangement_id'] for x in n['arrangements']]==['piano','synth_solo','synth_pad']
    assert n['evidence_streams']['synth']['chart_tracks']==['Synth Solo','Synth Pad']
    assert n['evidence_streams']['synth']['performance_midi']=='synth.mid'
    assert n['evidence_streams']['synth']['candidate_audio_source']=='other'
    assert n['evidence_streams']['synth']['deduplicate_symbolic_events'] is True

def test_project_config_legacy_roles_are_byte_equivalent_semantically():
    data=load('metadata_legacy_piano_hands.json')
    inventory=inv('Piano LH','Piano RH','Lead Guitar','Rhythm Guitar','Bass')
    assert pc.resolved_roles(data,inventory)=={
      'Piano LH':'piano_left','Piano RH':'piano_right','Lead Guitar':'guitar','Rhythm Guitar':'guitar','Bass':'bass'}
    result=pc.validate_song_config(data,inventory)
    assert result['valid'] is True and result['errors']==[] and result['warnings']==[]

def test_project_config_names_includes_new_keys_references():
    data=load('metadata_piano_and_synth.json');tracks=data['feedpak_project']['tracks']
    names=pc.names(tracks)
    assert 'Synth Solo' in names and 'Synth Pad' in names

def test_new_config_validates_available_tracks_and_stems():
    data=load('metadata_piano_and_synth.json')
    result=km.validate_keys_configuration(data,
      available_tracks=['Piano LH','Piano RH','Synth Solo','Synth Pad'],available_stems=['piano','other'])
    assert result['valid'] is True
    assert not any(x['code']=='configured_keys_audio_source_unavailable' for x in result['warnings'])

def test_missing_other_is_neutral_warning_not_error():
    data=load('metadata_piano_and_synth.json')
    result=km.validate_keys_configuration(data,
      available_tracks=['Piano LH','Piano RH','Synth Solo','Synth Pad'],available_stems=['piano'])
    assert result['valid'] is True
    assert {'code':'configured_keys_audio_source_unavailable','evidence_stream':'synth','audio_source':'other'} in result['warnings']

def test_missing_configured_track_is_clear_error():
    data=load('metadata_piano_and_synth.json')
    result=km.validate_keys_configuration(data,available_tracks=['Piano LH','Piano RH','Synth Solo'])
    assert result['valid'] is False
    assert any(x['code']=='configured_track_not_found' and x['track']=='Synth Pad' for x in result['errors'])

def test_conflicting_stream_assignment_is_rejected():
    data=load('metadata_piano_and_synth.json')
    data['feedpak_project']['tracks']['keys']['arrangements']['synth_pad']['evidence_stream']='pad'
    result=km.validate_keys_configuration(data,available_tracks=['Piano LH','Piano RH','Synth Solo','Synth Pad'])
    assert result['valid'] is False
    assert any(x['code']=='keys_track_in_conflicting_evidence_streams' and x['track']=='Synth Pad' for x in result['errors'])

def test_no_metadata_normalizes_to_empty_compatible_model():
    n=km.normalize_keys_configuration({})
    assert n['arrangements']==[] and n['evidence_streams']=={}
    assert n['input_schema_version']==1

def test_k3_wires_parsers_and_builder_but_not_alignment_engines():
    root=Path(__file__).resolve().parents[1]
    for name in ('alphatab_score.py','process_gp_alignment.py','build_feedpak.py'):
        assert 'import keys_metadata as km' in (root/name).read_text(encoding='utf-8')
    for name in ('midi_section_fit.py','continuous_anchors.py','structural_search.py','instrument_evidence.py'):
        text=(root/name).read_text(encoding='utf-8')
        assert 'import keys_metadata' not in text
        assert 'from keys_metadata' not in text
