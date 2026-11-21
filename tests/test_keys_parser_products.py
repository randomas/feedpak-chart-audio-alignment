import json
from pathlib import Path
import keys_metadata as km
import project_config as pc
import alphatab_score as ats

FIX=Path(__file__).parent/'fixtures'

def load(name):return json.loads((FIX/name).read_text(encoding='utf-8'))

def test_track_descriptors_preserve_separate_synth_arrangements_shared_stream():
    d=km.track_descriptors(load('metadata_piano_and_synth.json'))
    assert d['Synth Solo']['arrangement_id']=='synth_solo'
    assert d['Synth Pad']['arrangement_id']=='synth_pad'
    assert d['Synth Solo']['evidence_stream']==d['Synth Pad']['evidence_stream']=='synth'
    assert d['Piano LH']['evidence_stream']==d['Piano RH']['evidence_stream']=='piano'

def test_symbolic_deduplication_unions_duplicate_notes_but_keeps_unique_notes():
    rows=[
      {'t_gp':10.0,'end_gp':11.0,'midi':64,'source_track':'Synth Solo'},
      {'t_gp':10.0,'end_gp':12.0,'midi':64,'source_track':'Synth Pad'},
      {'t_gp':10.0,'end_gp':11.0,'midi':67,'source_track':'Synth Pad'},
      {'t_gp':11.0,'end_gp':12.0,'midi':64,'source_track':'Synth Solo'}]
    out=km.deduplicate_symbolic_events(rows)
    assert len(out)==3
    duplicate=next(x for x in out if x['t_gp']==10.0 and x['midi']==64)
    assert duplicate['source_tracks']==['Synth Pad','Synth Solo'] or duplicate['source_tracks']==['Synth Solo','Synth Pad']
    assert duplicate['duplicate_count']==2
    assert duplicate['end_gp']==12.0

def test_project_roles_remain_legacy_while_descriptors_add_keys_detail():
    data=load('metadata_piano_and_synth.json')
    inventory=[{'name':n,'automatic_role':'unsupported'} for n in ['Piano LH','Piano RH','Synth Solo','Synth Pad']]
    roles=pc.resolved_roles(data,inventory)
    assert roles['Piano LH']=='piano_left' and roles['Piano RH']=='piano_right'
    assert roles['Synth Solo']=='unsupported' and roles['Synth Pad']=='unsupported'
    assert km.track_descriptors(data)['Synth Solo']['family']=='keys'

def test_alphatab_and_gp_paths_expose_parser_neutral_products_and_legacy_aliases():
    root=Path(__file__).resolve().parents[1]
    alpha=(root/'alphatab_score.py').read_text(encoding='utf-8')
    gp=(root/'process_gp_alignment.py').read_text(encoding='utf-8')
    for text in (alpha,gp):
      assert 'keys_alignment_events_by_stream' in text
      assert 'keys_pitch_events_by_stream' in text
      assert 'keys_arrangements' in text
      assert 'piano_alignment_events' in text
      assert 'piano_pitch_events' in text

def test_k2_does_not_route_keys_products_into_alignment_methods_yet():
    root=Path(__file__).resolve().parents[1]
    text=(root/'process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'continuous_score_events["synth"]' not in text
    assert 'continuous_audio_paths["synth"]' not in text
    assert 'section_midi["synth"]' not in text
