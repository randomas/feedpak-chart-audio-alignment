import json
from pathlib import Path
import pytest
import build_feedpak as b
import process_gp_alignment as p


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')


def source(work,track_id,name,hand,staff,duration_field='d'):
    play=work/'arrangements'/f'{track_id}.json'
    notation=work/'arrangements'/f'notation_{track_id}.json'
    dump(play,{'name':name,'tuning':[0]*6,'capo':0,
        'notes':[{'t':1.25,'s':2,'f':12,duration_field:0.5}],
        'chords':[],'anchors':[{'time':0.0,'fret':1,'width':4}],
        'handshapes':[],'templates':[]})
    dump(notation,{'version':1,'instrument':'piano','source_track':name,
        'configured_hand':hand,'staves':[{'id':staff,'clef':'F4' if staff=='lh' else 'G2','label':name}],
        'measures':[{'idx':1,'t':0.0,'staves':{staff:{'voices':[{'beats':[{'t':1.25,'dur':4,'notes':[{'midi':60}]}]}]}}}]})
    return {'id':track_id,'name':name,'type':'piano',
        'file':f'arrangements/{track_id}.json','notation':f'arrangements/notation_{track_id}.json',
        'tuning':[0]*6,'capo':0}


def test_split_classic_recreates_historical_identity_and_d_wire(tmp_path):
    rh=source(tmp_path,'piano_right_hand','piano_right_hand','right','rh','sus')
    lh=source(tmp_path,'piano_left_hand','piano_left_hand','left','lh','d')
    out=b.apply_split_classic_contract(tmp_path,[rh,lh])
    assert [x['id'] for x in out]==['piano__rh','piano__lh']
    for suffix,hand,staff,name in [('rh','right','rh','Piano (RH)'),('lh','left','lh','Piano (LH)')]:
        entry=next(x for x in out if x['id']==f'piano__{suffix}')
        assert entry['name']==name
        assert entry['file']==f'arrangements/piano__{suffix}.json'
        assert entry['notation']==f'arrangements/notation_piano__{suffix}.json'
        play=json.loads((tmp_path/entry['file']).read_text())
        assert play['name']==name and play['notes'][0]['d']==0.5
        assert 'sus' not in play['notes'][0]
        notation=json.loads((tmp_path/entry['notation']).read_text())
        assert notation['source_track']==name
        assert notation['configured_hand']==hand
        assert notation['staves'][0]['id']==staff
        assert set(notation['measures'][0]['staves'])=={staff}
    assert not (tmp_path/'arrangements/piano_right_hand.json').exists()
    assert not (tmp_path/'arrangements/piano_left_hand.json').exists()


def test_split_classic_requires_both_hands(tmp_path):
    rh=source(tmp_path,'piano_right_hand','piano_right_hand','right','rh')
    with pytest.raises(RuntimeError,match='requires both hands'):
        b.apply_split_classic_contract(tmp_path,[rh])


def test_split_classic_playable_adapter_uses_d():
    notation={'source_track':'Piano (RH)','measures':[{'staves':{'rh':{'voices':[{'beats':[{'t':1.0,'notes':[{'midi':60,'d':0.5}]}]}]}}}]}
    out=p.playable_keyboard_from_notation(notation,piano_rendering_policy='split-classic')
    assert out['notes']==[{'t':1.0,'s':2,'f':12,'d':0.5}]
    assert out['piano_duration_diagnostics']['rendering_policy']=='split-classic'
