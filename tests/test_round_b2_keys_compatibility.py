import json
from pathlib import Path
import build_feedpak as b


def note_file(path,name,notes):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'name':name,'tuning':[0]*6,'capo':0,'notes':notes,
        'chords':[],'anchors':[],'handshapes':[],'templates':[]}),encoding='utf-8')


def notation(hand,staff,note):
    return {'version':1,'hand':hand,'measures':[{'idx':1,'t':0.0,
        'staves':{staff:{'voices':[{'beats':[{'t':note['t'],'notes':[{'midi':60,'d':1.0}]}]}]}}}]}


def test_compatibility_writer_merges_hands_and_caps_to_local_sixteenth(tmp_path):
    work=tmp_path;arr=work/'arrangements'
    note_file(arr/'piano_right_hand.json','RH',[{'t':0.0,'s':2,'f':12,'sus':1.0}])
    note_file(arr/'piano_left_hand.json','LH',[{'t':1.0,'s':1,'f':10,'sus':0.05}])
    entries=[{'id':'piano_right_hand','name':'RH','type':'piano','file':'arrangements/piano_right_hand.json'},
             {'id':'piano_left_hand','name':'LH','type':'piano','file':'arrangements/piano_left_hand.json'}]
    gp={'song_timeline':{'tempos':[{'time':0.0,'bpm':120.0}]},
        'notation':{'piano_right_hand':notation('right','rh',{'t':0.0}),
                    'piano_left_hand':notation('left','lh',{'t':1.0})}}
    out=b.write_merged_keys_compatibility(gp,work,entries)
    assert out==[{'id':'keys','name':'Keys','type':'piano','file':'arrangements/keys.json','notation':'arrangements/notation_keys.json'}]
    keys=json.loads((arr/'keys.json').read_text())
    assert keys['notes']==[{'t':0.0,'s':2,'f':12,'sus':0.125},{'t':1.0,'s':1,'f':10,'sus':0.05}]
    assert keys['anchors']==[{'time':0.0,'fret':1,'width':4}]
    merged=json.loads((arr/'notation_keys.json').read_text())
    assert merged['instrument']=='piano'
    assert set(merged['measures'][0]['staves'])=={'rh','lh'}
    assert not (arr/'piano_right_hand.json').exists()
    assert not (arr/'piano_left_hand.json').exists()


def test_pulse_has_safety_limits():
    assert b._bpm_at_time([],0)==120.0
    assert b._bpm_at_time([{'time':0,'bpm':10}],0)==20.0
    assert b._bpm_at_time([{'time':0,'bpm':999}],0)==400.0
