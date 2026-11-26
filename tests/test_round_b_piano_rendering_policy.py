import copy
import pytest
import process_gp_alignment as p


def notation():
    return {'source_track':'Piano RH','measures':[{'staves':{'rh':{'voices':[{'beats':[
        {'t':1.0,'notes':[{'midi':60,'d':0.0312}]},
        {'t':2.0,'notes':[{'midi':64,'d':0.5}]}
    ]}]}}}]}


def test_compatibility_pulse_uses_sus_when_explicit():
    out=p.playable_keyboard_from_notation(notation(),piano_rendering_policy="compatibility-pulse")
    assert out['notes'][0]=={'t':1.0,'s':2,'f':12,'sus':0.0312}
    assert all('d' not in n for n in out['notes'])
    assert out['piano_duration_diagnostics']['rendering_policy']=='compatibility-pulse'
    assert out['piano_duration_diagnostics']['playable_duration_field']=='sus'


def test_duration_aware_uses_sus_only():
    out=p.playable_keyboard_from_notation(notation(),piano_rendering_policy='duration-aware')
    assert out['notes'][0]=={'t':1.0,'s':2,'f':12,'sus':0.0312}
    assert all('d' not in n for n in out['notes'])
    assert out['piano_duration_diagnostics']['playable_duration_field']=='sus'


def test_policies_preserve_onsets_pitch_count_and_duration_value():
    source=notation(); before=copy.deepcopy(source)
    legacy=p.playable_keyboard_from_notation(source,piano_rendering_policy='d-experimental')['notes']
    aware=p.playable_keyboard_from_notation(source,piano_rendering_policy='duration-aware')['notes']
    assert source==before
    assert len(legacy)==len(aware)==2
    for a,b in zip(legacy,aware):
        assert (a['t'],a['s'],a['f'],a['d'])==(b['t'],b['s'],b['f'],b['sus'])


def test_invalid_policy_fails():
    with pytest.raises(ValueError):
        p.playable_keyboard_from_notation(notation(),piano_rendering_policy='both')
