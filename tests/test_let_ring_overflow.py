from types import SimpleNamespace as N
import expression_encoding as ee
import process_gp_alignment as pga

def test_overlong_let_ring_is_clipped_at_same_string_reattack():
    assert ee.apply_sustain_semantics(8.0,let_ring=True,next_same_string_delta=2.0,segment_remaining=12.0)==2.0

def test_short_let_ring_extends_to_first_interruption():
    assert ee.apply_sustain_semantics(.5,let_ring=True,next_same_string_delta=2.0,segment_remaining=3.0)==2.0

def test_segment_end_wins_before_next_attack():
    assert ee.apply_sustain_semantics(.5,let_ring=True,next_same_string_delta=4.0,segment_remaining=3.0)==3.0

def test_other_string_attack_does_not_clip():
    assert ee.apply_sustain_semantics(.5,let_ring=True)==.5

def test_gp5_parser_clips_at_next_same_string_attack(monkeypatch):
    monkeypatch.setattr(pga,"compute_anchors",lambda notes:[],raising=False)
    duration=N(time=960)
    ne=N(letRing=True,staccato=False,hammer=False,palmMute=False,ghostNote=False,fretHandMute=False,vibrato=False,accentuatedNote=False,heavyAccentuatedNote=False,slides=[],bend=None)
    be=N(letRing=True,palmMute=False,slapEffect=None,tremoloPicking=False,tremoloPickingEffect=None,pickStroke=None)
    beats=[N(start=0,duration=duration,notes=[N(string=1,value=0,type="normal",effect=ne)],effect=be),N(start=960,duration=duration,notes=[N(string=1,value=3,type="normal",effect=ne)],effect=be)]
    track=N(name="Guitar",strings=[N(number=1,value=64)],measures=[N(voices=[N(beats=beats)])],offset=0)
    out=pga.parse_fretted_track(track,[(0,120)],0)["notes"]
    assert out[0]["sus"]==.5
    assert out[0]["t_gp"]+out[0]["sus"]==out[1]["t_gp"]
