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

def test_non_let_ring_chord_texture_boundary_clips_cross_string_ring():
    assert ee.apply_sustain_semantics(.5, let_ring=True, next_same_string_delta=40.0, texture_boundary_delta=1.25, segment_remaining=80.0) == 1.25

def test_single_other_string_attack_remains_non_interrupting():
    assert ee.apply_sustain_semantics(.5, let_ring=True, next_same_string_delta=2.0, texture_boundary_delta=None) == 2.0

def test_gp5_merged_part_chord_interrupts_disjoint_string_let_ring(monkeypatch):
    monkeypatch.setattr(pga,"compute_anchors",lambda notes:[],raising=False)
    duration=N(time=960)
    ringing=N(letRing=True,staccato=False,hammer=False,palmMute=False,ghostNote=False,fretHandMute=False,vibrato=False,accentuatedNote=False,heavyAccentuatedNote=False,slides=[],bend=None)
    normal=N(letRing=False,staccato=False,hammer=False,palmMute=False,ghostNote=False,fretHandMute=False,vibrato=False,accentuatedNote=False,heavyAccentuatedNote=False,slides=[],bend=None)
    ring_beat=N(letRing=True,palmMute=False,slapEffect=None,tremoloPicking=False,tremoloPickingEffect=None,pickStroke=None)
    chord_beat=N(letRing=False,palmMute=True,slapEffect=None,tremoloPicking=False,tremoloPickingEffect=None,pickStroke=None)
    beats=[N(start=0,duration=duration,notes=[N(string=1,value=0,type="normal",effect=ringing)],effect=ring_beat),N(start=1920,duration=duration,notes=[N(string=2,value=5,type="normal",effect=normal),N(string=3,value=7,type="normal",effect=normal)],effect=chord_beat),N(start=7680,duration=duration,notes=[N(string=1,value=3,type="normal",effect=ringing)],effect=ring_beat)]
    strings=[N(number=i,value=64-(i-1)*5) for i in range(1,4)]
    track=N(name="Merged Rhythm",strings=strings,measures=[N(voices=[N(beats=beats)])],offset=0)
    out=pga.parse_fretted_track(track,[(0,120)],0)["notes"]
    assert out[0]["sus"] == 1.0
