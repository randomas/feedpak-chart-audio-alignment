from types import SimpleNamespace as N
import expression_encoding as ee
import process_gp_alignment as pga


def test_normal_short_note_has_explicit_feedforge_style_gate():
    assert ee.apply_sustain_semantics(0.4) == 0.38


def test_staccato_and_dead_note_release_rules():
    assert ee.apply_sustain_semantics(1.0, staccato=True) == 0.475
    assert ee.apply_sustain_semantics(1.0, dead=True) == 0.1


def test_let_ring_is_not_shortened():
    assert ee.apply_sustain_semantics(0.4, let_ring=True) == 0.4


def test_fretted_warp_maps_both_endpoints_and_keeps_short_sus():
    notes=[{"t_gp":1.0,"s":0,"f":5,"sus":0.1}]
    out=pga.apply_fretted_note_warp(notes,lambda t:t*t)
    assert out == [{"s":0,"f":5,"sus":0.21,"t":1.0}]


def test_bend_curve_relative_times_follow_warped_sustain():
    notes=[{"t_gp":1.0,"s":0,"f":5,"sus":1.0,"bnv":[{"t":0.0,"v":0},{"t":1.0,"v":2}]}]
    out=pga.apply_fretted_note_warp(notes,lambda t:2*t)[0]
    assert out["sus"] == 2.0
    assert out["bnv"][-1]["t"] == 2.0


def test_gp5_parser_emits_sus_for_short_notes(monkeypatch):
    monkeypatch.setattr(pga, "compute_anchors", lambda notes: [], raising=False)
    duration=N(time=120)
    effect=N(letRing=False,staccato=False,hammer=False,palmMute=False,ghostNote=False,
             fretHandMute=False,vibrato=False,accentuatedNote=False,heavyAccentuatedNote=False,
             slides=[],bend=None)
    beat_effect=N(letRing=False,palmMute=False,slapEffect=None,tremoloPicking=False,
                  tremoloPickingEffect=None,pickStroke=None)
    note=N(string=1,value=3,type="normal",effect=effect)
    beat=N(start=0,duration=duration,notes=[note],effect=beat_effect)
    measure=N(voices=[N(beats=[beat])])
    track=N(name="Guitar",strings=[N(number=1,value=64)],measures=[measure],offset=0)
    out=pga.parse_fretted_track(track,[(0,120)],0)
    assert "sus" in out["notes"][0]
    assert 0.03 <= out["notes"][0]["sus"] < 0.1
