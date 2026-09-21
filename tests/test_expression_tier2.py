import expression_encoding as ee


def test_dead_and_explicit_fret_hand_mute_are_distinct():
    dead = ee.encode_alphatab({"techniques": {"dead": True}}, {"fret": 5, "sus": .5})
    hand = ee.encode_alphatab({"techniques": {"fret_hand_mute": True}}, {"fret": 5, "sus": .5})
    assert dead == {"mt": True}
    assert hand == {"fhm": True}


def test_tremolo_is_not_trill():
    trem = ee.encode_alphatab({"techniques": {}}, {"fret": 5, "sus": .5}, beat={"state": {"tremolo": True}})
    trill_note = {"techniques": {"trill_value": 7}}
    trill = ee.encode_alphatab(trill_note, {"fret": 5, "sus": .5})
    assert trem["tr"] is True
    assert "tr" not in trill
    assert "trill" in ee.notation_only_alphatab(trill_note)


def test_staccato_shortens_and_let_ring_uses_safe_boundary():
    assert ee.apply_sustain_semantics(1.0, staccato=True) == .5
    assert ee.apply_sustain_semantics(.5, let_ring=True, next_same_string_delta=2.0, segment_remaining=3.0) == 2.0
    assert ee.apply_sustain_semantics(.5, let_ring=True, next_same_string_delta=4.0, segment_remaining=3.0) == 3.0


def test_pick_direction_and_fingering_propagate():
    out = ee.encode_alphatab(
        {"techniques": {"left_hand_finger": {"name": "RingFinger"}}},
        {"fret": 5, "sus": .5},
        beat={"rhythm": {"pick_stroke": {"name": "Up"}}},
    )
    assert out["pkd"] == 1
    assert out["fg"] == 3
