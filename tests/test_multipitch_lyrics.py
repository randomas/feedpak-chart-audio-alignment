import build_feedpak as b


def test_single_pitch_keeps_syllable_text_and_pitch_timing():
    words=[{"t":1.0,"d":0.5,"w":"word"}]
    notes=[{"t":1.0,"d":0.5,"midi":60}]
    assert b.expand_lyrics_with_pitch_continuations(words,notes)==[
        {"t":1.0,"d":0.5,"w":"word"}]


def test_multi_pitch_uses_plus_and_exact_pitch_blocks():
    words=[{"t":2.0,"d":0.6,"w":"eyes"}]
    notes=[{"t":2.0,"d":0.2,"midi":61},{"t":2.2,"d":0.15,"midi":64},
           {"t":2.35,"d":0.25,"midi":59}]
    assert b.expand_lyrics_with_pitch_continuations(words,notes)==[
        {"t":2.0,"d":0.2,"w":"eyes"},
        {"t":2.2,"d":0.15,"w":"+"},
        {"t":2.35,"d":0.25,"w":"+"}]


def test_hyphenated_syllable_retains_hyphen_then_plus():
    words=[{"t":3.0,"d":0.4,"w":"burn-"}]
    notes=[{"t":3.0,"d":0.2,"midi":60},{"t":3.2,"d":0.2,"midi":62}]
    got=b.expand_lyrics_with_pitch_continuations(words,notes)
    assert [x["w"] for x in got]==["burn-","+"]


def test_unpitched_syllable_is_preserved():
    assert b.expand_lyrics_with_pitch_continuations(
        [{"t":4.0,"d":0.1,"w":"a"}],[])==[{"t":4.0,"d":0.1,"w":"a"}]


def test_note_is_not_reused_by_adjacent_syllable():
    words=[{"t":0.0,"d":0.5,"w":"a-"},{"t":0.5,"d":0.5,"w":"part"}]
    notes=[{"t":0.0,"d":0.5,"midi":60},{"t":0.5,"d":0.5,"midi":62}]
    got=b.expand_lyrics_with_pitch_continuations(words,notes)
    assert [x["w"] for x in got]==["a-","part"]


















