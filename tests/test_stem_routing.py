import build_feedpak as b

def test_explicit_cross_instrument_stem_mapping():
    paths={"guitar":"Guitar.ogg","piano":"Piano.ogg"}
    assert b.select_stem_path(paths,"guitar",("piano",))=="Guitar.ogg"

def test_source_can_be_disabled_without_removing_packaged_stem():
    paths={"bass":"Bass.ogg"}
    assert b.select_stem_path(paths,"none",("bass",)) is None
    assert paths["bass"]=="Bass.ogg"

def test_automatic_alias_fallback_is_preserved():
    paths={"keys":"Keys.ogg"}
    assert b.select_stem_path(paths,None,("piano","keys","keyboard"))=="Keys.ogg"



