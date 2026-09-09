import process_gp_alignment as pga

def test_rb3_pitch_mapping():
    assert pga.midi_to_virtual_position(21)==(0,21)
    assert pga.midi_to_virtual_position(24)==(1,0)
    assert pga.midi_to_virtual_position(60)==(2,12)
    assert pga.midi_to_virtual_position(108)==(4,12)

def test_piano_fixed_anchor():
    notation={"source_track":"Piano (RH)","measures":[{"staves":{"rh":{"voices":[{"beats":[{"t":2.0,"notes":[{"midi":60,"d":0.5}]}]}]}}}]}
    out=pga.playable_keyboard_from_notation(notation)
    assert out["anchors"]==[{"time":0.0,"fret":1,"width":4}]
    assert out["notes"]==[{"t":2.0,"s":2,"f":12,"d":0.5}]















