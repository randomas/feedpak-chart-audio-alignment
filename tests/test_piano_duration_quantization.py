import process_gp_alignment as p

def notation():
    return {"source_track":"Piano", "measures":[{"staves":{"rh":{"voices":[{"beats":[
        {"t":1.0,"notes":[{"midi":60,"d":0.0312}]},
        {"t":2.0,"notes":[{"midi":62,"d":0.0316}]},
        {"t":3.0,"notes":[{"midi":64,"d":0.0504}]}
    ]}]}}}]}

def test_disabled_is_exact_and_uses_d():
    out=p.playable_keyboard_from_notation(notation(),0)
    assert [n["d"] for n in out["notes"]]==[0.0312,0.0316,0.0504]
    assert all("sus" not in n for n in out["notes"])

def test_one_ms_quantizes_only_duration():
    out=p.playable_keyboard_from_notation(notation(),1)
    assert [n["t"] for n in out["notes"]]==[1.0,2.0,3.0]
    assert [n["d"] for n in out["notes"]]==[0.031,0.032,0.05]
    d=out["piano_duration_diagnostics"]
    assert d["unique_before_count"]==3 and d["unique_after_count"]==3
