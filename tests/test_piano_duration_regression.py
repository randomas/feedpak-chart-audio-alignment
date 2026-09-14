import alphatab_score as ats
import process_gp_alignment as pga

def test_alphatab_duration_percent_ratio_and_percent():
    assert ats._effective_duration_ticks({"duration_ticks":960,"duration_percent":.5}) == 480
    assert ats._effective_duration_ticks({"duration_ticks":960,"duration_percent":50}) == 480

def test_alphatab_duration_percent_legacy_fallback():
    assert ats._effective_duration_ticks({"duration_ticks":960},{"duration":{"percent":.25}}) == 240

def test_short_positive_piano_duration_is_preserved():
    notation={"source_track":"Piano","measures":[{"staves":{"rh":{"voices":[{"beats":[{"t":1.0,"notes":[{"midi":60,"d":.0312}]}]}]}}}]}
    assert pga.playable_keyboard_from_notation(notation)["notes"][0] == {"t":1.0,"s":2,"f":12,"d":.0312}

def test_nonlinear_warp_uses_both_piano_endpoints():
    notation={"source_track":"Piano","measures":[{"idx":1,"t_gp":0.0,"staves":{"rh":{"voices":[{"beats":[{"t_gp":1.0,"notes":[{"midi":60,"end_gp":1.5}]}]}]}}}]}
    w=pga.shift_and_warp_notation(notation,2.0,lambda t:t*t)
    n=w["measures"][0]["staves"]["rh"]["voices"][0]["beats"][0]["notes"][0]
    assert n["d"] == 3.25
