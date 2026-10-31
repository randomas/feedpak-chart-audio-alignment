import incipit_three_clock as itc

def opening_region():
    return {"region_id":"opening","chart_start":3.,"chart_end":20.,"midi_start":5.,"midi_end":22.,
      "controls":[
       {"score_time":3.,"audio_time":5.,"roles":["piano"],"provenance":[{"role":"piano","window_id":"w0"}]},
       {"score_time":12.,"audio_time":14.,"roles":["piano"],"provenance":[{"role":"piano","window_id":"w1"}]}]}

def test_audio_corrects_transcription_lag_but_keeps_identity():
    fusion={"regions":[opening_region()]};chart={"piano":[{"t_gp":3.,"end_gp":20.}]}
    itc.reconcile_opening(fusion,chart,{"piano":3.1})
    r=fusion["regions"][0]
    assert fusion["three_clock_incipit"]["status"]=="validated"
    assert abs(r["midi_start"]-3.1)<1e-9
    assert abs(r["controls"][0]["raw_transcription_midi_time"]-5.)<1e-9
    assert r["acquisition_root"]["authority"]=="three_clock_role_incipit"

def test_one_candidate_cannot_establish_root():
    r=opening_region()
    for c in r["controls"]:c["provenance"]=[{"role":"piano","window_id":"same"}]
    fusion={"regions":[r]};itc.reconcile_opening(fusion,{"piano":[{"t_gp":3.}]},{"piano":3.})
    assert fusion["three_clock_incipit"]["status"]=="no_opening_candidate"

def test_late_repeat_cannot_establish_opening_root():
    r=opening_region();r["chart_start"]=30.;r["chart_end"]=45.
    fusion={"regions":[r]};itc.reconcile_opening(fusion,{"piano":[{"t_gp":3.}]},{"piano":3.})
    assert fusion["three_clock_incipit"]["status"]=="no_opening_candidate"

def test_audio_chart_disagreement_blocks_root():
    fusion={"regions":[opening_region()]};itc.reconcile_opening(fusion,{"piano":[{"t_gp":3.}]},{"piano":6.})
    assert fusion["three_clock_incipit"]["status"]=="no_opening_candidate"
