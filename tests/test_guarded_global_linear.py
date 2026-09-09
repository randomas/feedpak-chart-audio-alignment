import process_gp_alignment as p

def beats(step=2.5,n=4):
    return [{'t_gp':i*step,'measure':i+1,'ts_num':4} for i in range(n)]

def test_last_measure_end_extrapolates_final_bar():
    assert p.estimate_last_measure_end(beats()) == 10.0

def test_five_percent_gate_accepts_boundary(monkeypatch):
    monkeypatch.setattr(p,'detect_audio_content_end',lambda *a,**k:(10.5,{'available':True}))
    warp,r=p.build_guarded_global_span_warp(beats(), 'x', 0.0, minimum_scale=.95, maximum_scale=1.05)
    assert r['accepted'] and abs(r['applied_scale']-1.05)<1e-9
    assert abs(warp(10)-10.5)<1e-9

def test_five_percent_gate_rejects_excess(monkeypatch):
    monkeypatch.setattr(p,'detect_audio_content_end',lambda *a,**k:(10.51,{'available':True}))
    warp,r=p.build_guarded_global_span_warp(beats(), 'x', 0.0, minimum_scale=.95, maximum_scale=1.05)
    assert not r['accepted'] and r['applied_scale']==1.0
    assert warp(10)==10.0

def test_start_is_fixed(monkeypatch):
    b=[{'t_gp':2.5,'measure':1},{'t_gp':5.0,'measure':2},{'t_gp':7.5,'measure':3}]
    monkeypatch.setattr(p,'detect_audio_content_end',lambda *a,**k:(10.2,{'available':True}))
    warp,r=p.build_guarded_global_span_warp(b,'x',2.5)
    assert warp(2.5)==2.5
