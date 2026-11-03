import numpy as np
import bar_landmark_alignment as bla

def test_landmarks_choose_first_event_inside_each_bar():
    beats=[{"measure":1,"t_gp":0.0},{"measure":2,"t_gp":2.0},{"measure":3,"t_gp":4.0}]
    events={"drums":[{"t_gp":0.1},{"t_gp":2.2},{"t_gp":4.3}],"guitar":[{"t_gp":0.12}]}
    bars=bla._bar_landmarks(events,beats,0.4)
    assert [round(x["score_landmark_time"],2) for x in bars]==[0.1,2.2,4.3]
    assert bars[0]["score_roles"]==["drums","guitar"]

def test_residual_smoothing_rejects_single_bar_spike():
    cfg=bla.BarTrackingConfig(smoothing_window_bars=3,maximum_residual_step_seconds=.1)
    raw=np.array([.0,.05,.1,1.8,.15,.2,.25])
    fitted=bla._smooth_residuals(raw,np.ones(len(raw)),cfg)
    assert fitted[3] < .4
    assert np.max(np.abs(np.diff(fitted))) <= .100001

def test_piecewise_warp_is_monotonic():
    warp=bla.PiecewiseBarWarp([0,2,4],[1,3.1,5.0])
    values=[warp(x) for x in np.linspace(0,4,30)]
    assert all(b>a for a,b in zip(values,values[1:]))

def test_insufficient_observations_keeps_baseline(monkeypatch):
    monkeypatch.setattr(bla,"_load_onsets",lambda *a,**k:None)
    baseline=lambda t:t+1
    beats=[{"measure":1,"t_gp":0.0},{"measure":2,"t_gp":2.0}]
    warp,report=bla.build_bar_landmark_warp(baseline,{"drums":[]},{},beats)
    assert warp is baseline
    assert report["status"]=="insufficient_observations"
