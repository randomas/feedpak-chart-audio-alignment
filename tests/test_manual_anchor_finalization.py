import manual_anchors as ma
import method_orchestration as mo


def quality(*rows):
    return {'anchors':[{'id':i,'mode':'exact','nominal_time':x,'padded_audio_time':y}
                       for i,x,y in rows]}


def bars():
    return [{'t_gp':0.0},{'t_gp':10.0},{'t_gp':20.0},{'t_gp':30.0}]


def test_finalizer_preserves_shape_and_hits_internal_anchor():
    method=lambda t: float(t)+0.01*float(t)*float(t)
    warp,report=ma.constrain_warp_to_exact_anchors(method,quality(('C',20.0,23.0)),bars())
    assert report['strategy']=='shape_preserving_residual'
    assert abs(warp(20)-23.0)<1e-12
    # The method remains non-linear; finalization is not a flat manual-only line.
    assert abs((warp(15)-warp(10))-(warp(10)-warp(5)))>1e-4
    verified=mo.verify_manual_anchors(warp,{'quality':quality(('C',20.0,23.0))},'test')
    assert verified['manual_anchor_status']=='satisfied'


def test_exported_beatmap_verification_interpolates_fractional_beat_anchor():
    beatmap={'beats':[{'timing':{'score_time':0.0,'audio_time':2.0}},
                      {'timing':{'score_time':10.0,'audio_time':12.0}},
                      {'timing':{'score_time':20.0,'audio_time':22.0}}]}
    out=ma.verify_exported_beatmap(beatmap,quality(('half',15.0,17.0)))
    assert out['exported_manual_anchor_status']=='satisfied'
    assert out['maximum_exported_manual_anchor_error_ms']==0


def test_exported_beatmap_anchor_failure_is_visible():
    beatmap={'beats':[{'timing':{'score_time':0.0,'audio_time':0.0}},
                      {'timing':{'score_time':20.0,'audio_time':20.0}}]}
    out=ma.verify_exported_beatmap(beatmap,quality(('C',10.0,12.0)))
    assert out['exported_manual_anchor_status']=='export_failed'
    assert out['exported_manual_anchor_conflicts']
