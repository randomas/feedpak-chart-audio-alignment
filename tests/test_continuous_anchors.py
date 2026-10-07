import continuous_anchors as ca
import numpy as np

def bars():
    return [
        {"measure":1,"source_bar":1,"occurrence":1,"t_gp":0.0,"ts_num":4},
        {"measure":2,"source_bar":2,"occurrence":1,"t_gp":2.0,"ts_num":4},
        {"measure":3,"source_bar":3,"occurrence":1,"t_gp":4.0,"ts_num":4},
        {"measure":4,"source_bar":4,"occurrence":1,"t_gp":6.0,"ts_num":4},
    ]

def test_sustain_prevents_false_silence():
    r=ca.build_score_candidates({"guitar":[{"t_gp":0,"end_gp":3.5},{"t_gp":4,"end_gp":4.3}]},bars(),120)
    assert r["candidate_count"]==1

def test_full_measure_real_silence_is_retained():
    r=ca.build_score_candidates({"guitar":[{"t_gp":0,"end_gp":1},{"t_gp":4,"end_gp":4.3}]},bars(),120)
    assert r["candidate_count"]==2
    assert r["candidates"][1]["kind"]=="restart_after_measure_silence"
    assert r["candidates"][1]["score_silence_seconds"]==3.0

def test_chart_drives_corresponding_stem_search(monkeypatch):
    requested=[]
    monkeypatch.setattr(ca,"_features",lambda path,sr,hop:{"path":path} if path else None)
    def fake(features,prediction,source,cfg):
        requested.append((features["path"],prediction,source))
        return [{"time":prediction,"confidence":.9,"residual_seconds":0,"source":source,"kind":"silence_to_activity"}]
    monkeypatch.setattr(ca,"inspect_chart_event",fake)
    score={"guitar":[{"t_gp":0,"end_gp":.2},{"t_gp":3,"end_gp":3.2}]}
    report=ca.diagnose_continuous_anchors(score,{"guitar":"guitar.wav","drums":"drums.wav"},lambda t:t+1,bars(),120)
    assert requested==[("guitar.wav",1.0,"guitar"),("guitar.wav",4.0,"guitar")]
    assert report["search_strategy"]=="chart_first_corresponding_stem_local_search"
    assert report["timing_changes_applied"] is False

def test_strong_expected_stem_restart_needs_no_corroboration(monkeypatch):
    monkeypatch.setattr(ca,"_features",lambda path,sr,hop:{"path":path} if path else None)
    monkeypatch.setattr(ca,"inspect_chart_event",lambda features,prediction,source,cfg:[{"time":prediction,"confidence":.88,"residual_seconds":0,"source":source,"kind":"silence_to_activity"}])
    score={"guitar":[{"t_gp":0,"end_gp":.2},{"t_gp":3,"end_gp":3.2}]}
    report=ca.diagnose_continuous_anchors(score,{"guitar":"guitar.wav"},lambda t:t,bars(),120)
    assert report["matching"]["matches"][1]["best_candidate"]["single_stem_sufficient"] is True

def _candidate(time, confidence=.9, source='guitar', residual=0.0, single=True):
    return {'time':time,'confidence':confidence,'residual_seconds':residual,
            'source':source,'kind':'silence_to_activity','corresponding_stem':source,
            'evidence_source':source,'single_stem_sufficient':single}


def _match(event_id, score_time, candidates, source='guitar', kind='restart_after_measure_silence'):
    return {'score_event_id':event_id,'source':source,'kind':kind,'score_time':score_time,
            'score_position':{'playback_bar':int(score_time)},'candidates':candidates,
            'best_candidate':candidates[0] if candidates else None}


def _report(matches):
    return {'score_candidates':{'config':{'audio_search_radius_seconds':6.0}},
            'matching':{'matches':matches}}


def test_path_preview_is_monotonic_and_diagnostic_only():
    report=_report([
        _match('a',10,[_candidate(10.1),_candidate(14.0,.8)]),
        _match('b',20,[_candidate(20.2),_candidate(9.5,.99)]),
        _match('c',30,[_candidate(30.4)]),
    ])
    preview=ca.preview_monotonic_anchor_path(report,lambda t:t)
    times=[x['audio_time'] for x in preview['selected_anchors']]
    assert times==sorted(times)
    assert preview['validation']['monotonic'] is True
    assert preview['timing_changes_applied'] is False


def test_simultaneous_instruments_form_one_score_cluster_and_gain_support():
    report=_report([
        _match('g',10.00,[_candidate(11.00,.85,'guitar',1.0)],'guitar'),
        _match('d',10.08,[_candidate(11.05,.88,'drums',.97)],'drums'),
        _match('b',20.00,[_candidate(21.1,.84,'bass',1.1)],'bass'),
    ])
    preview=ca.preview_monotonic_anchor_path(report,lambda t:t)
    assert preview['score_cluster_count']==2
    first=preview['selected_anchors'][0]
    assert first['supporting_source_count']==2
    assert first['sources']==['drums','guitar']


def test_non_simultaneous_events_cannot_reuse_same_audio_candidate():
    report=_report([
        _match('a',10,[_candidate(12.0,.95)]),
        _match('b',20,[_candidate(12.04,.99),_candidate(22.0,.75)]),
    ])
    preview=ca.preview_monotonic_anchor_path(report,lambda t:t)
    chosen=[x['audio_time'] for x in preview['selected_anchors']]
    assert len(chosen)==len(set(chosen))
    assert not (12.0 in chosen and 12.04 in chosen)


def test_exact_manual_anchors_are_hard_interval_constraints():
    report=_report([
        _match('before',10,[_candidate(11.0)]),
        _match('after',30,[_candidate(31.0),_candidate(19.0,.99)]),
    ])
    manual={'anchors':[{'id':'manual','mode':'exact','nominal_time':20.0,
                        'padded_audio_time':20.5}]}
    preview=ca.preview_monotonic_anchor_path(report,lambda t:t,manual)
    assert preview['manual_hard_constraint_count']==1
    assert all(x['audio_time']<20.5 for x in preview['selected_anchors'] if x['score_time']<20)
    assert all(x['audio_time']>20.5 for x in preview['selected_anchors'] if x['score_time']>20)


def test_sparse_missing_event_can_be_skipped_without_breaking_path():
    report=_report([
        _match('a',10,[_candidate(10.0)]),
        _match('missing',20,[]),
        _match('c',30,[_candidate(31.0)]),
    ])
    preview=ca.preview_monotonic_anchor_path(report,lambda t:t)
    assert preview['selected_automatic_anchor_count']==2
    assert preview['skipped_cluster_count']==1


def test_moderate_tempo_regime_change_is_allowed_and_reported():
    report=_report([
        _match('a',0,[_candidate(0.0)]),
        _match('b',10,[_candidate(10.0)]),
        _match('c',20,[_candidate(21.0)]),
        _match('d',30,[_candidate(32.0)]),
    ])
    preview=ca.preview_monotonic_anchor_path(report,lambda t:t)
    scales=[x['implied_scale'] for x in preview['segments']]
    assert preview['validation']['monotonic'] is True
    assert all(.7<=x<=1.35 for x in scales)

def _beats_for_linear():
    return [
        {'measure':1,'source_bar':1,'occurrence':1,'t_gp':0.0,'ts_num':4},
        {'measure':2,'source_bar':2,'occurrence':1,'t_gp':10.0,'ts_num':4},
        {'measure':3,'source_bar':3,'occurrence':1,'t_gp':20.0,'ts_num':4},
        {'measure':4,'source_bar':4,'occurrence':1,'t_gp':30.0,'ts_num':4},
    ]


def _preview_for_linear(points, monotonic=True):
    return {
        'selected_anchors': [
            {'cluster_index':i,'score_time':s,'audio_time':a,'sources':['guitar'],
             'confidence':.9,'supporting_source_count':1}
            for i,(s,a) in enumerate(points)
        ],
        'validation': {'monotonic': monotonic},
    }


def test_continuous_linear_diagnostic_does_not_apply_timing():
    base=lambda t:t
    preview=_preview_for_linear([(0,0),(10,11),(20,22)])
    warp,report=ca.build_continuous_anchor_linear_warp(base,preview,_beats_for_linear(),apply=False)
    assert report['status']=='diagnostic_only'
    assert report['timing_changes_applied'] is False
    assert warp(15)==15
    assert report['comparison_by_playback_bar'][1]['proposed_time']==11.0


def test_continuous_linear_production_interpolates_and_holds_final_residual():
    base=lambda t:t
    preview=_preview_for_linear([(0,0),(10,10.5),(20,22.0)])
    warp,report=ca.build_continuous_anchor_linear_warp(base,preview,_beats_for_linear(),apply=True)
    assert report['status']=='applied'
    assert report['timing_changes_applied'] is True
    assert abs(warp(15)-16.25)<1e-9
    # After the final anchor, retain baseline slope with the +2 second residual.
    assert abs(warp(30)-32.0)<1e-9


def test_continuous_linear_manual_point_supersedes_nearby_automatic_control():
    base=lambda t:t
    preview=_preview_for_linear([(10.0,10.4),(20,20.5)])
    manual={'anchors':[{'id':'human','mode':'exact','nominal_time':10.0,
                        'padded_audio_time':11.0}]}
    warp,report=ca.build_continuous_anchor_linear_warp(
        base,preview,_beats_for_linear(),manual_anchor_quality=manual,apply=True)
    point=next(x for x in report['controls'] if abs(x['score_time']-10)<1e-9)
    assert point['kind']=='manual_exact'
    assert point['audio_time']==11.0
    assert warp(10)==11.0


def test_continuous_linear_invalid_path_falls_back_to_baseline():
    base=lambda t:t
    preview=_preview_for_linear([(10,20),(20,15)],monotonic=False)
    warp,report=ca.build_continuous_anchor_linear_warp(base,preview,_beats_for_linear(),apply=True)
    assert report['status']=='baseline_fallback_invalid_proposal'
    assert report['timing_changes_applied'] is False
    assert warp(15)==15


def test_force_invalid_continuous_warp_is_explicit_and_applied():
    baseline=lambda x: float(x)
    preview={"validation":{"monotonic":True},"selected_anchors":[
        {"score_time":0.0,"audio_time":0.0,"cluster_index":0,"sources":["guitar"],"confidence":0.9,"supporting_source_count":1},
        {"score_time":4.0,"audio_time":8.0,"cluster_index":1,"sources":["guitar"],"confidence":0.9,"supporting_source_count":1},
    ]}
    beats=[{"t_gp":0.0,"measure":1},{"t_gp":4.0,"measure":2}]
    safe, safe_report=ca.build_continuous_anchor_linear_warp(baseline,preview,beats,apply=True)
    forced, forced_report=ca.build_continuous_anchor_linear_warp(baseline,preview,beats,apply=True,force_invalid=True)
    assert safe_report["timing_changes_applied"] is False
    assert safe(4.0)==4.0
    assert forced_report["status"]=="forced_invalid_diagnostic"
    assert forced_report["forced_invalid_diagnostic"] is True
    assert forced_report["timing_changes_applied"] is True
    assert forced(4.0)==8.0


def test_score_fingerprint_contains_rhythm_and_sustained_chroma():
    events=[
        {"t_gp":0.0,"end_gp":1.5,"midi":60},
        {"t_gp":1.0,"end_gp":2.0,"midi":64},
        {"t_gp":2.0,"end_gp":2.5,"midi":67},
    ]
    fp=ca.build_score_fingerprint(events,1.25,1.25,bins=24)
    assert fp["event_count"]==3
    assert fp["pitched_event_count"]==3
    assert fp["chroma"][0].sum()>0
    assert fp["chroma"][4].sum()>0
    assert fp["chroma"][7].sum()>0


def test_temporal_prior_keeps_wide_candidates_possible_but_less_likely():
    cfg=ca.ContinuousAnchorConfig()
    near=ca._temporal_prior(.1,cfg)
    far=ca._temporal_prior(3.0,cfg)
    assert 0 < far < near <= 1


def test_dense_fingerprint_requires_both_rhythm_and_chroma():
    cfg=ca.ContinuousAnchorConfig(minimum_absolute_margin=0.0,minimum_score_ratio=1.0,
                                  minimum_posterior=0.0,minimum_rhythm_similarity=.1,
                                  minimum_chroma_similarity=.9,minimum_joint_identity=.1)
    times=np.linspace(0,10,1001)
    features={"times":times,"onset":np.sin(times*3)**2,"chroma_times":times,
              "chroma":np.zeros((12,len(times))),"sr":22050}
    events=[{"t_gp":4.5,"end_gp":5.0,"midi":60},{"t_gp":5.0,"end_gp":5.5,"midi":64}]
    item={"score_time":5.0,"source":"guitar","score_confidence":1.0}
    assert ca.inspect_dense_fingerprint(features,events,item,5.0,2.0,cfg)==[]

def test_force_mode_never_applies_a_non_monotonic_proposal():
    baseline = lambda value: float(value)
    # Simulate a malformed externally supplied preview: its validation flag is
    # true, but the final control geometry is non-monotonic. The final proposal
    # validator, not the preview flag, must remain the hard safety authority.
    preview = _preview_for_linear([(4.0, 4.1), (8.0, 3.9)])
    beats = [
        {"t_gp": 0.0, "measure": 1},
        {"t_gp": 4.0, "measure": 2},
        {"t_gp": 8.0, "measure": 3},
    ]

    warp, report = ca.build_continuous_anchor_linear_warp(
        baseline, preview, beats, apply=True, force_invalid=True
    )

    assert report["validation"]["preview_path_monotonic"] is True
    assert report["validation"]["proposal_monotonic"] is False
    assert report["status"] == "baseline_fallback_force_blocked_non_monotonic"
    assert report["force_requested"] is True
    assert report["forced_invalid_diagnostic"] is False
    assert report["force_blocked_non_monotonic"] is True
    assert report["force_blocked_reason"] == "proposal_not_monotonic"
    assert report["timing_changes_applied"] is False
    assert report["validation"]["fallback_used"] is True
    assert warp(4.0) == baseline(4.0)

