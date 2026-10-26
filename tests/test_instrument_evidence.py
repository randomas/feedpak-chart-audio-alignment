import numpy as np
import instrument_evidence as ie


def feature_for(events, residual, start=0.0, end=4.0, step=0.01):
    times=np.arange(start,end+step,step)
    novelty=np.zeros_like(times)
    for event in events:
        novelty += np.exp(-.5*((times-(event+residual))/.025)**2)
    novelty/=max(novelty.max(),1e-9)
    return {"status":"ok","times":times,"novelty":novelty}


def test_profiles_are_instrument_optimized_not_song_specific():
    assert ie.profile_for("drums").feature_family=="multiband_transient_pattern"
    assert ie.profile_for("bass").fmax==1200.0
    assert ie.profile_for("piano").harmonic_change_weight>ie.profile_for("drums").harmonic_change_weight
    assert ie.profile_for("guitar").spectral_flux_weight>ie.profile_for("bass").spectral_flux_weight


def test_measure_pattern_recovers_known_residual():
    events=[.4,1.2,2.0,3.1]
    result=ie.match_measure_pattern("drums",0,4,events,lambda x:x,
                                    feature_for(events,.12),1)
    assert result["accepted"] is True
    assert abs(result["residual_seconds"]-.12)<.025


def test_ambiguous_flat_audio_is_rejected():
    times=np.arange(0,4.01,.01)
    feature={"status":"ok","times":times,"novelty":np.ones_like(times)}
    result=ie.match_measure_pattern("guitar",0,4,[.5,1.5,2.5],lambda x:x,feature,1)
    assert result["accepted"] is False
    assert result["reason"]=="ambiguous_pattern_match"


def test_adapter_observations_keep_role_and_family():
    report={"roles":{"bass":{"feature_family":"low_frequency_attack_pattern","measures":[
        {"playback_measure":4,"accepted":True,"residual_ms":25.0,"best_score":.8,
         "score_event_count":3}]}}}
    obs=ie.to_shared_observations(report)[0]
    assert obs["evidence_family"]=="instrument_pattern"
    assert obs["source_roles"]==["bass"]
    assert obs["subtype"]=="low_frequency_attack_pattern"


def test_report_is_explicitly_diagnostic_only(monkeypatch):
    monkeypatch.setattr(ie,"extract_role_feature",lambda *a,**k:{"status":"unavailable","reason":"test"})
    report=ie.analyze_all({"drums":{"events":[0.1],"audio_path":"x"}},lambda x:x,
                          [{"t_gp":0,"measure":1},{"t_gp":2,"measure":2}])
    assert report["timing_changes_applied"] is False
