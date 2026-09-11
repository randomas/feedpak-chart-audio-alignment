import numpy as np
import process_gp_alignment as p


def test_sustained_activity_rejects_single_transient():
    times=np.arange(0,2.0,0.1); rms=np.zeros_like(times)+0.001
    rms[3]=1.0
    out=p._activity_span_from_features(rms,times,np.zeros_like(times),times)
    assert not out["available"]


def test_sustained_activity_finds_span_and_strong_onset():
    times=np.arange(0,3.0,0.1); rms=np.zeros_like(times)+0.001
    rms[10:20]=0.5
    onset=np.zeros_like(times); onset[10]=1.0
    out=p._activity_span_from_features(rms,times,onset,times)
    assert out["available"]
    assert 0.9 <= out["first_sustained_activity"] <= 1.1
    assert out["first_strong_onset"]==1.0


def test_missing_audio_is_diagnostic_only_and_timing_neutral():
    old=p.librosa; p.librosa=None
    try:
        out=p.diagnose_track_audio_coverage([10.0,11.0],None,lambda t:t+2)
    finally:
        p.librosa=old
    assert out["status"]=="diagnostic_only"
    assert out["gp"]["first_note_time"]==12.0
    assert not out["audio"]["available"]


















