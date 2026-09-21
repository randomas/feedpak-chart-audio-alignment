import process_gp_alignment as pga

def test_radius_doubles_whole_song(monkeypatch):
    calls=[]
    def fake(*args):
        radius=args[-2]; calls.append(radius)
        return {"summary":{"strong":1 if radius>=4 else 0}}
    monkeypatch.setattr(pga,"diagnose_chroma_checkpoints",fake)
    out=pga.adaptive_checkpoint_diagnostic({}, {}, lambda t:t, [], {}, {"measured_boundaries":0},4,1.0)
    assert calls==[1.0,2.0,4.0]
    assert out["trustworthy"]==1 and out["expanded"] is True



