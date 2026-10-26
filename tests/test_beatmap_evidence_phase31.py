import inspect
import beatmap_evidence as be


def trend(n=8, span=100, slope=1.5, offset=0):
    samples=[]
    for i in range(n):
        t=i*span/(n-1)
        samples.append({"audio_time":t,"residual_ms":offset+slope*t,"source_roles":["drums"],
                        "evidence_family":"rhythmic_onset"})
    return be._trend(samples)


def test_minor_short_variation_is_not_drift():
    t=trend(n=5,span=20,slope=1.5)
    result=be._interpret({"rhythmic_onset":t},{"drums":t},t,{"direction_reversal":False},{"source_weighting_conflict":False})
    assert result["classification"]=="minor_local_variation"
    assert result["recommended_alignment_strategy"]=="retain_global_map"


def test_long_accumulated_family_supported_drift():
    t=trend(n=9,span=110,slope=1.5)
    result=be._interpret({"rhythmic_onset":t,"path_hypothesis":t},{"drums":t},t,
                         {"direction_reversal":False},{"source_weighting_conflict":False})
    assert result["classification"]=="possible_drift"
    assert result["confidence"]=="high"
    assert result["recommended_alignment_strategy"]=="test_piecewise_regional_warp"


def test_source_scale_conflict_recommends_fusion():
    report={"decision":{"instrument_scale":{"candidates":[
        {"source":"drums","scale":1.009,"accepted":True},
        {"source":"bass","scale":0.989,"accepted":True}]}}}
    scale=be.source_scale_diagnostics(report)
    assert scale["source_weighting_conflict"] is True
    t=trend(n=8,span=100,slope=.2)
    result=be._interpret({"rhythmic_onset":t},{"drums":t,"bass":t},t,
                         {"direction_reversal":False},scale)
    assert result["evidence_behavior"]=="source_weighting_conflict"
    assert result["recommended_alignment_strategy"]=="test_regional_source_fusion"


def test_direction_reversal_recommends_structural_test():
    t=trend(n=8,span=100,slope=.2)
    result=be._interpret({"rhythmic_onset":t},{},t,{"direction_reversal":True},{"source_weighting_conflict":False})
    assert result["classification"]=="possible_nonlinear_timing"
    assert result["structural_correspondence"]=="possible_insertion_or_deletion"
    assert result["recommended_alignment_strategy"]=="test_structural_jump_then_regional_alignment"


def test_module_is_song_independent():
    source=inspect.getsource(be).lower()
    for name in ("all that i bleed","spellbound","against you","supremacy","don't stop me now","sound of love"):
        assert name not in source
