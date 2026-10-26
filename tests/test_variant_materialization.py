import variant_materialization as vm

def report():
    return {"decision":{"continuous_anchor_linear":{"status":"diagnostic_only","comparison_by_playback_bar":[
        {"baseline_time":0.0,"proposed_time":0.0},{"baseline_time":10.0,"proposed_time":11.0},{"baseline_time":20.0,"proposed_time":22.0}]}}}

def test_controls_make_monotonic_audio_time_map():
    controls,_=vm.continuous_comparison_controls(report());w=vm.AudioTimeMap(controls)
    assert w(5)==5.5 and w(25)==27.0
    assert all(w(b)>w(a) for a,b in zip(range(20),range(1,21)))

def test_event_duration_warps_both_endpoints():
    w=vm.AudioTimeMap([(0,0),(10,11),(20,22)])
    out=vm.rewrite_json_document("arrangements/guitar.json",{"notes":[{"t":5,"sus":2}]},w)
    assert out["notes"][0]=={"t":5.5,"sus":2.2}

def test_continuous_variant_requires_no_analysis_entrypoint():
    import inspect
    source=inspect.getsource(vm.materialize_continuous_variant)
    assert "process_gp_alignment" not in source
    assert "run_gp_script" not in source
