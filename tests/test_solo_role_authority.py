import solo_role_authority as sra


def test_only_expected_role_can_be_strong_and_root_opening():
    controls=[
      {"score_time":3.,"audio_time":5.,"roles":["piano"],"classification":"single_role_provisional","confidence":.9},
      {"score_time":10.,"audio_time":12.,"roles":["piano"],"classification":"single_role_provisional","confidence":.85}]
    fusion={"regions":[{"region_id":"opening","chart_start":3.,"chart_end":10.,"midi_start":5.,"midi_end":12.,"controls":controls}]}
    chart={"piano":[{"t_gp":3.,"end_gp":11.}],"bass":[],"drums":[],"guitar":[]}
    sra.annotate_fusion_report(fusion,chart)
    assert all(x["classification"]=="solo_role_expected_strong" for x in controls)
    assert fusion["regions"][0]["acquisition_root"]["validated"]


def test_inactive_roles_are_neutral():
    chart={"piano":[{"t_gp":1.,"end_gp":5.}],"drums":[{"t_gp":20.,"end_gp":25.}]}
    assert sra.expected_roles(chart,1.,4.)==["piano"]
    assert sra.expected_roles(chart,21.,22.)==["drums"]


def test_active_second_role_prevents_solo_promotion():
    controls=[{"score_time":3.,"audio_time":5.,"roles":["piano"],"classification":"single_role_provisional","confidence":.9}]
    fusion={"regions":[{"region_id":"ensemble","chart_start":3.,"chart_end":4.,"midi_start":5.,"midi_end":6.,"controls":controls}]}
    chart={"piano":[{"t_gp":3.,"end_gp":5.}],"drums":[{"t_gp":3.,"end_gp":5.}]}
    sra.annotate_fusion_report(fusion,chart)
    assert controls[0]["classification"]=="single_role_provisional"


def test_invalid_negative_baseline_uses_validated_root():
    fusion={"regions":[{"region_id":"opening","chart_start":3.,"midi_start":5.,"acquisition_root":{"validated":True},"controls":[{"score_time":3.},{"score_time":10.}]}]}
    warp,report=sra.safe_materialization_baseline(fusion,lambda t:t-100.)
    assert report["fallback"]=="acquisition_root_identity_clock"
    assert warp(0.)==2. and warp(3.)==5.


def test_invalid_baseline_without_root_blocks():
    fusion={"regions":[{"region_id":"late","chart_start":30.,"midi_start":32.,"acquisition_root":{"validated":False},"controls":[{"score_time":30.},{"score_time":40.}]}]}
    _,report=sra.safe_materialization_baseline(fusion,lambda t:t-100.)
    assert not report["valid"]
    assert report["blocked_reason"]=="invalid_baseline_without_acquisition_root"
