import audio_evidence_qualification as aeq
import evidence_contribution_ledger as ecl
import occurrence_audio_qualification as oaq


def observation(eid="ev-1", source="mix", adapter="complete", limitations=None):
    return {"evidence_id":eid,"evidence_type":"retained_local_audio_hypothesis","source":source,
        "adapter_status":adapter,"confidence":.92,"residual_seconds":.03,"limitations":limitations or [],
        "clocks":{"chart_time":15.,"padded_audio_time":105.02,"mapped_audio_time":105.02}}


def midi_report():
    c1={"window_id":"w1","chart_start":10.,"chart_end":20.,"midi_start":100.,"midi_end":110.,
        "chart_stretch":1.,"translation_seconds":90.,"combined_score":.91,
        "phase_diagnostics":{"occurrence_rank":1,"phase_distinctiveness":"ambiguous_best"}}
    c2={**c1,"midi_start":130.,"midi_end":140.,"translation_seconds":120.,"combined_score":.90,
        "phase_diagnostics":{"occurrence_rank":2,"phase_distinctiveness":"alternative_occurrence"}}
    return {"candidate_report":{"roles":{"guitar":{"windows":[{"window_id":"w1","nominal_start":10.,"nominal_end":20.,"candidates":[c1,c2]}]}}},
        "path_report":{"roles":{"guitar":{"streaks":[{"selected_candidates":[c1]}]}}},
        "fusion_report":{"regions":[]},"application_report":{"applied_regions":[]}}


def test_r4_exports_exact_normalized_component_fields():
    q=aeq.qualify_observation(observation())
    assert q["strict_eligible"] is True
    assert q["detector_identity_complete"] is True
    assert q["detector_identity_status"]=="explicit_complete"
    assert q["physical_source_identity_complete"] is True
    assert q["physical_source_identity_status"]=="explicit_complete"


def test_legacy_strict_row_inherits_missing_redundant_flags_without_false_veto():
    evidence_bundle={"evidence":[observation()]}
    legacy={"qualifications":[{"evidence_id":"ev-1","strict_eligible":True,
        "qualification":"near_baseline_support","clock_complete":True,"confidence":.92,
        "strict_ineligibility_reasons":[]}]}
    ledger=ecl.build_ledger(audio_evidence_bundle=evidence_bundle,midi_section_fit_report=midi_report(),
        audio_evidence_qualification=legacy)
    clock=next(c for e in ledger["events"] for c in e["audio_clocks"])
    assert clock["detector_identity_complete"] is True
    assert clock["physical_source_identity_complete"] is True
    assert clock["completeness_inherited_from_strict"] is True
    assert set(clock["qualification_schema_missing_fields"])=={"detector_identity_complete","physical_source_identity_complete"}
    out=oaq.build_report(midi_section_fit_report=midi_report(),evidence_contribution_ledger=ledger)
    assert out["qualifications"][0]["occurrence_identity_support"]=="phrase_and_occurrence_supported"


def bundle(): return {"evidence":[observation()]}


def test_explicit_component_failure_remains_veto():
    q={"qualifications":[{"evidence_id":"ev-1","strict_eligible":True,"qualification":"near_baseline_support",
        "clock_complete":True,"detector_identity_complete":False,"detector_identity_status":"explicit_incomplete",
        "physical_source_identity_complete":True,"physical_source_identity_status":"explicit_complete","confidence":.92}]}
    ledger=ecl.build_ledger(audio_evidence_bundle=bundle(),midi_section_fit_report=midi_report(),audio_evidence_qualification=q)
    out=oaq.build_report(midi_section_fit_report=midi_report(),evidence_contribution_ledger=ledger)
    assert out["qualifications"][0]["occurrence_identity_support"]=="phrase_supported_occurrence_unqualified"
    assert ledger["summary"]["contradictory_strict_audio_clock_count"]==1


def test_limitations_and_non_strict_evidence_remain_ineligible():
    q=aeq.qualify_bundle({"evidence":[observation(limitations=["contaminated"])]})
    assert q["qualifications"][0]["strict_eligible"] is False
    assert q["qualifications"][0]["detector_identity_complete"] is False
