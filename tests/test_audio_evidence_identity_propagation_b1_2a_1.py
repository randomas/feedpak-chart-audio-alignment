import evidence_contribution_ledger as ecl
import occurrence_audio_qualification as oaq


def bundle():
    return {"evidence":[{"evidence_id":"ev-1","evidence_type":"retained_local_audio_hypothesis",
        "detector_module":"structural_search","detector_method":"onset_cluster","detector_version":"1",
        "source":"Keys.ogg","physical_source_id":"file:keys","nominal_role":"synth",
        "observed_signal_type":"onset","confidence":.92,"limitations":[],
        "clocks":{"chart_time":15.0,"mapped_audio_time":105.02},
        "provenance":{"source_lineage_id":"stem:keys"}}]}


def qualification(strict=True):
    return {"qualifications":[{"evidence_id":"ev-1","strict_eligible":strict,
        "qualification":"near_baseline_support","clock_complete":True,"detector_complete":True,
        "physical_source_complete":True,"confidence":.92,"reasons":[] if strict else ["not_strict"]}]}


def midi_report():
    c1={"window_id":"w1","chart_start":10.,"chart_end":20.,"midi_start":100.,"midi_end":110.,
        "chart_stretch":1.,"translation_seconds":90.,"combined_score":.91,
        "phase_diagnostics":{"occurrence_rank":1,"phase_distinctiveness":"ambiguous_best"}}
    c2={**c1,"midi_start":130.,"midi_end":140.,"translation_seconds":120.,"combined_score":.90,
        "phase_diagnostics":{"occurrence_rank":2,"phase_distinctiveness":"alternative_occurrence"}}
    return {"candidate_report":{"roles":{"synth":{"windows":[{"window_id":"w1","nominal_start":10.,"nominal_end":20.,"candidates":[c1,c2]}]}}},
        "path_report":{"roles":{"synth":{"streaks":[{"selected_candidates":[c1]}]}}},
        "fusion_report":{"regions":[]},"application_report":{"applied_regions":[]}}


def query_eval():
    return {"evaluations":[{"query_id":"q1","bounded_query_status":"localized",
        "classification":"localized_with_strict_evidence","representative_audio_support":{"evidence_id":"ev-1"},
        "bridge_geometry":{"status":"safe"},"future_audio_only_anchor_candidate":False}]}


def test_identity_and_qualification_propagate_to_ledger_and_occurrence_resolution():
    ledger=ecl.build_ledger(audio_evidence_bundle=bundle(),midi_section_fit_report=midi_report(),
        audio_evidence_qualification=qualification(),query_result_evaluation=query_eval())
    clock=next(e["audio_clocks"][0] for e in ledger["events"] if e["audio_clocks"])
    assert clock["event_id"]=="ev-1"
    assert clock["physical_source_id"]=="file:keys"
    assert clock["source_lineage_id"]=="stem:keys"
    assert clock["detector_module"]=="structural_search"
    assert clock["strict_eligible"] is True
    assert clock["clock_complete"] is True
    assert clock["query_localized"] is True
    assert clock["query_links"][0]["query_id"]=="q1"
    out=oaq.build_report(midi_section_fit_report=midi_report(),evidence_contribution_ledger=ledger)
    assert out["summary"]["strict_audio_observation_count"]>=1
    assert out["qualifications"][0]["occurrence_identity_support"]=="phrase_and_occurrence_supported"


def test_non_strict_upstream_evidence_cannot_resolve_occurrence():
    ledger=ecl.build_ledger(audio_evidence_bundle=bundle(),midi_section_fit_report=midi_report(),
        audio_evidence_qualification=qualification(False),query_result_evaluation=query_eval())
    out=oaq.build_report(midi_section_fit_report=midi_report(),evidence_contribution_ledger=ledger)
    assert out["summary"]["strict_audio_observation_count"]==0
    assert out["qualifications"][0]["occurrence_identity_support"]=="phrase_supported_occurrence_unqualified"


def test_shared_physical_source_keeps_one_independence_group():
    b=bundle(); second=dict(b["evidence"][0]); second["evidence_id"]="ev-2"; second["nominal_role"]="piano"; b["evidence"].append(second)
    q=qualification(); q["qualifications"].append({**q["qualifications"][0],"evidence_id":"ev-2"})
    ledger=ecl.build_ledger(audio_evidence_bundle=b,midi_section_fit_report=midi_report(),audio_evidence_qualification=q)
    groups={c["source_independence_group"] for e in ledger["events"] for c in e["audio_clocks"]}
    assert groups=={"file:keys"}
