import json
from pathlib import Path
import occurrence_audio_qualification as oaq
import collect_run_diagnostics as crd


def candidate(rank,midi_start,score=.9):
    return {"window_id":"w1","role":"guitar","chart_start":10.0,"chart_end":20.0,
        "midi_start":midi_start,"midi_end":midi_start+10,"chart_stretch":1.0,
        "translation_seconds":midi_start-10.0,"combined_score":score,
        "phase_diagnostics":{"occurrence_rank":rank,"phase_distinctiveness":"ambiguous_best" if rank==1 else "alternative_occurrence"}}


def report():
    c1=candidate(1,100.0,.91); c2=candidate(2,130.0,.90)
    return {"candidate_report":{"roles":{"guitar":{"windows":[{"window_id":"w1","nominal_start":10.0,
        "nominal_end":20.0,"candidates":[c1,c2]}]}}},
        "path_report":{"roles":{"guitar":{"streaks":[{"selected_candidates":[c1]}]}}}}


def ledger(audio_time,confidence=.95,source="mix",event_id="a1"):
    return {"events":[{"event_id":"ledger-1","status":"exact_cross_modal_onset","chart_time":15.0,
        "audio_clocks":[{"event_id":event_id,"chart_time":15.0,"mapped_audio_time":audio_time,
            "confidence":confidence,"source":source,"physical_source_id":source,
            "strict_eligible":True,"clock_complete":True,"detector_identity_complete":True,
            "physical_source_identity_complete":True,"limitations":[]}]}]}


def test_selected_occurrence_supported_when_audio_discriminates():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(105.03))
    row=out["qualifications"][0]
    assert row["occurrence_identity_support"]=="phrase_and_occurrence_supported"
    assert row["production_status"]=="occurrence_supported"
    assert out["timing_changes_applied"] is False
    assert out["path_reselected"] is False


def test_alternative_occurrence_favored_without_reselection():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(135.02))
    row=out["qualifications"][0]
    assert row["occurrence_identity_support"]=="phrase_supported_alternative_occurrence_favored"
    assert row["favored_alternative_rank"]==2
    assert row["production_status"]=="occurrence_disputed"
    assert out["candidate_scores_changed"] is False


def test_missing_provenance_cannot_resolve_occurrence():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(105.01,source=None,event_id=None))
    row=out["qualifications"][0]
    assert row["phrase_identity_support"]=="supported"
    assert row["occurrence_identity_support"]=="phrase_supported_occurrence_unqualified"


def test_no_audio_is_explicitly_insufficient():
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger={"events":[]})
    assert out["qualifications"][0]["occurrence_identity_support"]=="insufficient_audio_for_occurrence_resolution"


def test_light_skeleton_preserves_occurrence_candidates(tmp_path):
    out=oaq.build_report(midi_section_fit_report=report(),evidence_contribution_ledger=ledger(135.02))
    src=tmp_path/"song.audio_qualified_occurrence_resolution.json"
    dst=tmp_path/"song.audio_qualified_occurrence_resolution.light.json"
    src.write_text(json.dumps(out),encoding="utf-8")
    crd.write_light_json(src,dst)
    light=json.loads(dst.read_text())
    assert light["qualifications"][0]["favored_alternative_rank"]==2
    assert light["safety"]["timing_changes_applied"] is False


def test_runtime_and_export_wiring_present():
    process=Path("process_gp_alignment.py").read_text()
    build=Path("build_feedpak.py").read_text()
    collect=Path("collect_run_diagnostics.py").read_text()
    assert "oaq.build_report" in process
    assert '"audio_qualified_occurrence_resolution":audio_qualified_occurrence_resolution' in process
    assert '.audio_qualified_occurrence_resolution.json' in build
    assert '*audio_qualified_occurrence_resolution.json' in collect
