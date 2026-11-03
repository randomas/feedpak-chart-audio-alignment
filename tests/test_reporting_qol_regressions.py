import json
from pathlib import Path
import process_gp_alignment as pga

ROOT=Path(__file__).resolve().parents[1]


def test_supremacy_successful_run_is_frozen_as_golden_contract():
    golden=json.loads((ROOT/'tests/fixtures/supremacy_continuous_dtw_golden.json').read_text())
    assert golden['selected_anchor_count']==7
    assert golden['linear_control_count']==7
    assert golden['trusted_interval_count']==16
    assert golden['applied_local_dtw_segments']==1
    assert golden['timeline_beat_mismatches']==0
    assert golden['timeline_tempo_mismatches']==0
    assert golden['repaired_measures']==0


def test_authoritative_production_timing_separates_provider_diagnostics():
    decision={
        'mode':'continuous-anchor-dtw-v1',
        'timing_output':'continuous-anchor-dtw-v1',
        'continuous_anchor_linear':{'status':'applied','timing_changes_applied':True},
        'continuous_anchor_dtw_production':{'status':'applied','timing_changes_applied':True,'applied_segment_count':1},
        'regional_source_fusion':{'timing_changes_applied':False},
    }
    record=pga.finalize_production_timing(decision,'continuous-anchor-dtw')
    assert record['timing_changes_applied'] is True
    assert record['linear_changes_applied'] is True
    assert record['local_dtw_changes_applied'] is True
    assert record['applied_local_dtw_segments']==1
    assert record['selected_production_method']=='continuous-anchor-dtw'
    assert decision['production_timing']==record


def test_bar_landmark_gets_truthful_final_method_label():
    source=(ROOT/'process_gp_alignment.py').read_text()
    assert 'decision["mode"] = "bar-landmark-regularized-v1"' in source
    assert 'decision["timing_output"] = "bar-landmark-regularized-v1"' in source
    assert 'FINAL PRODUCTION TIMING:' in source


def test_applied_dtw_segments_are_location_and_evidence_rich():
    source=(ROOT/'checkpoint_dtw.py').read_text()
    for field in ('start_measure','end_measure','baseline_median_residual_ms',
                  'candidate_median_residual_ms','held_out_evaluation',
                  'minimum_beat_stretch','maximum_beat_stretch'):
        assert repr(field) in source


def test_diagnostics_collector_creates_method_specific_snapshots_and_hashes():
    source=(ROOT/'collect_run_diagnostics.py').read_text()
    assert "song_root/'generated'/name" in source
    assert "'active_score_schema.json'" in source
    assert "'active_project_config'" in source
    assert "'sha256':digest" in source


def test_selective_reporting_distinguishes_coarse_checkpoint_and_local_layers():
    decision={
        'mode':'checkpoint-dtw-selective-v1',
        'timing_output':'checkpoint-linear-v1',
        'scale':0.9889,
        'opening_anchor':{'score_anchor':3.0,'audio_anchor':2.995},
        'checkpoint_linear':{'timing_changes_applied':True},
        'local_dtw_production':{
            'status':'checkpoint_linear_fallback_no_accepted_segments',
            'timing_changes_applied':False,
            'applied_segment_count':0,
        },
    }
    record=pga.finalize_production_timing(decision,'checkpoint-dtw-selective')
    assert record['selected_production_method']=='checkpoint-linear'
    assert record['base_timing_changed'] is True
    assert record['checkpoint_linear_changed'] is True
    assert record['local_dtw_changed'] is False
    assert record['timing_changes_applied'] is True
    assert record['fallback_reason']=='checkpoint_linear_fallback_no_accepted_segments'


def test_light_diagnostics_profile_is_analysis_focused():
    source=(ROOT/'collect_run_diagnostics.py').read_text()
    assert "--light" in source
    assert "skeleton_transcription" in source
    assert "incipit_role_tracking" in source
    assert "representative_controls" in source
    assert "unsupported_gaps" in source
    assert "Light diagnostics collected" in source


def test_test_runner_exposes_light_diagnostics_folder():
    source=(ROOT/'run_all_test_materials.ps1').read_text()
    assert '[switch]$LightDiagnostics' in source
    assert '"Diagnostics_light"' in source
    assert '$collectorArguments += "--light"' in source


def test_wide_monotonic_warp_switch_is_explicit_and_audio_refinement_is_opt_in():
    builder=(ROOT/'build_feedpak.py').read_text()
    runner=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    process=(ROOT/'process_gp_alignment.py').read_text()
    assert '--allow-wide-monotonic-warp' in builder
    assert '[switch]$AllowWideMonotonicWarp' in runner
    assert '[switch]$AllowCompatibleAudioRefinement' in runner
    assert 'force_monotonic_chain and not allow_compatible_audio_refinement' in process
    assert 'forced-wide-monotonic-chain' in process
