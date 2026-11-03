from pathlib import Path
import inspect
import continuous_anchors as structural
import continuous_anchors_legacy as historical

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'process_gp_alignment.py').read_text(encoding='utf-8')


def test_historical_and_structural_api_differences_are_routed_explicitly():
    historical_diag=inspect.signature(historical.diagnose_continuous_anchors)
    structural_diag=inspect.signature(structural.diagnose_continuous_anchors)
    historical_warp=inspect.signature(historical.build_continuous_anchor_linear_warp)
    structural_warp=inspect.signature(structural.build_continuous_anchor_linear_warp)
    assert 'feature_cache_dir' not in historical_diag.parameters
    assert 'feature_cache_dir' in structural_diag.parameters
    assert 'force_invalid' not in historical_warp.parameters
    assert 'force_invalid' in structural_warp.parameters
    assert 'if not historical_continuous:' in SOURCE
    assert 'continuous_diagnostic_kwargs["feature_cache_dir"]' in SOURCE
    assert 'continuous_warp_kwargs["force_invalid"]' in SOURCE


def test_process_cli_accepts_structural_method():
    assert '"continuous-anchor-structural","bar-landmark-regularized"' in SOURCE


def test_historical_modes_select_preserved_engine():
    assert 'historical_continuous = alignment_mode in ("continuous-anchor-linear", "continuous-anchor-dtw")' in SOURCE
    assert 'continuous_engine = ca_legacy if historical_continuous else ca' in SOURCE


def test_structural_second_pass_is_switchable_and_structural_only():
    assert '--structural-second-pass' in SOURCE
    assert 'alignment_mode=="continuous-anchor-structural" and structural_second_pass' in SOURCE
