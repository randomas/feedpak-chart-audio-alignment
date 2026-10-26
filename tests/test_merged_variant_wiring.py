from pathlib import Path
import inspect
import build_feedpak as bf
import phase3d_materialization as p3d
import variant_materialization as vm

ROOT=Path(__file__).resolve().parents[1]

def test_all_alignment_and_variant_modes_remain_wired():
    build=(ROOT/'build_feedpak.py').read_text(encoding='utf-8')
    runner=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    for token in ('bar-landmark-regularized','--phase3d-variants','--phase3d-safe-candidates',
                  '--phase3d-structural-candidates','--build-unsafe-continuous-comparison'):
        assert token in build or token in runner
    for token in ('Phase3DVariants','Phase3DSafeCandidates','Phase3DStructuralCandidates',
                  'BuildUnsafeContinuousComparison','bar-landmark-regularized'):
        assert token in runner

def test_materializers_do_not_call_analysis_entrypoints():
    for function in (p3d.materialize_from_work_dir,vm.materialize_continuous_variant):
        source=inspect.getsource(function)
        assert 'run_gp_script' not in source
        assert 'process_gp_alignment' not in source
        assert 'build(' not in source

def test_main_has_only_one_top_level_build_call():
    source=inspect.getsource(bf.main)
    assert source.count(' = build(')==1
    assert 'materialize_continuous_variant' in source

def test_phase3d_runs_inside_the_canonical_build_before_cleanup():
    source=inspect.getsource(bf.build)
    phase=source.index('materialize_from_work_dir')
    cleanup=source.index('if not keep_work_dir:')
    assert phase < cleanup
