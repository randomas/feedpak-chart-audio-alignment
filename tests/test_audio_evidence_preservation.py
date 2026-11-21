from pathlib import Path
import ast
import audio_evidence_schema as aes


def test_schema_module_has_no_audio_or_detector_imports():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'audio_evidence_schema.py').read_text(encoding='utf-8'))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    forbidden=('librosa','soundfile','numpy','continuous_anchors','structural_search','process_gp_alignment','instrument_evidence')
    assert not any(name.startswith(forbidden) for name in imports)


def test_r3_wiring_is_aggregation_only_and_absent_from_timing_engines():
    root=Path(__file__).resolve().parents[1]
    process=(root/'process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'import audio_evidence_schema as aes' in process
    assert 'build_read_only_evidence_bundle' in process
    for name in ('midi_section_fit.py','continuous_anchors.py','structural_search.py','phase3d_materialization.py'):
        text=(root/name).read_text(encoding='utf-8')
        assert 'import audio_evidence_schema' not in text
        assert 'from audio_evidence_schema' not in text


def test_all_evidence_classes_remain_diagnostic():
    samples=[]
    samples+=aes.adapt_opening_candidates([{'source':'guitar','audio_time':2.0}])
    samples+=aes.adapt_three_clock_incipit({'status':'validated','role':'guitar','root_offset_seconds':.1})
    samples+=aes.adapt_continuous_anchor_report({'matches':[{'source':'bass','time':4,'confidence':.9}]})
    samples+=aes.adapt_structural_search_report({'clusters':[{'cluster_id':'x','score':.9}]})
    samples+=aes.adapt_instrument_evidence_report({'events':[{'role':'piano','audio_time':3}]})
    assert samples
    assert all(row['authority']=='diagnostic' and row['timing_changes_applied'] is False and row['midi_authority_changed'] is False for row in samples)


def test_runtime_aggregation_does_not_call_original_detector_functions():
    root=Path(__file__).resolve().parents[1]
    schema=(root/'audio_evidence_schema.py').read_text(encoding='utf-8')
    assert 'librosa' not in schema
    assert 'soundfile' not in schema
    assert 'def detect_' not in schema
    process=(root/'process_gp_alignment.py').read_text(encoding='utf-8')
    call=process[process.index('audio_evidence_bundle=aes.build_read_only_evidence_bundle'):]
    assert 'detect_audio_content_end(' not in call.split('beatmap_report=',1)[0]
    assert 'diagnose_track_audio_coverage(' not in call.split('beatmap_report=',1)[0]


def test_bundle_safety_flags_are_literal_false_at_runtime_boundary():
    bundle=aes.build_read_only_evidence_bundle()
    assert bundle['timing_changes_applied'] is False
    assert bundle['midi_authority_changed'] is False
    assert bundle['detectors_executed'] is False
