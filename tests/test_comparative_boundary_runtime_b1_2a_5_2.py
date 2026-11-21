from pathlib import Path

def test_runtime_export_and_light_collection_wiring():
    process=Path('process_gp_alignment.py').read_text()
    build=Path('build_feedpak.py').read_text()
    collect=Path('collect_run_diagnostics.py').read_text()
    assert 'cbe.build_report' in process
    assert '"comparative_boundary_evidence_triage":comparative_boundary_evidence_triage' in process
    assert '.comparative_boundary_evidence_triage.json' in build
    assert '*comparative_boundary_evidence_triage.json' in collect
