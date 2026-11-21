from pathlib import Path

def test_runtime_export_and_light_collection_wiring():
    process=Path('process_gp_alignment.py').read_text()
    build=Path('build_feedpak.py').read_text()
    collect=Path('collect_run_diagnostics.py').read_text()
    assert 'tbe.build_report' in process
    assert '"targeted_boundary_evidence_acquisition":targeted_boundary_evidence_acquisition' in process
    assert '.targeted_boundary_evidence_acquisition.json' in build
    assert '*targeted_boundary_evidence_acquisition.json' in collect
