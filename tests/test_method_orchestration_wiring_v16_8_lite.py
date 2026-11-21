from pathlib import Path

def test_cli_runtime_and_envelope_wiring():
    process=Path('process_gp_alignment.py').read_text()
    build=Path('build_feedpak.py').read_text()
    assert 'import method_orchestration as mo' in process
    assert '--prepared-midi-mode' in process and '--prepared-midi-mode' in build
    assert 'mo.verify_manual_anchors' in process
    assert '"method_result_envelope":method_result_envelope' in process
