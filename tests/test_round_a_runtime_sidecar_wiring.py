from pathlib import Path

def test_runtime_contract_is_packaged_and_written_as_a_result_sidecar():
    text=Path('build_feedpak.py').read_text(encoding='utf-8')
    contract=text.index('runtime_contract=ar.build_runtime_contract(')
    package=text.index('package_feedpak(work_dir,manifest,output_path)',contract)
    sidecar=text.index('runtime_contract_path=output_path+".runtime_contract.json"',package)
    assertion=text.index('Runtime contract write failed',sidecar)
    assert contract < package < sidecar < assertion
