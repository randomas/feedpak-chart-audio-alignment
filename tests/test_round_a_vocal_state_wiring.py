from pathlib import Path

def test_vocal_runtime_state_is_defined_for_off_optional_and_required_paths():
    text=Path('build_feedpak.py').read_text(encoding='utf-8')
    init=text.index('vocals_data=None')
    branches=text.index('if gp_vocal_production:')
    products=text.index('vocal_products_present=bool(gp_vocal_production or vocals_data is not None)')
    contract=text.index('runtime_contract=ar.build_runtime_contract(')
    assert init < branches < products < contract
    assert 'runtime_failures=[]' in text[products:contract]
    assert 'required_vocals_unavailable' in text[products:contract]
