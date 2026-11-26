import json
from pathlib import Path
import pytest
import build_feedpak as b


def test_cleanup_removes_all_mutually_exclusive_piano_products(tmp_path):
    arr=tmp_path/'arrangements';arr.mkdir()
    for name in b.PIANO_GENERATED_ARTIFACTS:
        (arr/name).write_text('{}')
    (arr/'lead.json').write_text('{}')
    removed=b.clear_generated_piano_artifacts(tmp_path)
    assert set(removed)==set(b.PIANO_GENERATED_ARTIFACTS)
    assert (arr/'lead.json').is_file()
    assert not any((arr/name).exists() for name in b.PIANO_GENERATED_ARTIFACTS)


def test_inventory_accepts_only_referenced_split_classic_products(tmp_path):
    arr=tmp_path/'arrangements';arr.mkdir()
    expected=['piano__lh.json','piano__rh.json','notation_piano__lh.json','notation_piano__rh.json']
    for name in expected:(arr/name).write_text('{}')
    entries=[
      {'id':'piano__lh','file':'arrangements/piano__lh.json','notation':'arrangements/notation_piano__lh.json'},
      {'id':'piano__rh','file':'arrangements/piano__rh.json','notation':'arrangements/notation_piano__rh.json'}]
    result=b.validate_packaged_arrangement_inventory(tmp_path,entries)
    assert result['stale']==[]
    assert set(result['referenced'])==set(expected)


def test_inventory_rejects_stale_merged_files(tmp_path):
    arr=tmp_path/'arrangements';arr.mkdir()
    for name in ['piano__lh.json','piano__rh.json','notation_piano__lh.json','notation_piano__rh.json','keys.json','notation_keys.json']:
        (arr/name).write_text('{}')
    entries=[
      {'id':'piano__lh','file':'arrangements/piano__lh.json','notation':'arrangements/notation_piano__lh.json'},
      {'id':'piano__rh','file':'arrangements/piano__rh.json','notation':'arrangements/notation_piano__rh.json'}]
    with pytest.raises(RuntimeError,match='stale piano artifacts'):
        b.validate_packaged_arrangement_inventory(tmp_path,entries)
