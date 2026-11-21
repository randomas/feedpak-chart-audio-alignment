from pathlib import Path
import transcription_investigator as ti


def touch(folder,name):
    p=folder/name;p.write_bytes(b'x');return p


def test_generic_role_mid_files_are_discovered_case_insensitively(tmp_path):
    touch(tmp_path,'Bass.mid');touch(tmp_path,'drums.mid');touch(tmp_path,'Guitar.MID');touch(tmp_path,'piano.mid')
    found,diag=ti.discover_prepared_files(tmp_path,return_diagnostics=True)
    assert found['bass'].name=='Bass.mid'
    assert found['drums'].name=='drums.mid'
    assert found['guitar'].name=='Guitar.MID'
    assert found['piano'].name=='piano.mid'
    assert all(diag[x]['discovery_mode']=='generic_role_filename' for x in ('bass','drums','guitar','piano'))
    assert all(diag[x]['provider']=='prepared_midi' for x in ('bass','drums','guitar','piano'))


def test_canonical_wins_and_generic_is_reported_as_ignored(tmp_path):
    touch(tmp_path,'Bass.mid');touch(tmp_path,'Bass.muscriptor.mid')
    found,diag=ti.discover_prepared_files(tmp_path,return_diagnostics=True)
    assert found['bass'].name=='Bass.muscriptor.mid'
    assert diag['bass']['ignored_alternatives']==['Bass.mid']
    assert diag['bass']['provider']=='muscriptor'


def test_explicit_metadata_wins_over_canonical(tmp_path):
    touch(tmp_path,'custom.mid');touch(tmp_path,'Synth.muscriptor.mid')
    found,diag=ti.discover_prepared_files(tmp_path,{'synth':{'performance_midi':'custom.mid'}},True)
    assert found['synth'].name=='custom.mid'
    assert diag['synth']['discovery_mode']=='explicit_metadata'
    assert diag['synth']['ignored_alternatives']==['Synth.muscriptor.mid']


def test_piano_and_synth_aliases_are_supported(tmp_path):
    touch(tmp_path,'Keys.mid')
    found=ti.discover_prepared_files(tmp_path)
    assert found['piano'].name=='Keys.mid'
    assert found['synth'].name=='Keys.mid'


def test_generic_parser_provenance_is_provider_neutral(monkeypatch,tmp_path):
    path=touch(tmp_path,'Bass.mid')
    monkeypatch.setattr(ti,'mido',None)
    report=ti.parse_muscriptor_midi(path,'bass')
    assert report['provider']=='prepared_midi'
    assert report['discovery_mode']=='generic_role_filename'
