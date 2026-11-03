import json, os
from pathlib import Path
import pipeline_cache as plc
import build_feedpak as bf


def test_canonical_hash_is_order_and_set_stable():
    a={"b":{3,1,2},"a":[1,2]}; b={"a":[1,2],"b":{2,3,1}}
    assert plc.canonical_hash(a)==plc.canonical_hash(b)


def test_cache_lookup_reports_miss_hit_and_invalidation(tmp_path):
    artifact=tmp_path/'artifact.bin'; artifact.write_bytes(b'x')
    manifest=tmp_path/'manifest.json'; fp=plc.canonical_hash({'x':1})
    assert plc.cache_lookup(manifest,fp,[artifact])[1]=='cache_file_missing'
    plc.atomic_write_json(manifest,{'schema_version':plc.CACHE_SCHEMA,'fingerprint':fp})
    assert plc.cache_lookup(manifest,fp,[artifact])[1]=='exact_cache_hit'
    assert plc.cache_lookup(manifest,'wrong',[artifact])[1]=='fingerprint_mismatch'
    artifact.unlink()
    assert plc.cache_lookup(manifest,fp,[artifact])[1]=='cached_artifact_missing'


def test_prepared_stem_cache_reuses_exact_artifact(tmp_path,monkeypatch):
    song=tmp_path/'song'; work=song/'_feedpak_build'; song.mkdir()
    source=song/'Full.ogg'; source.write_bytes(b'synthetic-audio')
    monkeypatch.setattr(bf.pga,'get_initial_tempo',lambda _:120.)
    monkeypatch.setattr(bf.fc,'detect_leading_silence_seconds',lambda _:3.)
    stems=[{'id':'full','file':'Full.ogg'}]
    first=bf.prepare_stems_with_count_in(str(song),stems,None,str(work))
    assert first[4]['status']=='cache_file_missing'
    output=Path(first[3]['full']); before=output.read_bytes()
    stems2=[{'id':'full','file':'Full.ogg'}]
    second=bf.prepare_stems_with_count_in(str(song),stems2,None,str(work))
    assert second[4]['status']=='exact_cache_hit'
    assert Path(second[3]['full']).read_bytes()==before


def test_prepared_stem_cache_invalidates_when_source_changes(tmp_path,monkeypatch):
    song=tmp_path/'song';work=song/'_feedpak_build';song.mkdir();source=song/'Full.ogg'
    source.write_bytes(b'a');monkeypatch.setattr(bf.fc,'detect_leading_silence_seconds',lambda _:3.)
    bf.prepare_stems_with_count_in(str(song),[{'id':'full','file':'Full.ogg'}],None,str(work))
    source.write_bytes(b'b')
    result=bf.prepare_stems_with_count_in(str(song),[{'id':'full','file':'Full.ogg'}],None,str(work))
    assert result[4]['status']=='fingerprint_mismatch'


def test_pipeline_performance_wiring_present():
    text=Path(bf.__file__).read_text()
    for key in ('inspection_and_discovery_seconds','prepared_stems_seconds','vocal_processing_seconds',
                'gp_alignment_and_arrangement_seconds','manifest_seconds','packaging_seconds',
                'report_writing_seconds','total_build_seconds'):
        assert key in text


def test_prepared_stem_cache_records_actual_encoder_extension(tmp_path,monkeypatch):
    song=tmp_path/'song';work=song/'_feedpak_build';song.mkdir();source=song/'Bass.ogg'
    source.write_bytes(b'synthetic-audio')
    monkeypatch.setattr(bf.fc,'detect_leading_silence_seconds',lambda _:0.)
    monkeypatch.setattr(bf.fc,'pad_audio_with_silence',lambda src,dst,pad:Path(dst).write_bytes(b'wav'))
    def encode_ogg(wav_path,out_dir):
        out=Path(out_dir)/'Bass.ogg';out.write_bytes(b'ogg');Path(wav_path).unlink();return str(out)
    monkeypatch.setattr(bf,'compress_padded_stem',encode_ogg)
    first=bf.prepare_stems_with_count_in(str(song),[{'id':'bass','file':'Bass.ogg'}],None,str(work))
    assert Path(first[3]['bass']).suffix=='.ogg'
    manifest=plc.load_json(work/'cache'/'prepared_stems'/'manifest.json')
    assert manifest['artifacts'][0]['file']=='Bass.ogg'
    second=bf.prepare_stems_with_count_in(str(song),[{'id':'bass','file':'Bass.ogg'}],None,str(work))
    assert second[4]['status']=='exact_cache_hit'
    assert Path(second[3]['bass']).name=='Bass.ogg'
