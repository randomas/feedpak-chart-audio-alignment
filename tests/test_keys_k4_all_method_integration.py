from pathlib import Path
import transcription_investigator as ti

ROOT=Path(__file__).resolve().parents[1]


def test_k4_wires_synth_into_all_evidence_surfaces():
    text=(ROOT/'process_gp_alignment.py').read_text(encoding='utf-8')
    assert 'continuous_score_events[stream_id]' in text
    assert '"synth":synth_audio_path' in text
    assert 'prepared_midi_files["synth"]=synth_midi_path' in text
    assert 'prepared_files=prepared_midi_files' in text
    assert '--synth-audio' in text and '--synth-midi' in text


def test_builder_forwards_resolved_synth_sources():
    text=(ROOT/'build_feedpak.py').read_text(encoding='utf-8')
    assert 'synth_source=keys_evidence_sources.get("synth")' in text
    assert '["--synth-audio",synth_path]' in text
    assert '["--synth-midi",synth_midi_path]' in text


def test_investigator_accepts_configured_stream_without_legacy_filename(tmp_path):
    midi=tmp_path/'custom.keys.mid';midi.write_bytes(b'not-used-by-this-test')
    discovered={'bass':None,'drums':None,'guitar':None,'piano':None,'synth':str(midi)}
    # The configured role must not require a ROLE_FILENAMES entry when absent.
    # Parsing is intentionally replaced so this test remains fixture-free.
    original=ti.parse_muscriptor_midi
    try:
        ti.parse_muscriptor_midi=lambda path,role,padding:{'provider':'muscriptor','source':role,'status':'available','events':[]}
        report=ti.investigate({'synth':[]},str(tmp_path),lambda t:t,prepared_files=discovered)
    finally:
        ti.parse_muscriptor_midi=original
    assert report['summary']['prepared_role_count']==1
    assert 'synth' in report['role_activity_profiles']


def test_all_method_runner_has_four_song_and_guarded_midi_switches():
    text=(ROOT/'run_all_alignment_methods.ps1').read_text(encoding='utf-8')
    for folder in ('my_test_song_2_gp','my_test_song_3_gp','my_test_song_3b_gp','my_test_song_7_gp'):
        assert folder in text
    assert '[switch]$FourSongCorpus' in text
    assert '[switch]$MidiSectionFit' in text
    assert '$midiEnabled = $transcriptionMethods -contains $method' in text
    assert '$params.MidiSectionFit = $true' in text
    assert 'MIDI section fit skipped for unsupported method' in text
