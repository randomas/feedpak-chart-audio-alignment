from pathlib import Path


def test_prepared_midi_parser_uses_one_merged_track_pass_and_deque():
    source=Path("transcription_investigator.py").read_text(encoding="utf-8")
    start=source.index("def parse_muscriptor_midi")
    end=source.index("\ndef _score_event",start)
    body=source[start:end]
    assert body.count("mido.merge_tracks(midi.tracks)")==1
    assert "deque()" in body
    assert ".popleft()" in body
    assert ".pop(0)" not in body
