import alphatab_score as ats


def event(note_id, visit, source, start, duration=480):
    return {"playback_master_bar_index": visit, "source_master_bar_index": source,
            "track_index": 0, "staff_index": 0, "voice_index": 0,
            "note_id": note_id, "absolute_start_tick": start,
            "duration_ticks": duration, "duration_percent": 1}


def semantic(destination=None):
    return {"duration": {"percent": 1}, "links": {"tie_destination_id": destination}}


def test_standard_repeat_and_alternate_ending_create_segments():
    playback = {"master_bar_visits": [
        {"playback_index": 0, "source_master_bar_index": 18, "start_tick": 0, "end_tick": 3840},
        {"playback_index": 1, "source_master_bar_index": 19, "start_tick": 3840, "end_tick": 7680},
        {"playback_index": 2, "source_master_bar_index": 12, "start_tick": 7680, "end_tick": 11520},
        {"playback_index": 3, "source_master_bar_index": 13, "start_tick": 11520, "end_tick": 15360},
        {"playback_index": 4, "source_master_bar_index": 20, "start_tick": 15360, "end_tick": 19200},
    ]}
    segments, ends = ats._playback_segment_metadata(playback)
    assert segments == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    assert ends == {0: 7680.0, 1: 15360.0, 2: 19200.0}


def test_tie_cannot_cross_repeat_or_alternate_ending_jump():
    playback = {"master_bar_visits": [
        {"playback_index": 18, "source_master_bar_index": 18, "start_tick": 69120, "end_tick": 72960},
        {"playback_index": 19, "source_master_bar_index": 19, "start_tick": 72960, "end_tick": 76800},
        {"playback_index": 20, "source_master_bar_index": 12, "start_tick": 76800, "end_tick": 80640},
        {"playback_index": 21, "source_master_bar_index": 13, "start_tick": 80640, "end_tick": 84480},
        {"playback_index": 26, "source_master_bar_index": 20, "start_tick": 99840, "end_tick": 103680},
    ]}
    segments, ends = ats._playback_segment_metadata(playback)
    root = event(833, 18, 18, 72480, 480)
    within = event(857, 19, 19, 72960, 480)
    after_jump = event(897, 26, 20, 99840, 1920)
    notes = {(0, 833): semantic(857), (0, 857): semantic(897), (0, 897): semantic()}
    lookup = {ats._event_playback_identity(x, segments): x for x in (root, within, after_jump)}
    duration = ats._resolved_tied_duration_ticks(root, notes[(0, 833)], notes, lookup, segments, ends)
    assert duration == 960.0


def test_legitimate_long_tie_is_preserved_inside_continuous_segment():
    playback = {"master_bar_visits": [
        {"playback_index": 0, "source_master_bar_index": 0, "start_tick": 0, "end_tick": 3840},
        {"playback_index": 1, "source_master_bar_index": 1, "start_tick": 3840, "end_tick": 7680},
        {"playback_index": 2, "source_master_bar_index": 2, "start_tick": 7680, "end_tick": 11520},
    ]}
    segments, ends = ats._playback_segment_metadata(playback)
    root = event(1, 0, 0, 2400, 1440)
    continuation = event(2, 1, 1, 3840, 3840)
    notes = {(0, 1): semantic(2), (0, 2): semantic()}
    lookup = {ats._event_playback_identity(x, segments): x for x in (root, continuation)}
    duration = ats._resolved_tied_duration_ticks(root, notes[(0, 1)], notes, lookup, segments, ends)
    assert duration == 5280.0
