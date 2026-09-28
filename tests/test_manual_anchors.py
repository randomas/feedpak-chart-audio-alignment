import json
import pytest
import manual_anchors as m

B = [
    {'measure': 1, 'source_bar': 1, 'occurrence': 1, 't_gp': 0.0, 'ts_num': 4},
    {'measure': 2, 'source_bar': 2, 'occurrence': 1, 't_gp': 4.0, 'ts_num': 4},
    {'measure': 3, 'source_bar': 2, 'occurrence': 2, 't_gp': 8.0, 'ts_num': 4},
    {'measure': 4, 'source_bar': 3, 'occurrence': 1, 't_gp': 12.0, 'ts_num': 4},
]


def put(p, x):
    p = p / 'a.json'
    p.write_text(json.dumps(x), encoding='utf-8')
    return p


def test_clocks(tmp_path):
    a, _ = m.load_and_resolve(put(tmp_path, {'alignment': {'manual_anchor_time_basis': 'original', 'manual_anchors': [{'score': {'playback_bar': 2, 'beat': 1}, 'audio': {'time': 10}}]}}), B, 2)
    assert a[0]['padded_audio_time'] == 12


def test_repeat_occurrence(tmp_path):
    a, _ = m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'score': {'source_bar': 2, 'occurrence': 2, 'beat': 2, 'beat_fraction': .5}, 'audio': {'time': 9.5}}]}), B)
    assert a[0]['nominal_time'] == 9.5


def test_nearby(tmp_path):
    a, _ = m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'score': {'playback_bar': 2, 'beat': 1}, 'audio': {'time': 8.02}, 'mode': 'search'}]}), B)
    q = m.assess_anchors(a, lambda t: t, B)
    assert 'nearby_bar_fits_better' in q['anchors'][0]['warnings']


def test_exact(tmp_path):
    a, _ = m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'id': 'x', 'score': {'playback_bar': 2, 'beat': 1}, 'audio': {'time': 5}}]}), B)
    q = m.assess_anchors(a, lambda t: t, B)
    w, r = m.apply_exact_anchors(lambda t: t, q, B)
    assert w(4) == 5 and r['timing_changes_applied']


def test_final_playback_bar_beat_one_is_valid(tmp_path):
    a, _ = m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'id': 'final', 'score': {'playback_bar': 4, 'beat': 1}, 'audio': {'time': 12.5}}]}), B)
    assert a[0]['nominal_time'] == 12.0
    assert a[0]['playback_bar'] == 4


def test_final_source_bar_beat_one_is_valid(tmp_path):
    a, _ = m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'id': 'final', 'score': {'source_bar': 3, 'occurrence': 1, 'beat': 1}, 'audio': {'time': 12.5}}]}), B)
    assert a[0]['nominal_time'] == 12.0


def test_final_bar_fraction_rejected_without_end_boundary(tmp_path):
    with pytest.raises(ValueError, match='final-bar positions after beat 1'):
        m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'score': {'playback_bar': 4, 'beat': 2}, 'audio': {'time': 13}}]}), B)


def test_final_bar_fraction_accepted_with_explicit_end(tmp_path):
    bars = [dict(x) for x in B]
    bars[-1]['end_t_gp'] = 16.0
    a, _ = m.load_and_resolve(put(tmp_path, {'manual_anchors': [{'score': {'playback_bar': 4, 'beat': 2, 'beat_fraction': .5}, 'audio': {'time': 13.5}}]}), bars)
    assert a[0]['nominal_time'] == 13.5
