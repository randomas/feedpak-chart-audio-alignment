import checkpoint_dtw as c


def beats(n=17):
    return [
        {'measure': i + 1, 'source_bar': i + 1, 'occurrence': 1,
         't_gp': float(i), 'ts_num': 4}
        for i in range(n)
    ]


def linear_report():
    return {
        'controls': [
            {'score_time': 0.0, 'audio_time': 0.0, 'kind': 'automatic_continuous_anchor', 'id': 'a'},
            {'score_time': 8.0, 'audio_time': 8.2, 'kind': 'manual_exact', 'id': 'b'},
            {'score_time': 16.0, 'audio_time': 16.4, 'kind': 'automatic_continuous_anchor', 'id': 'c'},
        ]
    }


def test_trusted_controls_preserve_structural_points_and_bound_long_intervals():
    rows = c._trusted_controls(linear_report(), beats(), maximum_measures=4)
    times = [x['nominal_time'] for x in rows]
    assert 0.0 in times and 8.0 in times and 16.0 in times
    assert all(b['measure'] - a['measure'] <= 4 for a, b in zip(rows, rows[1:]))
    trusted = [x for x in rows if x['boundary_kind'] == 'trusted']
    assert {x['id'] for x in trusted} == {'a', 'b', 'c'}


def test_missing_audio_is_neutral_and_reports_segment_fallback():
    base = lambda t: t
    report = c.diagnose_trusted_anchor_segment_dtw(
        base, linear_report(), beats(),
        {'guitar': [1, 2, 3, 4, 5, 6, 7]},
        {'guitar': None},
    )
    assert report['version'] == '3.0'
    assert report['timing_changes_applied'] is False
    assert report['summary']['would_apply'] == 0
    assert report['segments']


def test_selective_layer_preserves_trusted_segment_endpoints():
    base = lambda t: t + 0.02 * t
    segment = {
        'segment_id': 'm001-m009',
        'start_nominal_time': 0.0,
        'end_nominal_time': 8.0,
        'decision': {'state': 'would_apply'},
        'path': {
            'control_nominal_times': [0, 2, 4, 6, 8],
            'control_residual_ms': [0, 40, 80, 40, 0],
        },
    }
    cfg = c.TrustedAnchorDTWConfig()
    warp, production = c.build_selective_warp(
        base, {'segments': [segment]}, beats(9), config=cfg
    )
    assert production['timing_changes_applied'] is True
    assert warp(0) == base(0)
    assert warp(8) == base(8)
    assert warp(4) > base(4)


def test_config_supports_wider_bounded_solo_refinement_than_legacy_mode():
    legacy = c.LocalDTWConfig()
    trusted = c.TrustedAnchorDTWConfig()
    assert trusted.corridor_interior_ms > legacy.corridor_interior_ms
    assert trusted.maximum_residual_excursion_ms > legacy.maximum_residual_excursion_ms
    assert trusted.minimum_beat_stretch < legacy.minimum_beat_stretch
    assert trusted.maximum_beat_stretch > legacy.maximum_beat_stretch
