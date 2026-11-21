import process_gp_alignment as pga


def test_complete_projection_does_not_drop_nonadvancing_measure_identity():
    warped=[{'time':0.0,'measure':1,'ts_num':4},{'time':4.0,'measure':2,'ts_num':4},
            {'time':3.5,'measure':3,'ts_num':4},{'time':8.0,'measure':4,'ts_num':4}]
    nominal=[{'t_gp':0.0,'measure':1,'ts_num':4},{'t_gp':4.0,'measure':2,'ts_num':4},
             {'t_gp':8.0,'measure':3,'ts_num':4},{'t_gp':12.0,'measure':4,'ts_num':4}]
    out=pga.build_explicit_feedpak_beats(warped,nominal)
    down=[x for x in out if x['measure']!=-1]
    assert [x['measure'] for x in down]==[1,2,3,4]
    assert len(out)==16  # every measure retains its complete beat identity inventory


def test_complete_projection_matches_canonical_even_subdivision_for_valid_input():
    warped=[{'time':0.0,'measure':1,'ts_num':4},{'time':4.0,'measure':2,'ts_num':3},
            {'time':7.0,'measure':3,'ts_num':4}]
    out=pga.build_explicit_feedpak_beats(warped,warped)
    assert len(out)==11
    assert out[0]=={'time':0.0,'measure':1}
    assert out[4]=={'time':4.0,'measure':2}


def test_final_measure_direct_warp_uses_previous_nominal_measure_duration():
    nominal=[{'t_gp':0.0,'measure':1,'ts_num':4},
             {'t_gp':4.0,'measure':2,'ts_num':4}]
    downbeats=[{'time':0.0,'measure':1,'ts_num':4},
               {'time':8.0,'measure':2,'ts_num':4}]
    warp=lambda t: float(t)*float(t)/2.0
    rows=pga.build_explicit_feedpak_beats(downbeats,nominal,warp_fn=warp)
    assert [x['time'] for x in rows[-4:]]==[8.0,12.5,18.0,24.5]
