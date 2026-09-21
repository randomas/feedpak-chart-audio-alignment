import process_gp_alignment as pga

def test_rebase_tempo_events_uses_one_clock():
    assert pga.rebase_tempo_events([(0,90),(16320,95),(43200,97)],960)==[(0,90.0),(15360,95.0),(42240,97.0)]

def test_rebase_tempo_event_at_origin_replaces_initial():
    assert pga.rebase_tempo_events([(0,90),(960,95),(2000,100)],960)==[(0,95.0),(1040,100.0)]

def test_chart_offset_avoids_double_count_in():
    assert abs(pga.choose_chart_offset(2.6666667,2.6666667))<1e-7
    assert pga.choose_chart_offset(2.6666667,0.0)==2.6666667

def test_trustworthy_excludes_onset_only():
    assert pga.trustworthy_checkpoint_count({"summary":{"strong":0}},{"measured_boundaries":0})==0



