import structural_repeat_diagnostics as srd

def measures(n=16,d=1.0):
    return [{'playback_index':i,'playback_measure_number':i+1,'written_measure_number':i+1,'start':i*d,'end':(i+1)*d} for i in range(n)]

def phrase(start,bars=2,d=1.0,base=60):
    rows=[]
    for bar in range(bars):
        for k,p in enumerate((base+bar,base+4+bar,base+7+bar,base+4+bar)):
            rows.append({'time':start+bar*d+k*d/4,'midi':p})
    return rows

def test_phrase_run_equal_counts_is_diagnostic():
    chart=[];midi=[]
    for i in range(3):chart+=phrase(i*2);midi+=phrase(10+i*2)
    report=srd.build(measures(6),{'piano':chart},{'piano':midi})
    rows=[x for x in report['phrase_run_correspondences'] if x['phrase_measures']==2]
    assert rows and rows[0]['classification']=='matching_phrase_run'
    assert rows[0]['confirmed_midi_occurrences']==3
    assert report['timing_changes_applied'] is False
    assert report['selection_behavior_changed'] is False

def test_phrase_run_mismatch_keeps_probable_and_unresolved_separate():
    chart=[];midi=[]
    for i in range(4):chart+=phrase(i*2)
    for i in range(3):midi+=phrase(10+i*2)
    report=srd.build(measures(8),{'piano':chart},{'piano':midi})
    row=next(x for x in report['phrase_run_correspondences'] if x['phrase_measures']==2)
    assert row['chart_occurrence_count']==4
    assert row['confirmed_midi_occurrences']<=3
    assert row['classification']!='matching_phrase_run'
    assert row['confirmed_midi_occurrences']+row['probable_additional_occurrences']+row['unresolved_occurrences']==4

def test_cross_role_corroboration_requires_same_phrase_level_and_classification():
    chart={'bass':[], 'guitar':[]};midi={'bass':[], 'guitar':[]}
    for role in chart:
        for i in range(3):chart[role]+=phrase(i*2,base=40 if role=='bass' else 60);midi[role]+=phrase(10+i*2,base=40 if role=='bass' else 60)
    report=srd.build(measures(6),chart,midi)
    assert any(set(x['roles'])=={'bass','guitar'} for x in report['cross_role_corroboration'])

def test_missing_midi_remains_inconclusive():
    chart=[]
    for i in range(3):chart+=phrase(i*2)
    report=srd.build(measures(6),{'guitar':chart},{'guitar':[]})
    assert all(x['classification']=='inconclusive' for x in report['phrase_run_correspondences'])
    assert not any(x['proposal_type']=='phrase_run_ordinal_constraint' for x in report['structural_anchor_proposals'])


def test_cleanup_excess_occurrences_are_not_called_complete():
    chart=[];midi=[]
    for i in range(2):chart+=phrase(i*2)
    for i in range(4):midi+=phrase(10+i*2)
    report=srd.build(measures(8),{'bass':chart},{'bass':midi})
    row=next(x for x in report['phrase_run_correspondences'] if x['phrase_measures']==2)
    assert row['confirmed_midi_occurrences']>row['chart_occurrence_count']
    assert row['classification'] in {'additional_midi_occurrences','multiple_possible_run_instances'}
    assert row['excess_confirmed_occurrences']>0
    assert not any(x['proposal_type']=='phrase_run_ordinal_constraint' for x in report['structural_anchor_proposals'])


def test_cleanup_shifted_overlapping_runs_cluster_as_alternative_origins():
    chart=[];midi=[]
    for i in range(4):chart+=phrase(i*2)
    for i in range(4):midi+=phrase(10+i*2)
    report=srd.build(measures(8),{'piano':chart},{'piano':midi})
    clusters=report['overlapping_hypothesis_clusters']
    assert any(x['alternative_phase_origin_count']>1 for x in clusters)
    cluster=next(x for x in clusters if x['alternative_phase_origin_count']>1)
    assert cluster['selected_run_id'] in cluster['member_run_ids']
    assert cluster['diagnostic_only'] is True


def test_cleanup_hierarchical_cross_role_corroboration_accepts_2m_and_4m():
    chart={'bass':[],'guitar':[]};midi={'bass':[],'guitar':[]}
    for i in range(4):
        chart['bass']+=phrase(i*2,base=40);midi['bass']+=phrase(10+i*2,base=40)
        chart['guitar']+=phrase(i*2,base=60);midi['guitar']+=phrase(10+i*2,base=60)
    report=srd.build(measures(8),chart,midi)
    groups=report['cross_role_corroboration']
    assert groups
    assert any(set(x['roles'])=={'bass','guitar'} for x in groups)
    assert all(x['diagnostic_only'] is True for x in groups)
