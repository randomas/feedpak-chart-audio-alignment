import sys,types
if 'instrument_evidence' not in sys.modules:
    sys.modules['instrument_evidence']=types.SimpleNamespace(extract_role_feature=lambda *a,**k:{'status':'unavailable'},measure_candidates=lambda *a,**k:{'candidates':[]})
import structural_search as ss
import numpy as np


def cluster(start,end,shift,score):
    return {'source_start_measure':start,'source_end_measure':end,'representative_measure_shift':shift,
            'best_score':score,'median_baseline_improvement':.2,'supporting_window_count':2}


def test_overlap_competition_marks_near_tie_ambiguous():
    a=cluster(47,50,-4,.6605);b=cluster(47,57,-8,.6588)
    out=ss._rank_competing_clusters([a,b])
    assert all(x['occurrence_uniqueness']=='ambiguous_repeated_occurrence' for x in out)
    assert abs(out[0]['competing_cluster_margin']-.0017)<.001


def test_no_competitor_does_not_mean_unique():
    out=ss._rank_competing_clusters([cluster(70,77,-2,.8)])
    assert out[0]['occurrence_uniqueness']=='uniqueness_not_established'
    assert out[0]['competing_cluster_margin'] is None


def test_shift_precision_separate_from_occurrence_uniqueness():
    c=cluster(10,18,3,.8);c['member_margin']=.01
    out=ss._rank_competing_clusters([c])[0]
    assert out['shift_precision']=='coarse'
    assert out['occurrence_uniqueness']=='uniqueness_not_established'


def test_common_similarity_uses_same_representation():
    times=np.arange(100)*.02;env=np.sin(times*3)**2
    left={'status':'ok','times':times,'envelopes':{'rms':env,'broadband_flux':env,'low_frequency_energy':env}}
    right={'status':'ok','times':times,'envelopes':{'rms':env.copy(),'broadband_flux':env.copy(),'low_frequency_energy':env.copy()}}
    assert ss._common_similarity(left,right,0,2)>.99


def test_regional_quality_can_differ_from_global():
    times=np.arange(200)*.02
    first=np.r_[np.sin(times[:100]*3)**2,np.zeros(100)]
    second=np.r_[np.sin(times[:100]*3)**2,np.random.default_rng(1).random(100)]
    prepared={'a':{'common_feature':{'status':'ok','times':times,'envelopes':{'rms':first}}},
              'b':{'common_feature':{'status':'ok','times':times,'envelopes':{'rms':second}}}}
    base={'sources':{},'pairwise_similarity':[{'sources':['a','b'],'observed_similarity':.3}]}
    q=ss._regional_quality(base,prepared,0,1.8)
    assert q['pairwise_similarity'][0]['observed_similarity']>.6


def test_near_miss_fields_exist_without_search():
    r=ss.analyse({},lambda x:x,[{'t_gp':0,'measure':1},{'t_gp':2,'measure':2}],{'coverage_gaps':[]})
    assert r['phase'] in ('3C.4.2','3C.4.2a','3C.4.3')
    assert 'rejected_candidate_summary' in r and 'near_miss_candidates' in r
