import numpy as np
import sys, types
if 'instrument_evidence' not in sys.modules:
    stub=types.SimpleNamespace(extract_role_feature=lambda *a,**k:{'status':'unavailable'},measure_candidates=lambda *a,**k:{'candidates':[]})
    sys.modules['instrument_evidence']=stub
import structural_search as ss


def quality(sim):
    return {'pairwise_similarity':[{'sources':['bass','drums'],'observed_similarity':sim}]}


def raw(start,end,shift,score=.8,window=4,roles=('bass','drums')):
    return {'source_start_measure':start,'source_end_measure':end,'measure_shift':shift,'audio_shift_seconds':shift*2,
            'combined_score':score,'median_improvement_over_local_baseline':.2,'window_size':window,
            'roles':list(roles),'effective_source_groups':1.8,'gap':{'start_playback_measure':1,'end_playback_measure':20}}


def test_highly_redundant_sources_are_discounted():
    assert ss._effective_source_groups(['bass','drums'],quality(.9)) < ss.MIN_EFFECTIVE_SOURCE_GROUPS
    assert ss._effective_source_groups(['bass','drums'],quality(.2)) >= ss.MIN_EFFECTIVE_SOURCE_GROUPS


def test_overlapping_nearby_shifts_cluster():
    clusters=ss._cluster_hypotheses([raw(10,13,3),raw(11,18,4,window=8)])
    assert len(clusters)==1
    assert clusters[0]['measure_shift_range']==[3,4]
    assert clusters[0]['persistence_status']=='multi_window'


def test_distant_shifts_remain_competing_clusters():
    clusters=ss._cluster_hypotheses([raw(10,13,-3),raw(10,13,4)])
    assert len(clusters)==2


def test_source_quality_marks_labels_unverified_and_detects_similarity():
    times=np.arange(0,20,.1);novelty=np.sin(times)**2
    prepared={'guitar':{'events':[1,2], 'feature':{'times':times,'novelty':novelty,'feature_family':'x'}},
              'piano':{'events':[1,2], 'feature':{'times':times,'novelty':novelty.copy(),'feature_family':'y'}}}
    result=ss.source_quality(prepared,lambda x:x,[{'start':0,'end':20}])
    assert result['sources']['guitar']['role_identity_status']=='unverified_declared_label'
    assert result['pairwise_similarity'][0]['dependence']=='high'
    assert sorted(result['dependence_groups'][0])==['guitar','piano']


def test_correspondence_output_has_no_temporary_timing_map():
    report={'proposals':[{'proposal_id':'global.baseline','method':'retain_baseline','comparison_score':{'total':0}}],'notes':[]}
    structural={'summary':{'retained_cluster_count':1},'clusters':[{
      'cluster_id':'c1','source_start_measure':10,'source_end_measure':18,'supporting_window_count':2,
      'best_score':.8,'median_baseline_improvement':.2,'persistence_status':'multi_window'}]}
    out=ss.append_to_alignment_proposals(report,structural,{})
    p=next(x for x in out['proposals'] if x['method']=='structural_correspondence_family')
    assert 'mapped_downbeats' not in p
    assert p['controls']==[]
    assert out['recommendation']=='retain_baseline'


def test_report_is_general_and_diagnostic(monkeypatch):
    import inspect
    source=inspect.getsource(ss).lower()
    for name in ('spellbound','supremacy','all that i bleed',"don't stop me now"):
        assert name not in source
    result=ss.analyse({},lambda x:x,[{'t_gp':0,'measure':1},{'t_gp':2,'measure':2}],{'coverage_gaps':[]})
    assert result['timing_changes_applied'] is False
    assert result['policy']['source_labels_are_unverified'] is True
