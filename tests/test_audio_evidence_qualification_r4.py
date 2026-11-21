from copy import deepcopy
import audio_evidence_qualification as q


def row(eid,residual=.05,confidence=.9,source='guitar',physical='stem-a',etype='retained_local_audio_hypothesis',limitations=None):
    return {'evidence_id':eid,'evidence_type':etype,'source':source,'residual_seconds':residual,
        'confidence':confidence,'adapter_status':'complete','limitations':limitations or [],
        'positive_factors':['retained'],'clocks':{'chart_time':10.0,'padded_audio_time':10.05,'mapped_audio_time':10.0},
        'provenance':{'physical_audio_source':physical}}


def test_qualification_is_read_only_and_strict_near_baseline():
    evidence=row('e1');original=deepcopy(evidence);result=q.qualify_observation(evidence)
    assert evidence==original
    assert result['qualification']=='near_baseline_support'
    assert result['strict_eligible'] is True
    assert result['timing_changes_applied'] is False and result['midi_authority_changed'] is False


def test_categories_preserve_alternatives_without_authority():
    assert q.qualify_observation(row('a',.2))['qualification']=='compatible_displaced_candidate'
    assert q.qualify_observation(row('b',.4))['qualification']=='ambiguous_alternative'
    assert q.qualify_observation(row('c',1.2,.95))['qualification']=='credible_displaced_candidate'
    structural=row('d',None,None,etype='retained_structural_occurrence_cluster')
    assert q.qualify_observation(structural)['qualification']=='structural_recurrence_support'
    assert q.qualify_observation(structural)['strict_eligible'] is False


def test_missing_clocks_confidence_or_limitations_is_not_strict():
    evidence=row('e1');evidence['clocks'].pop('padded_audio_time')
    result=q.qualify_observation(evidence)
    assert not result['strict_eligible'] and 'three_clock_mapping_incomplete' in result['strict_ineligibility_reasons']
    assert not q.qualify_observation(row('e2',.05,.5))['strict_eligible']
    assert not q.qualify_observation(row('e3',.05,.9,limitations=['contaminated']))['strict_eligible']


def test_bundle_groups_correlated_observations_by_physical_source():
    bundle={'schema_version':'r3.1','evidence':[row('solo',physical='keys.ogg'),row('pad',physical='keys.ogg'),row('bass',physical='bass.ogg')]}
    report=q.qualify_bundle(bundle)
    assert report['summary']['strict_eligible_count']==3
    assert report['summary']['physical_source_count']==2
    assert report['dependent_physical_sources']['keys.ogg']==['solo','pad']
    assert report['detectors_executed'] is False


def test_strict_query_counts_one_vote_per_physical_source():
    report=q.qualify_bundle({'evidence':[row('solo',physical='keys.ogg'),row('pad',physical='keys.ogg')]})
    answer=q.answer_strict_query(report,'materialization_gap_boundary')
    assert answer['status']=='strict_audio_support_available'
    assert answer['alternative_count']==2
    assert answer['independent_physical_source_count']==1
    empty=q.answer_strict_query(q.qualify_bundle({'evidence':[]}), 'terminal_mismatch')
    assert empty['status']=='insufficient_strict_audio_evidence'


def test_runtime_wiring_is_diagnostic_only():
    process=open('process_gp_alignment.py',encoding='utf-8').read()
    builder=open('build_feedpak.py',encoding='utf-8').read()
    assert 'aeq.qualify_bundle(audio_evidence_bundle)' in process
    assert 'audio_evidence_qualification_summary' in process
    assert '.audio_evidence_qualification.json' in builder
    assert 'timing_changes_applied":False' in open('audio_evidence_qualification.py',encoding='utf-8').read()
