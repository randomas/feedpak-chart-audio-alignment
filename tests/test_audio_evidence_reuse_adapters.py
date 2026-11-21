import copy
import pytest
import audio_evidence_schema as aes


def test_opening_adapter_preserves_order_and_origin_values():
    rows=[{'source':'guitar','score_start':2.0,'expected_time':3.0,'audio_time':2.8,'confidence':.7,'accepted':True},
          {'source':'bass','score_start':2.0,'expected_time':3.0,'audio_time':3.2,'confidence':.5,'accepted':False}]
    original=copy.deepcopy(rows)
    out=aes.adapt_opening_candidates(rows,padding_added=1.0)
    assert [x['source'] for x in out]==['guitar','bass']
    assert out[0]['clocks']['source_audio_time']==pytest.approx(1.8)
    assert out[0]['clocks']['padded_audio_time']==2.8
    assert out[0]['clocks']['chart_time']==2.0
    assert out[0]['clocks']['mapped_audio_time']==3.0
    assert abs(out[0]['residual_seconds']+.2)<1e-9
    assert 'not_selected_by_origin_detector' in out[1]['limitations']
    assert rows==original


def test_coverage_and_terminal_adapters_do_not_execute_detection():
    coverage={'available':True,'audio_activity_start':4.0,'audio_activity_end':20.0,'confidence':.8}
    terminal={'available':True,'content_end':30.0,'file_duration':31.0,'threshold_rms':.01}
    cov=aes.adapt_coverage_report(coverage,source='guitar',padding_added=1.5)
    term=aes.adapt_terminal_report(30.0,terminal,padding_added=1.5)
    assert [x['evidence_type'] for x in cov]==['source_activity_start','source_activity_end']
    assert cov[0]['clocks']['source_audio_time']==2.5
    assert term[0]['clocks']['source_audio_time']==28.5
    assert term[0]['detector']['method']=='detect_audio_content_end'


def test_competing_structural_hypotheses_are_not_collapsed():
    report={'version':'1','clusters':[{'cluster_id':'a','chart_start':10,'audio_start':12,'score':.8},
                                      {'cluster_id':'b','chart_start':10,'audio_start':16,'score':.78}]}
    out=aes.adapt_structural_search_report(report)
    assert len(out)==2
    assert [x['provenance']['cluster_id'] for x in out]==['a','b']


def test_missing_audio_evidence_is_neutral():
    bundle=aes.build_read_only_evidence_bundle()
    assert bundle['summary']['evidence_count']==0
    assert bundle['midi_authority_changed'] is False
    assert bundle['timing_changes_applied'] is False
    assert bundle['detectors_executed'] is False


def test_bundle_aggregates_without_modifying_origins():
    opening=[{'source':'piano','score_start':1.0,'expected_time':2.0,'audio_time':2.1}]
    coverage={'piano':{'available':True,'audio_start':2.1,'audio_end':40.0}}
    original=(copy.deepcopy(opening),copy.deepcopy(coverage))
    bundle=aes.build_read_only_evidence_bundle(opening_candidates=opening,coverage_reports=coverage,padding_added=1.0)
    assert bundle['summary']['evidence_count']==3
    assert opening==original[0] and coverage==original[1]


def test_exact_continuous_report_shape_maps_nested_retained_candidates():
    report={'version':4,'matches':[
        {'score_event_id':'e1','source':'guitar','kind':'restart_after_measure_silence','score_time':12.0,
         'baseline_prediction':13.0,'candidate_count':1,'candidates':[
             {'time':12.8,'confidence':.84,'residual_seconds':-.2,'evidence_source':'guitar',
              'corresponding_stem':'guitar','raw_audio_confidence':.9,'score_confidence':.8,
              'quiet_before':.7,'activity_rise':.8,'onset_support':.9,'single_stem_sufficient':True}]}]}
    out=aes.adapt_continuous_anchor_report(report,padding_added=2.0)
    assert len(out)==1
    row=out[0]
    assert row['evidence_type']=='retained_local_audio_hypothesis'
    assert row['clocks']=={'source_audio_time':10.8,'padded_audio_time':12.8,'chart_time':12.0,'mapped_audio_time':13.0}
    assert row['confidence']==.84
    assert row['native_quality']['raw_audio_confidence']==.9
    assert row['adapter_status']=='complete'


def test_exact_structural_report_shape_maps_clusters_not_raw_hypotheses():
    member={'source_start_measure':10,'target_start_measure':14,'audio_shift_seconds':8.0,'window_uniqueness_margin':.12}
    report={'schema_version':'4.3','hypotheses':[{'hypothesis_id':'raw'}],
        'clusters':[{'cluster_id':'c1','members':[member],'best_score':.81,'median_baseline_improvement':.2,
                     'persistence_status':'multi_window','occurrence_uniqueness':'locally_distinctive',
                     'sequence_validation':{'status':'supported'}}]}
    out=aes.adapt_structural_search_report(report)
    assert len(out)==1
    assert out[0]['provenance']['source_start_measure']==10
    assert out[0]['native_quality']['audio_shift_seconds']==8.0
    assert out[0]['confidence']==.81


def test_exact_instrument_report_shape_maps_only_accepted_selected_measures():
    selected={'residual_ms':120.0,'calibrated_confidence':.75}
    report={'schema_version':'3B.1','roles':{
      'guitar':{'feature_family':'harmonic','measures':[{'playback_measure':4,'accepted':True,'selected_candidate':selected,
          'neighbour_supported':True,'cross_role_supported':False}]},
      'bass':{'feature_family':'low_frequency','measures':[{'playback_measure':4,'accepted':True,
          'selected_candidate':{'residual_ms':100.0,'calibrated_confidence':.85},
          'neighbour_supported':False,'cross_role_supported':True},
          {'playback_measure':5,'accepted':False,'selected_candidate':selected}]}}}
    out=aes.adapt_instrument_evidence_report(report)
    assert len(out)==1
    assert out[0]['provenance']['playback_measure']==4
    assert out[0]['residual_seconds']==.11
    assert out[0]['confidence']==.8
    assert out[0]['source']=='bass+guitar'


def test_exact_opening_report_shape_maps_selected_anchor():
    report={'status':'fused_early_instrument_anchors','selected_sources':['piano','guitar'],
            'score_anchor':2.0,'audio_anchor':2.3,'warnings':[]}
    out=aes.adapt_opening_report(report,padding_added=.5)
    assert len(out)==1
    assert out[0]['clocks']['source_audio_time']==pytest.approx(1.8)
    assert out[0]['clocks']['padded_audio_time']==2.3
    assert out[0]['clocks']['chart_time']==2.0
    assert out[0]['clocks']['mapped_audio_time']==2.3
    assert out[0]['residual_seconds']==pytest.approx(.3)
