import dataclasses
import math
import statistics
import midi_section_fit as msf


def reference_align(chart,midi,cfg):
    n,m=len(chart),len(midi);neg=-1e12
    dp=[[neg]*(m+1) for _ in range(n+1)];back=[[None]*(m+1) for _ in range(n+1)]
    dp[0][0]=0.0
    for i in range(1,n+1):
        dp[i][0]=dp[i-1][0]-cfg.missing_chart_penalty;back[i][0]=(i-1,0,'skip_chart')
    for j in range(1,m+1):
        dp[0][j]=dp[0][j-1]-cfg.extra_midi_penalty;back[0][j]=(0,j-1,'skip_midi')
    for i in range(1,n+1):
        for j in range(1,m+1):
            sim=msf._set_similarity(set(chart[i-1]['pitches']),set(midi[j-1]['pitches']),cfg)
            options=[(dp[i-1][j]-cfg.missing_chart_penalty,i-1,j,'skip_chart'),
                     (dp[i][j-1]-cfg.extra_midi_penalty,i,j-1,'skip_midi')]
            if sim>0:options.append((dp[i-1][j-1]+sim,i-1,j-1,'match'))
            dp[i][j],pi,pj,op=max(options,key=lambda x:x[0]);back[i][j]=(pi,pj,op)
    pairs=[];i=n;j=m
    while i or j:
        pi,pj,op=back[i][j]
        if op=='match':
            sim=msf._set_similarity(set(chart[i-1]['pitches']),set(midi[j-1]['pitches']),cfg)
            pairs.append((i-1,j-1,sim))
        i,j=pi,pj
    pairs.reverse()
    return dp[n][m]/max(1,n),pairs


def window():
    chart=[{'time':i*.5,'pitches':{60+(i%5)}} for i in range(16)]
    return msf.MeasureWindow('w','piano',0,4,[1,2,3,4],[1,2,3,4],0.,8.,8.,1,
                             chart,chart,'observable','')


def midi_events():
    return [{'audio_time':.2+i*.5,'pitch':60+(i%5)} for i in range(40)]


def test_similarity_matrix_dp_is_exact_reference_parity():
    cfg=msf.SectionFitConfig();w=window();prepared=msf.prepare_grouped_midi(midi_events(),cfg)
    local=prepared['groups'][0:20]
    assert msf._align_ordered_groups(w.groups,local,cfg)==reference_align(w.groups,local,cfg)


def test_indexed_slice_matches_inclusive_reference_scan():
    cfg=msf.SectionFitConfig();prepared=msf.prepare_grouped_midi(midi_events(),cfg)
    lower=.2;upper=4.2
    indexed=prepared['groups'][msf.bisect_left(prepared['times'],lower):msf.bisect_right(prepared['times'],upper)]
    reference=[x for x in prepared['groups'] if lower<=x['time']<=upper]
    assert indexed==reference


def test_prepared_and_raw_search_are_candidate_identical():
    cfg=msf.SectionFitConfig(search_radius_seconds=1.,translation_step_seconds=.2,
                             stretch_min=.95,stretch_max=1.05,stretch_step=.05)
    w=window();raw=midi_events();prepared=msf.prepare_grouped_midi(raw,cfg)
    a=msf.search_section_candidates(w,raw,lambda t:t,cfg)
    b=msf.search_section_candidates(w,prepared,lambda t:t,cfg)
    assert [dataclasses.asdict(x) for x in a]==[dataclasses.asdict(x) for x in b]


def test_cache_reduces_dp_without_changing_candidates():
    cfg=msf.SectionFitConfig(search_radius_seconds=1.,translation_step_seconds=.1,
                             stretch_min=.95,stretch_max=1.05,stretch_step=.01)
    counters={'transform_trials':0,'midi_slice_lookups':0,'dp_evaluations':0,'alignment_cache_hits':0}
    cache={};w=window();prepared=msf.prepare_grouped_midi(midi_events(),cfg)
    candidates=msf.search_section_candidates(w,prepared,lambda t:t,cfg,'prepared_chart_clock',cache,counters)
    assert candidates
    assert counters['transform_trials']>counters['dp_evaluations']
    assert counters['alignment_cache_hits']>0
    assert counters['dp_evaluations']==len(cache)


def test_candidate_report_includes_actionable_timers_and_counters():
    measures=[{'playback_index':i,'playback_measure_number':i+1,'written_measure_number':i+1,
               'start_time':i*2.,'end_time':(i+1)*2.} for i in range(4)]
    chart={'piano':[{'time':i*.5,'midi':60+(i%5)} for i in range(16)]}
    midi={'piano':midi_events()}
    cfg=msf.SectionFitConfig(search_radius_seconds=.5,translation_step_seconds=.25,
                             stretch_min=.95,stretch_max=1.05,stretch_step=.05)
    report=msf.build_section_candidate_report(measures,chart,midi,cfg=cfg)
    perf=report['performance']
    for key in ('transform_trials','midi_slice_lookups','dp_evaluations','alignment_cache_hits',
                'unique_midi_slices','midi_grouping_seconds','candidate_search_seconds','cache_hit_rate'):
        assert key in perf
    assert perf['transform_trials']>=perf['dp_evaluations']


def test_materialization_report_is_json_serializable():
    import json
    region={"region_id":"r1","chart_start":0.,"chart_end":10.,"monotonic":True,
        "strong_control_count":3,"catch_up_events":[],"controls":[
        {"score_time":0.,"audio_time":0.,"classification":"single_role_strong","roles":["piano"]},
        {"score_time":5.,"audio_time":5.1,"classification":"single_role_strong","roles":["piano"]},
        {"score_time":10.,"audio_time":10.2,"classification":"single_role_strong","roles":["piano"]}]}
    _,report=msf.build_section_materialized_warp({"regions":[region]},lambda t:t,"permissive")
    encoded=json.dumps(report,allow_nan=False)
    assert '"tiers": ["A", "B"]' in encoded


def _cache_fixture():
    measures=[{'playback_index':i,'playback_measure_number':i+1,'written_measure_number':i+1,
               'start_time':i*2.,'end_time':(i+1)*2.} for i in range(4)]
    chart={'piano':[{'time':i*.5,'midi':60+(i%5)} for i in range(16)]}
    midi={'piano':midi_events()}
    cfg=msf.SectionFitConfig(search_radius_seconds=.5,translation_step_seconds=.25,
                             stretch_min=.95,stretch_max=1.05,stretch_step=.05)
    return measures,chart,midi,cfg


def test_persistent_candidate_cache_exact_hit_and_invalidation(tmp_path):
    measures,chart,midi,cfg=_cache_fixture();cache=tmp_path/'candidate-cache.json'
    first=msf.build_section_fusion_report(measures,chart,midi,candidate_config=cfg,
                                          candidate_cache_path=str(cache))
    assert cache.exists()
    assert first['performance']['candidate_cache']['hit'] is False
    second=msf.build_section_fusion_report(measures,chart,midi,candidate_config=cfg,
                                           candidate_cache_path=str(cache))
    assert second['performance']['candidate_cache']['hit'] is True
    assert second['performance']['candidate_cache']['reason']=='exact_cache_hit'
    assert first['candidate_report']==second['candidate_report']
    changed={**midi,'piano':midi['piano']+[{'audio_time':99.,'pitch':72}]}
    third=msf.build_section_fusion_report(measures,chart,changed,candidate_config=cfg,
                                          candidate_cache_path=str(cache))
    assert third['performance']['candidate_cache']['hit'] is False
    assert third['performance']['candidate_cache']['reason']=='input_fingerprint_changed'


def test_persistent_cache_rebuild_and_disable(tmp_path):
    measures,chart,midi,cfg=_cache_fixture();cache=tmp_path/'candidate-cache.json'
    msf.build_section_fusion_report(measures,chart,midi,candidate_config=cfg,candidate_cache_path=str(cache))
    rebuilt=msf.build_section_fusion_report(measures,chart,midi,candidate_config=cfg,
        candidate_cache_path=str(cache),candidate_cache_rebuild=True)
    assert rebuilt['performance']['candidate_cache']['hit'] is False
    assert rebuilt['performance']['candidate_cache']['reason']=='cache_rebuild_requested'
    disabled=msf.build_section_fusion_report(measures,chart,midi,candidate_config=cfg,
        candidate_cache_path=str(cache),candidate_cache_disabled=True)
    assert disabled['performance']['candidate_cache']['enabled'] is False
    assert disabled['performance']['candidate_cache']['reason']=='cache_disabled'
