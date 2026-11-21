import evidence_contribution_ledger as q


def audio(t,eid='a'):
 return {'evidence_id':eid,'evidence_type':'retained_local_audio_hypothesis','source':'guitar','confidence':.9,
  'clocks':{'chart_time':t,'mapped_audio_time':t+1,'padded_audio_time':t+1},'limitations':[]}

def midi_region(start,end,controls,applied=None,status='available'):
 return {'fusion_report':{'regions':[{'region_id':'source','chart_start':start,'chart_end':end,
  'controls':[{'score_time':c,'audio_time':a,'roles':['guitar']} for c,a in controls]}]},
  'application_report':{'status':status,'global_monotonicity_validated':status!='materialization_blocked',
  'applied_regions':applied or []}}


def test_coverage_incipit_is_exact_cross_modal_when_clocks_agree():
 coverage={'guitar':{'gp':{'first_note_time':142.768},'audio':{'first_sustained_activity':142.780},'comparison':{'classification':'matches'}}}
 report=midi_region(140,146,[(142.77,142.79)],applied=[{'region_id':'trim','source_region_id':'source','chart_start':140,'chart_end':146}])
 result=q.build_ledger(audio_evidence_bundle={'evidence':[]},midi_section_fit_report=report,coverage_reports=coverage)
 event=min(result['events'],key=lambda x:abs(x['chart_time']-142.768))
 assert event['status']=='exact_cross_modal_onset'
 assert event['midi_survived_materialization']
 assert event['audio_clocks'][0]['padded_audio_time']==142.780


def test_audio_inside_region_is_same_phrase_even_without_nearby_control():
 report=midi_region(140,155,[(140,141),(146,147)],status='materialization_blocked')
 result=q.build_ledger(audio_evidence_bundle={'evidence':[audio(149.6)]},midi_section_fit_report=report)
 events=[x for x in result['events'] if x['status']=='same_phrase_cross_modal_support']
 assert events
 assert events[0]['materialization_blocked_elsewhere']


def test_applied_interval_not_region_id_controls_survival():
 report=midi_region(0,20,[(5,6),(15,16)],applied=[{'region_id':'trimmed-different-id','source_region_id':'source','chart_start':0,'chart_end':10}])
 result=q.build_ledger(audio_evidence_bundle={'evidence':[]},midi_section_fit_report=report)
 by_time={round(x['chart_time']):x for x in result['events']}
 assert by_time[5]['midi_survived_materialization']
 assert not by_time[15]['midi_survived_materialization']


def test_distinguishes_same_phrase_from_exact_onset():
 report=midi_region(100,120,[(100,101),(120,121)],applied=[{'chart_start':100,'chart_end':120}])
 result=q.build_ledger(audio_evidence_bundle={'evidence':[audio(110)]},midi_section_fit_report=report)
 assert any(x['status']=='same_phrase_cross_modal_support' for x in result['events'])
 assert not any(x['status']=='exact_cross_modal_onset' for x in result['events'])


def test_runtime_passes_coverage_and_light_retains_clocks():
 process=open('process_gp_alignment.py',encoding='utf-8').read();collector=open('collect_run_diagnostics.py',encoding='utf-8').read()
 assert 'coverage_reports={"guitar":guitar_coverage}' in process
 assert '"audio_clocks":event.get("audio_clocks")' in collector
 assert '"midi_clocks":event.get("midi_clocks")' in collector
