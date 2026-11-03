import rooted_phrase_support as rps


def candidate(start=0,end=8,mstart=1,mend=9,score=.8):
    return {'window_id':'w','role':'piano','chart_start':start,'chart_end':end,
      'midi_start':mstart,'midi_end':mend,'combined_score':score}


def candidate_report(rows):
    return {'roles':{'piano':{'windows':[{'window':{'window_id':'w'},'candidates':rows}]}}}


def structural(selected='run1',classification='matching_phrase_run',boundary=.6,confidence=.85):
    return {'phrase_repeat_runs':[{'run_id':'run1','role':'piano','chart_occurrences':[
              {'ordinal':1,'start':0.0,'end':4.0},{'ordinal':2,'start':4.0,'end':8.0}]}],
      'phrase_run_correspondences':[{'run_id':'run1','role':'piano','classification':classification,
          'boundary_context_score':boundary,'joint_confidence':confidence,
          'midi_occurrences':[{'ordinal':1,'start':1.0,'end':5.0},{'ordinal':2,'start':5.0,'end':9.0}]}],
      'overlapping_hypothesis_clusters':[{'selected_run_id':selected}],
      'structural_anchor_proposals':[{'proposal_type':'phrase_run_ordinal_constraint','run_id':'run1','role':'piano'}]}


def fusion(rooted=True):
    return {'three_clock_incipit':{'status':'validated' if rooted else 'no_opening_candidate'},
      'solo_role_authority':{},'regions':[{'controls':[{'score_time':0.0,'audio_time':1.0}]}]}


def test_rooted_exact_phrase_rescores_existing_candidates_without_deleting():
    compatible=candidate();wrong=candidate(mstart=5,mend=13,score=.9)
    original=candidate_report([compatible,wrong])
    result,authority=rps.apply(original,structural(),fusion())
    rows=result['roles']['piano']['windows'][0]['candidates']
    assert len(rows)==2
    assert authority['eligible_run_count']==1
    assert authority['compatible_candidate_count']==1
    assert authority['incompatible_candidate_count']==1
    assert rows[0]['combined_score']>compatible['combined_score']
    assert rows[1]['combined_score']<wrong['combined_score']
    assert authority['candidates_deleted'] is False
    assert authority['phrase_controls_created'] is False
    assert original['roles']['piano']['windows'][0]['candidates'][0]['combined_score']==.8


def test_unrooted_song_is_neutral():
    original=candidate_report([candidate()])
    result,authority=rps.apply(original,structural(),fusion(False))
    assert authority['root_validated'] is False
    assert authority['adjusted_candidate_count']==0
    assert result==original


def test_nonselected_shifted_phrase_origin_is_rejected():
    result,authority=rps.apply(candidate_report([candidate()]),structural(selected='other'),fusion())
    assert authority['adjusted_candidate_count']==0
    assert authority['rejected_runs'][0]['reason']=='not_selected_cluster_representative'


def test_phrase_origin_must_agree_with_validated_root():
    data=structural();data['phrase_run_correspondences'][0]['midi_occurrences'][0]['start']=3.0
    _,authority=rps.apply(candidate_report([candidate()]),data,fusion())
    assert authority['adjusted_candidate_count']==0
    assert authority['rejected_runs'][0]['reason']=='phrase_origin_disagrees_with_validated_root'


def test_count_mismatch_never_receives_authority():
    _,authority=rps.apply(candidate_report([candidate()]),structural(classification='phrase_count_mismatch'),fusion())
    assert authority['adjusted_candidate_count']==0
    assert authority['rejected_runs'][0]['reason']=='occurrence_count_not_exact'
