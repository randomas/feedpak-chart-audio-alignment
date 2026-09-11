from process_gp_alignment import _parse_gp_lyric_items, allocate_measure_lyric_line, playable_keyboard_from_notation

def slot(t,m=1,p=60): return {'t_gp':t,'end_gp':t+.25,'midi':p,'measure':m}

def test_embedded_hyphens_become_note_syllables():
    assert [x['value'] for x in _parse_gp_lyric_items('To-ni-ght gon-na my-self _')]==['To-','ni-','ght','gon-','na','my-','self','+']

def test_lyrics_use_real_note_onsets_and_stay_in_measures():
    slots=[slot(.5,1),slot(1,1),slot(1.5,1),slot(3,2),slot(3.5,2)]
    words,pitch,d=allocate_measure_lyric_line(slots,'To-ni-ght gon-na',1)
    assert [x['t_gp'] for x in words]==[.5,1,1.5,3,3.5]
    assert [x['measure'] for x in words]==[1,1,1,2,2]
    assert [x['w'] for x in words]==['To-','ni-','ght','gon-','na']
    assert len(words)==len(pitch)

def test_piano_duration_is_d_and_release_precedes_resume():
    notation={'source_track':'Piano (RH)','measures':[
      {'idx':71,'staves':{'rh':{'voices':[{'beats':[{'t':121.9266,'notes':[{'midi':60,'d':.3852}]}]}]}}},
      *[{'idx':i} for i in range(72,81)],
      {'idx':81,'staves':{'rh':{'voices':[{'beats':[{'t':137.3344,'notes':[{'midi':65,'d':.3852}]}]}]}}}]}
    notes=playable_keyboard_from_notation(notation)['notes']
    assert notes[0]=={'t':121.9266,'s':2,'f':12,'d':.3852}
    assert 'sus' not in notes[0]
    assert notes[0]['t']+notes[0]['d']<notes[1]['t']



