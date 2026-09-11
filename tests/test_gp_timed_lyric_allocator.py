from process_gp_alignment import allocate_timed_gp_lyrics

def slot(t,midi=60,measure=1,d=.25):
    return {"t_gp":t,"end_gp":t+d,"midi":midi,"measure":measure}

def test_anchor_text_then_continuations_until_next_anchor():
    slots=[slot(1.0),slot(1.25),slot(1.5),slot(2.0,62)]
    anchors=[{"t_gp":1.0,"w":"burn-","measure":1},{"t_gp":2.0,"w":"ing","measure":2}]
    lyrics,pitch,diag=allocate_timed_gp_lyrics(slots,anchors)
    assert [x["w"] for x in lyrics]==["burn-","+","+","ing"]
    assert [x["t_gp"] for x in lyrics]==[x["t_gp"] for x in pitch]
    assert diag["unresolved_vocal_notes"]==0

def test_allocator_crosses_measure_boundary():
    slots=[slot(1.0,measure=1),slot(1.5,measure=2),slot(2.0,measure=2)]
    anchors=[{"t_gp":1.0,"w":"word","measure":1},{"t_gp":2.0,"w":"next","measure":2}]
    lyrics,_,_=allocate_timed_gp_lyrics(slots,anchors)
    assert [(x["measure"],x["w"]) for x in lyrics]==[(1,"word"),(2,"+"),(2,"next")]

def test_notes_before_first_anchor_are_not_guessed():
    slots=[slot(.5),slot(1.0)]
    anchors=[{"t_gp":1.0,"w":"start","measure":1}]
    lyrics,_,diag=allocate_timed_gp_lyrics(slots,anchors)
    assert [x["w"] for x in lyrics]==["start"]
    assert diag["unresolved_vocal_notes"]==1

def test_anchor_without_near_note_is_reported():
    slots=[slot(5.0)]
    anchors=[{"t_gp":1.0,"w":"lost","measure":1}]
    lyrics,pitch,diag=allocate_timed_gp_lyrics(slots,anchors)
    assert lyrics==[] and pitch==[]
    assert diag["anchors_without_notes"][0]["reason"]=="no_vocal_note_near_anchor"












