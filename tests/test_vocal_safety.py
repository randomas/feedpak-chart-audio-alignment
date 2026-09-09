from types import SimpleNamespace as N
import project_config as pc
from build_feedpak import validate_vocal_coverage

def _song(track=True,line=None,text=None):
    tracks=[N(name='Lead Vocals',measures=[N(voices=[N(beats=[N(start=0,text=text,notes=[N()])])])])] if track else []
    lines=[] if line is None else [N(lyrics=line,startingMeasure=1)]
    return N(tracks=tracks,lyrics=N(lines=lines))
INV=[{'name':'Lead Vocals','automatic_role':'lead_vocal'}]
ROLES={'Lead Vocals':'lead_vocal'}
def test_flat_is_rejected():
    c=pc.classify_vocal_capability(_song(line='flat blob'),INV,ROLES); assert c['classification']=='flat_untimed_lyrics' and not c['direct_gp_lyrics_supported']
def test_timed_is_accepted():
    c=pc.classify_vocal_capability(_song(line='hello',text='hello'),INV,ROLES); assert c['classification']=='timed_beat_text' and c['direct_gp_lyrics_supported']
def test_none_is_no_material():
    assert pc.classify_vocal_capability(_song(False),[],{})['classification']=='no_vocal_material'
def test_tail_warning():
    d={'speakers':{'s':{'words':[{'t':0,'d':1}],'pitch_notes':[{'t':0,'d':1}]}}}; assert validate_vocal_coverage(d,200)['status']=='warning'
