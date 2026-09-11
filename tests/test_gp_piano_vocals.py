import unittest
from types import SimpleNamespace as N
import process_gp_alignment as p

def dur(time=960,index=0): return N(time=time,index=index)
def beat(start,midis,text=None): return N(start=start,duration=dur(),notes=[N(value=x-60,string=1) for x in midis],text=text)
def measure(i,voices):
 h=N(start=(i-1)*3840,number=i,timeSignature=N(numerator=4,denominator=N(value=4)))
 return N(header=h,voices=[N(beats=x) for x in voices])
class GPInterpreterTests(unittest.TestCase):
 def test_separate_left_track_never_pitch_splits_and_preserves_voices(self):
  t=N(name='Piano (LH)',strings=[N(number=1,value=60)],measures=[measure(1,[[beat(0,[72])],[beat(960,[48])]])])
  n=p.parse_keyboard_track(t,[(0,120)],0,'left')
  self.assertEqual(set(n['measures'][0]['staves']),{'lh'})
  self.assertEqual([v['v'] for v in n['measures'][0]['staves']['lh']['voices']],[1,2])
 def test_note_duration_uses_warped_endpoints(self):
  t=N(name='Piano (RH)',strings=[N(number=1,value=60)],measures=[measure(1,[[beat(0,[60])]])])
  n=p.parse_keyboard_track(t,[(0,120)],0,'right')
  w=p.shift_and_warp_notation(n,2.0,lambda x:x*2)
  note=w['measures'][0]['staves']['rh']['voices'][0]['beats'][0]['notes'][0]
  self.assertEqual(note['d'],1.0)
 def test_bad_vocal_bar_is_skipped_not_whole_song(self):
  good=measure(1,[[beat(0,[60],'hello')]])
  bad=measure(2,[[beat(3840,[62,64],'bad')]])
  good2=measure(3,[[beat(7680,[65],'again')]])
  track=N(name='Lead Vocals',strings=[N(number=1,value=60)],measures=[good,bad,good2])
  song=N(lyrics=N(lines=[]))
  v=p.parse_gp_vocals(song,track,[(0,120)],0)
  self.assertEqual(v['diagnostics']['bars_written'],2)
  self.assertEqual(v['diagnostics']['bars_skipped'],1)
  self.assertEqual([x['w'] for x in v['lyrics']],['hello','again'])
if __name__=='__main__': unittest.main()





















