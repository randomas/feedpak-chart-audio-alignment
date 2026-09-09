import os,tempfile,unittest
from types import SimpleNamespace as N
import project_config as p
def tr(name,perc=False):return N(name=name,number=1,isPercussionTrack=perc,channel=N(instrument=0),strings=[],measures=[N(voices=[N(beats=[N(notes=[N()],text=None)])])])
class T(unittest.TestCase):
 def setUp(self):self.i=p.inventory_song(N(tracks=[tr('Lead Vocals'),tr('Backing Vocals'),tr('Lead Guitar'),tr('Bass'),tr('Piano (LH)'),tr('Piano (RH)'),tr('Drums',True),tr('Triangle',True),tr('Roger Walters')]))
 def test_create(self):
  with tempfile.TemporaryDirectory() as d:
   x,c=p.ensure_song_json(os.path.join(d,'metadata.json'),'song.gp5',self.i);self.assertTrue(c);self.assertIn('Roger Walters',x['feedpak_project']['tracks']['unused_tracks'])
 def test_typo(self):
  x={'feedpak_project':{'tracks':p.default_allocation(self.i)}};x['feedpak_project']['tracks']['lead_vocal']='Lead Vocal';self.assertFalse(p.validate_song_config(x,self.i)['valid'])
 def test_unknown(self):self.assertEqual(p.classify_name('Roger Walters'),'unsupported')
if __name__=='__main__':unittest.main()















