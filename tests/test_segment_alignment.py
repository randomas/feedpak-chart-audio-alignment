import unittest
from unittest.mock import patch
import numpy as np
import checkpoint_dtw as c
import process_gp_alignment as p
from types import SimpleNamespace as N

class FullFrameTests(unittest.TestCase):
    def test_full_frame_pins_known_start(self):
        fake=np.r_[np.zeros(1000),np.ones(8000)*.2,np.zeros(1000)]
        with patch.object(p.librosa,'load',return_value=(fake,1000)):
            w=p.compute_full_mix_frame_warp(2.5,10.0,'full.wav',2.5,sr=1000)
        self.assertAlmostEqual(w(2.5),2.5,places=6)
        self.assertEqual(w.alignment_diagnostics['selected_source'],'full')
        self.assertEqual(w.alignment_diagnostics['start_anchor_error_ms'],0.0)

class PitchTests(unittest.TestCase):
    def test_gp_note_value_is_added_to_open_string(self):
        track=N(strings=[N(number=1,value=64),N(number=2,value=59)])
        self.assertEqual(p.gp_note_to_midi(track,N(string=2,value=3)),62)
    def test_invalid_string_is_rejected(self):
        track=N(strings=[N(number=1,value=64)])
        with self.assertRaises(ValueError): p.gp_note_to_midi(track,N(string=2,value=3))

class SegmentBoundaryTests(unittest.TestCase):
    def test_four_measure_boundaries(self):
        beats=[{'measure':i+1,'t_gp':float(i),'ts_num':4} for i in range(10)]
        with patch.object(c,'_features',return_value=None):
            r=c.diagnose_segment_source_dtw(lambda t:t,beats,{}, {},segment_measures=4)
        self.assertEqual([(x['start_measure'],x['end_measure']) for x in r['segments']],[(1,5),(5,9),(9,10)])
        self.assertEqual(r['config']['global_frame_source'],'full')

if __name__=='__main__': unittest.main()
