import unittest
import checkpoint_dtw as c

class MeterGrid621Tests(unittest.TestCase):
    def test_inside_measure_segment_has_fractional_beats(self):
        beats=[{'measure':1,'t_gp':0.0,'ts_num':4},
               {'measure':2,'t_gp':4.0,'ts_num':4},
               {'measure':3,'t_gp':8.0,'ts_num':3}]
        p=c._meter_profile(beats,1.5,3.5)
        self.assertAlmostEqual(p['beat_units'],2.0)
        self.assertEqual(p['complete_intervals'],1)
        self.assertAlmostEqual(p['partial_start'],0.5)
        self.assertAlmostEqual(p['partial_end'],0.5)

    def test_cross_measure_mixed_meter(self):
        beats=[{'measure':1,'t_gp':0.0,'ts_num':3},
               {'measure':2,'t_gp':3.0,'ts_num':5},
               {'measure':3,'t_gp':8.0,'ts_num':4}]
        self.assertAlmostEqual(c._segment_beat_count(beats,1.0,7.0),6.0)

    def test_exact_measure_segment(self):
        beats=[{'measure':1,'t_gp':0.0,'ts_num':4},
               {'measure':2,'t_gp':4.0,'ts_num':4}]
        self.assertAlmostEqual(c._segment_beat_count(beats,0.0,4.0),4.0)

if __name__ == '__main__':
    unittest.main()





















