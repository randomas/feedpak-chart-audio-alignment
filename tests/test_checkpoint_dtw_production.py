import unittest
import checkpoint_dtw as c

BEATS=[{'measure':i+1,'t_gp':float(i),'ts_num':4} for i in range(9)]

def segment(state='would_apply', residuals=None):
    residuals=residuals or [0,0.002,0.004,0.006,0.008,0.006,0.004,0.002,0]
    return {'segment_id':'m001-m009','start_nominal_time':0.0,'end_nominal_time':8.0,
      'decision':{'state':state},'path':{'control_nominal_times':[0,1,2,3,4,5,6,7,8],
      'control_residual_ms':[x*1000 for x in residuals]}}

class ProductionWarpTests(unittest.TestCase):
 def test_zero_candidates_is_exact_baseline_fallback(self):
  base=lambda t:2.0+1.01*t
  warp,report=c.build_selective_warp(base,{'segments':[segment('rejected_no_meaningful_improvement')]},BEATS)
  self.assertEqual(report['status'],'checkpoint_linear_fallback_no_accepted_segments')
  self.assertFalse(report['timing_changes_applied'])
  for t in (0,.5,4,8):self.assertEqual(warp(t),base(t))

 def test_accepted_candidate_is_applied_inside_only(self):
  base=lambda t:t
  warp,report=c.build_selective_warp(base,{'segments':[segment()]},BEATS)
  self.assertEqual(report['status'],'applied');self.assertTrue(report['timing_changes_applied'])
  self.assertAlmostEqual(warp(4),4.008);self.assertEqual(warp(0),0);self.assertEqual(warp(8),8)

 def test_global_stretch_violation_falls_back_all_or_nothing(self):
  base=lambda t:t
  bad=[0,.2,.4,.6,.8,.6,.4,.2,0]
  warp,report=c.build_selective_warp(base,{'segments':[segment(residuals=bad)]},BEATS)
  self.assertEqual(report['status'],'checkpoint_linear_fallback_global_validation')
  self.assertFalse(report['timing_changes_applied']);self.assertEqual(warp(4),4)

 def test_overlapping_accepted_segments_fall_back(self):
  a=segment();b=segment();b.update(segment_id='m005-m009',start_nominal_time=4.0)
  b['path']['control_nominal_times']=[4,4.5,5,5.5,6,6.5,7,7.5,8]
  warp,report=c.build_selective_warp(lambda t:t,{'segments':[a,b]},BEATS)
  self.assertFalse(report['validation']['overlap_free']);self.assertFalse(report['timing_changes_applied'])

if __name__=='__main__':unittest.main()



