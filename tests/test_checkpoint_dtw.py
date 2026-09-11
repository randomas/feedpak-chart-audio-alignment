import unittest
import numpy as np
import checkpoint_dtw as c

class Checkpoint62Tests(unittest.TestCase):
 def test_metrics(self):
  m=c._metrics([-20,10,30]);self.assertEqual(m['matched_events'],3);self.assertEqual(m['median_residual_ms'],20);self.assertEqual(m['signed_median_residual_ms'],10)

 def test_fixed_observations_preserve_set(self):
  obs=c._fixed_observations([0,1,2],np.array([.01,1.02,2.03]),lambda t:t)
  self.assertEqual(len(obs),3);self.assertEqual(len(c._residuals(obs,lambda t:t+.01)),3)

 def test_meter_grid_uses_actual_numerators(self):
  beats=[{'measure':1,'t_gp':0.,'ts_num':3},{'measure':2,'t_gp':3.,'ts_num':5},{'measure':3,'t_gp':8.,'ts_num':4}]
  self.assertEqual(c._segment_beat_count(beats,0,8),8)
  grid=c._meter_grid(beats,0,8);self.assertEqual(len(grid[:3]),3);self.assertAlmostEqual(grid[0][1]-grid[0][0],1)

 def test_boundary_evidence_excludes_global_fixed_boundary(self):
  b={'measure':1,'nominal_time':0,'tier':'fixed','evidence':['fixed_start']}
  self.assertEqual(c._trusted_boundary_evidence(b,{}, {}, {'anchors':[]}),[])

 def test_interior_tier_a_boundary_is_assigned_once(self):
  bounds=[{'measure':1,'tier':'fixed'},{'measure':5,'tier':'A'},{'measure':9,'tier':'fixed'}]
  chroma={'checkpoints':[{'measure':5,'accepted':True,'combined_residual_ms':45}]}
  linear={'anchors':[{'measure':5,'tier':'A','evidence':['chroma','onset']}]}
  got=c._assign_boundary_activation(bounds,[.2,.9],chroma,{},linear,c.LocalDTWConfig())
  self.assertEqual(sum(len(v) for v in got.values()),1);self.assertEqual(len(got[1]),1)

 def test_assignment_tie_is_deterministic_to_left(self):
  bounds=[{'measure':1,'tier':'fixed'},{'measure':5,'tier':'A'},{'measure':9,'tier':'fixed'}]
  chroma={'checkpoints':[{'measure':5,'accepted':True,'combined_residual_ms':45}]}
  linear={'anchors':[{'measure':5,'tier':'A','evidence':['chroma']}]}
  got=c._assign_boundary_activation(bounds,[1,1],chroma,{},linear,c.LocalDTWConfig())
  self.assertEqual(len(got[0]),1);self.assertEqual(len(got[1]),0)

 def test_computational_boundaries_limit_gap(self):
  beats=[{'measure':i+1,'t_gp':float(i),'ts_num':4} for i in range(25)]
  pts=c._boundaries({'anchors':[]},{},beats,c.LocalDTWConfig())
  self.assertLessEqual(max(y['measure']-x['measure'] for x,y in zip(pts,pts[1:])),8)

 def test_missing_audio_is_timing_neutral(self):
  beats=[{'measure':i+1,'t_gp':float(i),'ts_num':4} for i in range(9)]
  r=c.diagnose_local_dtw(lambda t:t,{}, {}, {}, {'anchors':[]},beats,{'drums':[1,2,3,4]},{'drums':None})
  self.assertEqual(r['version'],'1.2.1');self.assertFalse(r['timing_changes_applied'])

if __name__=='__main__':unittest.main()





















