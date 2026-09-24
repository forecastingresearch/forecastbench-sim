import unittest,math
from fbsim_benchmark.adapters import CachedForecast
from fbsim_benchmark.scoring import pandemic,micropolis,freeciv
import numpy as np
class Contract(unittest.TestCase):
 def record(self,**kw):return dict(schema_version='1',world='starsim',metric='excess_brier',forecast=.4,truth=.6,**kw)
 def test_binary(self):self.assertAlmostEqual(CachedForecast.parse(self.record()).score(),.04)
 def test_reject(self):
  for k,v in [('forecast',float('nan')),('truth',1.1),('world','unknown'),('schema_version','2'),('forecast',True)]:
   d=self.record();d[k]=v
   with self.assertRaises(ValueError):CachedForecast.parse(d)
 def test_score_conventions(self):
  ys=np.array([0.,2.,4.,6.,8.]);q=[1.,2.,3.,5.,7.]
  self.assertAlmostEqual(pandemic.crps5(q,ys),micropolis.crps_distribution(q,ys))
  self.assertAlmostEqual(freeciv.bits(.4,.6)[1],pandemic.kl_bits(.6,.4))
 def test_floor(self):
  ys=np.array([0.,2.,4.,6.,8.]);q=np.quantile(ys,[.1,.25,.5,.75,.9],method='inverted_cdf')
  self.assertAlmostEqual(micropolis.crps_distribution(q,ys),micropolis.crps_floor(ys))
 def test_quantile_validation(self):
  d=dict(schema_version='1',world='freeciv',metric='quantile_crps',forecast=[0,1,2,3,4],truth=[1,2,3]);self.assertTrue(math.isfinite(CachedForecast.parse(d).score()))
  d['forecast']=[0,2,1,3,4]
  with self.assertRaises(ValueError):CachedForecast.parse(d)
if __name__=='__main__':unittest.main()
