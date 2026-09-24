"""Synthetic examples exercise native conventions, not private administered prompts."""
import unittest
from types import SimpleNamespace
from fbsim_benchmark.parsing import micropolis,freeciv_single,freeciv_batch,starsim_binary,starsim_continuous
from fbsim_benchmark.parsing.registry import parse_response
from fbsim_benchmark.worlds import micropolis as city,starsim
class NativeTests(unittest.TestCase):
 def test_freeciv_single_percent_convention(self):
  self.assertEqual(freeciv_single.parse_prob('<probability>75</probability>'),.75)
 def test_freeciv_last_answer(self):
  self.assertEqual(freeciv_single.parse_prob('<probability>0.2</probability><probability>0.8</probability>'),.8)
 def test_freeciv_batch(self):
  values,mode=freeciv_batch.parse_block('<<<PROBABILITIES>>>\nQ1: 0.2\nQ2: 0.8\n<<<END>>>',[1,2],'binary')
  self.assertEqual(values,{1:.2,2:.8})
 def test_city_requires_explicit_percent(self):
  self.assertIsNone(micropolis.parse_probability('75',quiet=True))
  self.assertEqual(micropolis.parse_probability('75%',quiet=True),.75)
 def test_city_rejects_nonmonotone(self):
  self.assertIsNone(micropolis.parse_percentiles('p10=5,p25=4,p50=3,p75=2,p90=1',quiet=True))
 def test_city_batch_positions(self):
  self.assertEqual(micropolis.parse_batch_probabilities('Q2: 0.8\nQ1: 0.2',['a','b'],quiet=True),[.2,.8])
 def test_city_line_trace(self):
  self.assertEqual(micropolis.parse_probability_with_line('discussion\n<<<PROBABILITIES>>>\nQ1: 0.8\n<<<END>>>',quiet=True),(.8,3))
 def test_starsim_historical_sorting(self):
  self.assertEqual(starsim_continuous.parse('{"a":5,"b":1,"c":3,"d":2,"e":4}',list('abcde')),dict(zip('abcde',[1,2,3,4,5])))
 def test_starsim_binary_rejects_outside_range(self):
  self.assertIsNone(starsim_binary.parse('{"a":2}',('a',)))
 def test_registry(self):
  self.assertEqual(parse_response('micropolis','binary-single','0.4',quiet=True),.4)
  with self.assertRaises(ValueError):parse_response('other','binary','0.4')
 def test_cached_city_world(self):
  sim=SimpleNamespace(log_data=[dict(tick=12,value=7)])
  self.assertEqual(city.to_world({'Example':sim},metrics=['value'],ticks_per_turn=4)['time_series']['value']['3']['0'],7)
 def test_cached_starsim_world(self):
  d={m:[1,2] for m in starsim.METRICS};w=starsim.to_world({0:d},{0:'Example'})
  self.assertEqual(w['time_series']['active_infections']['1']['0'],2)
