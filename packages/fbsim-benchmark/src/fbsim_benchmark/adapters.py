"""Version-1 cached-forecast interface. No simulator or provider imports."""
from dataclasses import dataclass
import json, math
import numpy as np
from .scoring import freeciv, micropolis, pandemic

@dataclass(frozen=True)
class CachedForecast:
    world: str
    metric: str
    forecast: object
    truth: object

    @classmethod
    def parse(cls, value):
        if not isinstance(value,dict) or set(value)!={'schema_version','world','metric','forecast','truth'}:
            raise ValueError('Expected exactly the version-1 cached forecast fields')
        if value['schema_version']!='1' or value['world'] not in {'freeciv','micropolis','starsim'}:
            raise ValueError('Unsupported schema version or world')
        metric=value['metric'];f=value['forecast'];q=value['truth']
        def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
        if metric in {'excess_brier','excess_bits'}:
            if not all(number(x) and 0<=x<=1 for x in [f,q]):raise ValueError('Probabilities must be finite and within [0,1]')
        elif metric=='quantile_crps':
            if not isinstance(f,list) or len(f)!=5 or not all(number(x) for x in f) or f!=sorted(f):raise ValueError('Five finite nondecreasing quantiles required')
            if not isinstance(q,list) or not q or not all(number(x) for x in q):raise ValueError('Nonempty finite replay outcomes required')
        else:raise ValueError('Unsupported metric')
        return cls(value['world'],metric,f,q)

    def score(self):
        if self.metric=='excess_brier':return (self.forecast-self.truth)**2
        if self.metric=='excess_bits':return pandemic.kl_bits(self.truth,self.forecast)
        if self.world=='freeciv':return freeciv.pinball(self.forecast,np.asarray(self.truth,float))
        if self.world=='micropolis':return micropolis.crps_distribution(self.forecast,self.truth)
        return pandemic.crps5(self.forecast,self.truth)

def load_record(path):
    def invalid(x):raise ValueError('Nonfinite JSON constant')
    with open(path) as f:return CachedForecast.parse(json.load(f,parse_constant=invalid))
