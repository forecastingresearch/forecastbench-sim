import argparse,json,math
from .adapters import CachedForecast,load_record
p=argparse.ArgumentParser(description='Offline cached-forecast scoring; never simulates or calls models.')
s=p.add_subparsers(dest='command',required=True);s.add_parser('smoke');v=s.add_parser('validate');v.add_argument('--input',required=True)
n=s.add_parser('parse');n.add_argument('--world',required=True);n.add_argument('--format',required=True);n.add_argument('--input',required=True);n.add_argument('--options',default='{}',help='JSON object of native parser arguments, e.g. keys or labels')
a=p.parse_args()
if a.command=='smoke':
 r=CachedForecast.parse(dict(schema_version='1',world='starsim',metric='excess_brier',forecast=.4,truth=.6));assert math.isclose(r.score(),.04)
 print(json.dumps(dict(status='PASS',scope='synthetic public fixture',schema_version='1',network=False,simulations=0)))
elif a.command=='parse':
 from pathlib import Path
 from .parsing.registry import parse_response
 try: value=parse_response(a.world,a.format,Path(a.input).read_text(),**json.loads(a.options))
 except (ValueError,TypeError,OSError) as e:p.error(str(e))
 print(json.dumps(value))
else:
 r=load_record(a.input);print(json.dumps(dict(schema_version='1',world=r.world,metric=r.metric,score=r.score())))
