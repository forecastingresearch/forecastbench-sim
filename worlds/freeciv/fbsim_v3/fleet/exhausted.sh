#!/usr/bin/env bash
# exhausted.sh — list tasks with >=3 failure markers (exhausted) across live workers, minus those since done anywhere.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
. "$HERE/paths.sh"
bash fleet_ssh.sh --workers 'ls /workspace/fleet/queue/failed 2>/dev/null | sed "s/\.[0-9a-z]*$//" | sort | uniq -c | awk "\$1>=3{print \$2}" | tr "\n" " "; echo; echo DONE:; ls /workspace/fleet/queue/done | tr "\n" " "' 2>/dev/null > "$STATE/.exh_raw"
STATE="$STATE" python3 - <<'PY'
import os, re
S = os.environ['STATE']
raw=open(f'{S}/.exh_raw').read()
exh=set(); done=set()
for line in raw.splitlines():
    body=line.split(': ',1)[1] if ': ' in line else line
    parts=body.split('|')
    for i,p in enumerate(parts):
        toks=p.replace('DONE:','').split()
        (done if (i>0 or 'DONE:' in p) else exh).update(t for t in toks if re.match(r'a\d+-rng\d+$',t))
exh -= done
print(len(exh), 'exhausted tasks not done anywhere:', sorted(exh))
open(f'{S}/exhausted.txt','w').write('\n'.join(f"{t[1:].split('-rng')[0]} {t.split('-rng')[1]}" for t in sorted(exh))+('\n' if exh else ''))
PY
