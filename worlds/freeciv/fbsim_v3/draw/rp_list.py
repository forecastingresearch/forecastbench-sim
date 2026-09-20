import json,sys
d=json.load(sys.stdin); d=d if isinstance(d,list) else d.get("pods",d.get("data",[]))
tot=0.0
for p in sorted(d,key=lambda p:(p.get("name",""),p.get("createdAt",""))):
    st=p.get("desiredStatus","")
    if st!="RUNNING": continue
    c=float(p.get("costPerHr") or 0); tot+=c
    m=p.get("machine") or {}; dc=m.get("dataCenterId","") if isinstance(m,dict) else ""
    pm=p.get("portMappings") or {}
    print("%-16s %-24s %-8s %3svcpu $%.3f/h %-10s %s:%s" % (p.get("id",""),p.get("name",""),st,p.get("vcpuCount",""),c,dc,p.get("publicIp","") or "-",pm.get("22","-")))
print("-- running total $%.2f/h" % tot)
