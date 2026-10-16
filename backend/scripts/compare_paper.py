#!/usr/bin/env python3
"""Compare final route metrics to the rounded FlowPlace row, without claiming equivalence."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
base=Path(__file__).resolve().parents[1]
manifest=json.loads((a.run/'manifest.json').read_text());obs=json.loads((a.run/'metrics.json').read_text())
ref=json.loads((base/'assets'/'paper_table2_flowplace.json').read_text())['designs'][manifest['design']]
rows=[]
for dst,src,scale in [('rWL_million_um','DRT_WL',1e-6),('WNS_ns','WNS',1),('TNS_ns','TNS',1),('Power_W','Power',1),('CellArea_um2','StdCellArea',1)]:
 actual=obs[src]*scale if obs and obs.get(src) is not None else None
 rows.append({'metric':dst,'paper_rounded':ref[dst],'observed':actual,'difference':None if actual is None else actual-ref[dst], 'relative_to_abs_paper_percent':None if actual is None else 100*(actual-ref[dst])/abs(ref[dst])})
(a.run/'paper_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print('| Metric | Paper (rounded) | This run | Delta |')
print('|---|---:|---:|---:|')
for r in rows:print(f"| {r['metric']} | {r['paper_rounded']} | {r['observed']} | {r['difference']} |")
