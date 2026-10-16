#!/usr/bin/env python3
"""Inspect raw released macro rectangles against released LEFs; never move them."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from export_macro_def import BASE, COMPONENT, POSITION
from grt_metrics import atomic_json

p = argparse.ArgumentParser()
p.add_argument('designs', nargs='+')
a = p.parse_args()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for design in a.designs:
    path = BASE / 'releases/flowplace_DEFS/openroad' / f'{design}.def'
    text = path.read_text()
    units = int(re.search(r'UNITS DISTANCE MICRONS (\d+)\s*;', text)[1])
    start = re.search(r'^COMPONENTS \d+\s*;', text, re.M).end()
    end = text.index('END COMPONENTS', start)
    macros, lefs = [], {}
    for ordinal, match in enumerate(COMPONENT.finditer(text[start:end])):
        name, master, body = match.groups()
        if not master.startswith('fakeram'):
            continue
        lef = BASE / 'thirdparty/OpenROAD-flow-scripts/platforms/nangate45/lef' / f'{master}.lef'
        size = re.search(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)', lef.read_text())
        w, h = map(float, size.groups())
        lefs[str(lef)] = sha(lef)
        position = POSITION.search(body)
        x, y = int(position[2]) / units, int(position[3]) / units
        orient = position[4]
        if orient in ['W', 'E', 'FW', 'FE']:
            w, h = h, w
        elif orient not in ['N', 'S', 'FN', 'FS']:
            raise RuntimeError(f'Unexpected orientation {orient}')
        macros.append({'component_ordinal': ordinal, 'name': name, 'master': master,
                       'orientation': orient, 'bbox_um': [x, y, x+w, y+h], 'size_um': [w, h]})
    overlaps = []
    for i, left in enumerate(macros):
        x0, y0, x1, y1 = left['bbox_um']
        for right in macros[:i]:
            a0, b0, a1, b1 = right['bbox_um']
            dx, dy = min(x1, a1)-max(x0, a0), min(y1, b1)-max(y0, b0)
            if dx > 1 / units and dy > 1 / units:
                overlaps.append({'a': left['name'], 'b': right['name'],
                                 'overlap_width_um': dx, 'overlap_height_um': dy,
                                 'overlap_area_um2': dx*dy})
    result = {'design': design, 'def_path': str(path), 'def_sha256': sha(path),
              'lef_sha256': lefs, 'dbu_per_um': units, 'macro_count': len(macros),
              'macros': macros, 'overlaps': overlaps, 'overlap_count': len(overlaps),
              'scope': 'Raw DEF component positions and released LEF rectangles, before OpenROAD snapping; no halo or optimization.'}
    output = BASE / 'results/released_def_geometry' / f'{design}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps({'design': design, 'macro_count': len(macros), 'overlap_count': len(overlaps), 'output': str(output)}))
