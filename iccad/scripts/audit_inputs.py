"""Audit actual ICCAD DEF/LEF inputs; no placement modifications."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "iccad"
DESIGNS = ["superblue1", "superblue3", "superblue4", "superblue5", "superblue7", "superblue10", "superblue16", "superblue18"]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def lef_blocks(path):
    blocks = {}
    name, body = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("MACRO "):
            name, body = line.split()[1], []
        elif name and line.strip() == "END " + name:
            text = "\n".join(body)
            size = re.search(r"\bSIZE\s+([\d.eE+-]+)\s+BY\s+([\d.eE+-]+)", text)
            assert size, name
            blocks[name] = tuple(map(float, size.groups()))
            name = None
        elif name:
            body.append(line)
    assert blocks, "No LEF BLOCK macros"
    return blocks

def compare_fixed(before, after):
    """Allow only decimal serialization noise, far below one database unit."""
    assert before['dbu_per_um'] == after['dbu_per_um']
    assert set(before['macros']) == set(after['macros'])
    max_delta = 0.0
    for name, old in before['macros'].items():
        new = after['macros'][name]
        assert all(old[k] == new[k] for k in ('master', 'status', 'orientation')), name
        max_delta = max(max_delta, *(abs(x-y) for x,y in zip(old['xy_dbu'],new['xy_dbu'])))
    # Absolute tolerance; no relative tolerance that grows with coordinate size.
    assert max_delta <= 1e-6, f'Fixed-coordinate change {max_delta} DBU'
    return dict(preserved=True, max_coordinate_difference_dbu=max_delta,
                absolute_tolerance_dbu=1e-6, relative_tolerance=0)

def inspect_def(path, blocks, selected_names=None):
    macros, statuses, digest = {}, Counter(), hashlib.sha256()
    section, entry, units, die, count, declared = False, [], None, None, 0, None
    identity_sum = 0
    def component(text):
        nonlocal count, identity_sum
        m = re.match(r"\s*-\s+(\S+)\s+(\S+)", text)
        assert m, text[:100]
        name, master = m.groups()
        digest.update((name + "\t" + master + "\n").encode())
        identity_sum = (identity_sum + int.from_bytes(hashlib.sha256((name+'\t'+master).encode()).digest(),'big')) % (1 << 256)
        count += 1
        p = re.search(r"\+\s+(FIXED|PLACED|COVER)\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s+(\S+)", text)
        status = p[1] if p else "UNPLACED"
        statuses[status] += 1
        if (selected_names is None and status != 'FIXED') or (selected_names is not None and name not in selected_names):
            return
        assert p, f"Unplaced macro {name}"
        w, h = blocks[master]
        if p[4] in {"W", "E", "FW", "FE"}:
            w, h = h, w
        else:
            assert p[4] in {"N", "S", "FN", "FS"}, p[4]
        x, y = float(p[2]) / units, float(p[3]) / units
        assert name not in macros
        macros[name] = dict(master=master, status=status, orientation=p[4],
                            xy_dbu=[float(p[2]), float(p[3])], bbox_um=[x, y, x+w, y+h])
    with Path(path).open() as f:
        for line in f:
            if line.startswith("UNITS DISTANCE MICRONS"):
                units = int(line.split()[3])
            if line.startswith("DIEAREA"):
                points = re.findall(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", line)
                assert len(points) == 2, "Only rectangular canvas supported"
                die = [int(v)/units for p in points for v in p]
            if line.startswith("COMPONENTS "):
                declared, section = int(line.split()[1]), True
                continue
            if line.startswith("END COMPONENTS"):
                break
            if section:
                entry.append(line)
                if ";" in line:
                    component("".join(entry)); entry = []
    assert units and die and count == declared and not entry
    overlaps, outside, not_fixed = [], [], []
    items = list(macros.items())
    eps = 1e-8
    for i, (name, m) in enumerate(items):
        x0,y0,x1,y1 = m["bbox_um"]
        if x0 < die[0]-eps or y0 < die[1]-eps or x1 > die[2]+eps or y1 > die[3]+eps:
            outside.append(name)
        if m["status"] != "FIXED": not_fixed.append(name)
        for other, n in items[:i]:
            a,b,c,d = n["bbox_um"]
            dx,dy = min(x1,c)-max(x0,a), min(y1,d)-max(y0,b)
            if dx > eps and dy > eps:
                overlaps.append(dict(a=name,b=other,width_um=dx,height_um=dy))
    return dict(path=str(path),sha256=sha(path),dbu_per_um=units,die_um=die,
                component_count=count,component_order_identity_sha256=digest.hexdigest(),
                component_identity_multiset_digest=f'{identity_sum:064x}',
                status_counts=dict(statuses),macro_count=len(macros),macros=macros,
                overlap_count=len(overlaps),overlaps=overlaps,out_of_die=outside,
                nonfixed_macros=not_fixed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--designs',nargs='+',default=DESIGNS)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for design in a.designs:
        folder=BASE/'assets/benchmarks/iccad2015.ot'/design
        lef=folder/f'{design}.lef'; blocks=lef_blocks(lef)
        released=inspect_def(ROOT/'backend/releases/flowplace_DEFS/iccad2015'/f'{design}.def',blocks)
        orig=inspect_def(folder/f'{design}.def',blocks,set(released['macros']))
        out=dict(design=design,lef_sha256=sha(lef),original=orig,released=released,
                 scope='All 512 FIXED components of the released DEF; matches the upstream n_macro=512 placement task, including CORE-class cells when selected. Original DEF uses the same names.',
                 component_identity_multiset_equal=orig['component_identity_multiset_digest']==released['component_identity_multiset_digest'],
                 component_identity_order_equal=orig['component_order_identity_sha256']==released['component_order_identity_sha256'],
                 macro_names_equal=set(orig['macros'])==set(released['macros']))
        (a.output/f'{design}.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(design=design,macros=released['macro_count'],
              overlap_count=released['overlap_count'],out_of_die=len(released['out_of_die']),
              nonfixed=len(released['nonfixed_macros']),identity_multiset_equal=out['component_identity_multiset_equal'])),flush=True)

if __name__=='__main__': main()
