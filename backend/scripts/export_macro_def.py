#!/usr/bin/env python3
"""Export FlowPlace cache-space macro coordinates, preserving released DEF identities.

The author's DREAMPlace apply() rounds the scaled lower-left coordinate, then
unscales and shifts to the placement-row origin. Recover that transform from
released LEF dimensions and DEF rows; never assume coordinates are microns.
"""
import argparse, hashlib, json, math, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
COMPONENT = re.compile(r'^\s*-\s+(\S+)\s+(\S+)(.*?);', re.M|re.S)
POSITION = re.compile(r'\+\s+(FIXED|PLACED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)')

def convert(design, coords_path, output, template=None):
    template = Path(template or BASE/'releases'/'flowplace_DEFS'/'openroad'/f'{design}.def')
    text = template.read_text()
    dbu = int(re.search(r'UNITS DISTANCE MICRONS (\d+)\s*;',text)[1])
    rows = [tuple(map(int,m)) for m in re.findall(r'^ROW \S+ \S+ (-?\d+) (-?\d+) \S+ DO (\d+) BY \d+ STEP (\d+) \d+\s*;',text,re.M)]
    if not rows: raise ValueError('No placement rows in template')
    xlo,ylo = min(r[0] for r in rows),min(r[1] for r in rows)
    row_ys=sorted({r[1] for r in rows})
    row_height=min(b-a for a,b in zip(row_ys,row_ys[1:]))
    xhi=max(x+nx*step for x,y,nx,step in rows); yhi=max(row_ys)+row_height
    start=text.index('\n',re.search(r'^COMPONENTS \d+\s*;',text,re.M).start())+1
    end=text.index('END COMPONENTS',start)
    section=text[start:end]; matches=list(COMPONENT.finditer(section))
    coords=json.loads(Path(coords_path).read_text())
    replacements=[]; macros=[]; scales=[]
    for key,xywh in coords.items():
        idx=int(key)
        if idx >= len(matches): raise ValueError(f'Node ID {idx} exceeds DEF component count')
        match=matches[idx];name,master,body=match.groups()
        lef=BASE/'thirdparty'/'OpenROAD-flow-scripts'/'platforms'/'nangate45'/'lef'/f'{master}.lef'
        if not lef.exists(): raise ValueError(f'Node ID {idx} maps to {master}, not a released macro LEF')
        size=re.search(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)',lef.read_text())
        if size is None: raise ValueError(f'Missing LEF size: {lef}')
        w,h=map(float,size.groups());px,py,cw,ch=map(float,xywh)
        if not all(math.isfinite(v) for v in [px,py,cw,ch]) or cw<=0 or ch<=0:raise ValueError('Invalid coordinates')
        position=POSITION.search(body)
        if position is None:raise ValueError(f'Missing position for {name}')
        orient=position[4]
        if orient not in ['N','S','FN','FS']:raise ValueError(f'Unsupported rotated macro orientation {orient}')
        scales += [w/cw,h/ch]
        macros.append({'node_id':idx,'name':name,'master':master,'orientation':orient,'lower_left_cache':[px,py], 'size_cache':[cw,ch], 'size_um':[w,h]})
        replacements.append((match,position))
    scale=sum(scales)/len(scales)
    if any(abs(s-scale)>scale*1e-5 for s in scales):raise ValueError(f'Inconsistent cache/LEF scales {scales}')
    # The archive uses scaled placement coordinates in standard-cell site widths.
    site_um=rows[0][3]/dbu
    if abs(scale-site_um)>site_um*1e-5:raise ValueError(f'LEF-derived scale {scale} differs from site width {site_um}')
    scale=site_um
    actual_macro_count=sum('fakeram' in m[2] for m in matches)
    if len(macros)!=actual_macro_count:raise ValueError(f'Expected all {actual_macro_count} macros, received {len(macros)}')
    updated=section
    for macro,(match,position) in sorted(zip(macros,replacements), key=lambda x:x[1][0].start(), reverse=True):
        px,py=macro['lower_left_cache']
        x=round(round(px)*scale*dbu+xlo);y=round(round(py)*scale*dbu+ylo)
        macro['lower_left_dbu']=[x,y];macro['lower_left_um']=[x/dbu,y/dbu]
        replacement=POSITION.sub(f'+ FIXED ( {x} {y} ) {macro["orientation"]}',match.group(0),count=1)
        updated=updated[:match.start()]+replacement+updated[match.end():]
    bad_bounds=[]; overlaps=[]
    for i,a in enumerate(macros):
        x,y=a['lower_left_dbu'];w,h=[v*dbu for v in a['size_um']]
        if x<xlo-1 or y<ylo-1 or x+w>xhi+1 or y+h>yhi+1:bad_bounds.append(a['name'])
        for b in macros[:i]:
            bx,by=b['lower_left_dbu'];bw,bh=[v*dbu for v in b['size_um']]
            overlap_w=min(x+w,bx+bw)-max(x,bx);overlap_h=min(y+h,by+bh)-max(y,by)
            if overlap_w>1 and overlap_h>1:overlaps.append({'a':a['name'],'b':b['name'],'area_um2':overlap_w*overlap_h/dbu**2})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(text[:start]+updated+text[end:])
    audit={'design':design,'template':str(template.resolve()),'template_sha256':hashlib.sha256(text.encode()).hexdigest(),'input_json':str(Path(coords_path).resolve()),'output':str(output.resolve()),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'dbu_per_um':dbu,'cache_to_um_scale':scale,'origin_um':[xlo/dbu,ylo/dbu],'row_bounds_um':[xlo/dbu,ylo/dbu,xhi/dbu,yhi/dbu],'rounding':'round cache-space lower-left to integer, then unscale/shift (author apply())','orientation_source':'released DEF unchanged; original source DEF is not released','macros':sorted(macros,key=lambda m:m['node_id']),'out_of_bounds':bad_bounds,'overlaps':overlaps,'legal_no_halo':not bad_bounds and not overlaps}
    output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='macros'},indent=2))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--design',required=True);p.add_argument('--coordinates',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--template',type=Path)
    a=p.parse_args();audit=convert(a.design,a.coordinates,a.output,a.template)
    if not audit['legal_no_halo']:raise SystemExit('Exported macro geometry is illegal; audit retained.')
