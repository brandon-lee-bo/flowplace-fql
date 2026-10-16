"""Fetch the exact pip dry-run artifact URLs, verify their SHA256 hashes.

Parallel network I/O avoids a slow single connection; no package code executes.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/wheels';OUT.mkdir(parents=True,exist_ok=True)
PLAN=json.loads((ROOT/'artifacts/pip_plan.json').read_text())['install']


def fetch(entry):
    info=entry['download_info'];url=info['url']
    dst=OUT/urllib.parse.unquote(url.split('/')[-1])
    sha=info['archive_info']['hashes']['sha256']
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==sha:
        return {'file':dst.name,'cached':True}
    s=requests.Session();s.trust_env=False
    for attempt in range(4):
        try:
            with s.get(url,stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                h=hashlib.sha256();n=0;t=time.monotonic()
                with dst.with_suffix(dst.suffix+'.part').open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b);h.update(b);n+=len(b)
                if h.hexdigest()!=sha:raise ValueError('SHA mismatch '+dst.name)
            dst.with_suffix(dst.suffix+'.part').rename(dst)
            return {'file':dst.name,'bytes':n,'seconds':time.monotonic()-t,'sha256':sha}
        except Exception:
            if attempt==3:raise


if __name__=='__main__':
    # Skip torch only if pip currently downloading; caller can choose to retain
    # pip's transfer while other CUDA packages are prefetched in parallel.
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--skip-torch',action='store_true');args=p.parse_args()
    entries=[e for e in PLAN if not(args.skip_torch and e['metadata']['name']=='torch')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(fetch,entries):print(json.dumps(result),flush=True)
