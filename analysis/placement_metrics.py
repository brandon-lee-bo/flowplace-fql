"""Independent layout validation and explicitly macro-only hypernet HPWL.

Coordinates are macro centers in [-1,1], sizes in [0,2]. HPWL is computed
in cache physical units by undoing per-axis normalization. It is not routed
wirelength or full-design HPWL: nets containing <2 modeled macros contribute 0.
"""
import numpy as np


def legality(centers, sizes, padding=0.0, tolerance=1e-6):
    centers = np.asarray(centers, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if centers.shape != sizes.shape or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('centers and sizes must both have shape [N,2]')
    if not np.isfinite(centers).all() or not np.isfinite(sizes).all():
        return {'legal': False, 'finite': False, 'overlap_pairs': None,
                'out_of_bounds': None, 'overlap_area': None}
    if np.any(sizes < 0):
        raise ValueError('negative macro size')
    half = sizes / 2 + padding
    lo, hi = centers - half, centers + half
    out = np.any((lo < -1 - tolerance) | (hi > 1 + tolerance), axis=1)
    ext = np.minimum(hi[:, None, :], hi[None, :, :]) - np.maximum(lo[:, None, :], lo[None, :, :])
    pairs = np.triu(np.all(ext > tolerance, axis=-1), 1)
    area = np.maximum(ext[..., 0], 0) * np.maximum(ext[..., 1], 0)
    return {'legal': bool(not out.any() and not pairs.any()), 'finite': True,
            'overlap_pairs': int(pairs.sum()), 'out_of_bounds': int(out.sum()),
            'overlap_area': float(area[pairs].sum()),
            'max_boundary_excess': float(max(0., np.max(-1 - lo), np.max(hi - 1)))}


class MacroHPWL:
    def __init__(self, cache, grid=224):
        self.names = list(cache['node_id_to_name'])
        self.name_to_id = {n: i for i, n in enumerate(self.names)}
        self.canvas = np.array([cache['ratio_x'] * grid, cache['ratio_y'] * grid], dtype=np.float64)
        self.sizes = np.array([[cache['node_info'][n]['x'], cache['node_info'][n]['y']]
                              for n in self.names], dtype=np.float64) * 2 / self.canvas
        self.nets = []
        self.ignored_nets = 0
        for net_name, net in cache['net_info'].items():
            ids, offsets = [], []
            for name, pin in net['nodes'].items():
                if name not in self.name_to_id:
                    continue
                ids.append(self.name_to_id[name])
                offsets.append([pin.get('x_offset', 0), pin.get('y_offset', 0)])
            if len(ids) >= 2:
                self.nets.append((np.array(ids, dtype=np.int64), np.array(offsets, dtype=np.float64)))
            else:
                self.ignored_nets += 1

    def __call__(self, centers):
        x = np.asarray(centers, dtype=np.float64)
        single = x.ndim == 2
        if single:
            x = x[None]
        if x.shape[1:] != self.sizes.shape:
            raise ValueError('layout shape does not match cache macro order')
        physical = (x + 1) * self.canvas / 2
        result = np.zeros(x.shape[0], dtype=np.float64)
        for ids, offsets in self.nets:
            pins = physical[:, ids] + offsets
            result += np.ptp(pins, axis=1).sum(axis=-1)
        return float(result[0]) if single else result

    def metadata(self):
        return {'metric': 'macro_only_hypernet_hpwl_cache_units',
                'included_nets': len(self.nets), 'ignored_nets': self.ignored_nets,
                'canvas': self.canvas.tolist(), 'num_macros': len(self.names),
                'offset_convention': 'center-relative, as official graph conversion',
                'excludes': 'standard cells and explicit fixed I/O terminal positions; not routed PPA'}


def summarize(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('refusing to silently summarize nonfinite values')
    if not x.size:
        return {'count': 0}
    return {'count': len(x), 'mean': float(x.mean()), 'std': float(x.std(ddof=1)) if len(x)>1 else 0.,
            'median': float(np.median(x)), 'min': float(x.min()), 'max': float(x.max())}
