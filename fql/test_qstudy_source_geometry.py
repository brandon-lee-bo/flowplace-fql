"""Small coordinate-audit invariants; run under a bounded Slurm CPU job."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.placement_metrics import legality
from .qstudy_data import action_sha, sha
from .qstudy_source_geometry import analyze, load_old, nearest, stats


class Metric:
    sizes = np.array([[.2, .2], [.2, .2], [0., 0.]])

    def __call__(self, array):
        array = np.asarray(array)
        return np.abs(array[:, 0]-array[:, 1]).sum(axis=-1)


def row(index, family, raw, executed=None):
    raw = np.asarray(raw, dtype=np.float32)
    executed = raw.copy() if executed is None else np.asarray(executed, dtype=np.float32)
    return dict(source='toy', split='train', family=family, variant=index % 2,
                index=index, source_row=index, collection_kind='toy', raw=raw,
                executed=executed, raw_sha256=action_sha(raw), executed_sha256=action_sha(executed))


class CoordinateTests(unittest.TestCase):
    def test_rms_uses_movable_entries_only(self):
        query = np.array([[[0., 0.], [100., 100.]]])
        reference = np.array([[[3., 4.], [-100., -100.]]])
        distance, _ = nearest(query, reference, np.array([True, False]))
        self.assertAlmostEqual(distance[0], np.sqrt(12.5))

    def test_same_family_exclusion_not_just_diagonal(self):
        coordinates = np.array([[[0., 0.]], [[.01, 0.]], [[1., 0.]]])
        families = np.array([5, 5, 6])
        distances, ids = nearest(coordinates, coordinates, np.array([True]),
                                 families[:, None] == families[None, :])
        self.assertEqual(ids.tolist(), [2, 2, 1])
        self.assertTrue(np.all(distances > 0))

    def test_no_eligible_neighbor_rejected(self):
        array = np.zeros((1, 1, 2))
        with self.assertRaisesRegex(ValueError, 'eligible'):
            nearest(array, array, np.array([True]), np.ones((1, 1), dtype=bool))

    def test_validation_cannot_change_train_threshold(self):
        a = row(0, 0, [[-.5, -.5], [.5, .5], [0., 0.]])
        b = row(1, 1, [[-.3, -.3], [.3, .3], [0., 0.]])
        groups = dict(old_train=[a, b], old_validation=[a], teacher_train=[b], teacher_validation=[a])
        _, _, first, _ = analyze(groups, np.array([True, True, False]), Metric())
        groups['teacher_validation'] = [row(8, 8, [[9., 9.], [8., 8.], [0., 0.]])]
        _, _, second, _ = analyze(groups, np.array([True, True, False]), Metric())
        self.assertEqual([x['old_train_lfo_thresholds'] for x in first],
                         [x['old_train_lfo_thresholds'] for x in second])

    def test_raw_overlap_bounds_use_boxes_not_centers_only(self):
        geometry = legality(np.array([[.95, 0.], [.9, 0.]]), np.full((2, 2), .2))
        self.assertEqual(geometry['overlap_pairs'], 1)
        self.assertEqual(geometry['out_of_bounds'], 1)

    def test_nonfinite_diagnostics_rejected(self):
        with self.assertRaises(ValueError):
            stats([1., np.nan])

    def fixture_old(self, folder):
        arrays, provenance = {}, []
        for split, family in [('train', 0), ('validation', 9)]:
            raw = np.array([[-.5, -.5], [.5, .5]], dtype=np.float32)
            r = dict(design='bp_fe', split=split, family=family, variant=0, source_row=family,
                     collection_kind='toy', raw_action=raw.tolist(), executed_action=raw.tolist(),
                     raw_action_sha256=action_sha(raw), action_sha256=action_sha(raw))
            arrays.update({split+'_raw_action': raw.reshape(1, -1), split+'_executed_action': raw.reshape(1, -1),
                split+'_family': np.array([family]), split+'_variant': np.array([0]),
                split+'_source_row': np.array([family]), split+'_provenance_index': np.array([len(provenance)]),
                # Loading this forbidden label array with allow_pickle=False fails.
                split+'_metrics': np.array([{'must_not_be_loaded': True}], dtype=object)})
            provenance.append(r)
        path = folder/'dataset.npz'
        np.savez_compressed(path, **arrays)
        rows_path = folder/'rows.jsonl'
        rows_path.write_text(''.join(json.dumps(r)+'\n' for r in provenance))
        manifest = dict(design='bp_fe', dataset_sha256=sha(path), rows_sha256=sha(rows_path),
            no_test_or_heldout_exported=True, historical_online_included=False,
            split_counts=dict(train=1, validation=1), source_bindings=[], validation_status='historical')
        (folder/'manifest.json').write_text(json.dumps(manifest))
        return path

    def test_loader_never_decompresses_metric_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.fixture_old(Path(temporary))
            groups, _ = load_old(path)
            self.assertEqual(len(groups['old_train']), 1)
            self.assertEqual(groups['old_validation'][0]['family'], 9)

    def test_loader_binds_coordinate_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = self.fixture_old(folder)
            rows = [json.loads(x) for x in (folder/'rows.jsonl').read_text().splitlines()]
            rows[0]['raw_action'][0][0] = 99
            (folder/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest = json.loads((folder/'manifest.json').read_text())
            manifest['rows_sha256'] = sha(folder/'rows.jsonl')
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_old(path)


if __name__ == '__main__':
    unittest.main()
