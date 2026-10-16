import numpy as np
from placement_metrics import legality, MacroHPWL


def test_geometry_known_overlap_and_touch():
    sizes = np.array([[1., 1.], [1., 1.]])
    assert legality([[-.5, 0], [.5, 0]], sizes)['legal']
    r = legality([[-.25, 0], [.25, 0]], sizes)
    assert not r['legal'] and r['overlap_pairs'] == 1
    assert abs(r['overlap_area'] - .5) < 1e-12
    assert legality([[0, 0], [1, 0]], sizes)['out_of_bounds'] == 1
    assert not legality([[float('nan'), 0], [0, 0]], sizes)['finite']
    assert not legality([[-.5, 0], [.5, 0]], sizes, padding=.01)['legal']


def test_true_hypernet_bbox_not_clique_edge_sum():
    c = {'node_id_to_name': ['a', 'b', 'c'], 'ratio_x': 10/224, 'ratio_y': 20/224,
         'node_info': {n: {'x': 1, 'y': 1} for n in 'abc'},
         'net_info': {'n': {'nodes': {n: {} for n in 'abc'}}}}
    m = MacroHPWL(c)
    # Physical centers (0,0),(5,10),(10,20): hypernet HPWL=30, not pair sum=60.
    assert abs(m([[-1,-1],[0,0],[1,1]]) - 30) < 1e-12
    assert np.allclose(m([[[-1,-1],[0,0],[1,1]]]*2), [30,30])
