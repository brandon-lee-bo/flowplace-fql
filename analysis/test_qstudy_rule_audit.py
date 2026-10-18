"""Small geometry/rank contracts; run only inside a Slurm CPU allocation."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from .qstudy_rule_audit import structure,rules,describe,rank_association,associations,restoration_inventory,sha


def cache():
    return dict(node_id_to_name=['a','b'],ratio_x=1.,ratio_y=1.,
        node_info={n:dict(x=22.4,y=22.4) for n in ['a','b']},net_info={'n':dict(nodes={})},
        macro_clusters=[['a','b']],dataflow_mat=np.array([[0,1],[1,0]]),id2index={'a':0,'b':1},
        port_pos={'p':[0,112]},pin_blocking_rectangles=[[0,0,22.4,224]])


class RulesTest(unittest.TestCase):
    def test_directions_group_dataflow_and_edge(self):
        c,_=structure(cache());hp=lambda _:0.
        close=rules(np.array([[-.8,0],[-.5,0]]),c,hp)
        spread=rules(np.array([[-.8,0],[.8,0]]),c,hp)
        self.assertLess(close['cluster_bbox_fraction'],spread['cluster_bbox_fraction'])
        self.assertLess(close['dataflow_distance'],spread['dataflow_distance'])
        edge=rules(np.array([[-.9,-.9],[.9,.9]]),c,hp)
        inner=rules(np.array([[-.3,-.3],[.3,.3]]),c,hp)
        self.assertLess(edge['edge_clearance'],inner['edge_clearance'])
        self.assertLess(edge['corner_clearance'],inner['corner_clearance'])

    def test_port_rectangle_fraction_and_no_ports(self):
        c,_=structure(cache());hp=lambda _:0.
        x=np.array([[-.9,0],[.8,0]])
        self.assertAlmostEqual(rules(x,c,hp)['port_blockage_fraction'],.5)
        empty=cache();empty['pin_blocking_rectangles']=[];empty['port_pos']={}
        ctx,_=structure(empty)
        self.assertEqual(rules(x,ctx,hp)['port_blockage_fraction'],0.)

    def test_channel_risk_is_facing_gap_only(self):
        c,_=structure(cache());hp=lambda _:0.
        touch=rules(np.array([[-.1,0],[.1,0]]),c,hp)
        far=rules(np.array([[-.3,0],[.3,0]]),c,hp)
        diagonal=rules(np.array([[-.1,-.4],[.1,.4]]),c,hp)
        self.assertGreater(touch['narrow_channel_risk'],0)
        self.assertEqual(far['narrow_channel_risk'],0)
        self.assertEqual(diagonal['narrow_channel_risk'],0)

    def test_missing_flow_constant_groups_and_bad_geometry(self):
        v=cache();v['dataflow_mat']=None;v['macro_clusters']=[]
        c,m=structure(v)
        r=rules(np.array([[-.8,0],[.8,0]]),c,lambda _:0.)
        self.assertIsNone(r['dataflow_distance']);self.assertEqual(r['cluster_bbox_fraction'],0)
        self.assertIsNotNone(m['dataflow_missing_reason'])
        v['pin_blocking_rectangles']=[[223,0,22,3]]
        with self.assertRaises(ValueError):structure(v)

    def test_constant_and_missing_ranks_and_family_unit(self):
        self.assertIsNone(rank_association([0,0,0],[1,2,3])['spearman'])
        self.assertTrue(describe([0,0,0])['constant'])
        self.assertEqual(describe([np.nan])['missing'],1)
        values=[{k:float(i) for k in ('edge_clearance','corner_clearance','cluster_bbox_fraction',
            'dataflow_distance','port_blockage_fraction','narrow_channel_risk','macro_hpwl')} for i in range(6)]
        metrics=np.ones((6,7));metrics[:,0]=np.arange(6);metrics[:,1:3]=-np.arange(6)[:,None]
        result=associations(values,metrics,np.ones((6,7),bool),np.repeat([0,1,2],2))
        a=result['edge_clearance']['WNS_ns']
        self.assertAlmostEqual(a['family_mean']['spearman'],1.)
        self.assertEqual(a['family_mean']['n'],3)
        self.assertEqual(a['row_level']['n'],6)

    def test_actual_archive_names_and_missing_checkpoint_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(design_config='/immutable/designs/nangate45/bp_fe_top/config_eval.mk')))
            (root/'grt_artifact_archive.json').write_text(json.dumps(dict(status='durable_archive_and_verified_scratch_cache')))
            key='orfs_work/results/nangate45/bp_fe/eval/5_1_grt.odb'
            inventory={key:dict(bytes=5,sha256='a'*64)}
            (root/'grt_intermediates_inventory.json').write_text(json.dumps(inventory))
            row=dict(family=0,variant=0,collection_kind='fixed_TRAIN',bindings=dict(
                physical_manifest=dict(path=str(manifest),sha256=sha(manifest))))
            result=restoration_inventory([row])[0]
            self.assertEqual(result['config_parent'],'bp_fe_top')
            self.assertEqual(result['design_nickname'],'bp_fe')
            self.assertIn('not the results nickname',result['continuation_path_basis'])
            self.assertEqual(len(result['archive_metadata']),2)
            self.assertFalse(result['all_continuation_files_present'])
            self.assertFalse(result['all_missing_continuation_files_in_archive_inventory'])
            checkpoint=next(v for v in result['continuation_files'] if v['path'].endswith('5_1_grt.odb'))
            self.assertEqual(checkpoint['archive_expected'],inventory[key])
            self.assertFalse(checkpoint['actual_content_hash_verified_here'])


if __name__=='__main__':unittest.main()
