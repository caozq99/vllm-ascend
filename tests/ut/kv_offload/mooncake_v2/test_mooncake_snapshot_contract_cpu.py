"""CPU-only contract for Mooncake's saved snapshot block projection."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
STORE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
WORKER = STORE / "layerwise/pool_worker.py"


def helper():
    tree = ast.parse(WORKER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayerwiseKVPoolWorker")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name == "_layerwise_load_snapshot_start"]
    if len(methods) != 1:
        raise AssertionError("Mooncake _load_entries requires the snapshot projection helper")
    node = methods[0]
    node.decorator_list = []
    ns = {}
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), ns)
    return ns[node.name]


class MooncakeSnapshotContractTest(unittest.TestCase):
    def setUp(self):
        self.worker = SimpleNamespace(
            layerwise_offload=True, group_uses_align_state=[False],
            grouped_block_size=[8], cache_transfer_granularity=16,
        )
        self.spec = SimpleNamespace(layerwise_snapshot_end_token=48,
                                    layerwise_snapshot_start_block=2,
                                    layerwise_snapshot_block_index=3)
        self.request = SimpleNamespace(load_spec=self.spec)

    def project(self, cached=40):
        return helper()(self.worker, self.request, 0, cached)

    def test_saved_snapshot_start_projects_transfer_lcm_to_group_blocks(self):
        self.assertEqual(self.project(), 4)

    def test_legacy_snapshot_index_remains_supported(self):
        self.spec.layerwise_snapshot_start_block = None
        self.assertEqual(self.project(), 6)

    def test_older_snapshot_cannot_cover_newer_prefix(self):
        self.spec.layerwise_snapshot_end_token = 32
        self.assertIsNone(self.project())

    def test_recurrent_state_uses_separate_exact_endpoint_path(self):
        self.worker.group_uses_align_state = [True]
        self.assertIsNone(self.project())

    def test_non_offload_has_no_private_projection(self):
        self.worker.layerwise_offload = False
        self.assertIsNone(self.project())

    def test_missing_load_spec_has_no_projection(self):
        self.request.load_spec = None
        self.assertIsNone(self.project())

    def test_missing_identity_has_no_projection(self):
        self.spec.layerwise_snapshot_end_token = None
        self.assertIsNone(self.project())

    def test_missing_start_has_no_projection(self):
        self.spec.layerwise_snapshot_start_block = None
        self.spec.layerwise_snapshot_block_index = None
        self.assertIsNone(self.project())

    def test_empty_prefix_has_no_projection(self):
        self.assertIsNone(self.project(0))


def actual_stage_globals(layout):
    tree = ast.parse(WORKER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayerwiseKVPoolWorker")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_init_layerwise_config")
    assignments = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "stage_globals" for t in n.targets)]
    if len(assignments) != 1:
        raise AssertionError("Expected one stage coordinate definition")
    ns = {"layout": layout}
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])),
                 str(WORKER), "exec"), ns)
    return ns["stage_globals"]


def actual_remap(layout):
    layout_path = STORE / "layerwise/layerwise_cache_layout.py"
    helper_tree = ast.parse(layout_path.read_text())
    reuse = next(n for n in ast.walk(helper_tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "reuse_sources")
    tree = ast.parse(WORKER.read_text())
    remap = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "_remap_layout_to_stage_local")
    remap.decorator_list = []
    ns = {}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), reuse, remap]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(WORKER), "exec"), ns)
    return ns[remap.name](layout.prefetch_layer_map, [], actual_stage_globals(layout))[0]


class MixedGroupStageOrderTest(unittest.TestCase):
    @staticmethod
    def layout(order):
        first, second, third, fourth = sorted(order)
        return SimpleNamespace(layer_cache_specs=dict.fromkeys(order),
                               prefetch_layer_map={third: (first,), fourth: (second,)})

    def test_group_insertion_order_is_not_the_physical_layer_order(self):
        self.assertEqual(actual_stage_globals(self.layout([0, 2, 1, 3])), [0, 1, 2, 3])

    def test_hybrid_predecessors_match_the_physical_layer_index(self):
        self.assertEqual(actual_remap(self.layout([0, 2, 1, 3])), {2: (0,), 3: (1,)})

    def test_pipeline_stage_offset_preserves_the_same_local_predecessors(self):
        self.assertEqual(actual_remap(self.layout([8, 10, 9, 11])), {2: (0,), 3: (1,)})

    def test_already_ordered_layers_keep_the_existing_mapping(self):
        self.assertEqual(actual_remap(self.layout([0, 1, 2, 3])), {2: (0,), 3: (1,)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
