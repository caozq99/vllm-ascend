# SPDX-License-Identifier: Apache-2.0
"""Real Qwen descriptors through Mooncake range save/load, CPU storage only."""
import ctypes
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from tests.ut.models.qwen4_exp.test_cache_reuse import _groups, _runtime
from tests.ut.distributed.ascend_store.test_mooncake_hybrid import MemoryRangeStore
from vllm_ascend.models.qwen4_exp.cache_config import (
    build_qwen4_kv_cache_layout, get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_kv_cache_config, get_qwen4_exp_pool_bytes_per_block, make_slab_view, make_qsa_kernel_page_cache,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import apply_layerwise_kv_cache_plan
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_views import allocate_cache_backing, get_cache_views
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import get_layerwise_kv_cache_specs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    LayerBatchBuilder, KVCacheStoreLayerSendingThread, KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData, LayerLoadTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta, LoadSpec


class TestQwenMooncakeDescriptorRoundtrip(unittest.TestCase):
    def run_roundtrip(self, packed):
        runtime = _runtime(buffers=8, independent=[])
        runtime.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
        runtime.cache_config.get_resolved_kv_cache_layout = lambda: SimpleNamespace(is_block_outermost=packed)
        runtime.kv_transfer_config.kv_connector_extra_config['backend'] = 'mooncake'
        runtime.kv_transfer_config.kv_role = 'kv_producer'
        specs = {name: group.kv_cache_spec.kv_cache_specs[name]
                 if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else group.kv_cache_spec
                 for group in _groups() for name in group.layer_names}
        groups = get_qwen4_exp_kv_cache_groups(runtime, specs, get_kv_cache_groups)
        cost = get_qwen4_exp_pool_bytes_per_block(groups)
        config = get_qwen4_exp_kv_cache_config(runtime, groups, 3 * cost)
        native = build_qwen4_kv_cache_layout(groups, config.num_blocks)
        apply_layerwise_kv_cache_plan(config, runtime, native_backing_bytes=3 * cost)
        layout = config.layerwise_reuse_layout
        target = 15 if packed else 3
        main = f'model.layers.{target}.self_attn.attn'
        compressed = f'model.layers.{target}.self_attn.indexer.compressed_key_cache'
        self.assertNotIn(main, layout.reused_cache_names)
        self.assertIn(compressed, layout.reused_cache_names)
        group_id = next(i for i,g in enumerate(groups) if main in g.layer_names and compressed in g.layer_names)
        names = groups[group_id].layer_names
        cache_specs = get_layerwise_kv_cache_specs(config)
        backing = allocate_cache_backing(config, cache_specs, lambda size: torch.zeros(size, dtype=torch.int8))
        windows = get_cache_views(config, cache_specs)
        caches = {}
        # Reconstruct the actual main K/V and compressed views from descriptor
        # anchors, component offsets and physical block stride, not dense stand-ins.
        for name in names:
            spec = cache_specs[name]
            descriptor = config.kv_cache_tensors[windows[name][0]]
            if name.endswith('.attn'):
                caches[name] = make_qsa_kernel_page_cache(
                    backing[name], dtype=spec.dtype, num_blocks=config.num_blocks,
                    manager_block_size=spec.block_size, kernel_block_size=128,
                    num_kv_heads=spec.num_kv_heads, head_size=spec.head_size,
                    head_size_v=spec.head_size_v, storage_offset=0,
                    block_stride=descriptor.block_stride)
            else:
                self.assertTrue(name.endswith('.compressed_key_cache'))
                caches[name] = (make_slab_view(backing[name], dtype=spec.dtype,
                    num_blocks=config.num_blocks,
                    item_shape=(spec.block_size // spec.tokens_per_state, spec.num_kv_heads, spec.head_size),
                    storage_offset=0, block_stride=descriptor.block_stride),)
            self.assertEqual(caches[name][0].data_ptr(), backing[name].data_ptr())
            region = native.region(native.region_name_for_owner(name))
            self.assertLessEqual(sum(t[0].numel()*t.element_size() for t in caches[name]), region.page_size_bytes)

        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker.num_blocks = config.num_blocks
        worker.num_layers = 48
        worker.hf_config = SimpleNamespace(num_hidden_layers=48)
        worker.use_layerwise = True
        worker.use_kvpp = False
        worker._global_to_local_layer = {i:i for i in range(48)}
        worker.kv_caches = caches
        for attr in ('group_kv_caches_base_addr','group_block_len','group_block_stride',
                     'group_layer_cache_entry_offsets','group_cache_entry_segments',
                     'group_cache_entry_names','group_num_layers'):
            setattr(worker, attr, {})
        worker._infer_cache_group_metadata(group_id, names)
        layers = sorted({worker._extract_physical_layer_index(n) for n in names})
        layer_index = layers.index(target)
        builder = LayerBatchBuilder(worker, page_size_bytes=sum(worker.group_block_len[group_id]),
                                    num_layers=len(layers), group_id=group_id)
        owners = worker.group_cache_entry_names[group_id]
        self.assertEqual(len(owners), len(worker.group_block_len[group_id]))
        self.assertEqual(owners.count(main), 2 * (cache_specs[main].block_size // 128))
        self.assertEqual(owners.count(compressed), 1)
        total = sum(worker.group_block_len[group_id])
        keys = ['private-old-snapshot', 'public-fresh-prefix']
        store = MemoryRangeStore()
        store.batch_put_start(keys, [total, total])
        shared = SharedBlockData(np.array([0,1]), None, ['old','fresh'], [False,False], block_keys=keys)
        sender = KVCacheStoreLayerSendingThread.__new__(KVCacheStoreLayerSendingThread)
        sender.m_store = store
        sender._active_put_keys = None
        sender.sync_save_events = [MagicMock() for _ in range(48)]
        sender.max_transfer_blocks = sender.max_transfer_bytes = 0
        sender.raise_on_transfer_failure = True
        sender.final_layer_id = layers[-1]
        sender._session_tracker = None
        sender._put_started_keys = set(keys)
        sender._put_started_keys_lock = threading.Lock()
        expected = [bytearray(total), bytearray(total)]
        for index, physical in enumerate(layers):
            for name in names:
                if worker._extract_physical_layer_index(name) == physical:
                    for component, tensor in enumerate(caches[name]):
                        for row in (0,1):
                            scale = tensor.shape[0] // config.num_blocks
                            tensor[row * scale:(row+1) * scale].fill_(2 + index * 3 + component + row * 50)
            metadata = builder.build_addrs(shared, index)
            metadata.layer_id = physical
            first,last = worker.group_layer_cache_entry_offsets[group_id][index:index+2]
            wire_offsets = np.cumsum([0] + worker.group_block_len[group_id][:-1]).tolist()
            for row in (0,1):
                # Independent oracle: registration's complete entry order, not
                # the builder's selected/range offsets used by both endpoints.
                self.assertEqual(metadata.all_offsets[row],wire_offsets[first:last])
                for entry in range(first,last):
                    size = worker.group_block_len[group_id][entry]
                    offset = wire_offsets[entry]
                    addr = worker.group_kv_caches_base_addr[group_id][entry] + row * worker.group_block_stride[group_id][entry]
                    expected[row][offset:offset+size] = ctypes.string_at(addr,size)
            sender._handle_range_request(metadata, final_group_layer=index == len(layers)-1)
        self.assertEqual(store.complete, set(keys))
        for key, content in zip(keys,expected,strict=True):
            self.assertEqual(store.objects[key], content, 'Full published object differs from descriptor-derived bytes')

        # Poison all shared memory, then preserve an old request's resident KV.
        for tensor in backing.values():
            tensor.fill_(-1)
        for tensor in caches[main]:
            tensor[:tensor.shape[0] // config.num_blocks].fill_(99)
        resident_meta = builder.build_addrs(shared,layer_index)
        owner_start = worker.group_layer_cache_entry_offsets[group_id][layer_index]
        resident = {local:ctypes.string_at(addr,size) for local,(addr,size) in enumerate(
            zip(resident_meta.all_buffers[0],resident_meta.all_sizes[0],strict=True))
            if owners[owner_start+local] == main}
        worker.layerwise_offload = True
        worker.use_block_key_layerwise = True
        worker.grouped_block_size = [g.kv_cache_spec.block_size for g in groups]
        worker.independent_layers = []
        worker._reused_layers_by_group = {group_id: {worker._extract_physical_layer_index(name) for name in names if name in layout.reused_cache_names}}
        worker._layerwise_reuse_layout = layout
        worker.layer_load_tasks = [[] for _ in range(48)]
        requests = []
        for row, previous in ((0,True),(1,False)):
            req = ReqMeta('old' if previous else 'fresh', block_ids=[row],
                load_spec=LoadSpec(0,256,can_load=True,layerwise_previous_step=previous),
                is_last_chunk=False)
            req.load_block_keys = [keys[row]]
            req.block_ids_by_group = [[row] for _ in groups]
            req.load_keys = [keys[row]]
            requests.append(req)
        worker._process_load_for_layer_batch(requests, target, group_id=group_id, layer_idx_in_group=layer_index)
        tasks = worker.layer_load_tasks[target]
        self.assertEqual(len(tasks),2)
        selectors = {task.block_ranges[0].request.req_id:task.component_indices for task in tasks}
        self.assertIsNone(selectors['fresh'])
        begin,end = worker.group_layer_cache_entry_offsets[group_id][layer_index:layer_index+2]
        self.assertEqual(selectors['old'],tuple(i for i,n in enumerate(owners[begin:end]) if n in layout.reused_cache_names))
        for task in tasks:
            task.shared_block_data = builder.build_shared(task,is_save=False)
        store.batch_get_start(keys)
        receiver = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        receiver.m_store = store
        receiver.group_builders = {group_id:builder}
        receiver._group_load_indices = {}
        receiver._active_load_indices = None
        receiver._load_abort_event = threading.Event()
        receiver.max_transfer_blocks = receiver.max_transfer_bytes = 0
        receiver.h2d_stagger_us = 0
        receiver.final_layer_id = 47
        receiver.external_slot_release_waiter = None
        receiver.save_failure_checker = None
        receiver.layer_load_finished_events = [threading.Event() for _ in range(48)]
        receiver.get_event = threading.Event()
        receiver.request_queue = queue.Queue()
        load = LayerLoadTask(wait_for_save_layers=(), transfer_tasks=tasks,layer_id=target)
        receiver.request_queue.put(load)
        receiver._handle_range_layer_task(load)
        store.batch_get_end(keys)
        self.assertTrue(receiver.layer_load_finished_events[target].is_set())
        self.assertFalse(store.open_reads)
        full = builder.build_addrs(shared,layer_index)
        for row in (0,1):
            for local,(addr,size,offset) in enumerate(zip(full.all_buffers[row],full.all_sizes[row],full.all_offsets[row],strict=True)):
                owner = owners[begin+local]
                actual = ctypes.string_at(addr,size)
                if row == 0 and owner == main:
                    self.assertEqual(actual,resident[local])
                else:
                    self.assertEqual(actual,bytes(expected[row][offset:offset+size]),(packed,row,owner,offset))
        print('QWEN_REAL_DESCRIPTOR_ROUNDTRIP',packed,target,'bytes',total,'selectors',selectors)

    def test_nonpacked_real_descriptor_roundtrip(self):
        self.run_roundtrip(False)

    def test_packed_real_descriptor_roundtrip(self):
        self.run_roundtrip(True)