# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component identity must survive native registration and running-step restore."""
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


def make_worker():
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.layerwise_offload = True
    worker.use_block_key_layerwise = True
    worker.grouped_block_size = [16]
    worker.independent_layers = []
    worker._reused_layers_by_group = {0: {0}}
    worker.group_num_layers = {0: 1}
    worker.num_layers = 1
    worker.layer_load_tasks = [[]]
    worker._layerwise_reuse_layout = SimpleNamespace(reused_cache_names=frozenset({'compressed'}))
    worker.group_cache_entry_names = {0: ['main', 'main', 'compressed']}
    worker.group_layer_cache_entry_offsets = {0: [0, 3]}
    return worker


def request(previous):
    return SimpleNamespace(load_spec=SimpleNamespace(layerwise_previous_step=previous), load_block_keys=['key'])


def test_running_restore_selects_reused_entries_but_external_hit_restores_all():
    worker = make_worker()
    old, fresh = request(True), request(False)
    worker._process_load_for_layer_batch([old, fresh], 0)
    tasks = worker.layer_load_tasks[0]
    assert len(tasks) == 2
    assert {task.component_indices for task in tasks} == {(2,), None}
    for task in tasks:
        assert [r.request for r in task.block_ranges] == ([old] if task.component_indices else [fresh])


def test_resident_only_running_restore_does_not_enqueue_empty_task():
    worker = make_worker()
    worker._layerwise_reuse_layout.reused_cache_names = frozenset()
    worker._process_load_for_layer_batch([request(True)], 0)
    assert worker.layer_load_tasks == [[]]


def test_multiple_entries_from_one_reused_owner_are_all_selected():
    worker = make_worker()
    worker._layerwise_reuse_layout.reused_cache_names = frozenset({'main'})
    assert worker._reused_component_indices(0, 0) == (0, 1)


def test_registration_tracks_each_segment_owner_in_wire_order():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

    worker = make_worker()
    worker._global_to_local_layer = {}
    worker._extract_physical_layer_index = lambda name: 0
    worker._as_cache_tuple = lambda caches: caches
    worker._get_combined_cache_transfer_segments = lambda caches, group: None
    worker._get_cache_transfer_segments = lambda cache: cache
    worker.kv_caches = {
        'main': ([(10, 2, 20)], [(30, 3, 30)]),
        'compressed': ([(40, 1, 10), (50, 1, 10)],),
    }
    for attr in ('group_kv_caches_base_addr', 'group_block_len', 'group_block_stride', 'group_cache_entry_segments'):
        setattr(worker, attr, {})
    BaseKVPoolWorker._infer_cache_group_metadata(worker, 0, ['main', 'compressed'])
    assert worker.group_cache_entry_names[0] == ['compressed', 'compressed', 'main', 'main']
    assert worker.group_kv_caches_base_addr[0] == [40, 50, 10, 30]
    assert worker._reused_component_indices(0, 0) == (0, 1)


def _partition_fixture():
    import threading
    from types import MethodType
    from unittest.mock import Mock

    import numpy as np

    from tests.ut.distributed.ascend_store.test_component_selector_cpu import TestComponentSelectorCPU
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import KVCacheStoreLayerRecvingThread
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
        LayerLoadTask, LayerTransferTask, SharedBlockData,
    )

    received = []
    lane = SimpleNamespace(
        _wait_for_reuse_saves=lambda data: True,
        group_builders=None,
        layer_batch_builder=TestComponentSelectorCPU.make_builder(),
        _group_load_indices={},
        _active_load_indices=None,
        _load_abort_event=threading.Event(),
        _stagger_h2d_submit=lambda layer: None,
        max_transfer_blocks=0,
        max_transfer_bytes=0,
        final_layer_id=9,
        layer_load_finished_events=[threading.Event() for _ in range(3)],
        request_queue=Mock(), get_event=threading.Event(),
    )
    lane._range_transfer_batches = lambda keys, buffers, sizes, offsets, *args: [(keys, buffers, sizes, offsets)]
    lane.m_store = SimpleNamespace(batch_copy_get=lambda keys, *args: received.extend(keys) or [0] * len(keys))
    lane._handle_range_request = MethodType(KVCacheStoreLayerRecvingThread._handle_range_request, lane)
    tasks = []
    for keys, selection in [(['old'], (1,)), (['fresh1', 'fresh2'], None)]:
        shared = SharedBlockData(np.array([2] * len(keys)), None, keys, [False] * len(keys), block_keys=keys)
        tasks.append(LayerTransferTask(1, [], shared_block_data=shared, component_indices=selection))
    lane.run = lambda layer, batch: KVCacheStoreLayerRecvingThread._handle_range_layer_task(
        lane, LayerLoadTask((), batch, layer)
    )
    return lane, tasks, received


def test_partitioned_range_loads_do_not_share_row_indices():
    lane, tasks, received = _partition_fixture()
    lane.run(1, tasks)
    assert received == ['old', 'fresh1', 'fresh2']


def test_partition_failures_survive_other_partition_until_group_end_and_next_step():
    lane, tasks, received = _partition_fixture()
    lane._mark_invalid_range_indices = lambda meta, indices: None
    def copy(keys, *args):
        received.extend(keys)
        return [-1 if key == 'old' else 0 for key in keys]
    lane.m_store.batch_copy_get = copy
    lane.run(1, tasks)
    assert received == ['old', 'fresh1', 'fresh2']
    received.clear()
    # A different key partition must neither inherit nor erase this failure.
    # A later layer may select different components of the same failed row.
    tasks[0].component_indices = None
    lane.run(2, tasks)
    assert received == ['fresh1', 'fresh2']
    tasks[-1].final_group_layer = True
    lane.run(2, tasks)
    assert not lane._group_load_indices
    # The next step can read the same keys again, with no stale failed rows.
    received.clear()
    lane.run(1, tasks)
    assert received == ['old', 'fresh1', 'fresh2']


def test_partition_identity_preserves_duplicate_key_rows_and_order():
    lane, tasks, received = _partition_fixture()
    tasks[1].shared_block_data.block_keys = ['old', 'old']
    lane.run(1, tasks)
    assert received == ['old', 'old', 'old']



def test_partition_identity_preserves_key_order():
    lane, tasks, received = _partition_fixture()
    lane._mark_invalid_range_indices = lambda meta, indices: None
    lane.m_store.batch_copy_get = lambda keys, *args: received.extend(keys) or [-1, 0]
    lane.run(1, [tasks[1]])
    assert received == ['fresh1', 'fresh2']
    received.clear()
    tasks[1].shared_block_data.block_keys = ['fresh2', 'fresh1']
    lane.m_store.batch_copy_get = lambda keys, *args: received.extend(keys) or [0] * len(keys)
    lane.run(2, [tasks[1]])
    assert received == ['fresh2', 'fresh1']


def test_same_key_different_destination_does_not_inherit_failed_row():
    from copy import deepcopy

    lane, tasks, received = _partition_fixture()
    old = tasks[0]
    fresh = deepcopy(old)
    fresh.component_indices = None
    fresh.shared_block_data.block_ids_arr[0] = 3
    fresh.shared_block_data.req_ids = ['fresh-request']
    lane._mark_invalid_range_indices = lambda meta, indices: None
    calls = []
    def copy(keys, *args):
        calls.append(keys)
        return [-1] if len(calls) == 1 else [0]
    lane.m_store.batch_copy_get = copy
    lane.run(1, [old, fresh])
    assert calls == [['old'], ['old']]


def test_range_abort_clears_all_partition_state():
    import pytest

    lane, tasks, _ = _partition_fixture()
    lane._group_load_indices[(0, ('old-step',), (7,), ('request',))] = {0}
    lane._mark_invalid_transfer_task_blocks = lambda tasks: None
    def fail(*args):
        raise RuntimeError('copy failed')
    lane.m_store.batch_copy_get = fail
    with pytest.raises(RuntimeError, match='copy failed'):
        lane.run(1, tasks)
    assert not lane._group_load_indices
    assert lane._load_abort_event.is_set()
