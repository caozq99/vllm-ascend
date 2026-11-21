# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import MooncakePushConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerLoadTask


def test_store_reuse_waits_for_both_source_push_writes_without_consuming_store_events():
    receiver = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
    push = MooncakePushConnectorWorker.__new__(MooncakePushConnectorWorker)
    receiver._stopping = threading.Event()
    receiver.save_failure_checker = None
    receiver.layer_save_finished_events = [threading.Event(), threading.Event()]
    receiver.sync_save_events = [SimpleNamespace(synchronize=Mock()), SimpleNamespace(synchronize=Mock())]
    first, second = Future(), Future()
    first.set_result(None)
    push._shared_layer_writes = {0: (first,), 1: (second,)}
    second_push_wait = threading.Event()
    finished_sources = []

    def wait_for_push(source):
        if source == 1:
            second_push_wait.set()
        push.wait_for_layer_reuse(source)
        finished_sources.append(source)

    receiver.external_slot_release_waiter = wait_for_push
    task = LayerLoadTask(wait_for_save_layers=(0, 1), transfer_tasks=[], layer_id=2)
    receiver.layer_save_finished_events[0].set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(receiver._wait_for_reuse_saves, task)
        try:
            assert not second_push_wait.is_set()
            assert not result.done()
            receiver.layer_save_finished_events[1].set()
            assert second_push_wait.wait(timeout=3)
            assert finished_sources == [0]
            assert not result.done()
        finally:
            receiver.layer_save_finished_events[1].set()
            second.set_result(None)
        assert result.result(timeout=3)
    assert finished_sources == [0, 1]
    # A second successor can observe these same save completions.
    assert receiver._wait_for_reuse_saves(task)
    assert finished_sources == [0, 1, 0, 1]
    for sync in receiver.sync_save_events:
        assert sync.synchronize.call_count == 2
