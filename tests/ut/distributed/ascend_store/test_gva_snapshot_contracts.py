# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from tests.ut.distributed.ascend_store.test_reuse_snapshot_blocks import worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta, RequestTracker


@pytest.mark.parametrize("saved_end", [None, 18, 20, 24])
def test_recurrent_gva_restore_requires_exact_checkpoint_endpoint(saved_end):
    block_size, computed = 16, 20
    w = worker()
    w.grouped_block_size = [block_size]
    w.hash_block_size = block_size
    w.group_uses_align_state = [True]
    w._reused_layers_by_group = {0: {1}}
    w._snapshot_keys = {}
    tracker = RequestTracker("r", computed, allocated_block_ids_by_group=[[1, 2, 3]])
    hashes = [b"hash0"]
    if saved_end is not None:
        save_tracker = RequestTracker("r", saved_end, allocated_block_ids_by_group=[[1, 2, 3]])
        saved = ReqMeta.from_request_tracker(save_tracker, block_size, block_hashes=hashes, save_partial_block=True)
        w._data_plane._alloc_gvas_for_save([saved])
        assert w._snapshot_keys["r"][0][1] == f"r:snapshot:1:{saved_end}"

    scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
    scheduler.layerwise_offload = True
    load = scheduler._running_load_spec(tracker, computed)
    tracker.token_len = 32
    request = ReqMeta.from_request_tracker(
        tracker, block_size, block_hashes=hashes, load_spec=load, save_partial_block=True
    )
    plan = w._data_plane._plan_load_gvas_for_request(request)
    if saved_end == computed:
        assert plan is not None
        assert f"r:snapshot:1:{computed}" in plan[0][0].keys
    else:
        # A failed plan takes GVA's normal failure path, never issuing a read.
        assert plan is None
        with patch.object(w.m_store, "batch_get_key_info") as query:
            w._data_plane._prepare_load_gvas([request])
        query.assert_not_called()
        assert not request.load_spec.can_load
        assert not request.can_save


def test_gva_aligned_public_lookup_never_falls_back_to_a_stale_private_page():
    block_size = 16
    w = worker()
    w.grouped_block_size = [block_size]
    w.hash_block_size = block_size
    w.group_uses_align_state = [True]
    w._reused_layers_by_group = {0: {1}}
    w._snapshot_keys = {}
    block_table = [[1, 2, 3]]
    tracker = RequestTracker("r", 24, allocated_block_ids_by_group=block_table)
    saved = ReqMeta.from_request_tracker(tracker, block_size, block_hashes=[b"hash0"], save_partial_block=True)
    w._data_plane._alloc_gvas_for_save([saved])
    assert w._snapshot_keys["r"][0][1] == "r:snapshot:1:24"

    scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
    scheduler.layerwise_offload = True
    load = scheduler._running_load_spec(tracker, 32)
    tracker.token_len = 33
    request = ReqMeta.from_request_tracker(
        tracker, block_size, block_hashes=[b"hash0", b"hash1"], load_spec=load, save_partial_block=True
    )
    plan = w._data_plane._plan_load_gvas_for_request(request)
    if plan is not None:
        groups, _ = plan
        assert "r:snapshot:1:24" not in groups[0].snapshot_fallbacks.values()
