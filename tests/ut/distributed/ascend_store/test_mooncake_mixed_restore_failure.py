# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import weakref
from unittest.mock import Mock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)


def test_mixed_range_restore_keeps_failed_rows_local_to_the_step_batch():
    """Full first hits and compressed-only previous-step loads can alternate."""
    block_size, num_layers = 16, 3
    database = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], [block_size], None)
    # Two region entries per physical layer: resident and rotating.
    database.set_group_buffers(
        {0: [1000, 2000, 3000, 4000, 5000, 6000]},
        {0: [block_size] * (num_layers * 2)},
        group_num_layers={0: num_layers},
    )
    backend = Mock()
    seen = []

    def copy_get(keys, _addresses, _sizes, offsets):
        seen.append((tuple(keys), tuple(tuple(row) for row in offsets)))
        return [-1, 0] if len(seen) == 1 else [0] * len(keys)

    backend.batch_copy_get.side_effect = copy_get
    invalid_blocks = set()
    recv = KVCacheStoreLayerRecvingThread(
        m_store=backend,
        token_database=database,
        block_size=block_size,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        page_size_bytes=block_size * 2,
        ready_event=threading.Event(),
        layer_load_finished_events=[threading.Event() for _ in range(num_layers)],
        layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
        sync_save_events=[],
        num_layers=num_layers,
        invalid_block_ids=invalid_blocks,
    )

    def make_request(req_id, block_ids, keys, previous_step):
        request = ReqMeta(
            req_id,
            block_ids_by_group=[block_ids],
            load_spec=LoadSpec(0, len(keys) * block_size, True, layerwise_previous_step=previous_step),
        )
        request.load_block_keys = keys
        return request

    def task(layer, request, indices=None, shared=None, final=False):
        result = LayerTransferTask(
            layer_id=layer,
            layer_idx_in_group=layer,
            block_ranges=[LayerBlockRange(request, 0, len(request.load_block_keys))],
            group_id=0,
            use_key_major_ranges=True,
            final_group_layer=final,
            component_indices=indices,
        )
        result.shared_block_data = shared if shared is not None else recv.build_shared_data(result)
        assert result.shared_block_data is not None
        return result

    def run_layer(layer, tasks):
        data = LayerLoadTask(wait_for_save_layers=(), transfer_tasks=tasks, layer_id=layer)
        recv.request_queue.put(data)
        recv._handle_request(recv.request_queue.get_nowait())

    first_hit = make_request("first", [7, 8], ["shared-key", "good-key"], False)
    # Same object key, different destination block: filter by batch and row, not key.
    previous_step = make_request("previous", [9], ["shared-key"], True)
    a0, b0 = task(0, first_hit), task(0, previous_step, indices=(1,))
    run_layer(0, [a0, b0])
    # Only B has a real load on the last physical layer of this group.
    run_layer(
        1,
        [
            task(1, first_hit, shared=a0.shared_block_data),
            task(1, previous_step, indices=(1,), shared=b0.shared_block_data),
        ],
    )
    run_layer(2, [task(2, previous_step, indices=(1,), shared=b0.shared_block_data, final=True)])
    assert invalid_blocks == {7}

    # The scheduler consumes invalid blocks and a new step builds new shared data.
    invalid_blocks.clear()
    # Start above layer zero, where an implicit layer-id reset would not help.
    next_step = task(1, first_hit)
    next_step_shared = weakref.ref(next_step.shared_block_data)
    assert next_step.shared_block_data is not a0.shared_block_data
    run_layer(1, [next_step])
    # With no marked group tail, the receiver still must not retain the selection.
    del next_step
    assert next_step_shared() is None
    assert seen == [
        (("shared-key", "good-key"), ((0, 16), (0, 16))),
        (("shared-key",), ((16,),)),
        (("good-key",), ((32, 48),)),
        (("shared-key",), ((48,),)),
        (("shared-key",), ((80,),)),
        (("shared-key", "good-key"), ((32, 48), (32, 48))),
    ]
    assert not invalid_blocks
