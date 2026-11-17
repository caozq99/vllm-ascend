# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU address-planning contracts for resident/reused component selection."""

import unittest

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData


class TestComponentSelectorCPU(unittest.TestCase):
    @staticmethod
    def make_builder():
        builder = LayerBatchBuilder.__new__(LayerBatchBuilder)
        builder._block_len_np = np.array([10, 20, 30], dtype=np.int64)
        builder._kv_caches_base_addr_np = np.array([1000, 2000, 3000], dtype=np.int64)
        builder._block_stride_np = np.array([100, 200, 300], dtype=np.int64)
        builder._layer_cache_entry_offsets_np = np.array([0, 2, 3], dtype=np.int64)
        builder.layer_byte_offset = 0
        builder.page_size_bytes = 60
        builder.num_layers = 2
        return builder

    @staticmethod
    def shared(block_key=False):
        return SharedBlockData(
            block_ids_arr=np.array([2], dtype=np.int64),
            block_gvas_arr=np.array([500], dtype=np.int64),
            req_ids=["request"],
            is_last_chunks=[True],
            block_keys=["key"] if block_key else None,
        )

    def assert_gva(self, result, addresses, sizes, gvas):
        np.testing.assert_array_equal(result.addr_array, addresses)
        np.testing.assert_array_equal(result.size_array, sizes)
        np.testing.assert_array_equal(result.gvas_array, gvas)

    def assert_ranges(self, result, addresses, sizes, offsets):
        np.testing.assert_array_equal(result.all_buffers, addresses)
        np.testing.assert_array_equal(result.all_sizes, sizes)
        np.testing.assert_array_equal(result.all_offsets, offsets)

    def test_baseline_gva_without_selector(self):
        result = self.make_builder().build_addrs(self.shared(), 0)
        self.assert_gva(result, [1200, 2400], [10, 20], [500, 510])

    def test_baseline_block_key_without_selector(self):
        result = self.make_builder().build_addrs(self.shared(True), 0)
        self.assert_ranges(result, [[1200, 2400]], [[10, 20]], [[0, 10]])

    def test_none_preserves_all_components_in_both_protocols(self):
        for block_key in (False, True):
            with self.subTest(block_key=block_key):
                result = self.make_builder().build_addrs(self.shared(block_key), 0, component_indices=None)
                if block_key:
                    self.assert_ranges(result, [[1200, 2400]], [[10, 20]], [[0, 10]])
                else:
                    self.assert_gva(result, [1200, 2400], [10, 20], [500, 510])

    def test_gva_second_component_keeps_original_remote_offset(self):
        result = self.make_builder().build_addrs(self.shared(), 0, component_indices=(1,))
        self.assert_gva(result, [2400], [20], [510])

    def test_block_key_second_component_keeps_original_wire_offset(self):
        result = self.make_builder().build_addrs(self.shared(True), 0, component_indices=(1,))
        self.assert_ranges(result, [[2400]], [[20]], [[10]])

    def test_empty_gva_selector_has_no_copy_ranges(self):
        result = self.make_builder().build_addrs(self.shared(), 0, component_indices=())
        self.assertEqual(np.asarray(result.addr_array).size, 0)
        self.assertEqual(np.asarray(result.size_array).size, 0)
        self.assertEqual(np.asarray(result.gvas_array).size, 0)

    def test_empty_block_key_selector_has_no_copy_ranges(self):
        result = self.make_builder().build_addrs(self.shared(True), 0, component_indices=())
        self.assertEqual(np.asarray(result.all_buffers).size, 0)
        self.assertEqual(np.asarray(result.all_sizes).size, 0)
        self.assertEqual(np.asarray(result.all_offsets).size, 0)

    def test_invalid_relative_component_indices_fail_closed(self):
        # Layer 0 has two entries. Global entry 2 belongs to layer 1 and must
        # not be addressable through layer 0's relative component selector.
        for block_key in (False, True):
            for indices in ((-1,), (2,)):
                with self.subTest(block_key=block_key, component_indices=indices):
                    with self.assertRaises((ValueError, IndexError)):
                        self.make_builder().build_addrs(self.shared(block_key), 0, component_indices=indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
