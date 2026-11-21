# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.models.qwen4_exp.test_cache_reuse import _groups, _runtime
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_pool_bytes_per_block
from vllm_ascend.platform import NPUPlatform


@pytest.fixture(autouse=True)
def use_ascend_cache_builder(monkeypatch):
    monkeypatch.setattr(
        current_platform, "get_kv_cache_config_builder_cls", NPUPlatform.get_kv_cache_config_builder_cls
    )


def _setup_admission(auto_fit, backend="mooncake"):
    config = _runtime()
    config.kv_transfer_config.kv_connector_extra_config["backend"] = backend
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.enable_extensible_kv_cache = False
    max_len = 4096
    config.model_config.max_model_len = max_len
    config.model_config.original_max_model_len = -1 if auto_fit else max_len
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=max_len, max_num_seqs=1)
    config.max_in_flight_tokens = max_len
    config.attention_config = SimpleNamespace(hisparse_config=None)
    config.device_config = SimpleNamespace(device_type="cpu")
    groups = _groups()
    specs = {
        name: group.kv_cache_spec.kv_cache_specs[name]
        if hasattr(group.kv_cache_spec, "kv_cache_specs")
        else group.kv_cache_spec
        for group in groups
        for name in group.layer_names
    }
    return config, groups, specs


def _required_blocks(config, groups):
    return sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(config), group.kv_cache_spec.page_size_bytes)
        for group in groups
    )


def _compiled_budget(config, groups):
    block_count = _required_blocks(config, groups) + 1  # Core reserves one null block.
    compiled = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(config, groups, num_blocks=block_count)
    budget = get_layerwise_reuse_backing_bytes(compiled)
    assert budget is not None
    native_bytes = get_qwen4_exp_pool_bytes_per_block(groups)
    assert native_bytes is not None
    assert budget < block_count * native_bytes
    return block_count, budget


@pytest.mark.parametrize("backend", ["mooncake", "memcache"])
@pytest.mark.parametrize("auto_fit", [False, True], ids=["fixed-length", "auto-fit"])
def test_compiled_qwen_shared_pool_admits_full_context_at_physical_budget(auto_fit, backend):
    config, groups, specs = _setup_admission(auto_fit, backend)
    length = config.model_config.max_model_len
    block_count, budget = _compiled_budget(config, groups)

    [result] = get_kv_cache_configs(config, [specs], [budget], groups)

    assert config.model_config.max_model_len == length
    assert result.num_blocks == block_count
    assert get_layerwise_reuse_backing_bytes(result) == budget


def test_qwen_shared_pool_cannot_spend_the_null_block():
    config, groups, specs = _setup_admission(auto_fit=False)
    _, budget = _compiled_budget(config, groups)
    with pytest.raises(ValueError, match="larger than the available KV cache memory"):
        get_kv_cache_configs(config, [specs], [budget - 1], groups)


@pytest.mark.parametrize("mode", ["disabled", "all-resident"])
def test_qwen_without_compiled_reuse_keeps_native_admission(mode):
    config, groups, specs = _setup_admission(auto_fit=False)
    _, shared_budget = _compiled_budget(config, groups)
    extra = config.kv_transfer_config.kv_connector_extra_config
    if mode == "disabled":
        extra["use_layerwise"] = False
    else:
        extra["layerwise_independent_layers"] = "all"
    block_count = _required_blocks(config, groups) + 1
    native_bytes = get_qwen4_exp_pool_bytes_per_block(groups)
    assert native_bytes is not None

    with pytest.raises(ValueError, match="larger than the available KV cache memory"):
        get_kv_cache_configs(config, [specs], [shared_budget], groups)
    [result] = get_kv_cache_configs(config, [specs], [block_count * native_bytes], groups)
    assert result.num_blocks == block_count
    assert get_layerwise_reuse_backing_bytes(result) is None


def test_unregistered_model_keeps_ordinary_budget_with_layerwise_option_set():
    config, _, _ = _setup_admission(auto_fit=False, backend="memcache")
    config.model_config.hf_config.model_type = "llama"
    config.model_config.hf_text_config.model_type = "llama"
    config.cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout.LBHNC
    spec = FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=64, dtype=torch.float16)
    specs = {"model.layers.0.attn": spec}
    groups = [KVCacheGroupSpec(list(specs), spec)]
    block_count = _required_blocks(config, groups) + 1
    budget = block_count * spec.page_size_bytes

    with pytest.raises(ValueError, match="larger than the available KV cache memory"):
        get_kv_cache_configs(config, [specs], [budget - 1], groups)
    [result] = get_kv_cache_configs(config, [specs], [budget], groups)
    assert result.num_blocks == block_count
    assert get_layerwise_reuse_backing_bytes(result) is None
