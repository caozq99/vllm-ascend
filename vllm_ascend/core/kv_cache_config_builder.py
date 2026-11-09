# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend KV cache planning hooks for vLLM's pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). Ascend-specific grouping and
pool accounting live here; materialization stays in the patched planning
functions (see ``patch_kv_cache_utils.py``), which
``DefaultKVCacheConfigBuilder.get_kv_cache_config_from_groups`` calls with the
Core-resolved ``num_blocks``.
"""

from collections import defaultdict

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _approximate_gcd,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
