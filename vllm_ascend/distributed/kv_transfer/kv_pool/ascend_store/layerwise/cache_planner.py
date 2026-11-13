# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan layerwise cache reuse from native cache groups."""

from collections import defaultdict
from math import lcm
from types import MappingProxyType
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_backend_class,
)

from .cache_views import (
    LayerwiseCacheConfig,
    LayerwiseReuseLayout,
    SpecReuseRules,
    get_cache_views,
)
from .layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
    get_physical_layer_index,
)

SPEC_REUSE_RULES = MappingProxyType(
    {
        **dict.fromkeys(
            ("glm5_next", "glm5_next_text", "glm_moe_dsa"),
            "vllm_ascend.models.glm5next.cache_reuse.GLM5_NEXT_SPEC_REUSE_RULES",
        ),
        **dict.fromkeys(
            ("qwen4_exp", "qwen4_exp_text"),
            "vllm_ascend.models.qwen4_exp.cache_reuse.QWEN4_EXP_SPEC_REUSE_RULES",
        ),
    }
)


def get_spec_reuse_rules(vllm_config: VllmConfig) -> SpecReuseRules | None:
    text_config = getattr(vllm_config.model_config, "hf_text_config", None) or getattr(
        vllm_config.model_config, "hf_config", None
    )
    path = SPEC_REUSE_RULES.get(getattr(text_config, "model_type", None))
    return resolve_obj_by_qualname(path) if path is not None else None


def get_layerwise_reuse_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Use one transport gate for both GVA reuse and producer-only Mooncake."""
    from .layerwise_cache_layout import get_layerwise_reuse_config as get_config

    return get_config(kv_transfer_config)


# Existing Mooncake call sites use this name for the same opt-in policy.
def get_layerwise_offload_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    return get_layerwise_reuse_config(kv_transfer_config)


class LayerwiseCachePlanner:
    @staticmethod
    def can_reuse_spec(spec: KVCacheSpec, rules: SpecReuseRules | None) -> bool:
        """Apply the rule for the concrete spec type, defaulting to false."""
        rule = False if rules is None else rules.get(type(spec), False)
        result = rule(spec) if callable(rule) else rule
        if not isinstance(result, bool):
            raise TypeError("Spec reuse rules must return bool")
        return result

    def __init__(
        self,
        config: KVCacheConfig,
        vllm_config: VllmConfig,
        extra_config: dict[str, Any],
        excluded_layer_names: set[str] | None = None,
        *,
        spec_reuse_rules: SpecReuseRules,
        native_backing_bytes: int | None = None,
    ) -> None:
        self.config = config
        self.vllm_config = vllm_config
        self.extra_config = extra_config
        self.spec_reuse_rules = spec_reuse_rules
        self.native_backing_bytes = native_backing_bytes
        self.specs = get_layerwise_kv_cache_specs(config)
        self.group_ids = {
            name: index for index, group in enumerate(config.kv_cache_groups) for name in group.layer_names
        }
        self.transfer_groups = {
            i for i, group in enumerate(config.kv_cache_groups) if getattr(group, "enable_kv_transfer", True)
        }
        topology = getattr(config, "layerwise_cache_topology", None)
        if topology is not None and dict(topology.group_ids) != self.group_ids:
            raise ValueError("Layerwise cache topology does not match the native cache groups")
        get_total = getattr(vllm_config.model_config, "get_total_num_hidden_layers", None)
        base_layers = (
            get_total() if callable(get_total) else vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        )
        self.total_layers = base_layers
        self.layer_indices = {name: get_physical_layer_index(name, base_layers) for name in self.specs}
        registered = set(getattr(config, "dspark_draft_layer_names", ()))
        self.draft_names = {
            name
            for name, spec in self.specs.items()
            if name in registered or getattr(spec, "is_draft", False) or self.layer_indices[name] >= base_layers
        }
        self.layers = tuple(
            sorted({index for name, index in self.layer_indices.items() if name not in self.draft_names})
        )
        resident = set(getattr(topology, "resident_layer_names", ()))
        self.excluded_layer_names = (
            self.draft_names | resident | (frozenset(excluded_layer_names or ()) & self.specs.keys())
        )

    def _alignment(self, packs: list[tuple[str, ...]]) -> int:
        """Return the element alignment of the packs' data and scale components."""
        dtypes = []
        for pack in packs:
            for name in pack:
                spec = self.specs[name]
                dtypes.extend(getattr(spec, "dtypes", (getattr(spec, "dtype", torch.uint8),)))
                if getattr(spec, "scale_dim", 0):
                    dtypes.append(spec.scale_dtype)
        return lcm(*(get_dtype_size(dtype) for dtype in dtypes))

    def build(self) -> LayerwiseReuseLayout:
        indices, layers = self.layer_indices, self.layers
        settings = LayerwiseCacheConfig.parse(self.total_layers, self.extra_config, stage_layers=layers)
        independent = settings.independent_layers
        tensors = self.config.kv_cache_tensors
        buckets, aliases = defaultdict(list), defaultdict(list)
        regions = {}
        members = set()
        views = {}
        # 1. Bucket layer names by block stride, independent of native slots.
        if settings.has_layer_reuse:
            members = {
                name
                for name, spec in self.specs.items()
                if name not in self.excluded_layer_names
                and indices[name] not in independent
                and self.group_ids[name] in self.transfer_groups
                and is_prefix_cacheable(spec)
                and self.can_reuse_spec(spec, self.spec_reuse_rules)
            }
            views = get_cache_views(self.config, self.specs)
            backings = defaultdict(list)
            allocation_ids = getattr(self.config, "native_allocation_ids", range(len(tensors)))
            bucket_ids = {}
            for name, (index, start, end) in views.items():
                stride = tensors[index].block_stride
                bucket_ids[name] = stride
                buckets[stride].append(name)
                if name in members:
                    regions[name] = (0, end - start)
                else:
                    backings[allocation_ids[index]].append(name)
            # Preserve overlaps only between resident views. Reusable views
            # are relocated to their pack origin.
            for names in backings.values():
                units, unit, end = [], [], -1
                for name in sorted(names, key=lambda name: views[name][1:]):
                    if unit and views[name][1] >= end:
                        units.append(unit)
                        unit = []
                    unit.append(name)
                    end = max(end, views[name][2])
                units.append(unit)
                for unit in units:
                    start = views[unit[0]][1]
                    stride = bucket_ids[unit[0]]
                    if any(bucket_ids[name] != stride for name in unit):
                        # A shared interval with incompatible page strides cannot be relocated.
                        break
                    size = max(views[name][2] for name in unit) - start
                    aliases[stride].append(unit)
                    for name in unit:
                        regions[name] = (views[name][1] - start, size)
                else:
                    continue
                buckets.clear()
                break
        # Buckets occupy disjoint regions in one device backing. Resident
        # packs reserve private bytes in that backing.
        candidate = [tensor for tensor in tensors if tensor.host_resident]
        dependencies = defaultdict(set)
        reused_names = set()
        capacity = 0
        for stride, names in buckets.items():
            # 2. Greedily take the longest layer-ordered run without a repeated
            # group. A repeated group starts the next pack, including at the
            # same physical layer; groups are never inferred from spec types.
            packs, pack, groups = [], [], set()
            for name in sorted(names, key=lambda name: (indices[name], self.group_ids[name], name)):
                if name not in members:
                    continue
                group = self.group_ids[name]
                if group in groups:
                    packs.append(tuple(pack))
                    pack, groups = [], set()
                pack.append(name)
                groups.add(group)
            if pack:
                packs.append(tuple(pack))
            packs.extend(tuple(unit) for unit in aliases[stride])
            packs.sort(key=lambda pack: min(indices[name] for name in pack))
            descriptors, cursor = {}, 0
            dtype_alignment = self._alignment(packs)
            # Align in per-block units so probing one block gives exactly the
            # same cost as planning the final number of blocks.
            alignment = self.config.num_blocks * dtype_alignment
            capacity = (capacity + alignment - 1) // alignment * alignment
            buffer_size = self.config.num_blocks * stride
            buffers, previous, slot = {}, {}, 0
            slot_count = settings.num_shared_buffers
            for pack in packs:
                # 3. Rotate complete packs, growing the ring only when every
                # slot conflicts. Resident packs have private regions.
                group_layers = {(self.group_ids[name], indices[name]) for name in pack}
                size = max(regions[name][1] for name in pack)
                selected_slot = None
                if size <= buffer_size and all(name in members for name in pack):
                    for distance in range(slot_count):
                        candidate_slot = (slot + distance) % slot_count
                        if all(
                            layer > previous.get((candidate_slot, group), (-1, ()))[0] for group, layer in group_layers
                        ):
                            selected_slot = candidate_slot
                            break
                    else:
                        selected_slot = slot_count
                        slot_count += 1
                if selected_slot is not None and selected_slot in buffers:
                    offset = buffers[selected_slot]
                else:
                    offset = (cursor + alignment - 1) // alignment * alignment
                    cursor = offset + (buffer_size if selected_slot is not None else size)
                    if selected_slot is not None:
                        buffers[selected_slot] = offset
                group_names = defaultdict(set)
                for name in pack:
                    block_stride = tensors[views[name][0]].block_stride
                    cache_offset = offset + regions[name][0]
                    key = (cache_offset, block_stride)
                    if key not in descriptors:
                        descriptors[key] = KVCacheTensor(
                            size=0,
                            layers=[],
                            layer_stride=0,
                            block_stride=block_stride,
                            offset=capacity + cache_offset,
                        )
                    descriptors[key].layers.append(name)
                    group_names[self.group_ids[name]].add(name)
                if selected_slot is not None:
                    for group, layer in group_layers:
                        key = (selected_slot, group)
                        if key in previous:
                            source, names = previous[key]
                            dependencies[layer].add(source)
                            reused_names.update(names | group_names[group])
                        previous[key] = (layer, group_names[group])
                    slot = (selected_slot + 1) % slot_count
            capacity += cursor
            candidate.extend(descriptors.values())
        for tensor in candidate:
            if not tensor.host_resident:
                tensor.size = capacity
        if not dependencies or (self.native_backing_bytes is not None and capacity >= self.native_backing_bytes):
            candidate = tensors
            dependencies.clear()
            reused_names.clear()
        return LayerwiseReuseLayout(
            layers=tuple(layers),
            cache_layer_indices=indices,
            kv_cache_tensors=tuple(candidate),
            cache_view_sizes={name: end - start for name, (_, start, end) in views.items()},
            prefetch_layer_map={layer: tuple(sorted(sources)) for layer, sources in dependencies.items()},
            independent_layers=independent,
            num_prefetch_layers=settings.num_prefetch_layers,
            reused_cache_names=frozenset(reused_names),
        )

    def apply(self) -> None:
        layout = self.build()
        for name in layout.reused_cache_names:
            spec = self.specs[name]
            if isinstance(spec, MambaSpec) and (
                spec.mamba_cache_mode != "align" or getattr(spec, "num_prefill_checkpoint_blocks", 0) != 0
            ):
                raise NotImplementedError(
                    f"Reusable cache {name} has unsupported state semantics: {type(spec).__name__}"
                )
        if layout.has_layer_reuse:
            if getattr(getattr(self.vllm_config, "cache_config", None), "num_gpu_blocks_override", None) is not None:
                raise ValueError(
                    "Layerwise offload does not support num_gpu_blocks_override; use automatic block sizing."
                )
            if any(t.host_resident for t in self.config.kv_cache_tensors) or any(
                getattr(spec, "store_on_host", False) for spec in self.specs.values()
            ):
                raise ValueError("Layerwise offload requires device-resident cache tensors.")
            parallel = getattr(self.vllm_config, "parallel_config", None)
            backend = get_backend_class(str(self.extra_config.get("backend", "mooncake")))
            # Mooncake's GLM topology already describes PCP-local native pages.
            # Do not extend that capability to Qwen/GVA layouts implicitly.
            pcp_supported = (
                getattr(self.config, "layerwise_cache_topology", None) is not None
                and backend is not None
                and backend.layerwise_data_plane == "block_key"
            )
            if getattr(parallel, "prefill_context_parallel_size", 1) > 1 and not pcp_supported:
                raise NotImplementedError("Layerwise offload does not support prefill context parallelism.")
            logger.info(
                "Layerwise reuse uses one %d-byte device backing; independent layers=%s.",
                next(tensor.size for tensor in layout.kv_cache_tensors if not tensor.host_resident),
                layout.independent_layers,
            )
            self.config.kv_cache_tensors = list(layout.kv_cache_tensors)
            self.config.layerwise_kv_cache_reuse = True
        self.config.layerwise_reuse_layout = layout
        # Store must validate reusable prefixes, and allocation must use exactly
        # the same single backing and byte windows as the physical planner.
        self.config.layerwise_component_plan = layout.has_layer_reuse
        self.config.layerwise_store_verified_prefix = layout.has_layer_reuse
        if layout.has_layer_reuse and self.draft_names:
            self.config.layerwise_draft_layer_names = tuple(sorted(self.draft_names))


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    *,
    excluded_layer_names: set[str] | None = None,
    native_backing_bytes: int | None = None,
) -> None:
    """Plan whole-pack reuse for models with registered rules."""
    layout = getattr(kv_cache_config, "layerwise_reuse_layout", None)
    if layout is not None:
        if excluded_layer_names and layout.reused_cache_names.intersection(excluded_layer_names):
            raise ValueError("Reuse exclusions must be supplied before the cache layout is planned.")
        return
    # An empty device pool has no byte views to reuse.
    if not kv_cache_config.kv_cache_tensors or kv_cache_config.num_blocks == 0 or not kv_cache_config.kv_cache_groups:
        return
    extra_config = get_layerwise_reuse_config(getattr(vllm_config, "kv_transfer_config", None))
    if extra_config is None:
        return
    rules = get_spec_reuse_rules(vllm_config)
    if rules is None:
        return
    planner = LayerwiseCachePlanner(
        kv_cache_config,
        vllm_config,
        extra_config,
        excluded_layer_names,
        spec_reuse_rules=rules,
        native_backing_bytes=native_backing_bytes,
    )
    parallel = vllm_config.parallel_config
    pp_size = getattr(parallel, "pipeline_parallel_size", 1)
    if pp_size > 1:
        complete_stage = any(
            planner.layers == tuple(range(*get_pp_indices(planner.total_layers, rank, pp_size)))
            for rank in range(pp_size)
        )
    else:
        complete_stage = len(planner.layers) >= vllm_config.model_config.get_num_layers(parallel)
    if not complete_stage:
        # Hook cursors require every local physical layer; a partial probe is
        # not permission to reuse buffers across omitted layers.
        planner.extra_config = {**extra_config, "layerwise_independent_layers": "all"}
    planner.apply()
