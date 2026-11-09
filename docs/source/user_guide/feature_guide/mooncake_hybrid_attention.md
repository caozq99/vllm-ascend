# Mooncake Layerwise Hybrid Attention

## Feature Introduction

Mooncake layerwise hybrid attention extends KV cache pooling to models with
multiple cache groups. It saves and restores each group's cache layer by layer,
so that a request can reuse a prefix together with the window, compressed cache,
or recurrent state needed to continue computation.

This feature uses `AscendStoreConnector` with `backend="mooncake"` and
`use_layerwise=true`. Cache groups are detected from the model's runtime cache
configuration; no separate hybrid-pooling switch is required.

For the basic transfer mechanism, Mooncake preparation, and common parameters,
see [Mooncake Layerwise KV Cache Pooling](mooncake_layerwise_adaptation_and_optimization.md).

### Usage Scenarios

- Models combining full attention and sliding-window attention.
- Models with compressed KV, indexer caches, and compressor state, such as DeepSeek-V4.
- Models combining attention with linear or recurrent layers that use aligned state checkpoints.

### Constraints and Limitations

The common Mooncake dependency and deployment requirements are listed in the
[layerwise pooling guide](mooncake_layerwise_adaptation_and_optimization.md#constraints-and-limitations).

| Category | Constraint |
| --- | --- |
| Attention backend | Compute/transfer overlap is integrated with Ascend FA, SFA, and DSA paths. Other attention backends need their own integration. |
| Execution mode | Eager and graph-mode compatibility follows the selected model and attention backend. |
| Recurrent state | Recurrent and linear-attention groups, including GDN and KDA state, require `mamba_cache_mode="align"`. |
| Parallelism | Use matching TP, PP, PCP, and DCP sizes and the same PP layer partitions on sharing instances. PCP and DCP cannot be enabled together. TP mismatch is not supported. |
| PP | Uneven partitions and stage-local groups are supported only when every stage owns layers in every required group. Empty stage-local groups are not supported. |
| PCP | Replicated cache groups share Store objects and only one PCP rank saves them. PCP-sharded Mamba groups retain a distinct state shard per PCP rank. |
| KVPP | Hybrid layerwise pooling does not support KVPP. Mooncake layerwise PCP or DCP also cannot be combined with `enable_kvpp`. |
| Cache storage | Producers can opt into [prefill buffer reuse](mooncake_prefill_offload.md), including private chunk snapshots. Ordinary pooling publishes complete aligned snapshots. |
