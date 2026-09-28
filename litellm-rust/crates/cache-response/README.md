# Response cache

`ResponseCache<B>` adds request keys, independent read/write controls, response envelopes, and freshness checks to any `B: BaseCache<Value = CacheEntry>`

## Ownership

`litellm-cache` defines typed storage, codec, and capability traits. `BaseCache` is only get, set, TTL, and pipeline writes. Everything else is an optional capability a backend implements only where its Python class defines the method: `DisconnectCache`, `ConnectionCache` (`test_connection`), `PingCache`, `BatchCache`, `DeleteCache`, `FlushCache`, counters, queues, TTL, scan, and scripts. Memory, Redis, disk, S3, GCS, and Azure Blob implement those traits without depending on response policy, so other consumers can store their own value types in the same backends

Semantic backends (Redis, Valkey, Qdrant) are generic over their embedder and codec, and share one prompt and embedding contract from `litellm_cache::semantic`. They take a `SemanticCacheContext`, so `ResponseCache` drives them the same way it drives exact backends

`litellm-cache-response` owns response keys, controls, entries, the Python-compatible response codec, and `WriteBuffer`, the backend-neutral deferred-write policy. It has no runtime dependency on a specific cache backend or Python

`ExactResponseCache` is the object-safe view of a `ResponseCache` over an exact backend. `ConnectionProbe` is the object-safe `test_connection`, implemented only when the backend implements `ConnectionCache`, so a host holds one next to its `ExactResponseCache` and reports the operation as unsupported otherwise, as Python's `BaseCache` does. Lookup, store, batch, and flush never require it

## Native Rust use

```rust
use std::{sync::Arc, time::Duration};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheKeyInput, ResponseCache, ResponseCacheRequest};
use serde_json::json;

let cache = ResponseCache::new(Arc::new(InMemoryCache::default()));
let request = ResponseCacheRequest::new(CacheKeyInput {
    preset: Some("example:key".into()),
    ..Default::default()
});
let now = Duration::from_secs(100);
cache.store(&request, json!({"answer": 7}), now)?;
assert_eq!(cache.async_lookup(&request, now).await?, Some(json!({"answer": 7})));
```

For Redis, inject `RedisCache::new(url, ttl, ResponseCacheCodec)` instead. Namespaces are optional and existing namespace prefixes are preserved

Callers supply Unix time for response freshness. Backend TTL uses its own clock. A read can reject an entry through `max_age` even while the backend still retains it

## Python integration

The assigned cache object selects its implementation. The legacy Python `Cache` constructor keeps its Python backend. `_v2.Cache` factories return that same Python facade with a Rust storage adapter. Cache selection does not use the rollout catalog and does not change inference dispatch

Python inference uses the facade's existing key generation, response reconstruction, streaming and callback logic. The adapter delegates storage to Rust. Native inference extracts the adapter's handle at the bridge boundary and passes it to the shared core cache orchestration. Both paths share the backend, but use separate keys and response formats

A legacy or custom Python cache without a native handle causes native inference admission to decline before the call starts, allowing the existing Python fallback to honor that cache. Calling arbitrary Python cache overrides from native core remains outside this v2 integration

Native cache handles must be recreated after fork. Storage failures retain the caller's existing fail-open policy

## Adding another backend

Implement `BaseCache` for the backend with its associated value type and the capability traits its Python class supports, and accept a `CacheCodec` when wire serialization is needed. `ResponseCache<B>` then works without another response implementation

Run the `litellm-cache-testing` contract checks the backend's capabilities allow, and run response fixtures with `ResponseCacheCodec`, including both Python envelope encodings, before exposing the backend through a cache factory

## Experimental inference cache

`InferenceCache` shares an exact response backend across Chat Completions, Messages, and HTTP Responses. Configure a route before calling either `machine` or `execute`:

```rust
use litellm_cache_response::{InferenceCache, InferenceCacheOptions, ResponseCache};

let cache = InferenceCache::new(
    Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
        Some(200),
        Some(Duration::from_secs(600)),
    )))),
    "my-service".into(),
    4 * 1024 * 1024,
);
let machine = route
    .with_cache(cache, InferenceCacheOptions::default())
    .machine(request);
```

The gateway accepts the same handle through `Gateway::new(...)?.with_cache(cache)`. Its handlers authorize each call before lookup and partition entries by the authenticated caller. Request keys also include the API surface, canonical request parameters, provider, explicit credentials, base URL, and headers. Use separate namespaces for separate applications and trust boundaries, especially when credentials come from environment or secret-source defaults

Core stores normalized provider results before host response transformations. Cache hits still run host response processing and success callbacks. The Python bridge reports `cache_hit=True` and zero provider cost. Errors from cache reads or writes leave inference available

Streaming entries are written only after successful exhaustion with the API's terminal event. Errors, incomplete streams, cancellation, and oversized entries are not stored. Replay preserves SSE bytes, not transport chunk boundaries or upstream headers. Existing streaming support is unchanged: native Chat Completions and Python Responses currently accept non-streaming calls only

Python callers opt in explicitly:

```python
import litellm
from litellm import _v2

litellm.cache = _v2.Cache.memory(ttl=600, capacity=200)
response = await litellm.acompletion(
    model=model,
    messages=messages,
)
```

`_v2` exposes only `Cache`. Assign it to the existing `litellm.cache` global, then call the existing completion, Messages, or Responses API. The inference rollout independently selects Python or Rust execution; choosing v2 storage does not force a native route. Redis uses `litellm.cache = _v2.Cache.redis(url, namespace="my-service", ttl=600)`

Cache hits carry the existing `x-litellm-cache-key` header, including streamed responses before the first chunk. Misses omit it. Python preserves the key in response metadata for the existing proxy header path, while the native gateway emits it directly

Per-call controls are uniform across these surfaces: `caching=False` disables reads and writes, `cache={"no-cache": True}` skips reads, and `cache={"no-store": True}` skips writes. `cache={"ttl": seconds}` sets write expiry and `cache={"s-max-age": seconds}` limits read freshness; `s-maxage` remains an alias. Durations must be positive and finite

Legacy constructors and custom cache subclasses keep their existing behavior. New native backends can be injected behind `ExactResponseCache`; neither core orchestration nor the `_v2.Cache` interface needs to change
