use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{InferenceCache, InferenceCacheOptions, ResponseCache};
use litellm_core::{
    RouteError,
    caching::{CacheFormat, execute},
};
use litellm_host::{
    call::{CallOutput, OutputOf},
    protocol::Protocol,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

struct TestRoute;

impl Protocol for TestRoute {
    type Request = Value;
    type Response = Value;
    type Error = RouteError;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ();
}

impl CacheFormat for TestRoute {
    const SURFACE: &'static str = "test";
    const TERMINAL_EVENT: &'static str = "message_stop";
    fn cache_input(request: &Value) -> Result<Value, RouteError> {
        Ok(request.clone())
    }
    fn replay(data: Bytes) -> Option<OutputOf<Self>> {
        Some(CallOutput::Stream {
            head: (),
            chunks: stream::iter([Ok(data)]).boxed(),
        })
    }
    fn bytes(chunk: &Bytes) -> &[u8] {
        chunk
    }
}

#[fixture]
fn cache() -> InferenceCache {
    cache_with_limit(4096)
}

fn cache_with_limit(max_entry_bytes: usize) -> InferenceCache {
    InferenceCache::new(
        Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
            Some(100),
            Some(Duration::from_secs(60)),
        )))),
        "test".into(),
        max_entry_bytes,
    )
}

async fn call(
    cache: &InferenceCache,
    options: InferenceCacheOptions,
    calls: &AtomicUsize,
    request: Value,
) -> Value {
    let output =
        execute::<TestRoute, _, _>(request, Some((cache.clone(), options)), &(), |_| async {
            Ok(CallOutput::Complete(
                json!({"call": calls.fetch_add(1, Ordering::SeqCst)}),
            ))
        })
        .await
        .unwrap();
    let CallOutput::Complete(response) = output else {
        panic!("expected a response");
    };
    response
}

#[rstest]
#[case::normal(InferenceCacheOptions::default(), true, true)]
#[case::no_cache(InferenceCacheOptions { no_cache: true, ..Default::default() }, false, true)]
#[case::no_store(InferenceCacheOptions { no_store: true, ..Default::default() }, true, false)]
#[case::disabled(InferenceCacheOptions { caching: Some(false), ..Default::default() }, false, false)]
#[tokio::test]
async fn cache_controls_apply_to_both_reads_and_writes(
    cache: InferenceCache,
    #[case] options: InferenceCacheOptions,
    #[case] reads: bool,
    #[case] writes: bool,
) {
    let calls = AtomicUsize::new(0);
    let first = call(&cache, options.clone(), &calls, json!({"model":"test"})).await;
    let second = call(&cache, Default::default(), &calls, json!({"model":"test"})).await;
    assert_eq!(first == second, writes);
    let third = call(&cache, options, &calls, json!({"model":"test"})).await;
    assert_eq!(second == third, reads);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1 + usize::from(!writes) + usize::from(!reads)
    );
}

#[rstest]
#[tokio::test]
async fn request_identity_is_canonical_and_scoped(cache: InferenceCache) {
    let calls = AtomicUsize::new(0);
    let first = call(
        &cache,
        Default::default(),
        &calls,
        json!({"model":"m", "input":{"a":1,"b":2}}),
    )
    .await;
    let second = call(
        &cache,
        Default::default(),
        &calls,
        json!({"input":{"b":2,"a":1}, "model":"m"}),
    )
    .await;
    assert_eq!(first, second);
    let other = call(
        &cache,
        InferenceCacheOptions {
            scope: "other-tenant".into(),
            ..Default::default()
        },
        &calls,
        json!({"model":"m", "input":{"a":1,"b":2}}),
    )
    .await;
    assert_ne!(first, other);
    let changed = call(
        &cache,
        Default::default(),
        &calls,
        json!({"model":"m", "input":{"a":2,"b":2}}),
    )
    .await;
    assert_ne!(first, changed);
}

async fn streamed(
    cache: &InferenceCache,
    calls: &AtomicUsize,
    text: &str,
    fail: bool,
) -> OutputOf<TestRoute> {
    execute::<TestRoute, _, _>(
        json!({"stream":true}),
        Some((cache.clone(), Default::default())),
        &(),
        |_| async {
            calls.fetch_add(1, Ordering::SeqCst);
            let chunks = text
                .as_bytes()
                .chunks(3)
                .map(|bytes| Ok(Bytes::copy_from_slice(bytes)))
                .collect::<Vec<_>>();
            let ending = fail.then_some(Err(RouteError::Unsupported("test transport failure")));
            Ok(CallOutput::Stream {
                head: (),
                chunks: stream::iter(chunks.into_iter().chain(ending)).boxed(),
            })
        },
    )
    .await
    .unwrap()
}

async fn consume(output: OutputOf<TestRoute>) -> Result<Vec<u8>, RouteError> {
    let CallOutput::Stream { chunks, .. } = output else {
        panic!("expected a stream");
    };
    chunks
        .try_fold(Vec::new(), |mut bytes, chunk| async move {
            bytes.extend_from_slice(&chunk);
            Ok(bytes)
        })
        .await
}

#[rstest]
#[case::complete("data: {\"type\":\"message_stop\"}\n\n", false, true)]
#[case::truncated("data: {\"type\":\"content_block_delta\"}\n\n", false, false)]
#[case::error_then_stop(
    "data: {\"type\":\"error\"}\n\ndata: {\"type\":\"message_stop\"}\n\n",
    false,
    false
)]
#[case::trailing_incomplete("data: {\"type\":\"message_stop\"}\n\ndata: {", false, false)]
#[case::transport_failure("data: {\"type\":\"message_stop\"}\n\n", true, false)]
#[tokio::test]
async fn stream_replay_requires_successful_exhaustion(
    cache: InferenceCache,
    #[case] text: &str,
    #[case] fail: bool,
    #[case] cached: bool,
) {
    let calls = AtomicUsize::new(0);
    let first = consume(streamed(&cache, &calls, text, fail).await).await;
    assert_eq!(first.is_err(), fail);
    let second = consume(streamed(&cache, &calls, text, fail).await).await;
    assert_eq!(second.is_err(), fail);
    if !fail {
        assert_eq!(first.unwrap(), second.unwrap());
    }
    assert_eq!(calls.load(Ordering::SeqCst), if cached { 1 } else { 2 });
}

#[rstest]
#[tokio::test]
async fn abandoning_a_partially_consumed_stream_does_not_store(cache: InferenceCache) {
    let calls = AtomicUsize::new(0);
    let text = "data: {\"type\":\"message_stop\"}\n\n";
    let CallOutput::Stream { mut chunks, .. } = streamed(&cache, &calls, text, false).await else {
        panic!();
    };
    assert!(chunks.next().await.unwrap().is_ok());
    drop(chunks);
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn oversized_streams_are_delivered_without_being_stored() {
    let cache = cache_with_limit(8);
    let calls = AtomicUsize::new(0);
    let text = "data: {\"type\":\"message_stop\"}\n\n";
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(
        consume(streamed(&cache, &calls, text, false).await)
            .await
            .unwrap(),
        text.as_bytes()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn a_provider_failure_never_populates_the_cache(cache: InferenceCache) {
    let calls = AtomicUsize::new(0);
    let first = execute::<TestRoute, _, _>(
        json!({}),
        Some((cache.clone(), Default::default())),
        &(),
        |_| async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(RouteError::Unsupported("test provider failure"))
        },
    )
    .await;
    assert!(first.is_err());
    let successful = call(&cache, Default::default(), &calls, json!({})).await;
    let replayed = call(&cache, Default::default(), &calls, json!({})).await;
    assert_eq!(successful, replayed);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn an_invalid_cached_envelope_is_replaced_by_a_provider_result(cache: InferenceCache) {
    let request = json!({"input":"hello"});
    cache
        .session(TestRoute::SURFACE, request.clone(), Default::default())
        .store(json!({"unexpected":"old-format"}))
        .await
        .unwrap();
    let calls = AtomicUsize::new(0);
    let first = call(&cache, Default::default(), &calls, request.clone()).await;
    let second = call(&cache, Default::default(), &calls, request).await;
    assert_eq!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct UnavailableCache;

impl litellm_cache::BaseCache for UnavailableCache {
    type Value = litellm_cache_response::CacheEntry;
    type Context = litellm_cache::ExactCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        Some(Duration::from_secs(60))
    }

    fn get_cache(
        &self,
        _: &str,
        _: &Self::Context,
    ) -> Result<Option<Self::Value>, litellm_cache::Error> {
        Err(litellm_cache::Error::Unavailable)
    }

    fn set_cache(
        &self,
        _: &str,
        _: Self::Value,
        _: &Self::Context,
    ) -> Result<(), litellm_cache::Error> {
        Err(litellm_cache::Error::Unavailable)
    }
}

impl litellm_cache::BatchCache for UnavailableCache {}

impl litellm_cache::FlushCache for UnavailableCache {
    fn flush_cache(&self) -> Result<(), litellm_cache::Error> {
        Err(litellm_cache::Error::Unavailable)
    }
}

#[rstest]
#[tokio::test]
async fn backend_failures_do_not_fail_inference() {
    let cache = InferenceCache::new(
        Arc::new(ResponseCache::new(Arc::new(UnavailableCache))),
        "test".into(),
        4096,
    );
    let calls = AtomicUsize::new(0);
    let first = call(&cache, Default::default(), &calls, json!({})).await;
    let second = call(&cache, Default::default(), &calls, json!({})).await;
    assert_ne!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
