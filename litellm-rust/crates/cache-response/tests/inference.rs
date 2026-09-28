use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{InferenceCache, InferenceCacheOptions, ResponseCache};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[tokio::test]
async fn surfaces_namespaces_and_per_call_expiry_are_independent() {
    let clock = Arc::new(AtomicU64::new(0));
    let cache_clock = clock.clone();
    let backend = Arc::new(ResponseCache::new(Arc::new(InMemoryCache::with_clock(
        Some(100),
        Some(Duration::from_secs(60)),
        move || Duration::from_secs(cache_clock.load(Ordering::SeqCst)),
    ))));
    let cache = InferenceCache::new(backend.clone(), "first".into(), 4096);
    let request = json!({"model":"test", "input":"hello"});
    let messages = cache.session(
        "messages",
        request.clone(),
        InferenceCacheOptions {
            ttl: Some(Duration::from_secs(5)),
            ..Default::default()
        },
    );
    messages.store(json!({"answer":"cached"})).await.unwrap();
    assert_eq!(
        messages.lookup().await.unwrap(),
        Some(json!({"answer":"cached"}))
    );
    let responses = cache.session("responses", request.clone(), Default::default());
    assert_eq!(responses.lookup().await.unwrap(), None);
    let other = InferenceCache::new(backend, "second".into(), 4096).session(
        "messages",
        request,
        Default::default(),
    );
    assert_eq!(other.lookup().await.unwrap(), None);
    responses
        .store(json!({"answer":"longer-lived"}))
        .await
        .unwrap();
    clock.store(6, Ordering::SeqCst);
    assert_eq!(messages.lookup().await.unwrap(), None);
    assert_eq!(
        responses.lookup().await.unwrap(),
        Some(json!({"answer":"longer-lived"}))
    );
}
