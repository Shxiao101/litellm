mod support;

use std::{sync::Arc, time::Duration};

use axum::body::to_bytes;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{InferenceCache, ResponseCache};
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[rstest]
#[case::chat("/v1/chat/completions", "anthropic/test-model", false)]
#[case::messages("/v1/messages", "anthropic/test-model", false)]
#[case::responses("/v1/responses", "openai/test-model", false)]
#[case::messages_stream("/v1/messages", "anthropic/test-model", true)]
#[case::responses_stream("/v1/responses", "openai/test-model", true)]
#[tokio::test]
async fn all_inference_endpoints_share_native_cache(
    #[case] path: &str,
    #[case] model: &str,
    #[case] stream: bool,
    #[values("s-maxage", "s-max-age")] max_age: &str,
) {
    let upstream = MockServer::start().await;
    let is_responses = path.ends_with("responses");
    let provider_body = if is_responses {
        json!({"id":"response-1", "model":"test-model", "status":"completed", "output":[]})
    } else {
        json!({"id":"message-1", "model":"test-model", "type":"message", "role":"assistant", "content":[{"type":"text","text":"hello"}], "stop_reason":"end_turn", "usage":{"input_tokens":1,"output_tokens":1}})
    };
    let terminal = if is_responses {
        "response.completed"
    } else {
        "message_stop"
    };
    let events = format!("event: {terminal}\ndata: {{\"type\":\"{terminal}\"}}\n\n");
    let template = if stream {
        ResponseTemplate::new(200).set_body_raw(events.clone(), "text/event-stream")
    } else {
        ResponseTemplate::new(200).set_body_json(provider_body)
    };
    Mock::given(method("POST"))
        .respond_with(template)
        .expect(1)
        .mount(&upstream)
        .await;
    let cache = InferenceCache::new(
        Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
            Some(100),
            Some(Duration::from_secs(60)),
        )))),
        "gateway-test".into(),
        4096,
    );
    let app = support::app_with_cache(model, &upstream.uri(), cache);
    let request = if is_responses {
        json!({"model":"public/model", "input":"hello", "stream":stream, "cache":{(max_age):600}})
    } else {
        json!({"model":"public/model", "messages":[{"role":"user","content":"hello"}], "max_tokens":16, "stream":stream, "cache":{(max_age):600}})
    };
    let first = support::post(app.clone(), path, request.clone()).await;
    assert_eq!(first.status(), 200);
    assert!(!first.headers().contains_key("x-litellm-cache-key"));
    let first = to_bytes(first.into_body(), 4096).await.unwrap();
    let second = support::post(app, path, request).await;
    assert_eq!(second.status(), 200);
    let cache_key = second.headers().get("x-litellm-cache-key").unwrap();
    assert!(!cache_key.as_bytes().is_empty());
    let second = to_bytes(second.into_body(), 4096).await.unwrap();
    if stream {
        assert_eq!(first, events);
        assert_eq!(second, first);
    } else {
        assert_eq!(
            serde_json::from_slice::<Value>(&first).unwrap(),
            serde_json::from_slice::<Value>(&second).unwrap()
        );
    }
}
