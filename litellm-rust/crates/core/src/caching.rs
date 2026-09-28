use std::future::Future;

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::{InferenceCache, InferenceCacheOptions, InferenceCacheSession};
use litellm_host::{
    call::{CallOutput, OutputOf},
    event::MachineEvent,
    hooks::RouteHooks,
    protocol::Protocol,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio_util::codec::Decoder;

use crate::RouteError;

pub trait CacheFormat: Protocol<Error = RouteError> {
    const SURFACE: &'static str;
    const TERMINAL_EVENT: &'static str;

    fn cache_input(request: &Self::Request) -> Result<Value, RouteError>;
    fn replay(data: Bytes) -> Option<OutputOf<Self>>;
    fn bytes(chunk: &Self::Chunk) -> &[u8];

    fn reusable(response: &Value) -> bool {
        let _ = response;
        true
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
enum CachedOutput<R> {
    Response(R),
    Stream(String),
}

pub async fn execute<P, F, Fut>(
    request: P::Request,
    cache: Option<(InferenceCache, InferenceCacheOptions)>,
    hooks: &impl RouteHooks<RouteError>,
    provider: F,
) -> Result<OutputOf<P>, RouteError>
where
    P: CacheFormat,
    P::Response: Serialize + DeserializeOwned,
    F: FnOnce(P::Request) -> Fut,
    Fut: Future<Output = Result<OutputOf<P>, RouteError>>,
{
    let Some((cache, options)) = cache.filter(|(_, options)| {
        options.caching != Some(false) && !(options.no_cache && options.no_store)
    }) else {
        return provider(request).await;
    };
    let session = cache.session(P::SURFACE, P::cache_input(&request)?, options);
    match session.lookup().await {
        Ok(Some(value)) => {
            if let Ok(hit) = serde_json::from_value::<CachedOutput<P::Response>>(value) {
                let output = match hit {
                    CachedOutput::Response(response) => Some(CallOutput::Complete(response)),
                    CachedOutput::Stream(data) => P::replay(Bytes::from(data)),
                };
                if let Some(output) = output {
                    hooks
                        .on_event(MachineEvent::CacheHit { key: session.key() })
                        .await?;
                    return Ok(output);
                }
            }
        }
        Ok(None) => {}
        Err(_) => tracing::warn!("response cache lookup failed"),
    }
    let output = provider(request).await?;
    if !session.writes() {
        return Ok(output);
    }
    match output {
        CallOutput::Complete(response) => {
            if let Ok(value) = serde_json::to_value(&response)
                && P::reusable(&value)
                && let Ok(entry) = serde_json::to_value(CachedOutput::Response(value))
            {
                store(&session, entry).await;
            }
            Ok(CallOutput::Complete(response))
        }
        CallOutput::Stream { head, chunks } => {
            let captured = stream::try_unfold(
                (chunks, Some(Vec::<u8>::new()), session),
                |(mut chunks, captured, session)| async move {
                    match chunks.try_next().await? {
                        Some(chunk) => {
                            let captured = captured.and_then(|mut data| {
                                let bytes = P::bytes(&chunk);
                                if data.len().saturating_add(bytes.len())
                                    > session.max_entry_bytes()
                                {
                                    return None;
                                }
                                data.extend_from_slice(bytes);
                                Some(data)
                            });
                            Ok(Some((chunk, (chunks, captured, session))))
                        }
                        None => {
                            if let Some(data) = captured
                                && let Ok(text) = String::from_utf8(data)
                                && successful_stream(&text, P::TERMINAL_EVENT)
                                && let Ok(entry) =
                                    serde_json::to_value(CachedOutput::<Value>::Stream(text))
                            {
                                store(&session, entry).await;
                            }
                            Ok::<_, RouteError>(None)
                        }
                    }
                },
            )
            .boxed();
            Ok(CallOutput::Stream {
                head,
                chunks: captured,
            })
        }
    }
}

async fn store(session: &InferenceCacheSession, entry: Value) {
    if session.store(entry).await.is_err() {
        tracing::warn!("response cache write failed");
    }
}

fn successful_stream(text: &str, terminal: &str) -> bool {
    let mut pending = BytesMut::from(text.as_bytes());
    let mut codec = litellm_framing::sse::SseCodec::default();
    let mut complete = false;
    loop {
        let event = match codec.decode(&mut pending) {
            Ok(Some(event)) => event,
            Ok(None) => return complete && pending.is_empty(),
            Err(_) => return false,
        };
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            return false;
        };
        let Some(kind) = value.get("type").and_then(Value::as_str) else {
            return false;
        };
        if matches!(kind, "error" | "response.failed" | "response.incomplete") {
            return false;
        }
        complete |= kind == terminal;
    }
}
