use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_types::responses::main::ResponsesApiResponse;

use super::{
    Error, ResponsesRoute,
    types::{ResponsesCall, ResponsesStreamHead},
};

pub struct Responses;

impl Protocol for Responses {
    type Response = ResponsesApiResponse;
    type Error = Error;
    type Request = ResponsesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ResponsesStreamHead;
}

impl ResponsesRoute {
    pub fn machine(self, call: ResponsesCall) -> HostedMachine<Responses> {
        hosted_call(call, move |call, _, hooks| async move {
            self.run(call, &hooks).await
        })
    }
}

impl crate::caching::CacheFormat for Responses {
    const SURFACE: &'static str = "responses";
    const TERMINAL_EVENT: &'static str = "response.completed";

    fn cache_input(request: &Self::Request) -> Result<serde_json::Value, crate::RouteError> {
        Ok(serde_json::json!({
            "model": request.model,
            "input": request.input,
            "params": request.optional_params,
            "provider": request.custom_llm_provider,
            "api_key": request.api_key,
            "api_base": request.api_base,
            "headers": request.extra_headers
        }))
    }

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        Some(litellm_host::call::CallOutput::Stream {
            head: ResponsesStreamHead {
                headers: Vec::new(),
            },
            chunks: Box::pin(futures_util::stream::iter([Ok(data)])),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }

    fn reusable(response: &serde_json::Value) -> bool {
        response.get("status").and_then(serde_json::Value::as_str) == Some("completed")
    }
}
