use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedCompletion, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;

use super::{Error, MessagesCall};

pub type MessagesOutput = HostedCompletion<Box<AnthropicMessagesResponse>>;

/// The upstream response as the caller sees it at stream hand-off, before any chunk.
pub struct MessagesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub struct Messages;

impl Protocol for Messages {
    type Response = Box<AnthropicMessagesResponse>;
    type Error = Error;
    type Request = MessagesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub type MessagesMachine = HostedMachine<Messages>;

impl super::MessagesRoute {
    pub fn machine(self, call: MessagesCall) -> HostedMachine<Messages> {
        hosted_call(call, move |call, _, hooks| async move {
            self.run(call, &hooks).await
        })
    }
}

impl crate::caching::CacheFormat for Messages {
    const SURFACE: &'static str = "messages";
    const TERMINAL_EVENT: &'static str = "message_stop";

    fn cache_input(request: &Self::Request) -> Result<serde_json::Value, crate::RouteError> {
        Ok(serde_json::json!({
            "body": request.body,
            "provider": request.custom_llm_provider,
            "api_key": request.api_key,
            "api_base": request.api_base,
            "headers": request.extra_headers,
            "provider_headers": request.provider_specific_header,
            "shaping": request.shaping
        }))
    }

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        Some(litellm_host::call::CallOutput::Stream {
            head: MessagesStreamHead {
                headers: Vec::new(),
            },
            chunks: Box::pin(futures_util::stream::iter([Ok(data)])),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }
}
