use std::convert::Infallible;

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_types::utils::ChatCompletionsResponse;

use super::{
    ChatCompletionsRoute, Error,
    types::{ChatCompletionsCall, ChatCompletionsRequest},
};

pub struct ChatCompletions;

impl Protocol for ChatCompletions {
    type Response = ChatCompletionsResponse;
    type Error = Error;
    type Request = ChatCompletionsCall;
    type HostCall = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

impl ChatCompletionsRoute {
    pub fn machine(self, call: ChatCompletionsCall) -> HostedMachine<ChatCompletions> {
        hosted_call(call, move |call, _, hooks| async move {
            self.run_call(call, &hooks).await.map(CallOutput::Complete)
        })
    }

    pub(super) async fn run_call(
        &self,
        call: ChatCompletionsCall,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let output = crate::caching::execute::<ChatCompletions, _, _>(
            call,
            self.cache.clone(),
            hooks,
            |call| async move {
                let request = ChatCompletionsRequest {
                    model: &call.model,
                    messages: call.messages,
                    optional_params: call.optional_params,
                    api_key: call.api_key.as_deref(),
                    api_base: call.api_base.as_deref(),
                    custom_llm_provider: call.custom_llm_provider.as_deref(),
                    extra_headers: call.extra_headers,
                    timeout: call.timeout,
                };
                self.run(request, hooks).await.map(CallOutput::Complete)
            },
        )
        .await?;
        match output {
            CallOutput::Complete(response) => Ok(response),
            CallOutput::Stream { head, .. } => match head {},
        }
    }
}

impl crate::caching::CacheFormat for ChatCompletions {
    const SURFACE: &'static str = "chat_completions";
    const TERMINAL_EVENT: &'static str = "";

    fn cache_input(request: &Self::Request) -> Result<serde_json::Value, crate::RouteError> {
        Ok(serde_json::json!({
            "model": request.model,
            "messages": request.messages,
            "params": request.optional_params,
            "provider": request.custom_llm_provider,
            "api_key": request.api_key,
            "api_base": request.api_base,
            "headers": request.extra_headers
        }))
    }

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        let _ = data;
        None
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        match *chunk {}
    }
}
