import asyncio
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, Literal

import pytest
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm import _v2
from litellm._v2.cache import NativeBackend
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.messages.dispatch import MessagesResult
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION, LiteLLMChatCompletionsRequest
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import call_hook
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, LiteLLMMessagesRequest
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES, LiteLLMResponsesRequest
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.test_inference import RESPONSES_MODEL, RESPONSES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


def payload(value: object) -> object:
    if isinstance(value, ModelResponse):
        return value.model_dump_json(exclude=MappingProxyType({"id": True, "created": True}))
    if isinstance(value, dict):
        fields: Final = TypeAdapter(dict[str, object]).validate_python(value)
        return {name: field for name, field in fields.items() if name != "_hidden_params"}
    return value.model_dump_json() if isinstance(value, BaseModel) else value


def cache_key(response: object) -> object:
    hidden: Final = get_hidden_params_dict(response)
    headers: Final = TypeAdapter(dict[str, object]).validate_python(hidden.get("additional_headers", {}))
    return headers.get("x-litellm-cache-key")


async def invoke(
    route: Literal["chat", "messages", "responses"],
    server: RecordingServer,
    options: Mapping[str, object],
) -> object:
    common: Final = {"api_key": "test-key", "api_base": server.base_url, **options}
    if route == "responses":
        server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        arguments: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        request: Final = LiteLLMResponsesRequest(
            RESPONSES_MODEL, "hello", None, "test-key", server.base_url, "openai", None, arguments
        )
        return await runtime.arun(
            RouteContext(Route.RESPONSES),
            binding=NATIVE_ARESPONSES,
            native=lambda hook: call_hook(hook, request, (), arguments),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),),
        )
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    parameters: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
    if route == "chat":
        chat: Final = LiteLLMChatCompletionsRequest(
            MESSAGES_MODEL, list(MESSAGES), None, "test-key", server.base_url, None, None, parameters
        )
        return await runtime.arun(
            RouteContext(Route.CHAT_COMPLETIONS),
            binding=NATIVE_ACOMPLETION,
            native=lambda hook: call_hook(hook, chat, (), parameters),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),),
        )
    messages: Final = LiteLLMMessagesRequest(
        MESSAGES_MODEL, list(MESSAGES), 32, None, "test-key", server.base_url, "anthropic", parameters
    )
    return await runtime.arun(
        RouteContext(Route.MESSAGES),
        binding=NATIVE_AMESSAGES,
        native=lambda hook: call_hook(hook, messages, (), parameters),
        python=runtime.NO_PYTHON,
        rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "messages", "responses"))
async def test_v2_cache_skips_provider_and_reports_one_success_per_call(
    recording_server: RecordingServer, route: Literal["chat", "messages", "responses"]
) -> None:
    litellm.cache = _v2.Cache.memory()
    recorder: Final = RecordingLogger()
    first: Final = await invoke(route, recording_server, {"callbacks": [recorder]})
    second: Final = await invoke(route, recording_server, {"callbacks": [recorder]})
    assert payload(first) == payload(second)
    assert cache_key(first) is None
    assert isinstance(cache_key(second), str)
    assert cache_key(second) == get_hidden_params_dict(second)["cache_key"]
    assert len(recording_server.requests) == 1
    await drain_logging()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    assert len(successes) == 2
    cached_log: Final = TypeAdapter(dict[str, object]).validate_python(successes[-1].kwargs)
    assert cached_log["cache_hit"] is True
    assert cached_log["response_cost"] == 0


@pytest.mark.asyncio
async def test_v2_global_cache_leaves_legacy_only_calls_usable() -> None:
    litellm.cache = _v2.Cache.memory()
    response: Final = await litellm.aembedding(
        model="openai/cache-test-embedding",
        input=["hello"],
        api_key="test-key",
        mock_response=[0.25, 0.75],
    )
    assert response.model_dump(include={"data"}) == {
        "data": [{"embedding": [0.25, 0.75], "index": 0, "object": "embedding"}]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "messages", "responses"))
async def test_v2_cache_controls_and_credentials_isolate_requests(
    recording_server: RecordingServer, route: Literal["chat", "messages", "responses"]
) -> None:
    recording_server.expected_requests = 4
    litellm.cache = _v2.Cache.memory()
    await invoke(route, recording_server, {"cache": {"no-store": True}})
    await invoke(route, recording_server, {})
    await invoke(route, recording_server, {})
    assert len(recording_server.requests) == 2
    await invoke(route, recording_server, {"cache": {"no-cache": True}})
    await invoke(route, recording_server, {"api_key": "another-key"})
    assert len(recording_server.requests) == 4


async def collect(stream: MessagesResult) -> bytes:
    assert isinstance(stream, AsyncIterator)
    return b"".join([chunk_bytes(chunk) async for chunk in stream])


def chunk_bytes(value: object) -> bytes:
    assert isinstance(value, bytes)
    return value


@pytest.mark.asyncio
async def test_v2_messages_replays_a_completed_stream(recording_server: RecordingServer) -> None:
    recording_server.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
    litellm.cache = _v2.Cache.memory()
    recorder: Final = RecordingLogger()
    parameters: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
        "stream": True,
        "callbacks": [recorder],
    }
    first_stream: Final = await litellm.anthropic_messages(**parameters)
    assert cache_key(first_stream) is None
    first: Final = await collect(first_stream)
    second_stream: Final = await litellm.anthropic_messages(**parameters)
    assert isinstance(cache_key(second_stream), str)
    assert cache_key(second_stream) == get_hidden_params_dict(second_stream)["cache_key"]
    second: Final = await collect(second_stream)
    assert payload(first) == payload(second)
    assert first == b"".join(recording_server.default_response.payloads())
    assert len(recording_server.requests) == 1
    await drain_logging()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    cached_log: Final = TypeAdapter(dict[str, object]).validate_python(successes[-1].kwargs)
    assert cached_log["cache_hit"] is True
    assert cached_log["response_cost"] == 0


@pytest.mark.parametrize("route", ("chat", "responses"))
def test_v2_cache_works_through_python_inference(
    recording_server: RecordingServer, route: Literal["chat", "messages", "responses"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    litellm.cache = _v2.Cache.memory()
    common: Final = {"api_key": "test-key", "api_base": recording_server.base_url}
    if route == "responses":
        recording_server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        parameters: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        first: Final = litellm.responses(**parameters)
        second: Final = litellm.responses(**parameters)
        assert payload(first) == payload(second)
    else:
        recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
        arguments: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
        initial: Final = litellm.completion(**arguments)
        cached: Final = litellm.completion(**arguments)
        assert isinstance(initial, ModelResponse) and isinstance(cached, ModelResponse)
        assert (
            initial.choices[0].message.content
            == cached.choices[0].message.content
            == MESSAGES_RESPONSE["content"][0]["text"]
        )
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
async def test_v2_facade_and_backend_share_storage_and_management() -> None:
    cache: Final = _v2.Cache.memory()
    await cache.async_add_cache({"answer": 7}, cache_key="shared")
    assert cache.get_cache(cache_key="shared") == {"answer": 7}
    assert await cache.ping() is True
    await cache.delete_cache_keys(["shared"])
    assert await cache.async_get_cache(cache_key="shared") is None
    cache.add_cache({"answer": 8}, cache_key="flush")
    backend: Final = cache.cache
    assert isinstance(backend, NativeBackend)
    backend.flush_cache()
    assert cache.get_cache(cache_key="flush") is None
    await cache.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ("s-maxage", "s-max-age"))
async def test_v2_native_cache_accepts_existing_freshness_aliases(
    recording_server: RecordingServer, control: str
) -> None:
    litellm.cache = _v2.Cache.memory()
    first: Final = await invoke("responses", recording_server, {})
    second: Final = await invoke("responses", recording_server, {"cache": {control: 600}})
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
async def test_v2_cache_does_not_force_native_responses_streaming(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    litellm.cache = _v2.Cache.memory()
    recording_server.default_response = ResponseSpec(
        body=None,
        events=(
            ("response.created", {"type": "response.created", "sequence_number": 0, "response": RESPONSES_RESPONSE}),
            (
                "response.completed",
                {"type": "response.completed", "sequence_number": 1, "response": RESPONSES_RESPONSE},
            ),
        ),
    )
    response: Final = await litellm.aresponses(
        model=RESPONSES_MODEL,
        input="hello",
        stream=True,
        caching=False,
        api_key="test-key",
        api_base=recording_server.base_url,
    )
    assert isinstance(response, AsyncIterator)
    chunks: Final = [chunk async for chunk in response]
    assert chunks[-1].type == "response.completed"
    assert chunks[-1].response.output[0].content[0].text == "native response"


@pytest.mark.asyncio
async def test_v2_cache_works_through_python_messages(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    litellm.cache = _v2.Cache.memory()
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    parameters: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
    }
    first: Final = await litellm.anthropic_messages(**parameters)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    second: Final = await litellm.anthropic_messages(**parameters)
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
async def test_python_cache_interface_can_delete_a_native_inference_entry(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 2
    cache: Final = _v2.Cache.memory()
    litellm.cache = cache
    await invoke("responses", recording_server, {})
    cached: Final = await invoke("responses", recording_server, {})
    key: Final = cache_key(cached)
    assert isinstance(key, str)
    await cache.delete_cache_keys([key])
    fresh: Final = await invoke("responses", recording_server, {})
    assert cache_key(fresh) is None
    assert len(recording_server.requests) == 2


@pytest.mark.asyncio
async def test_rust_messages_fallback_honors_a_legacy_cache(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setenv("LITELLM_RUST", "1")
    litellm.cache = Cache()
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    parameters: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
    }
    first: Final = await litellm.anthropic_messages(**parameters)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    second: Final = await litellm.anthropic_messages(**parameters)
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1
