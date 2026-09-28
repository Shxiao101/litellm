from litellm.litellm_core_utils.cost_discount import (
    parse_cost_discount_key,
    resolve_cost_discount,
)


def test_parse_cost_discount_key_bare_provider():
    parsed = parse_cost_discount_key("vertex_ai")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern is None


def test_parse_cost_discount_key_splits_on_first_slash():
    parsed = parse_cost_discount_key("vertex_ai/claude-*")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern == "claude-*"


def test_parse_cost_discount_key_pattern_containing_slash_stays_in_pattern():
    parsed = parse_cost_discount_key("vertex_ai/a/b")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern == "a/b"


def test_parse_cost_discount_key_empty_pattern_after_slash():
    parsed = parse_cost_discount_key("vertex_ai/")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern == ""


def test_resolve_cost_discount_exact_key_beats_glob_and_bare():
    config = {"vertex_ai/claude-sonnet-4-5": 0.3, "vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.3


def test_resolve_cost_discount_glob_beats_bare_provider():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.2


def test_resolve_cost_discount_longest_literal_glob_wins():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai/claude-sonnet-*": 0.25, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.25


def test_resolve_cost_discount_strips_provider_prefix_from_model():
    config = {"vertex_ai/claude-*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "vertex_ai/claude-sonnet-4-5") == 0.2


def test_resolve_cost_discount_non_matching_model_falls_back_to_bare():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-3-pro-preview") == 0.05


def test_resolve_cost_discount_non_matching_model_no_bare_returns_none():
    config = {"vertex_ai/claude-*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-3-pro-preview") is None


def test_resolve_cost_discount_other_provider_untouched():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "openai", "claude-sonnet-4-5") is None


def test_resolve_cost_discount_none_provider_returns_none():
    config = {"vertex_ai": 0.05}
    assert resolve_cost_discount(config, None, "claude-sonnet-4-5") is None


def test_resolve_cost_discount_empty_provider_returns_none():
    config = {"vertex_ai": 0.05}
    assert resolve_cost_discount(config, "", "claude-sonnet-4-5") is None


def test_resolve_cost_discount_none_model_only_bare_matches():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", None) == 0.05
    assert resolve_cost_discount({"vertex_ai/claude-*": 0.2}, "vertex_ai", None) is None


def test_resolve_cost_discount_char_class_counts_as_wildcard():
    config = {"vertex_ai/claude-sonnet-*": 0.3, "vertex_ai/claude-[abcdefghijklmnopqrstuvwxyz]*": 0.1}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.3


def test_resolve_cost_discount_unclosed_bracket_matches_literally():
    config = {"vertex_ai/weird[": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "weird[") == 0.2
    assert resolve_cost_discount(config, "vertex_ai", "weirdx") is None


def test_resolve_cost_discount_char_class_with_leading_bracket():
    config = {"vertex_ai/a[]x]*": 0.1, "vertex_ai/a]*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "a]q") == 0.2
    assert resolve_cost_discount(config, "vertex_ai", "axq") == 0.1


def test_resolve_cost_discount_glob_crosses_slash_in_model():
    config = {"bedrock/*anthropic.claude-*": 0.15, "bedrock": 0.05}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-east-1/anthropic.claude-v2:1") == 0.15


def test_resolve_cost_discount_exact_pattern_key_does_not_count_as_exact():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "vertex_ai") == 0.05
