<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai/models/ Guidelines

## API Design

- Mirror patterns across provider implementations — ensures consistent APIs and prevents fragmentation — Users expect all providers to behave similarly; consistent patterns reduce learning curve and make providers interchangeable
- Expose provider-specific metadata via `provider_details` fields, always setting `provider_name` — Keeps the core API provider-agnostic while surfacing valuable metadata like logprobs, safety ratings, and token counts in a consistent, discoverable way
- Use `snake_case` versions of provider's official API parameter names in config classes — Ensures provider parameters match official documentation, making them discoverable and reducing confusion when users reference vendor docs

## General

- Silently ignore unsupported `ModelSettings` when docs already state compatibility — prevents warning noise — Users already know from documentation which models support which settings; runtime warnings are redundant and clutter logs unnecessarily
- Keep provider-specific code in provider modules (e.g., `models/openai.py`) not shared utils — Prevents coupling between providers and maintains clear module boundaries, making code easier to maintain and extend
- Prefix provider-specific config fields with provider name (e.g., `anthropic_cache_tools`, `openrouter_engine`) — Distinguishes provider-specific settings from generic cross-provider options, preventing configuration ambiguity

## Unified Thinking Settings

When adding or modifying a model that supports thinking/reasoning:

1. **Profile** (`profiles/{provider}.py`): Set `supports_thinking=True`. If thinking can't be disabled (always-on models like o-series, DeepSeek R1), also set `thinking_always_enabled=True`.
2. **Resolver**: Add a `_resolve_{provider}_thinking()` method that calls `resolve_thinking_config(settings, self.profile)` and translates to the provider's native format. See existing providers for the pattern.
3. **OpenAIChatModel subclasses**: If inheriting from `OpenAIChatModel`, strip `openai_reasoning_effort` in `prepare_request()` after calling `super()` — the parent injects it, but your provider has its own mechanism.
4. **Tests** (`tests/test_unified_thinking.py`): Add tests for `thinking=True`, `thinking=False`, each effort level, silent drop on unsupported models, and provider-specific precedence.

<!-- /braindump -->
