# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BigTalk (`big-talk-ai` on PyPI) is a lightweight, middleware-first LLM/agent framework for Python (3.12+), inspired by
Starlette/FastAPI's middleware pipeline pattern. It provides a provider-agnostic abstraction over LLMs (Anthropic,
OpenAI) with tool calling, an agentic loop, and streaming.

The repo is an Nx monorepo (JS tooling orchestrating Python projects via `uv` and the `@dev-tales/nx-python` plugin)
with two Python projects:

- `libs/big-talk` — the `big-talk-ai` package itself.
- `example` — a FastAPI app demonstrating usage (depends on `dinkleberg` for DI).

Both are `uv` workspace members declared in the root `pyproject.toml`.

## Commands

Python deps are managed by `uv` across the workspace; JS deps (Nx, husky, commitlint, prettier) via `npm`.

```bash
npm install                 # installs JS tooling + sets up husky (npm run prepare)
npm run sync                 # uv sync --all-packages --all-groups --all-extras
```

Per-project targets are provided by the `@dev-tales/nx-python` Nx plugin (test/lint/build executors backed by
pytest/flake8/uv):

```bash
npx nx test big-talk         # run the big-talk-ai test suite (pytest)
npx nx lint big-talk         # flake8 (max-line-length = 120, see .flake8)
npx nx build big-talk        # build the package via uv

npx nx test example
npx nx run-many -t test      # run tests for every project
```

To run a single test file/case directly with pytest (from `libs/big-talk`, inside the uv-managed venv):

```bash
uv run pytest tests/test_bigtalk.py
uv run pytest tests/test_bigtalk.py::test_name -v
```

Formatting is handled by Prettier via lint-staged on commit (`*.*`: `prettier --write`), and commit messages are
linted by commitlint using `@commitlint/config-conventional` (enforced by the husky `commit-msg` hook) — so commits
must follow Conventional Commits (`feat:`, `fix:`, `chore:`, etc.).

## Architecture

### Middleware pipeline (the core abstraction)

Everything in BigTalk is built on one generic primitive, `MiddlewareStack[C, R]` (`libs/big-talk/src/big_talk/middleware/`):

- A `Middleware[C, R]` is `(next_handler, ctx, **kwargs) -> R`, i.e. an onion-style wrapper around a handler,
  matching the Starlette/FastAPI middleware shape. Plain async callables are auto-adapted into `Middleware` via
  `_CallableMiddlewareAdapter`.
- `.use(mw)` appends middleware; `.build()` composes them (in registration order, outermost first) around a
  `_base_handler`, producing a single callable handler.

`BigTalk` (`big_talk.py`) owns three independent stacks, each with its own context/handler types and its own base
handler that contains the actual default behavior:

| Stack            | Property               | Context                                          | Base handler does                                                                                                                                                         |
| ---------------- | ---------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Streaming        | `app.streaming`        | `StreamContext` (`stream.py`)                    | Runs the full outer agent loop: iterate stream-iteration → collect tool uses → execute tools → append `ToolMessage` → repeat until no more tool calls or `max_iterations` |
| Stream iteration | `app.stream_iteration` | `StreamIterationContext` (`stream_iteration.py`) | Resolves the LLM provider for `ctx.model` and delegates to `provider.stream(...)` for a single generation                                                                 |
| Tool execution   | `app.tool_execution`   | `ToolExecutionContext` (`tool_execution.py`)     | Maps tool names to `Tool` objects and builds one asyncio task per `ToolUse`, catching exceptions into `ToolResult(is_error=True)`                                         |

`BigTalk.send()` (non-streaming) runs an equivalent but simpler loop directly, without the stream-iteration stack.

Middleware is where cross-cutting behavior lives: logging, human-in-the-loop approval (block tool execution by
returning `[]` instead of calling `handler`), retries, request/response shaping, etc. New features that touch
"what happens around a generation" or "what happens around tool execution" should usually be a middleware, not a
change to a base handler.

### Agent loop mechanics (`loop.py`)

`extract_tool_uses(message)` pulls `ToolUse` blocks out of an `AssistantMessage`, tagged with the parent message id
(a message can contain multiple tool calls; tool calls across parallel iterations are also grouped). `use_tools(...)`
invokes the tool-execution handler, gathers the resulting awaitables, and groups `ToolResult`s back by parent id so
each `ToolMessage` in history correctly follows its originating assistant message.

### Suspension (human-in-the-loop) (`exceptions.py`)

A tool (or middleware) can raise `SuspensionError` to pause the agent loop mid-execution (e.g. to await human
approval) — this is intentionally caught and converted rather than treated as a plain tool error. When multiple
tools in the same parallel batch suspend, `use_tools` raises `BatchSuspendedException`, carrying both the pending
`suspensions` (by parent id) and any `partial_results` that already completed in that batch, so callers can persist
state and resume the loop later.

### Messages (`message.py`)

All messages are `TypedDict`s, not classes — `UserMessage`, `SystemMessage`, `ToolMessage` (`InputMessage`) vs.
`AssistantMessage`, `AppMessage` (`OutputMessage`); together, `Message`. Streaming can yield either full
`AssistantMessage`s (`is_aggregate=True`) or granular `AssistantMessageDelta`s (`is_aggregate=False`) depending on
`stream_deltas`; consumers of `app.stream(...)` should check `is_aggregate`/`role` to distinguish deltas, aggregated
assistant messages, tool-result messages, and `AppMessage`s (provider/framework-level side-channel events).

### Tools (`tool.py`)

`Tool.from_func` builds a JSON-schema-based `Tool` from a plain Python callable via signature inspection + docstring
parsing (`docstring-parser`) — no Pydantic required, but Pydantic models are duck-typed in automatically if a type
has `model_json_schema()`. Supports `Annotated[...]` descriptions, `Literal` → enum, `TypedDict` → nested object
schema, and nested `$defs` hoisting to a flat top-level dict. `hidden_default_types`/`hidden_default_values` let a
tool have Python-only parameters (e.g. an injected context object) that are hidden from the LLM-facing schema.
`BigTalk._normalize_tools` accepts either raw callables or pre-built `Tool`s everywhere tools are passed in.

### LLM providers (`llm/`)

`LLMProvider` (`llm_provider.py`) is the abstract interface (`count_tokens`, `send`, `stream`, `close`) each provider
must implement; `anthropic.py` and `openai.py` are the built-in implementations, lazily instantiated by
`BigTalk._anthropic_provider_factory`/`_openai_provider_factory` only when their model prefix (`"anthropic/..."`,
`"openai/..."`) is first used, so the optional `anthropic`/`openai` extras are only required if actually invoked.
Custom providers can be registered via `BigTalk.add_provider(name, factory)`; model strings are always
`"provider/model_name"`.

## Testing conventions

Tests live in `libs/big-talk/tests/`. `tests/helpers.py` defines `TestLLMProvider`, a scriptable fake provider used
instead of hitting real APIs — tests configure it with canned `responses` and register it via
`bigtalk.add_provider(...)`. Shared fixtures (`bigtalk`, `create_provider`, `simple_message`) live in
`tests/conftest.py`. Tests are organized by concern (`test_middleware.py`, `test_tool_execution.py`,
`test_tool_schema.py`, `test_stream_deltas.py`, etc.) rather than one-file-per-module.
