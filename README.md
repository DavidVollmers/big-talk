# 🦜 BigTalk

> The lightweight, middleware-first LLM framework for Python.

BigTalk is designed for developers who want the power of agents (tool calling, loops, memory) without the bloat of
heavy frameworks. It provides a clean, type-safe abstraction over LLMs with a robust middleware pipeline inspired by
Starlette/FastAPI.

## Why BigTalk?

Most LLM frameworks are either too simple (just a wrapper around API calls) or too complex (forcing you into their
"chains" and "graphs").

**BigTalk strikes the balance:**

- **Middleware-First Architecture**: Intercept requests, modify streams, and control tool execution using a familiar
  pipeline pattern.

- **True Async & Streaming**: Built from the ground up for `asyncio`. Tools run in parallel automatically.

- **Zero-Dependency Core**: We don't force Pydantic or NumPy on you. Use them if you want; we support them via duck
  typing.

- **Provider Agnostic**: Swap OpenAI for Anthropic (or your own) with a single line of config.

- **Type-Safe**: leveraged modern Python (3.12+) generics for excellent IDE autocompletion.

## The Architecture

BigTalk separates the **Generation Phase** (talking to the LLM) from the **Execution Phase** (running tools). Each has
its own middleware stack.

## Quick Start

```bash
pip install big-talk-ai[anthropic]

uv add big-talk-ai[anthropic]
```

```python
import asyncio
from big_talk import BigTalk


# 1. Define your tools (Standard Python functions)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."


# 2. Initialize
app = BigTalk()


# 3. Add Middleware (Optional)
@app.streaming.use
async def log_stream(handler, ctx, **kwargs):
    print(f"🤖 Generating for model: {ctx.model}")
    async for msg in handler(ctx, **kwargs):
        yield msg


@app.tooling.use
async def human_approval(handler, ctx, **kwargs):
    # Ask permission before running tools!
    print(f"✋ Tool Request: {ctx.tool_uses[0]['name']}")
    if input("Allow? (y/n) > ") == "y":
        return await handler(ctx, **kwargs)  # Proceed
    return []  # Block execution


# 4. Run the Agent Loop
async def main():
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    # The stream() method handles the loop:
    # Generation -> Tool Call -> Execution -> Result -> Generation
    async for chunk in app.stream("anthropic/claude-sonnet-4-5", messages, tools=[get_weather], max_tokens=100):
        if chunk.get('is_aggregate'):
            print(f"\nLast Message: {chunk['content']}")
        else:
            print(chunk['content'][0].get('text', ''), end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```
