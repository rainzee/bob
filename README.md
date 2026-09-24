# Bub

Bub is a small Python runtime for building agents in shared environments. It started in group chats, where multiple humans and agents had to work in the same conversation without hidden state, hand-wavy memory, or framework-specific magic.

Bub owns the turn loop and the tape. Everything else — which provider to call, which tools to offer, what the system prompt says, where settings come from — is passed in by the host. There are no builtin tools, no plugins, no settings layer, and no telemetry.

[GitHub](https://github.com/rainzee/bob)

## Quick Start

```bash
uv add bub
```

```python
import asyncio
from pathlib import Path
from typing import Any

from bub import Agent, BubFramework, ChatRequest, FileTapeStore, Tool, tool


class OpenAIClient:
    """The host's model call, in whichever SDK it prefers"""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def stream(self, request: ChatRequest):
        stream = await self._client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            max_tokens=request.max_tokens,
            stream=True,
            **request.options,
        )
        async for chunk in stream:
            yield chunk.model_dump()


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


async def main() -> None:
    framework = BubFramework(workspace=Path.cwd(), home=Path("./run"))
    framework.add_tape_store(FileTapeStore(Path("./run/tapes")))

    agent = Agent(
        framework,
        model="gpt-5",
        client=OpenAIClient(openai.AsyncOpenAI()),
        tools=[read_file],
    )
    async with framework.running():
        stream = await agent.run_stream(session_id="demo", prompt="Hello")
        async for event in stream:
            if event.kind == "text":
                print(event.data.get("delta", ""), end="")


asyncio.run(main())
```

Tools are plain functions or `Tool.from_callable(...)`. Storage is any object
that implements `TapeStore`; `FileTapeStore` and `InMemoryTapeStore` ship with
the library, and the runtime falls back to memory when neither the framework nor
the agent is given one.

Everything is instance-scoped and explicit. `BubFramework` takes `workspace` and
`home`; `Agent` takes its model, its client, its tools and its hooks. The library
reads no environment variables, touches no user directory, keeps no
process-wide registry, loads no configuration file, and configures no logging:
where anything comes from is the host's decision.

## Why Bub

- **One boundary for the model.** `ChatClient` is a protocol your SDK already
  nearly satisfies. The library never imports a provider.
- **Composable by design.** Every turn stage is an ordered list of callbacks.
  Override one stage or replace the whole flow without forking the runtime.
- **Tape context.** Context is rebuilt from append-only records, not carried
  around as mutable session state. Easier to inspect, replay, and hand off.
- **Surface-agnostic.** The runtime owns the turn; the host owns I/O. No channel,
  REPL, transport, message envelope, or tool set is baked in.
- **No options we invented.** No token cap, no step limit, no timeout, no
  search heuristics: unset means unset, and every default you get is the one the
  layer below chose.
- **Operator equivalence.** Humans and agents work inside the same runtime
  boundaries, with the same evidence trail and handoff model. No hidden operator
  class.

## How It Works

A turn is one call. `Agent.run_stream(session_id, prompt)` resolves the turn state
through hooks, forks the session tape, then loops on the model until no tool calls
remain:

```
build_state → agent loop → client.stream(ChatRequest) → tape
                 ↑                          ↓
            (same tape)                tool calls
```

Each stage is a slot on a `Hooks` value, so the host contributes state, system
prompt, or tool interception without forking the runtime. Callbacks run in list
order and later entries win; async and sync callbacks are both accepted; a failing
callback raises rather than being swallowed. A continuation step sends no new user
message: the tape already ends with the assistant tool calls and their results, so
the loop just asks the model again.

Key source files:

- Composition root: [`src/bub/framework.py`](https://github.com/rainzee/bob/blob/main/src/bub/framework.py)
- Hook contract: [`src/bub/hooks.py`](https://github.com/rainzee/bob/blob/main/src/bub/hooks.py)
- Agent loop: [`src/bub/agent.py`](https://github.com/rainzee/bob/blob/main/src/bub/agent.py)
- Model boundary: [`src/bub/model_runner.py`](https://github.com/rainzee/bob/blob/main/src/bub/model_runner.py)
- Tape and stores: [`src/bub/tape.py`](https://github.com/rainzee/bob/blob/main/src/bub/tape.py), [`src/bub/store.py`](https://github.com/rainzee/bob/blob/main/src/bub/store.py)

## The Model Boundary

`ChatClient` is one method:

```python
class ChatClient(Protocol):
    def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]: ...
```

The request carries `run_id`, `model`, `messages`, `tools`, `max_tokens`,
`reasoning_effort` and `options`. Messages and tool schemas use the OpenAI
chat-completions shape, which is also the shape the tape records and
`Tool.to_schema()` produces. The client yields chunks as JSON mappings: each
chunk carries `choices[i].delta`, and the last one may carry `usage`.

That is the whole contract. Which SDK you use, how credentials are supplied, what
`model` means, whether you retry another model on failure, and how a provider's
quirks are spelled are all on your side of the line.

## Extend It

There is no registry, no discovery and no name lookup. Append callbacks to the
slots you care about:

```python
from pathlib import Path

from bub import Agent, BubFramework, Hooks


def one_paragraph(prompt, state):
    return "Answer in one paragraph."


async def audit(call, result, state):
    print("tool ran:", call.tool, "->", result.error or result.result)


framework = BubFramework(workspace=Path("."), home=Path(".bub"))
framework.add_hooks(Hooks(system_prompt=[one_paragraph], after_tool_call=[audit]))

agent = Agent(framework, model="gpt-5", client=my_client)
async with framework.running():
    ...
```

The six slots are `load_state`, `system_prompt`, `before_llm_call`,
`after_llm_call`, `before_tool_call` and `after_tool_call`. Supply-type values
(tape store, sidecars, lifespans) are plain constructor slots on
`BubFramework`, not callbacks.

## Parameters

There is no settings layer. Every knob is a keyword argument on the object that
uses it, so `Agent` is the whole surface for one agent:

| Parameter               | Default   | Description                                             |
| ----------------------- | --------- | ------------------------------------------------------- |
| `model`                 | required  | Model identifier; the client decides what it means        |
| `client`                | required  | The host's `ChatClient`                                  |
| `chat_options`          | —         | Extra options added to every chat request                |
| `max_tokens`            | client    | Per-call output cap; unset sends no cap                  |
| `max_steps`             | unlimited | Tool-use loop limit                                      |
| `model_timeout_seconds` | —         | Model call timeout (seconds)                             |
| `tools`                 | —         | Tools available to this agent                            |
| `tape_store`            | framework | Store override; falls back to the framework's or memory  |
| `tape_context`          | chat replay | How tape entries become messages                       |
| `sidecars`              | —         | Extra sidecars mounted after the framework's              |
| `hooks`                 | —         | Extra callbacks appended after the framework's            |

`BubFramework(workspace=..., home=...)` adds `add_hooks`, `add_tape_store`,
`add_sidecars` and `add_lifespans`. Nothing is read from the environment, no file
is loaded, and nothing is validated against a schema: build a `dict` however you
like (YAML, argparse, a secret store) and pass the keys you need.

```python
import yaml
from pathlib import Path

from bub import Agent, FileTapeStore

settings = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}

agent = Agent(
    framework,
    model=settings["model"],
    client=OpenAIClient(openai.AsyncOpenAI(api_key=settings["api_key"])),
    max_tokens=settings.get("max_tokens"),
)
```

## Background

Bub is shaped by one constraint: real collaboration is messier than a solo demo. In shared environments, operators need visible boundaries, auditable history, and extension points that do not collapse into framework sprawl.

## Development

```bash
make install
make check
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
