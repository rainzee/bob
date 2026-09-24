# Bub

<div align="center">

<picture>
  <source srcset="https://raw.githubusercontent.com/bubbuild/bub/refs/heads/main/website/src/assets/bub-logo-dark.png" media="(prefers-color-scheme: dark)">
  <img alt="Bub logo" src="https://raw.githubusercontent.com/bubbuild/bub/refs/heads/main/website/src/assets/bub-logo.png" width="200">
</picture>

<p><strong>A tiny, plugin-free agent runtime.</strong></p>

</div>

Bub is a small Python runtime for building agents in shared environments. It started in group chats, where multiple humans and agents had to work in the same conversation without hidden state, hand-wavy memory, or framework-specific magic.

Built on [agents.md](https://agents.md/) and [Agent Skills](https://agentskills.io/) , Bub stays intentionally small. Every turn stage is a plain callback you pass in. Builtins are included but replaceable. Bub is a library: channels, CLIs, and deployment shells are the host application's business.

[GitHub](https://github.com/rainzee/bob)

## Quick Start

```bash
uv add bub
```

```python
import asyncio
from pathlib import Path

from bub import BubFramework, Config, Hooks
from bub.builtin import Agent, Battery, BuiltinHooks, battery_tools


async def main() -> None:
    framework = BubFramework(
        workspace=Path.cwd(),
        home=Path("./run"),
        config=Config({"model": "openai:gpt-5", "api_key": "sk-..."}),
    )
    framework.add_hooks(BuiltinHooks(framework).hooks)

    # Batteries are optional and come apart: take any of store, sidecars and lifespans.
    battery = Battery(home=framework.home, config=framework.config)
    framework.add_hooks(battery.hooks)
    framework.add_tape_store(battery.tape_store)
    framework.add_sidecars(*battery.sidecars)
    framework.add_lifespans(*battery.lifespans)

    agent = Agent(framework, tools=battery_tools(), skill_dirs=[Path("./skills")])
    async with framework.running():
        stream = await agent.run_stream(session_id="demo", prompt="Hello")
        async for event in stream:
            if event.kind == "text":
                print(event.data.get("delta", ""), end="")


asyncio.run(main())
```

`battery_tools()` is the builtin tool set (bash, fs.*, tape.*, web.fetch, subagent,
spill reader). `BuiltinHooks` owns only the turn pipeline; without `Battery` the
runtime has no tape store and `Agent` starts with whatever tools you pass. Build
your own with `@tool` / `Tool.from_callable` and inject storage with
`tape_store=FileTapeStore("sessions")`.

Everything is instance-scoped and explicit. `BubFramework` takes `workspace`,
`home`, and an optional `Config`; `Agent` takes its tools and skill roots; skill
discovery only reads the roots you pass. The library reads no environment
variables, touches no user directory, and keeps no process-wide registry.
Build configuration from a mapping or a file:

```python
Config({"model": "openai:gpt-5"})
Config.from_file(Path("./config.yml"))
```

From a checkout, `uv sync` is enough; `make install` is a thin wrapper around it.

## Why Bub

- **Composable by design.** Every turn stage is an ordered list of callbacks. Override one stage or replace the whole flow without forking the runtime.
- **Tape context.** Context is rebuilt from append-only records, not carried around as mutable session state. Easier to inspect, replay, and hand off.
- **Surface-agnostic.** The runtime owns the turn; the host owns I/O. No channel, REPL, transport, or message envelope is baked in.
- **Batteries optional.** Tools, skills, tape stores, and model execution ship with the runtime, but the batteries are opt-in and the tool set is passed to each agent explicitly.
- **Operator equivalence.** Humans and agents work inside the same runtime boundaries, with the same evidence trail and handoff model. No hidden operator class.

## How It Works

A turn is one call. `Agent.run_stream(session_id, prompt)` resolves the turn
state through hooks, forks the session tape, then loops on the model until no
tool calls remain:

```
build_state → agent loop → model stream
                 ↑              ↓
          (same tape)      tool calls
```

Each stage is a slot on a `Hooks` value, so the host contributes state, system
prompt, or tool interception without forking the runtime. Callbacks run in list
order and later entries win; async and sync callbacks are both accepted; a
failing callback raises. A continuation step sends no new user message: the tape
already ends with the assistant tool calls and their results, so the loop just
asks the model again.

Key source files:

- Composition root: [`src/bub/framework.py`](https://github.com/bubbuild/bub/blob/main/src/bub/framework.py)
- Hook contract: [`src/bub/hooks.py`](https://github.com/bubbuild/bub/blob/main/src/bub/hooks.py)
- Agent loop: [`src/bub/builtin/agent.py`](https://github.com/bubbuild/bub/blob/main/src/bub/builtin/agent.py)
- Builtin hooks: [`src/bub/builtin/hooks.py`](https://github.com/bubbuild/bub/blob/main/src/bub/builtin/hooks.py)
- Skill discovery: [`src/bub/skills.py`](https://github.com/bubbuild/bub/blob/main/src/bub/skills.py)

## Extend It

There is no registry, no discovery and no name lookup. Append callbacks to the
slots you care about:

```python
from pathlib import Path

from bub import BubFramework, Hooks
from bub.builtin import Agent, Battery, BuiltinHooks

framework = BubFramework(workspace=Path("."), home=Path(".bub"), config=Config.from_file(Path("config.yml")))
framework.add_hooks(BuiltinHooks(framework).hooks)

battery = Battery(home=framework.home, config=framework.config)
framework.add_hooks(battery.hooks)
framework.add_tape_store(battery.tape_store)
framework.add_sidecars(*battery.sidecars)
framework.add_lifespans(*battery.lifespans)


def one_paragraph(prompt, state):
    return "Answer in one paragraph."


async def audit(call, result, state):
    print("tool ran:", call.tool)


framework.add_hooks(Hooks(system_prompt=[one_paragraph], after_tool_call=[audit]))

agent = Agent(framework, tools=battery_tools())
async with framework.running():
    ...
```

`Battery` is optional and comes apart: `hooks`, `tape_store`, `sidecars` and
`lifespans` each go into their own slot, so you can take the file store without
the shell manager, or the spill sidecar without the store.

## Configuration

Configuration is a mapping passed to `Config`, either directly or through
`Config.from_file(path)`. Nothing is read from the environment. The root section
is `AgentSettings`; other sections belong to `Settings` subclasses that declare
their own `section`:

| Key                     | Default    | Description                                             |
| ----------------------- | ---------- | ------------------------------------------------------- |
| `model`                 | required   | Model identifier, `provider:model_id`                    |
| `fallback_models`       | —          | Additional models tried when the primary call fails      |
| `api_key`               | —          | Provider key, or a mapping keyed by provider             |
| `api_base`              | —          | Custom provider endpoint, or a mapping keyed by provider |
| `client_args`           | —          | Extra arguments for the underlying model client          |
| `completion_args`       | —          | Extra arguments forwarded to each completion call        |
| `max_steps`             | unlimited  | Tool-use loop limit; must be a positive integer          |
| `max_tokens`            | `16384`    | Max tokens per model call                                |
| `model_timeout_seconds` | —          | Model call timeout (seconds)                             |

```yaml
model: openai:gpt-5
api_key:
  openai: sk-...
max_tokens: 8192
spill:
  threshold: 4096
```

```python
from typing import ClassVar

from bub import Config, Settings


class SpillSettings(Settings):
    section: ClassVar[str] = "spill"

    threshold: int = 4096


spill = Config.from_file(path).ensure(SpillSettings)
```

## Background

Bub is shaped by one constraint: real collaboration is messier than a solo demo. In shared environments, operators need visible boundaries, auditable history, and extension points that do not collapse into framework sprawl.

Read more:

- [Why We Rewrote Bub](https://bub.build/posts/why-rewrite-bub/)
- [Socialized Evaluation and Agent Partnership](https://bub.build/posts/socialized-evaluation/)
- [Context from Tape](https://tape.systems)

## Docs

- [Getting Started](https://bub.build/docs/getting-started/) — install Bub and run the first turn
- [Concepts](https://bub.build/docs/concepts/) — the mental model behind the runtime
- [Skills](https://bub.build/docs/build/skills/) — discover, inspect, and author Agent Skills in Bub
- [Build](https://bub.build/docs/build/) — write hooks, ship tools and skills

Some of these pages describe the upstream CLI and channel packages that this repository no longer ships.

## Development

```bash
make install
make check
make test
```

See [CONTRIBUTING.md](https://github.com/bubbuild/bub/blob/main/CONTRIBUTING.md).

## License

[Apache-2.0](https://github.com/bubbuild/bub/blob/main/LICENSE)
