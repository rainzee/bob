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

import yaml

from bub import BubFramework
from bub.builtin import Agent, Battery, BuiltinHooks, battery_tools


async def main() -> None:
    settings = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}

    framework = BubFramework(workspace=Path.cwd(), home=Path("./run"))
    framework.add_hooks(BuiltinHooks(framework).hooks)

    # Batteries are optional and come apart: take any of store, sidecars and lifespans.
    battery = Battery(home=framework.home)
    framework.add_hooks(battery.hooks)
    framework.add_tape_store(battery.tape_store)
    framework.add_sidecars(*battery.sidecars)
    framework.add_lifespans(*battery.lifespans)

    agent = Agent(
        framework,
        model=settings["model"],
        api_key=settings["api_key"],
        tools=battery_tools(),
        skill_dirs=[Path("./skills")],
    )
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

Everything is instance-scoped and explicit. `BubFramework` takes `workspace` and
`home`; `Agent` takes its model, credentials, tools and skill roots; skill
discovery only reads the roots you pass. The library reads no environment
variables, touches no user directory, keeps no process-wide registry, and loads
no configuration file: where settings come from is the host's decision.

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
from bub.builtin import Agent, Battery, BuiltinHooks, battery_tools

framework = BubFramework(workspace=Path("."), home=Path(".bub"))
framework.add_hooks(BuiltinHooks(framework).hooks)

battery = Battery(home=framework.home)
framework.add_hooks(battery.hooks)
framework.add_tape_store(battery.tape_store)
framework.add_sidecars(*battery.sidecars)
framework.add_lifespans(*battery.lifespans)


def one_paragraph(prompt, state):
    return "Answer in one paragraph."


async def audit(call, result, state):
    print("tool ran:", call.tool)


framework.add_hooks(Hooks(system_prompt=[one_paragraph], after_tool_call=[audit]))

agent = Agent(framework, model="openai:gpt-5", tools=battery_tools())
async with framework.running():
    ...
```

`Battery` is optional and comes apart: `hooks`, `tape_store`, `sidecars` and
`lifespans` each go into their own slot, so you can take the file store without
the shell manager, or the spill sidecar without the store.

## Parameters

There is no settings layer. Every knob is a keyword argument on the object that
uses it, so `Agent` is the whole surface for one agent:

| Parameter               | Default   | Description                                              |
| ----------------------- | --------- | -------------------------------------------------------- |
| `model`                 | required  | Model identifier, `provider:model_id`                     |
| `fallback_models`       | —         | Additional models tried when the primary call fails       |
| `api_key`               | —         | Provider key, or a mapping keyed by provider              |
| `api_base`              | —         | Custom provider endpoint, or a mapping keyed by provider  |
| `client_args`           | —         | Extra arguments for the underlying model client           |
| `completion_args`       | —         | Extra arguments forwarded to each completion call         |
| `max_tokens`            | provider  | Max tokens per model call; unset sends no cap             |
| `max_steps`             | unlimited | Tool-use loop limit                                       |
| `model_timeout_seconds` | —         | Model call timeout (seconds)                              |
| `tools`                 | —         | Tools available to this agent                             |
| `skill_dirs`            | —         | Skill roots, first root wins on a name collision           |
| `tape_store`            | framework | Store override; falls back to the framework's or memory    |
| `sidecars`              | —         | Extra sidecars mounted after the framework's               |
| `hooks`                 | —         | Extra callbacks appended after the framework's             |

`Battery(home=..., spill_threshold=4096)` covers the batteries; `spill_threshold=0`
disables spilling. Nothing is read from the environment, no file is loaded, and
nothing is validated against a schema: build a `dict` however you like (YAML,
argparse, a secret store) and pass the keys you need.

```python
import yaml
from pathlib import Path

from bub.builtin import Agent, Battery, battery_tools

settings = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}

agent = Agent(
    framework,
    model=settings["model"],
    api_key=settings.get("api_key"),
    max_tokens=settings.get("max_tokens"),
    tools=battery_tools(),
)
battery = Battery(home=framework.home, spill_threshold=settings.get("spill", {}).get("threshold", 4096))
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
