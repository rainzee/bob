# Bub

<div align="center">

<picture>
  <source srcset="https://raw.githubusercontent.com/bubbuild/bub/refs/heads/main/website/src/assets/bub-logo-dark.png" media="(prefers-color-scheme: dark)">
  <img alt="Bub logo" src="https://raw.githubusercontent.com/bubbuild/bub/refs/heads/main/website/src/assets/bub-logo.png" width="200">
</picture>

<p><strong>A tiny agent runtime, composable with plugins.</strong></p>

</div>

Bub is a small Python runtime for building agents in shared environments. It started in group chats, where multiple humans and agents had to work in the same conversation without hidden state, hand-wavy memory, or framework-specific magic.

Built on [agents.md](https://agents.md/) and [Agent Skills](https://agentskills.io/) , Bub stays intentionally small. Every turn stage is a [pluggy](https://pluggy.readthedocs.io/) hook. Builtins are included but replaceable. Bub is a library: channels, CLIs, and deployment shells are the host application's business.

[GitHub](https://github.com/rainzee/bob)

## Quick Start

```bash
uv add bub
```

```python
import asyncio

from bub import BubFramework
from bub.builtin import Agent, battery_tools


async def main() -> None:
    framework = BubFramework()
    # batteries=True adds the file tape store, tool-output spill, and shell lifecycle.
    framework.load_builtin_hooks(batteries=True)
    agent = Agent(framework, tools=battery_tools())
    async with framework.running():
        stream = await agent.run_stream(session_id="demo", prompt="Hello")
        async for event in stream:
            if event.kind == "text":
                print(event.data.get("delta", ""), end="")


asyncio.run(main())
```

`battery_tools()` is the builtin tool set (bash, fs.*, tape.*, web.fetch, subagent,
spill reader). Without `batteries=True` the default hooks own only the turn
pipeline and provide no tape store, and `Agent` starts with whatever tools you
pass; build your own with `@tool` / `Tool.from_callable` and inject storage with
`tape_store=FileTapeStore("sessions")`.

`BubFramework(config_file=..., home=...)` owns this instance's config and home
directory (`BUB_HOME` is read only when neither is given). There are no
process-wide configuration or tool registries.

From a checkout, `uv sync` is enough; `make install` is a thin wrapper around it.

## Why Bub

- **Composable by design.** Every turn stage is a plugin hook. Override one stage or replace the whole flow without forking the runtime.
- **Tape context.** Context is rebuilt from append-only records, not carried around as mutable session state. Easier to inspect, replay, and hand off.
- **Surface-agnostic.** The runtime owns the turn; the host owns I/O. No channel, REPL, or transport is baked in.
- **Batteries optional.** Tools, skills, tape stores, and model execution ship with the runtime, but the batteries are opt-in and the tool set is passed to each agent explicitly.
- **Operator equivalence.** Humans and agents work inside the same runtime boundaries, with the same evidence trail and handoff model. No hidden operator class.

## How It Works

Every inbound message goes through one turn pipeline. Each stage is a hook.

```
resolve_session → load_state → build_prompt → run_model → save_state
```

Builtins are registered first. External plugins load after them. At runtime, later plugins take precedence. There are no framework-only shortcuts.

Key source files:

- Turn orchestrator: [`src/bub/framework.py`](https://github.com/bubbuild/bub/blob/main/src/bub/framework.py)
- Hook contract: [`src/bub/hooks/specs.py`](https://github.com/bubbuild/bub/blob/main/src/bub/hooks/specs.py)
- Builtin hooks: [`src/bub/builtin/hook_impl.py`](https://github.com/bubbuild/bub/blob/main/src/bub/builtin/hook_impl.py)
- Skill discovery: [`src/bub/skills.py`](https://github.com/bubbuild/bub/blob/main/src/bub/skills.py)

## Extend It

```python
from bub import hookimpl
from bub.envelope import content_of


class EchoPlugin:
    @hookimpl
    def build_prompt(self, message, session_id, state):
        return f"[echo] {content_of(message)}"

    @hookimpl
    async def run_model(self, prompt, session_id, state):
        return prompt


echo_plugin = EchoPlugin()
```

```toml
[project.entry-points."bub"]
echo = "my_package.plugin:echo_plugin"
```

See the [Build docs](https://bub.build/docs/build/) for hook guides, packaging, and plugin structure.

## Configuration

| Variable                    | Default                      | Description                                          |
| --------------------------- | ---------------------------- | ---------------------------------------------------- |
| `BUB_MODEL`                 | required                     | Model identifier, `provider:model_id`                |
| `BUB_API_KEY`               | —                            | Provider key, or a JSON object keyed by provider     |
| `BUB_API_BASE`              | —                            | Custom provider endpoint, or a JSON object keyed by provider |
| `BUB_CLIENT_ARGS`           | —                            | JSON object forwarded to the underlying model client |
| `BUB_COMPLETION_ARGS`       | —                            | JSON object forwarded to each completion call         |
| `BUB_MAX_STEPS`             | unlimited                    | Tool-use loop limit; must be a positive integer      |
| `BUB_MAX_TOKENS`            | `16384`                      | Max tokens per model call                            |
| `BUB_MODEL_TIMEOUT_SECONDS` | —                            | Model call timeout (seconds)                         |
| `BUB_SPILL_THRESHOLD`       | `4096`                       | Estimated tokens before tool output spills; `0` disables |
| `BUB_HOME`                  | `~/.bub`                     | Home used for tapes and the default config file; ignored when passed to the constructor |

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
- [Build](https://bub.build/docs/build/) — write plugins, override hooks, ship tools and skills

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
