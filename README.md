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
from bub.builtin import Agent
from bub.store import FileTapeStore


async def main() -> None:
    framework = BubFramework()
    framework.load_builtin_hooks()
    agent = Agent(framework, tape_store=FileTapeStore("sessions"))
    async with framework.running():
        stream = await agent.run_stream(session_id="demo", prompt="Hello")
        async for event in stream:
            if event.kind == "text":
                print(event.data.get("delta", ""), end="")


asyncio.run(main())
```

From a checkout, `uv sync` is enough; for local development use `make install` so `prek` hooks are installed too.

## Why Bub

- **Composable by design.** Every turn stage is a plugin hook. Override one stage or replace the whole flow without forking the runtime.
- **Tape context.** Context is rebuilt from append-only records, not carried around as mutable session state. Easier to inspect, replay, and hand off.
- **Surface-agnostic.** The runtime owns the turn; the host owns I/O. No channel, REPL, or transport is baked in.
- **Batteries included.** Tools, skills, tape stores, and model execution ship with the runtime. Use the defaults first, replace them later.
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

## Internal commands

Prompts starting with `,` run an internal command instead of a model turn (`,help`, `,skill name=my-skill`, `,fs.read path=README.md`).

## Configuration

| Variable                    | Default                      | Description                                          |
| --------------------------- | ---------------------------- | ---------------------------------------------------- |
| `BUB_MODEL`                 | `openrouter:openrouter/free` | Model identifier                                     |
| `BUB_API_KEY`               | —                            | Provider key                                         |
| `BUB_API_BASE`              | —                            | Custom provider endpoint                             |
| `BUB_CLIENT_ARGS`           | —                            | JSON object forwarded to the underlying model client |
| `BUB_COMPLETION_ARGS`       | —                            | JSON object forwarded to each completion call         |
| `BUB_MAX_STEPS`             | unlimited                    | Tool-use loop limit; must be a positive integer      |
| `BUB_MAX_TOKENS`            | `16384`                      | Max tokens per model call                            |
| `BUB_MODEL_TIMEOUT_SECONDS` | —                            | Model call timeout (seconds)                         |
| `BUB_SPILL_THRESHOLD`       | `4096`                       | Estimated tokens before tool output spills; `0` disables |

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
