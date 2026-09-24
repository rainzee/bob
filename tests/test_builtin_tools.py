from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from bub.builtin.shell_manager import ShellManager
from bub.builtin.tools import (
    bash,
    bash_output,
    kill_bash,
    tape_info,
)
from bub.errors import ErrorKind
from bub.store import AsyncTapeStoreAdapter, InMemoryTapeStore
from bub.tape import Tape, TapeContext
from bub.tools import Tool, ToolContext, ToolExecutor, render_tools_prompt, resolve_tool_names, tool

_TEST_SHELL_MANAGER = ShellManager()


def _tool_context(tmp_path, **state) -> ToolContext:
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    return ToolContext(
        tape=tape,
        run_id="test-run",
        state={"_runtime_workspace": str(tmp_path), "_runtime_shell_manager": _TEST_SHELL_MANAGER, **state},
    )


def _python_shell(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _shell_id(result: str) -> str:
    return result.split("shell_id: ", 1)[1].splitlines()[0].strip()


@pytest.mark.asyncio
async def test_tape_info_formats_token_cache_hit_rate(tmp_path) -> None:
    context = _tool_context(tmp_path)
    await context.tape.record_chat(
        run_id="run-1",
        system_prompt=None,
        new_messages=[],
        response_text=None,
        usage={
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    )

    result = await tape_info.run(context=context)

    assert "last_token_cache_hit_rate: 37.50%" in result


def test_render_tools_prompt_renders_available_tools_block() -> None:
    first_name = "tests.prompt_one"
    second_name = "tests.prompt_two"

    @tool(name=first_name, description="First tool")
    def prompt_one() -> str:
        return "one"

    @tool(name=second_name)
    def prompt_two() -> str:
        return "two"

    rendered = render_tools_prompt([prompt_one, prompt_two])

    assert rendered == "<available_tools>\n- tests_prompt_one(): First tool\n- tests_prompt_two()\n</available_tools>"


def test_render_tools_prompt_includes_model_name_and_parameter_signature() -> None:
    tool_name = "tests.prompt_signature"

    @tool(name=tool_name, description="Read a file")
    def prompt_signature(path: str, offset: int = 0) -> str:
        return f"{path}:{offset}"

    rendered = render_tools_prompt([prompt_signature])

    assert rendered == "<available_tools>\n- tests_prompt_signature(path, offset?): Read a file\n</available_tools>"


def test_render_tools_prompt_returns_empty_string_for_empty_input() -> None:
    assert render_tools_prompt([]) == ""


def test_render_tools_prompt_includes_plain_tools() -> None:
    internal_tool = Tool(name="tests.internal", handler=lambda: None)

    assert render_tools_prompt([internal_tool]) != ""


def test_resolve_tool_names_accepts_runtime_names_and_model_aliases() -> None:
    dotted_name = "tests.resolve_alias"
    underscored_name = "tests_with_underscore"
    excluded_name = "tests.excluded_tool"
    available = (dotted_name, underscored_name, excluded_name)

    @tool(name=dotted_name)
    def resolve_alias() -> str:
        return "alias"

    @tool(name=underscored_name)
    def resolve_runtime_name() -> str:
        return "runtime"

    @tool(name=excluded_name)
    def excluded_tool() -> str:
        return "excluded"

    assert resolve_tool_names(
        [" tests_resolve_alias ", " tests_with_underscore "],
        exclude={" tests_excluded_tool "},
        all_names=available,
    ) == {
        dotted_name,
        underscored_name,
    }
    assert dotted_name not in resolve_tool_names(None, exclude={" tests_resolve_alias "}, all_names=available)
    assert excluded_name not in resolve_tool_names(None, exclude={" tests_excluded_tool "}, all_names=available)
    assert resolve_tool_names(None, exclude={" tests_resolve_alias "}, all_names=available) >= {underscored_name}


def test_resolve_tool_names_rejects_unknown_names() -> None:
    available = ("tests_present_tool",)

    with pytest.raises(ValueError, match="tests_missing_tool"):
        resolve_tool_names([" tests_missing_tool "], all_names=available)

    with pytest.raises(ValueError, match="tests_missing_tool"):
        resolve_tool_names(None, exclude={" tests_missing_tool "}, all_names=available)


def test_bash_schema_exposes_command_parameter() -> None:
    assert bash.parameters["properties"]["command"]["type"] == "string"
    assert bash.parameters["required"] == ["command"]
    assert "cmd" not in bash.parameters["properties"]


@pytest.mark.asyncio
async def test_bash_returns_stdout_for_foreground_command(tmp_path) -> None:
    result = await bash.run(command=_python_shell("print('hello')"), context=_tool_context(tmp_path))

    assert result == "hello"


@pytest.mark.asyncio
async def test_foreground_bash_releases_shell_from_shell_manager(tmp_path, monkeypatch) -> None:
    manager = ShellManager()
    monkeypatch.setattr(sys.modules[__name__], "_TEST_SHELL_MANAGER", manager)

    result = await bash.run(command=_python_shell("print('hello')"), context=_tool_context(tmp_path))

    assert result == "hello"
    assert manager._shells == {}


@pytest.mark.asyncio
async def test_foreground_bash_releases_shell_when_command_fails(tmp_path, monkeypatch) -> None:
    manager = ShellManager()
    monkeypatch.setattr(sys.modules[__name__], "_TEST_SHELL_MANAGER", manager)

    with pytest.raises(RuntimeError, match="command exited with code"):
        await bash.run(command=_python_shell("import sys; sys.exit(2)"), context=_tool_context(tmp_path))

    assert manager._shells == {}


@pytest.mark.asyncio
async def test_foreground_bash_terminates_shell_when_cancelled(tmp_path, monkeypatch) -> None:
    manager = ShellManager()
    monkeypatch.setattr(sys.modules[__name__], "_TEST_SHELL_MANAGER", manager)

    task = asyncio.create_task(
        bash.run(
            command=_python_shell("import time; time.sleep(10)"),
            context=_tool_context(tmp_path, session_id="session:target"),
        )
    )
    await asyncio.sleep(0.1)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert manager._shells == {}


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
@pytest.mark.asyncio
@pytest.mark.parametrize("shell_exits", [False, True])
@pytest.mark.parametrize("ignore_term", [False, True])
async def test_timed_out_bash_keeps_descendants_until_killed(tmp_path, monkeypatch, shell_exits, ignore_term) -> None:
    manager = ShellManager()
    monkeypatch.setattr(sys.modules[__name__], "_TEST_SHELL_MANAGER", manager)
    monkeypatch.setattr(manager, "TERMINATE_TIMEOUT", 0.2)
    pid_file = tmp_path / "child.pid"
    code = (
        "import os, signal, time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    command = f"{_python_shell(code)} & " + ("exit 0" if shell_exits else "wait")
    try:
        async with asyncio.timeout(5):
            result = await bash.run(command=command, timeout_seconds=1, context=_tool_context(tmp_path))
        assert "continuing in background" in result
        shell_id = _shell_id(result)
        shell = manager.get(shell_id)
        assert all(not task.cancelled() for task in shell.read_tasks)
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
        await kill_bash.run(shell_id=shell_id)
        assert manager._shells == {}
        # An orphan can briefly remain a zombie until the OS reaps it.
        ps = shutil.which("ps")
        assert ps is not None
        status = subprocess.run([ps, "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
        assert not status.stdout.strip() or status.stdout.strip().startswith("Z")
    finally:
        await asyncio.gather(*(manager.terminate(shell_id) for shell_id in list(manager._shells)))
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


@pytest.mark.asyncio
async def test_timed_out_bash_preserves_output_and_completes_in_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ShellManager()
    monkeypatch.setattr(sys.modules[__name__], "_TEST_SHELL_MANAGER", manager)
    gate = tmp_path / "continue"
    command = _python_shell(
        "import sys, time\n"
        "from pathlib import Path\n"
        "print('before timeout', flush=True)\n"
        f"while not Path({str(gate)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        "print('after timeout', flush=True)\n"
        "print('stderr after timeout', file=sys.stderr, flush=True)\n"
    )
    try:
        result = await bash.run(
            command=command, timeout_seconds=1, context=_tool_context(tmp_path, session_id="session:target")
        )
        assert "continuing in background" in result
        shell_id = _shell_id(result)
        shell = manager.get(shell_id)
        assert shell.returncode is None
        assert shell.session_id == "session:target"
        output = await bash_output.run(shell_id=shell_id)
        assert "status: running" in output
        assert "before timeout" in output

        gate.touch()
        async with asyncio.timeout(5):
            await shell.process.wait()
            await asyncio.gather(*shell.read_tasks)
        output = await bash_output.run(shell_id=shell_id)
        assert "status: exited" in output
        assert "exit_code: 0" in output
        assert "before timeout" in output
        assert "after timeout" in output
        assert "stderr after timeout" in output
        assert manager._shells == {}
    finally:
        gate.touch()
        await asyncio.gather(*(manager.terminate(shell_id) for shell_id in list(manager._shells)))


@pytest.mark.asyncio
async def test_shell_termination_bounds_output_drain(monkeypatch) -> None:
    manager = ShellManager()
    shell = await manager.start(cmd=_python_shell("pass"), cwd=None)
    await shell.process.wait()
    await asyncio.gather(*shell.read_tasks)
    reader = asyncio.create_task(asyncio.Event().wait())
    shell.read_tasks.append(reader)
    monkeypatch.setattr(manager, "DRAIN_TIMEOUT", 0.01)

    async with asyncio.timeout(2):
        await manager.terminate(shell.shell_id)

    assert reader.cancelled()
    assert manager._shells == {}


@pytest.mark.asyncio
async def test_bash_non_zero_exit_is_returned_as_tool_error(tmp_path) -> None:
    command = _python_shell("import sys; print('boom'); sys.exit(7)")
    executor = ToolExecutor()

    result = await executor.execute_async([(bash, {"command": command})], context=_tool_context(tmp_path))

    assert result.error is not None
    assert result.error.kind is ErrorKind.TOOL
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result["kind"] == "tool"
    assert tool_result["message"] == "Tool 'bash' execution failed."
    error_detail = tool_result["details"]["error"]
    assert "command exited with code 7" in error_detail
    assert "boom" in error_detail


@pytest.mark.asyncio
async def test_background_bash_exposes_output_via_bash_output(tmp_path) -> None:
    command = _python_shell(
        "import sys, time; print('start'); sys.stdout.flush(); time.sleep(0.2); print('done'); sys.stdout.flush()"
    )

    started = await bash.run(command=command, background=True, context=_tool_context(tmp_path))
    shell_id = _shell_id(started)

    await asyncio.sleep(0.35)
    output = await bash_output.run(shell_id=shell_id)

    assert output.startswith(f"id: {shell_id}\nstatus: exited\n")
    assert "exit_code: 0" in output
    assert "start" in output
    assert "done" in output


@pytest.mark.asyncio
async def test_kill_bash_terminates_background_process_and_releases_shell(tmp_path) -> None:
    started = await bash.run(
        command=_python_shell("import time; time.sleep(10)"),
        background=True,
        context=_tool_context(tmp_path),
    )
    shell_id = _shell_id(started)

    killed = await kill_bash.run(shell_id=shell_id)

    assert killed.startswith(f"id: {shell_id}\nstatus: exited\nexit_code: ")
    assert "exit_code: null" not in killed
    with pytest.raises(KeyError, match="unknown shell id"):
        await bash_output.run(shell_id=shell_id)


@pytest.mark.asyncio
async def test_kill_bash_returns_status_when_process_already_finished(tmp_path) -> None:
    started = await bash.run(
        command=_python_shell("print('done')"),
        background=True,
        context=_tool_context(tmp_path),
    )
    shell_id = _shell_id(started)

    await asyncio.sleep(0.1)
    result = await kill_bash.run(shell_id=shell_id)

    assert result == f"id: {shell_id}\nstatus: exited\nexit_code: 0"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_bash_cleanup_does_not_return_a_released_shell_id_or_mask_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    manager = ShellManager()
    monkeypatch.setattr(sys.modules[__name__], "_TEST_SHELL_MANAGER", manager)
    monkeypatch.setattr(manager, "TERMINATE_TIMEOUT", 1.2)
    pid_file = tmp_path / "child.pid"
    child = _python_shell(
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    wait_ready = _python_shell(
        f"import time; from pathlib import Path\nwhile not Path({str(pid_file)!r}).exists(): time.sleep(0.01)"
    )
    task = asyncio.create_task(
        bash.run(command=f"{child} >/dev/null 2>&1 & {wait_ready}", timeout_seconds=1, context=_tool_context(tmp_path))
    )
    try:
        if cancel:
            async with asyncio.timeout(5):
                while not manager._shells or not next(iter(manager._shells.values())).termination_task:
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == "(no output)"
        assert manager._shells == {}
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
        await asyncio.gather(task, return_exceptions=True)
        for shell_id in list(manager._shells):
            await manager.terminate(shell_id)
