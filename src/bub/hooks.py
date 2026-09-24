"""每一步的回调契约, 以及按序列执行它们的实现"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from bub.turn import TurnState
from bub.utils import MaybeAwait


@dataclass(frozen=True)
class LlmCallRequest:
    """暴露给回调的一次模型请求

    回调可以返回 ``dataclasses.replace`` 出来的副本以改写模型或消息; 工具对象不暴露,
    ``tool_names`` 只用于观察, 改工具集不在回调的职责内
    """

    run_id: str
    model: str
    messages: list[dict[str, Any]]
    tool_names: tuple[str, ...] = ()
    max_tokens: int | None = None


@dataclass(frozen=True)
class LlmCallResult:
    """暴露给回调的一次模型调用的最终结果

    流式调用给的是完全累积后的状态, 不是逐块视图, 调用失败时 ``error`` 是原始异常
    """

    run_id: str
    text: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error: Exception | None = None
    duration_ms: int = 0


@dataclass(frozen=True)
class LlmCallDecision:
    """``before_llm_call`` 的短路裁决"""

    action: Literal["finish"] = "finish"
    text: str = ""

    @classmethod
    def finish(cls, text: str) -> LlmCallDecision:
        """跳过这次 provider 调用, 把 ``text`` 当作它的最终输出"""

        return cls(action="finish", text=text)


@dataclass(frozen=True)
class ToolCall:
    """暴露给回调的一次工具调用"""

    run_id: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallDecision:
    """``before_tool_call`` 的裁决"""

    action: Literal["proceed", "replace", "deny"] = "proceed"
    arguments: dict[str, Any] | None = None
    result: Any = None
    message: str | None = None

    @classmethod
    def proceed(cls, arguments: dict[str, Any] | None = None) -> ToolCallDecision:
        """带着可选的新参数继续执行工具"""

        return cls(action="proceed", arguments=arguments)

    @classmethod
    def replace(cls, result: Any) -> ToolCallDecision:
        """跳过工具本身, 直接用 ``result`` 作为结果"""

        return cls(action="replace", result=result)

    @classmethod
    def deny(cls, message: str) -> ToolCallDecision:
        """跳过工具本身, 把 ``message`` 作为工具错误抛出"""

        return cls(action="deny", message=message)


@dataclass
class ToolCallResult:
    """暴露给回调的工具调用最终结果, 回调可以直接改写 ``result``"""

    run_id: str
    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: Exception | None = None
    duration_ms: int = 0


class LoadState(Protocol):
    """会话状态的补全者, 返回 ``None`` 或一个待合并的部分状态"""

    def __call__(self, session_id: str, state: TurnState) -> MaybeAwait[TurnState | None]: ...


class SystemPrompt(Protocol):
    """system prompt 的贡献者, 只有非空字符串会被拼接进去"""

    def __call__(self, prompt: str | list[dict], state: TurnState) -> MaybeAwait[str | None]: ...


class BeforeLlmCall(Protocol):
    """模型请求的查看者或改写者"""

    def __call__(
        self, request: LlmCallRequest, state: TurnState
    ) -> MaybeAwait[LlmCallRequest | LlmCallDecision | None]: ...


class AfterLlmCall(Protocol):
    """模型调用结果的观察者, 返回值被忽略"""

    def __call__(self, request: LlmCallRequest, result: LlmCallResult, state: TurnState) -> MaybeAwait[None]: ...


class BeforeToolCall(Protocol):
    """工具调用的查看者或改写者"""

    def __call__(self, call: ToolCall, state: TurnState) -> MaybeAwait[ToolCallDecision | None]: ...


class AfterToolCall(Protocol):
    """工具调用结果的观察者, 返回值被忽略"""

    def __call__(self, call: ToolCall, result: ToolCallResult, state: TurnState) -> MaybeAwait[None]: ...


async def maybe_await(value: Any) -> Any:
    """回调可以是同步的也可以是异步的, 统一在这里等一次"""

    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(frozen=True)
class Hooks:
    """一套回调, 每个槽位按序列顺序执行, 空槽位什么也不做

    序列顺序就是执行顺序, 也是覆盖顺序: 同一个槽位里靠后的回调看到的是靠前的回调
    改过的东西, 供应类的结果则由靠后的覆盖靠前的
    """

    load_state: Sequence[LoadState] = ()
    system_prompt: Sequence[SystemPrompt] = ()
    before_llm_call: Sequence[BeforeLlmCall] = ()
    after_llm_call: Sequence[AfterLlmCall] = ()
    before_tool_call: Sequence[BeforeToolCall] = ()
    after_tool_call: Sequence[AfterToolCall] = ()

    def __add__(self, other: Hooks) -> Hooks:
        """逐槽位拼接两套回调, 顺序即执行顺序"""

        return Hooks(
            load_state=(*self.load_state, *other.load_state),
            system_prompt=(*self.system_prompt, *other.system_prompt),
            before_llm_call=(*self.before_llm_call, *other.before_llm_call),
            after_llm_call=(*self.after_llm_call, *other.after_llm_call),
            before_tool_call=(*self.before_tool_call, *other.before_tool_call),
            after_tool_call=(*self.after_tool_call, *other.after_tool_call),
        )

    async def run_load_state(self, session_id: str, state: TurnState) -> TurnState:
        """依次让每个回调补全状态, 后返回的键覆盖先前的"""

        for hook in self.load_state:
            update = await maybe_await(hook(session_id, state))
            if update:
                state.update(update)
        return state

    async def run_system_prompt(self, prompt: str | list[dict], state: TurnState) -> str:
        """按顺序拼接每个回调返回的非空块, 块之间空一行"""

        blocks: list[str] = []
        for hook in self.system_prompt:
            block = await maybe_await(hook(prompt, state))
            if block:
                blocks.append(block)
        return "\n\n".join(blocks)

    async def run_before_llm_call(
        self, request: LlmCallRequest, state: TurnState
    ) -> tuple[LlmCallRequest, LlmCallDecision | None]:
        """把请求依次交给回调查看, 每个回调都看到前一个改过的请求

        返回 ``LlmCallDecision`` 会短路剩下的回调
        """

        for hook in self.before_llm_call:
            value = await maybe_await(hook(request, state))
            if value is None:
                continue
            if isinstance(value, LlmCallRequest):
                request = value
            elif isinstance(value, LlmCallDecision):
                return request, value
            else:
                raise TypeError(
                    f"before_llm_call must return LlmCallRequest | LlmCallDecision | None, got {type(value).__name__}"
                )
        return request, None

    async def run_after_llm_call(self, request: LlmCallRequest, result: LlmCallResult, state: TurnState) -> None:
        """通知每个观察者; 返回值被忽略"""

        for hook in self.after_llm_call:
            await maybe_await(hook(request, result, state))

    async def run_before_tool_call(self, call: ToolCall, state: TurnState) -> tuple[ToolCall, ToolCallDecision]:
        """把调用依次交给回调查看, 每个回调都看到前一个改过的参数

        ``proceed`` 可以带着新参数继续, ``replace`` 和 ``deny`` 短路剩下的回调
        """

        for hook in self.before_tool_call:
            value = await maybe_await(hook(call, state))
            if value is None:
                continue
            if not isinstance(value, ToolCallDecision):
                raise TypeError(f"before_tool_call must return ToolCallDecision | None, got {type(value).__name__}")
            if value.action == "proceed":
                if value.arguments is not None:
                    call = replace(call, arguments=dict(value.arguments))
                continue
            return call, value
        return call, ToolCallDecision.proceed()

    async def run_after_tool_call(self, call: ToolCall, result: ToolCallResult, state: TurnState) -> None:
        """通知每个观察者; 返回值被忽略, 但可以改写 ``result.result``"""

        for hook in self.after_tool_call:
            await maybe_await(hook(call, result, state))
