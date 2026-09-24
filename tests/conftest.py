from __future__ import annotations

from bub.builtin.hooks import Battery, BuiltinHooks
from bub.framework import BubFramework


def install_builtin(framework: BubFramework, *, batteries: bool = False) -> Battery | None:
    """把 builtin 回调装到 framework 上, batteries=True 时一并装可选电池"""

    framework.add_hooks(BuiltinHooks(framework).hooks)
    if not batteries:
        return None
    battery = Battery(home=framework.home)
    framework.add_hooks(battery.hooks)
    framework.add_tape_store(battery.tape_store)
    framework.add_sidecars(*battery.sidecars)
    framework.add_lifespans(*battery.lifespans)
    return battery
