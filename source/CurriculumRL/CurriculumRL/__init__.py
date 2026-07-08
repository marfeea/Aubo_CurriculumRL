# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""


def _missing_optional_runtime(error: ModuleNotFoundError) -> bool:
    """判断导入失败是否仅由未启动 Isaac 运行环境造成。"""

    module_root = (error.name or "").split(".", maxsplit=1)[0]
    return module_root in {"carb", "gymnasium", "isaaclab", "isaaclab_tasks", "omni", "pxr"}


# Isaac 启动环境中保持模板的自动注册行为；纯配置检查不强制加载 Isaac。
try:
    from .tasks import *  # noqa: F403
except ModuleNotFoundError as error:
    if not _missing_optional_runtime(error):
        raise

try:
    from .ui_extension_example import *  # noqa: F403
except ModuleNotFoundError as error:
    if not _missing_optional_runtime(error):
        raise
