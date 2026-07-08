"""仓库脚本使用的本地源码路径引导。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "source" / "CurriculumRL"
PRODUCT_MODULE_ROOT = PACKAGE_SOURCE / "CurriculumRL"


def add_package_source() -> None:
    """允许按 ``CurriculumRL`` 包名导入产品代码。"""

    source = str(PACKAGE_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)


def add_product_module_root() -> None:
    """允许无 Isaac 环境的检查入口只导入纯配置模块。"""

    source = str(PRODUCT_MODULE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
