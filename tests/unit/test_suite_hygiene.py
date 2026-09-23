"""测试套件自检：同一文件里**不允许出现重名的 test 函数**。

为什么值得一条用例：Python 里后定义的函数直接覆盖同名的前者，pytest 只会收集到最后那个，
前面那条**静默消失**——不报错、不警告、计数还少一条，谁也不会注意。

2026-09-23 真踩到：`tests/unit/test_temp_ak_extend_not_before.py` 里
`test_extend_missing_original_not_before_falls_back_to_now` 被定义了两次（改名时撞车），
「老记录 + 表单没填 start」那条从此不再执行。重名最容易发生在改名/合并用例的时候，
而那正是覆盖面最容易被悄悄削掉的时刻。
"""
import ast
import collections
import pathlib

import pytest

_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent
_FILES = sorted(p for p in _TESTS_DIR.rglob("test_*.py"))


def _dupes(path: pathlib.Path):
    # utf-8-sig：仓库里有带 BOM 的测试文件，普通 utf-8 读进来首行会是 U+FEFF 而解析失败
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = [n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test")]
    return sorted(k for k, v in collections.Counter(names).items() if v > 1)


def test_test_files_discovered():
    assert len(_FILES) > 50, f"只发现 {len(_FILES)} 个测试文件，路径推导可能坏了"


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_no_duplicate_test_function_names(path):
    dup = _dupes(path)
    assert not dup, (
        f"{path} 里有重名 test 函数 {dup} —— 后者覆盖前者，前面那条**不会被执行**。"
        "改名/合并用例时最常发生，请改成两个不同的名字（而不是删掉其中一条）")
