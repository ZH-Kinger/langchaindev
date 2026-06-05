"""测试脚本包。运行方式：python -m tests.test_xxx 或 python tests/test_xxx.py。"""
import sys
from pathlib import Path

# 把项目根加入 sys.path，使测试脚本无论从哪运行都能 import 主代码
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
