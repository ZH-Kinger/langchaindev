"""
Jira / GitHub workflow 工具连通性验证（集成测试 — 默认 skip）。
跑：pytest -m integration tests/integration/test_workflow.py

USERNAME / GH_LOGIN 改成自己的账号后再跑。
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

pytestmark = pytest.mark.integration

from config.settings import settings

# 通过环境变量传入用户名，避免改文件污染 git
USERNAME = os.environ.get("TEST_JIRA_USER", "")
GH_LOGIN = os.environ.get("TEST_GH_LOGIN", "")


def test_jira_my_activity():
    if not settings.JIRA_PAT or not USERNAME:
        pytest.skip("JIRA_PAT 或 TEST_JIRA_USER 未配置")
    from tools.jira.workflow import query_my_activity
    result = query_my_activity(USERNAME, "yesterday")
    assert isinstance(result, str) and result, "应返回非空字符串"


def test_jira_in_progress():
    if not settings.JIRA_PAT or not USERNAME:
        pytest.skip("JIRA_PAT 或 TEST_JIRA_USER 未配置")
    from tools.jira.workflow import query_in_progress
    result = query_in_progress(USERNAME)
    assert isinstance(result, str)


def test_jira_sprint_stories():
    if not settings.JIRA_PAT:
        pytest.skip("JIRA_PAT 未配置")
    from tools.jira.workflow import query_sprint_stories
    result = query_sprint_stories()
    assert isinstance(result, str)


def test_github_my_commits():
    if not (settings.GITHUB_TOKEN and GH_LOGIN):
        pytest.skip("GITHUB_TOKEN 或 TEST_GH_LOGIN 未配置")
    from tools.github.workflow import query_my_commits
    result = query_my_commits(GH_LOGIN, "yesterday")
    assert isinstance(result, str)


def test_github_open_prs():
    if not (settings.GITHUB_TOKEN and GH_LOGIN):
        pytest.skip("GITHUB_TOKEN 或 TEST_GH_LOGIN 未配置")
    from tools.github.workflow import query_my_prs
    result = query_my_prs(GH_LOGIN, state="open")
    assert isinstance(result, str)
