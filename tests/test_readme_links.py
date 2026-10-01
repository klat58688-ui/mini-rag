"""README 链接的离线校验（README §15.29）。

README 是本项目的主要交付物（1800+ 行）。链接烂掉是它最容易发生的退化：
- 目录锚点会随标题改名而失效；
- 文件链接会随重命名而失效；
- 编辑器补全很容易写进 `file:///C:/Users/...` 这种**对任何读者都无效**的绝对路径
  （本项目确实出现过 9 处，见 §15.29）。

这三类都能在不联网的前提下挡住。

**关键实现细节**：抽取链接前必须先把**代码**剔掉——包括 ``` 围栏块与行内 `` `...` ``。
理由有两个，都是踩过的：
  1. §15.25 的示例代码块里有一行 `## Token`，GitHub 不为它建锚点，
     不跳过就会把"真实失效的锚点"误判为有效；
  2. 本文件对应的 §15.29 正文里写了字面量 ``](#anchor)`` 作为说明，
     不剔除行内代码就会被当成一条真实链接，导致误报。
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TEXT = README.read_text(encoding="utf-8")


def _strip_code(text: str) -> str:
    """剔掉 ``` 围栏块与行内 `` `...` ``，只留正文。"""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


PROSE = _strip_code(TEXT)


def _github_slug(heading: str) -> str:
    """复刻 GitHub 的标题锚点规则：小写、去标点、空格转连字符，保留中英数字。"""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", s)
    return s.replace(" ", "-")


def _headings() -> set[str]:
    return {
        _github_slug(line.lstrip("#").strip())
        for line in PROSE.splitlines()
        if line.startswith("#")
    }


def test_readme_has_a_table_of_contents():
    assert "### 目录" in TEXT, "README 缺少目录"
    anchors = re.findall(r"\]\(#([^)]+)\)", PROSE)
    assert len(anchors) >= 15, f"目录锚点太少（{len(anchors)}）"


def test_all_anchor_links_resolve():
    slugs = _headings()
    refs = re.findall(r"\]\(#([^)]+)\)", PROSE)
    broken = [r for r in refs if r not in slugs]
    assert not broken, f"失效的锚点：{broken}"


def test_all_relative_file_links_exist():
    refs = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", PROSE)
    broken = []
    for ref in refs:
        if ref.startswith(("http://", "https://", "mailto:")):
            continue
        if not (ROOT / ref).exists():
            broken.append(ref)
    assert not broken, f"失效的文件链接：{broken}"


def test_no_absolute_local_path_links():
    """`file:///C:/Users/...` 之类的绝对路径对任何其他读者都是坏的。"""
    bad = re.findall(r"\]\((file:///[^)]*)\)", PROSE)
    assert not bad, f"README 里还有本地绝对路径链接：{bad}"


@pytest.mark.parametrize("needle", ["## 当前状态速览", "## 怎么读这份 README"])
def test_orientation_sections_exist(needle):
    """新读者应能立刻看到当前状态与阅读指引，而不是翻过整本审计日志。"""
    assert needle in TEXT
