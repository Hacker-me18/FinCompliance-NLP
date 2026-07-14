# -*- coding: utf-8 -*-
"""
文本智能清洗模块（文档第 5.2 节 - 文本智能清洗预处理模块）

职责：
  作为项目统一的文本清洗函数库。所有解析、合并、打标脚本都应调用本模块，
  而不应在各自内部重复实现清洗逻辑。

清洗规则（针对政采合同 / 司法文书解析出的原始文本）：
  1. 去除 PDF 页眉页脚 / 页码标注，如 "第X页 共Y页"、"-$ -"
  2. 去除勾选框占位符：  ☑ ☐ □ ☒ ✓ ✔ 等 Unicode/Symbol 符号
  3. 去除下划线 / 省略号占位："___"、"……"、"XX"
  4. 压缩非换行的多余空白（空格/制表符），但保留段落换行结构
  5. 合并中英文混用的多余标点（如连续的 "。。"、"，，"）
  6. 去除行首 / 行尾的多余空白

输入：单条原始文本字符串
输出：清洗后的标准化纯文本
"""

import re


# 需要整体移除的符号（在政府招标文件 PDF 解析中常见）
_NOISE_SYMBOLS = re.compile(
    r"["
    r"☐☑☒□■▣◇◈"
    r"✓✔✗✘❓❔︀]+"
)

# 填空占位：连续下划线、省略号、XX/xx 占位
_PLACEHOLDER_RE = re.compile(r"(?:_{3,}|…{2,}|\\.{3,}|…_)")
_PLACEHOLDER_XX_RE = re.compile(r"(?:X{2,}|x{2,})")

# PDF 页眉页脚、页码
_PAGE_MARK_RE = re.compile(r"第\s*\d+\s*页\s*共\s*\d+\s*页|-\s*\d+\s*-|^\d+$")

# 重复标点
_REPEAT_PUNCT = re.compile(r"([。！？，、；：」』）)\s])\1+")

# 一行内只含空白 / 噪音太短的占位行
_SHORT_NOISE_RE = re.compile(r"^[\s-]{1,4}$")


def clean_text(raw: str) -> str:
    """
    对单条原始文本执行清洗流水线。保留段落换行结构。

    Args:
        raw: 原始文本（如 pdfplumber / docx 抽取结果）

    Returns:
        str: 清洗后的标准化纯文本
    """
    if not raw:
        return ""

    # 1) 去噪音符号（勾选框、私有区字符等）
    text = _NOISE_SYMBOLS.sub("", raw)

    # 2) 去占位符（下划线 / 省略号填空）
    text = _PLACEHOLDER_RE.sub("", text)
    text = _PLACEHOLDER_XX_RE.sub("", text)

    # 3) 去重复标点
    text = _REPEAT_PUNCT.sub(r"\1", text)

    # 4) 按行处理：保留段落结构，标题行去页码
    cleaned_lines = []
    for line in text.split("\n"):
        # 去页码标记
        line = _PAGE_MARK_RE.sub("", line)
        # 压缩行内空白
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        # 去掉太短的占位行
        if _SHORT_NOISE_RE.match(line):
            continue
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_filename(name: str) -> str:
    """
    清洗下载文件名：去掉路径分隔符、控制字符，避免落到非预期目录。

    Args:
        name: 原始文件名（从 URL / Content-Disposition 解析出的）

    Returns:
        str: 安全的文件名（空则返回 "unnamed"）
    """
    bad = '\\/:*?"<>|\r\n\t'
    cleaned = "".join(c for c in name if c not in bad).strip()
    return cleaned or "unnamed"


def is_noise_line(line: str) -> bool:
    """
    判断一行文本是否属于噪音行（清洗后为空 / 太短 / 全是占位符）。

    Args:
        line: 单行文本

    Returns:
        bool: True 表示是噪音，应丢弃
    """
    s = line.strip()
    if not s:
        return True
    if len(s) < 5:
        return True
    if _SHORT_NOISE_RE.match(s):
        return True
    return False
