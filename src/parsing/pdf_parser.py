# -*- coding: utf-8 -*-
"""
PDF 合同文本提取模块（pdfplumber 引擎）

技术方案：
  - 默认 pdfplumber，表格与中文文本提取效果更好
  - 扫描件（图片型）页会拿不到文本，跳过并在结果中 warning
  - 保留段落结构：每页文本自然换行，便于下游 splitter / filter 按行处理

输入：PDF 文件路径
输出：提取后的全文文本字符串（保留换行结构）
"""

import os
import re
import json
import pdfplumber
from docx import Document

from src.data_process.clean_data import clean_text

# ---------------------- 全局配置：批量过滤关键词 ----------------------
# 段落包含以下任意词，判定为合同有效段落（保留）
KEEP_KEYS = {"政府采购合同", "合同通用条款", "合同专用条款", "货款", "价款",
             "付款", "保证金", "质保金", "增值税发票", "验收", "逾期", "违约金"}
# 段落包含以下任意词，直接丢弃（非合同条款）
FILTER_KEYS = {"投标人须知", "技术参数", "规格型号", "评分标准", "资格审查",
               "报价清单", "图纸", "维保方案", "中小企业声明", "投标文件格式"}
# 文件名含以下标识，整份跳过（纯投标须知分册，没有合同条款）
SKIP_FILE_NAME = {"第一册", "投标人须知册"}


# ---------------------- 分句 ----------------------
def split_clause(text: str) -> list:
    """按数字标号 / 中文分号切分，过滤过短 / 过长片段"""
    splits = re.split(r"\d+\.\d+|（\d+）|；", text)
    res = []
    for s in splits:
        s = s.strip()
        if 15 <= len(s) <= 130:
            res.append(s)
    return res


# ---------------------- 弱监督打标（严格规则：0~5 风险标签 / discard / review） ----------------------
# 正向触发词
_LABEL_HIT = {
    1: ["财政资金到位再付", "无固定尾款日期", "长期扣押保证金", "质保金无退还期限",
        "按财政资金到位情况支付", "待财政拨款后支付", "财政拨款到账后"],
    2: ["交货数月后开票", "税费全由乙方承担", "发票损失乙方全责", "乙方承担全部税费",
        "开票时间晚于交货", "乙方负责一切税费"],
    3: ["异议期3日", "异议期5日", "3日内异议", "5日内异议", "未提异议直接视为合格",
        "未书面提出视为合格", "仅甲方单方判定", "视为验收合格"],
    4: ["只约束乙方赔付", "甲方延期无违约金", "瑕疵扣除全款", "乙方逾期扣款",
        "仅罚乙方", "甲方不承担违约", "乙方向甲方支付违约金"],
    5: ["私户收款", "仅现金结算", "禁止对公转账", "法人个人账户打款",
        "个人账户", "现金结算", "转入个人账户", "汇款至个人"],
}
# 反向排除词：命中则取消对应标签
_LABEL_EXCLUDE = {
    1: [("日内付清", 15), ("验收合格后", 30), ("质保期满", 20), ("固定期限退质保金", 0),
        ("X日内付清", 0), ("xx日内付清", 0)],
    2: [("交货同步开专票", 0), ("税费各自承担", 0), ("开具发票后付款", 0)],
    3: [("异议期≥7天", 0), ("双方共同核验", 0), ("免费复检", 0),
        ("异议期7日", 0), ("异议期10日", 0), ("异议期15日", 0), ("异议期30日", 0)],
    4: [("甲乙赔付标准一致", 0), ("甲方逾期同样赔偿", 0), ("双方违约责任对等", 0),
        ("甲方延期", 8), ("甲方逾期", 8)],
    5: [("强制公对公", 0), ("禁止现金私户", 0), ("必须转账至对公", 0)],
}
# 无关业务章节词 → discard，出现就丢弃
_DISCARD_SECTIONS = ["技术参数", "图纸", "包装", "知识产权", "诉讼管辖", "保密条款",
                     "投标函", "开标一览表", "货物说明一览表", "投标分项报价表"]
_DISCARD_PLACEHOLDER = ["________", "……", "XX", "xx", "___", "空白填写"]


def weak_label(text: str) -> str:
    """
    严格规则打标。
    返回：'0'/'1'/'2'/'3'/'4'/'5' / 'discard' / 'review' / None（不在业务范围内）
    """
    # 前置 1：长度过滤
    if not (18 <= len(text) <= 120):
        return "discard"
    # 前置 2：无关章节 / 纯占位
    for d in _DISCARD_SECTIONS:
        if d in text:
            return "discard"
    if all(c in "_…×X " for c in text):
        return "discard"
    for ph in _DISCARD_PLACEHOLDER:
        if text.count(ph) >= 2:
            return "discard"
    # 只处理支付/发票/验收/保证金/违约
    _Biz = ["付款", "支付", "开票", "发票", "税费", "验收", "异议", "保证金", "质保金",
            "违约", "赔付", "违约金", "赔偿", "退款", "结算", "对公", "账户", "现金"]
    if not any(b in text for b in _Biz):
        return None  # 不在业务范围内，让上层丢弃
    # 多风险冲突检测
    hit_labels = []
    for lbl in [5, 1, 2, 3, 4]:
        pos = any(p in text for p in _LABEL_HIT.get(lbl, []))
        if not pos:
            continue
        excluded = False
        for ex, _ in _LABEL_EXCLUDE.get(lbl, []):
            if ex in text:
                excluded = True
                break
        if not excluded:
            hit_labels.append(lbl)
    # 标签 5 最高优先级，即使多标签命中也直接取 5
    if 5 in hit_labels:
        return "5"
    if len(hit_labels) >= 2:
        return "review"
    if len(hit_labels) == 1:
        return str(hit_labels[0])
    return "0"


# ---------------------- 段落过滤 ----------------------
def filter_paragraph(para_text: str) -> bool:
    """True=保留，False=丢弃"""
    for f_word in FILTER_KEYS:
        if f_word in para_text:
            return False
    for k_word in KEEP_KEYS:
        if k_word in para_text:
            return True
    return False


# ---------------------- 单文件全文提取（兼容 __init__.py / 下游 predict） ----------------------
def parse_pdf(file_path: str) -> str:
    """提取 PDF 合同全文，按页拼接并保留段落换行（保留给 parse_document 调用）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
    pages_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    txt = page.extract_text() or ""
                except Exception as e:
                    pages_text.append(f"\n[警告] 第{i}页文本抽取失败: {e}\n")
                    continue
                if txt.strip():
                    pages_text.append(txt)
    except Exception as e:
        raise ValueError(f"无法打开 PDF 文件 {file_path}: {e}") from e
    if not pages_text:
        raise ValueError(f"PDF 未提取到任何文本（可能为扫描件/图片型 PDF）: {file_path}")
    return "\n".join(pages_text)


# ---------------------- 读取 PDF 全文（批量过滤版本，内部使用） ----------------------
def read_pdf(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_txt = page.extract_text()
            if page_txt:
                # 每页用公共清洗模块做简单清洗（保留换行结构）
                page_txt = clean_text(page_txt)
                if page_txt.strip():
                    pages.append(page_txt)
    return "\n".join(pages)


# ---------------------- 读取 DOCX 全文 ----------------------
def read_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    paras = []
    for para in doc.paragraphs:
        # 公共清洗：去噪音符号、压缩空白
        t = clean_text(para.text)
        if t:
            paras.append(t)
    return "\n".join(paras)


# ---------------------- 读取老 .doc（antiword 兜底，失败返回空串） ----------------------
def read_old_doc(doc_path: str) -> str:
    import shutil, subprocess
    antiword = shutil.which("antiword")
    if not antiword:
        return ""
    try:
        out = subprocess.check_output([antiword, doc_path], stderr=subprocess.DEVNULL, timeout=30)
        return clean_text(out.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


# ---------------------- 去掉 PDF 每页重复的页眉 ----------------------
def _strip_page_headers(raw_text: str) -> str:
    """
    pdfplumber 会把每页都完整抽出来，页眉（如"XX政府采购项目 招标文件（第X册）"）在每一行都出现。
    思路：按换行切分 → 去重（保留顺序）→ 再拼回来，整段重复的行会被合并成一次。
    """
    lines = raw_text.split("\n")
    seen = set()
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return "\n".join(out)


# ---------------------- 单文件解析入口 ----------------------
def parse_single_file(file_path: str) -> list:
    """解析单个 PDF/DOC/DOCX，返回样本列表。单文件异常不中断批量流程。"""
    fname = os.path.basename(file_path)
    # 快速跳过纯须知分册
    for skip_tag in SKIP_FILE_NAME:
        if skip_tag in fname:
            print(f"跳过纯须知文件：{fname}")
            return []
    # 按实际扩展名读全文（老 .doc 用 antiword）
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    try:
        if ext == "pdf":
            raw_text = read_pdf(file_path)
        elif ext == "docx":
            raw_text = read_docx(file_path)
        elif ext == "doc":
            raw_text = read_old_doc(file_path)
            if not raw_text:
                print(f"{fname} 是老 .doc 且 antiword 提取失败，跳过")
                return []
        else:
            return []
    except Exception as e:
        print(f"{fname} 读取失败（{type(e).__name__}: {e}），跳过")
        return []
    # 1) 去掉 PDF 每页重复的页眉
    raw_text = _strip_page_headers(raw_text)
    # 2) 按章节丢弃整段无关内容（投标人须知 / 评分标准 / 报价清单等是章节标题，不是合同条款）
    _skip_re = re.compile(r"^(第[一二三四五六七八九十\d]+[章节篇部分篇])\s*(投标人须知|评分标准|资格审查|报价清单"
                          r"|图纸|维保方案|中小企业声明|投标文件格式|招标公告|采购公告"
                          r"|合同专用条款|合同通用条款|$)")
    _keep_re = re.compile(r"^(第[一二三四五六七八九十\d]+[章节篇部分篇])\s*(政府采购合同|政府采购)"
    )
    paragraphs = raw_text.split("\n")
    valid_parts = []
    skip_mode = False
    for p in paragraphs:
        s = p.strip()
        if not s:
            continue
        if _skip_re.match(s):
            skip_mode = True
            continue
        if _keep_re.match(s):
            skip_mode = False
            continue
        if re.match(r"^第[一二三四五六七八九十\d]+[章节篇部分篇]", s):
            # 遇到新章节标题，自动退出 skip_mode
            skip_mode = False
        if skip_mode:
            continue
        if filter_paragraph(s):
            valid_parts.append(s)
    valid_text = " ".join(valid_parts)
    if not valid_text:
        print(f"{fname} 未提取到有效合同条款")
        return []
    # 3) 分句、打标、组装样本
    clause_list = split_clause(valid_text)
    sample_list = []
    for clause in clause_list:
        label = weak_label(clause)
        sample = {
            "text": clause,
            "label": label,
            "confidence": 0.75,
            "source": "gov_bid_file"
        }
        sample_list.append(sample)
    print(f"{fname} 提取有效条款 {len(sample_list)} 条")
    return sample_list


# ---------------------- 批量遍历文件夹 ----------------------
def batch_parse_folder(folder: str, output_jsonl: str) -> None:
    """批量处理整个文件夹，输出 JSONL"""
    all_samples = []
    for fname in os.listdir(folder):
        fp = os.path.join(folder, fname)
        if fp.endswith((".pdf", ".doc", ".docx")):
            samples = parse_single_file(fp)
            all_samples.extend(samples)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n批量完成，总计提取 {len(all_samples)} 条招标合同样本 → {output_jsonl}")


# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    from pathlib import Path

    # 项目根目录 / src / parsing / pdf_parser.py  →  向上 2 级即项目根目录
    project_root = Path(__file__).resolve().parents[2]
    data_folder = project_root / "data" / "raw" / "public" / "contract_pdf"
    out_file = data_folder / "bid_contract_samples.jsonl"

    print(f"开始解析文件夹：{data_folder}\n")
    batch_parse_folder(str(data_folder), str(out_file))
