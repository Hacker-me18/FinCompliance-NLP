# -*- coding: utf-8 -*-
"""
合同文件（PDF / DOC / DOCX）批量下载模块

技术方案：
  - url_list 中每条 url 都是可直接下载的直链
  - 文件名按 url_list 中的顺序命名为 001.pdf / 002.doc / 003.docx ...
  - 后缀按「Content-Disposition → Content-Type → URL 路径后缀」三级推断
  - 单条失败跳过并记录，最后统一汇总

输入：url_list，纯 url 字符串列表
    ["https://xxx/aa.pdf", "https://xxx/download?id=123", ...]

输出：下载后的本地文件路径列表；失败记录写入 download_failed.json
"""

import os
import json
from urllib.parse import urlparse, unquote

from src.data_process.clean_data import clean_filename


# 浏览器请求头：模拟真实浏览器
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 域名 → Referer：部分站点会校验 Referer 才放行下载，按域名分组匹配
# 匹配顺序：按域名长度从长到短，长域名优先
_REFERER_BY_HOST = [
    ("download.ccgp.gov.cn", "https://download.ccgp.gov.cn/"),   # 中国政府采购网
    ("ccgp.gov.cn",         "https://www.ccgp.gov.cn/"),
    ("222.75.70.90",        "https://www.ccgp-ningxia.gov.cn/"),# 宁夏站
    ("zbcg.mas.gov.cn",     "https://zbcg.mas.gov.cn/"),         # 马鞍山
    ("ggzyjy.ahsz.gov.cn",  "https://ggzyjy.ahsz.gov.cn/"),      # 宿州
    ("60.173.73.4",         "http://60.173.73.4/"),
    ("dengzhou.zfcg.henan.gov.cn", "https://dengzhou.zfcg.henan.gov.cn/"),
]


def _build_headers(url: str) -> dict:
    """构造请求头：自动为已知域名加上对应 Referer 绕过反爬"""
    headers = {"User-Agent": _DEFAULT_UA, "Accept": "*/*", "Accept-Language": "zh-CN,zh;q=0.9"}
    host = urlparse(url).netloc.split(":")[0]      # 去掉端口号，只留域名
    for domain, referer in _REFERER_BY_HOST:
        if host.endswith(domain):
            headers["Referer"] = referer
            break
    return headers


# MIME 类型 → 文件后缀，用于 Content-Type 兜底
_MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

# 魔术字节 → 文件后缀，用于没有响应头信息时兜底（通过读取文件真实字节判断）
# PDF=%PDF, DOC/DOCX/ZIP=PK, RTF={\rtf
_MAGIC_TO_EXT = [
    (b"%PDF", ".pdf"),
    (b"PK\x03\x04", ".docx"),   # docx 本质是 zip；doc 是 OLE2 更难识别，先用 docx 兜底
    (b"{\\rtf", ".rtf"),
]


def _guess_ext_from_headers(url: str, resp) -> str:
    """从响应头推断后缀：先 Content-Disposition，再 Content-Type，最后 URL 路径"""
    # 1) Content-Disposition
    #    普通格式：filename="xxx.pdf"
    #    RFC 5987 格式：filename*=utf-8''xxx.doc  (编码过的中文)
    cd = resp.headers.get("Content-Disposition", "")
    fname = ""
    if "filename*=" in cd:                                       # 优先 RFC 5987 编码格式
        raw = cd.split("filename*=")[-1].strip('"\'')
        # 格式：utf-8'zh_cn'<url编码文件名>
        parts = raw.split("'", 2)
        if len(parts) >= 3:
            fname = unquote(parts[-1])
    elif "filename=" in cd:                                      # 普通格式
        raw = cd.split("filename=")[-1].strip('"\'')
        fname = unquote(raw)

    if fname and "." in fname:
        return "." + fname.rsplit(".", 1)[-1].lower()

    # 2) Content-Type
    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if ctype in _MIME_TO_EXT:
        return _MIME_TO_EXT[ctype]

    # 3) URL 路径里的后缀
    path = unquote(urlparse(url).path)
    if "." in path:
        return "." + path.rsplit(".", 1)[-1].lower()
    return ""


def _guess_ext_from_bytes(data: bytes) -> str:
    """读真实字节判断文件类型（魔术字节兜底）"""
    for magic, ext in _MAGIC_TO_EXT:
        if data[:len(magic)] == magic:
            return ext
    return ""


def get_pdf(url_list: list, save_dir: str, timeout: int = 60) -> list:
    """
    批量下载合同文件（PDF / DOC / DOCX）。

    Args:
        url_list: 链接列表，每项可以是字典 {"url":..., "file_name":...} 或纯 url 字符串
        save_dir: 下载保存根目录，不存在则自动创建
        timeout:  单次请求超时秒数

    Returns:
        list[dict]: 形如
            {"url": ..., "file_name": ..., "local_path": ..., "ok": True}
            失败项带 "error"
    """
    import requests

    os.makedirs(save_dir, exist_ok=True)
    results = []

    for idx, item in enumerate(url_list):
        url = item.get("url") if isinstance(item, dict) else item
        if not url:
            continue

        rec = {
            "url": url,
            "file_name": "",
            "local_path": "",
            "ok": False,
        }

        try:
            resp = requests.get(url, headers=_build_headers(url), timeout=timeout, stream=True)
            resp.raise_for_status()

            # 一次性读进内存便于魔术字节兜底再写盘（合同单份几 MB 内，吃得消）
            data = resp.content

            # 推断后缀：响应头优先 → 魔术字节兜底
            ext = _guess_ext_from_headers(url, resp)
            if not ext:
                ext = _guess_ext_from_bytes(data) or ".bin"

            local_name = f"{idx + 1:03d}{ext}"                   # 001.pdf / 002.doc ...
            local_path = os.path.join(save_dir, local_name)
            rec["file_name"] = local_name
            rec["local_path"] = local_path

            with open(local_path, "wb") as f:
                f.write(data)

            rec["ok"] = True
            rec["size"] = os.path.getsize(local_path)
            print(f"[成功] {url} → {local_path} ({rec['size']} bytes)")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[失败] {url} → {rec['error']}")

        # 回写到原始 item，下游可直接拿到 local_path
        if isinstance(item, dict):
            item["local_path"] = local_path
            item["ok"] = rec["ok"]
            if not rec["ok"]:
                item["error"] = rec["error"]
        results.append(item if isinstance(item, dict) else rec)

    ok_cnt = sum(1 for r in results if isinstance(r, dict) and r.get("ok"))
    fail_cnt = len(results) - ok_cnt
    print(f"\n下载完成: 成功 {ok_cnt} / 失败 {fail_cnt} / 总计 {len(results)}")

    if fail_cnt:
        fail_path = os.path.join(save_dir, "download_failed.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(
                [r for r in results if isinstance(r, dict) and not r.get("ok")],
                f, ensure_ascii=False, indent=2,
            )
        print(f"失败清单已写入: {fail_path}")

    return results

if __name__ == "__main__":
    from pathlib import Path

    # 文件层级：项目根目录 / src / data_process / download_pdf.py
    #resolve()让路径不依赖运行位置，结果稳定可预期
    # 向上 3 级就是项目根目录
    project_root = Path(__file__).resolve().parents[2]
    save_dir = project_root / "data/raw/public/contract_pdf"
    url_path = project_root / "data" / "url_list"

    with open(url_path, "r", encoding="utf-8") as f:
        url_list = [line.strip() for line in f if line.strip()]

    print(f"从 {url_path} 读取到 {len(url_list)} 条链接，开始下载...\n")
    get_pdf(url_list,save_dir)
    #print(save_dir)
