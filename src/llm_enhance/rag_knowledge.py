# -*- coding: utf-8 -*-
"""
轻量化 RAG 知识库（文档第 8 节 Stage4 - 轻量化 RAG 知识库）

职责：
  对接企业财务制度、合同审核规范等内部文档，构建向量索引。
  在 LLM 风险解释时，通过检索增强生成让解释贴合企业内部合规要求。

知识来源：
  - data/knowledge/ 目录下的企业财务制度、合同审核规范文档
  - 支持 PDF/DOCX/TXT 格式

工作流：
  1. build_knowledge_base: 文档切片 → 向量化 → 构建索引
  2. retrieve: 根据风险条款检索 Top-K 条相关知识
  3. 检索结果注入 LLM Prompt，提升解释的专业性与合规性
"""


def build_knowledge_base(config: dict):
    """
    构建向量知识库索引。

    Args:
        config: 全局配置（含 knowledge_dir、embedding 配置）
    """
    raise NotImplementedError("知识库构建待实现")


def retrieve(config: dict, query: str, top_k: int = 3) -> list:
    """
    检索与风险条款相关的知识片段。

    Args:
        config: 全局配置
        query: 查询文本（风险条款）
        top_k: 返回 Top-K 条知识

    Returns:
        list[str]: 相关知识片段列表
    """
    raise NotImplementedError("知识检索待实现")
