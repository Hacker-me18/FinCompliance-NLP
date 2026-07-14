# -*- coding: utf-8 -*-
"""
FastAPI 请求/响应数据模型（Pydantic）

接口设计：
  POST /predict       单条/批量条款风险分类推理
  POST /upload        上传合同文件 → 解析 → 全量风险识别
  POST /explain       风险解释（LLM 增强）
  POST /report        生成标准化财务风险审核报告
  GET  /health        服务健康检查
"""

from pydantic import BaseModel
from typing import Optional, List


# ---------- 请求模型 ----------

class PredictRequest(BaseModel):
    """单条/批量条款推理请求"""
    texts: List[str]  # 条款文本列表（支持批量）


class ExplainRequest(BaseModel):
    """风险解释请求"""
    text: str  # 风险条款原文
    risk_type: Optional[str] = None  # 风险类型（可选）


class ReportRequest(BaseModel):
    """报告生成请求"""
    file_path: str  # 合同文件路径


# ---------- 响应模型 ----------

class PredictResult(BaseModel):
    """单条条款推理结果"""
    text: str
    risk_type: str          # 风险类型（payment_risk 等）
    risk_label_id: int      # 风险标签 ID（0-5）
    confidence: float       # 置信度（0-1）
    risk_level: str         # 风险等级（无风险/低/中/高）


class PredictResponse(BaseModel):
    """推理响应"""
    results: List[PredictResult]
    total: int


class ExplainResult(BaseModel):
    """风险解释结果"""
    risk_cause: str         # 风险成因
    risk_impact: str        # 风险影响
    rectification: str      # 整改建议
    risk_level: str         # 风险等级


class ExplainResponse(BaseModel):
    """风险解释响应"""
    text: str
    explanation: ExplainResult


class ReportResponse(BaseModel):
    """报告生成响应"""
    file_path: str
    report_path: str
    summary: dict           # 风险汇总统计


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    model_name: str
