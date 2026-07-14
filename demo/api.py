# -*- coding: utf-8 -*-
"""
FastAPI 线上推理服务

提供 RESTful API 接口：
  GET  /               渲染 index.html 网页
  POST /predict        单条条款风险分类推理（返回结构化结果 + 概率分布）
  GET  /health         服务健康检查
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from src.inference.predict import predict_text
from src.utils.config import get_config

# 加载配置
_root, config = get_config()

# 网页目录
TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(cfg: dict = None) -> FastAPI:
    """创建 FastAPI 应用实例"""
    if cfg is None:
        cfg = config

    app = FastAPI(
        title="ContractGuard-NLP",
        description="制造业合同财务风险智能识别系统",
        version="1.0.0",
    )

    # ── 网页首页 ──────────────────────────────────────
    @app.get("/")
    async def index(request: Request):
        return FileResponse(TEMPLATE_DIR / "index.html")

    # ── 预测接口 ──────────────────────────────────────
    @app.post("/predict")
    async def predict_endpoint(request: Request):
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "文本为空"}, status_code=400)

        result = predict_text(cfg, text)
        return JSONResponse({
            "text": result["text"],
            "risk_type": result["risk_type"],
            "risk_label_id": result["risk_label_id"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "confidence_array": result["confidence_array"],
        })

    # ── 健康检查 ──────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": True, "model_type": "baseline"}

    return app


# 默认应用实例（供 uvicorn 直接加载）
app = create_app()
