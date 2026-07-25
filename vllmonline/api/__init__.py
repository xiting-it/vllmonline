"""api 子包：REST API schemas 与路由。

模块：
    schemas.py  Pydantic request/response 模型
    routes.py   FastAPI 路由（按 SPEC §8 注册）

按 Phase 渐进挂载：
    P2: /api/models/*          模型管理
    P3: /api/canary/*          灰度管理
    P4: /api/eval/*            评测
"""
