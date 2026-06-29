# 工业时序异常诊断 Agent 平台（Relation-EVGAT Demo）

这是一个独立版 MVP：项目内包含 Relation-EVGAT 必要算法脚本、WaDI/SMD 样例数据和默认 outputs，不需要运行时引用旧项目目录。

## 启动后端

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173/dashboard`。

## 主要功能

- `/dashboard`：实时监控、异常分数、报警时间线、Agent 面板。
- `/relations`：传感器关系图、Top 退化边、边向量对比。
- `/root-cause`：Top-K 根因候选、证据卡片、历史曲线。
- `/report`：规则 Agent 诊断对话、报告生成、工具调用日志。

## 后端接口

- `GET /api/health`
- `GET /api/datasets`
- `POST /api/jobs/train`
- `GET /api/jobs/{job_id}`
- `GET /api/overview?dataset=WaDI_A2_ds10`
- `GET /api/timeseries?dataset=WaDI_A2_ds10`
- `GET /api/relation-graph?dataset=WaDI_A2_ds10&event_id=1`
- `GET /api/root-cause?dataset=WaDI_A2_ds10&event_id=1`
- `POST /api/agent/ask`
- `GET /api/report?dataset=WaDI_A2_ds10&event_id=1`

## 训练说明

前端“启动轻量训练”会调用本项目内的 `relation_evgat/run_top_ready_relation_gat.py`，默认使用 `epochs=1`、`max_train_windows=1000` 做快速闭环验证。首次演示无需等待训练，系统会直接读取复制进来的 `outputs/top_ready_relation_gat/WaDI_A2_ds10/full_joint`。
