from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import data_service as svc
from . import diagnosis_tasks
from .model import ModelFactory
from .rag import get_knowledge_store


app = FastAPI(title="Relation-EVGAT Industrial Diagnosis Agent", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    epochs: int = Field(default=1, ge=1, le=12)
    max_train_windows: int = Field(default=1000, ge=100, le=20000)
    eval_stride: int = Field(default=8, ge=1, le=64)
    edge_mode: Literal["none", "corr", "corr_lag", "full"] = "full"
    use_relation_degradation: bool = True


class AgentRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    question: str
    event_id: int | None = None


class DiagnosisRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    event_id: int | None = 1
    question: str = "为什么报警？请给出根因和排查建议。"
    use_llm: bool = False


class KnowledgeUploadRequest(BaseModel):
    filename: str = "note.txt"
    content: str


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.get("/api/health")
def health():
    payload = svc.health()
    payload["knowledge"] = get_knowledge_store().status
    payload["agent"] = ModelFactory().status().__dict__
    return payload


@app.get("/api/datasets")
def datasets():
    return {"datasets": svc.available_datasets()}


@app.post("/api/jobs/train")
def train(req: TrainRequest):
    try:
        job = svc.create_train_job(req.dataset, req.model_dump())
        return {"job_id": job.job_id, "status": job.status, "dataset": job.dataset}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    try:
        return svc.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


@app.get("/api/overview")
def overview(dataset: str = "WaDI_A2_ds10"):
    try:
        return svc.overview(dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/timeseries")
def timeseries(dataset: str = "WaDI_A2_ds10", start: int | None = None, end: int | None = None):
    try:
        return svc.timeseries(dataset, start, end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/relation-graph")
def relation_graph(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.relation_graph(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/root-cause")
def root_cause(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.root_cause(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/agent/ask")
def agent(req: AgentRequest):
    try:
        return svc.agent_answer(req.dataset, req.question, req.event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/report")
def report(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.report(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/diagnosis/tasks")
def create_diagnosis_task(req: DiagnosisRequest):
    try:
        return diagnosis_tasks.create_task(req.dataset, req.event_id, req.question, req.use_llm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/diagnosis/tasks/{task_id}")
def get_diagnosis_task(task_id: str):
    try:
        return diagnosis_tasks.task_summary(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Diagnosis task not found: {task_id}") from exc


@app.get("/api/diagnosis/tasks/{task_id}/tool-calls")
def get_diagnosis_tool_calls(task_id: str):
    try:
        return diagnosis_tasks.task_tool_calls(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Diagnosis task not found: {task_id}") from exc


@app.get("/api/diagnosis/tasks/{task_id}/thinking/stream")
def stream_diagnosis_thinking(task_id: str):
    try:
        diagnosis_tasks.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Diagnosis task not found: {task_id}") from exc
    return StreamingResponse(
        diagnosis_tasks.format_sse(diagnosis_tasks.stream_events(task_id, "thinking")),
        media_type="text/event-stream",
    )


@app.get("/api/diagnosis/tasks/{task_id}/report/stream")
def stream_diagnosis_report(task_id: str):
    try:
        diagnosis_tasks.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Diagnosis task not found: {task_id}") from exc
    return StreamingResponse(
        diagnosis_tasks.format_sse(diagnosis_tasks.stream_events(task_id, "report")),
        media_type="text/event-stream",
    )


@app.get("/api/knowledge/documents")
def knowledge_documents():
    store = get_knowledge_store()
    return {"documents": store.list_documents(), "status": store.status}


@app.post("/api/knowledge/upload")
async def knowledge_upload(request: Request):
    store = get_knowledge_store()
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        body = KnowledgeUploadRequest(**payload)
        return store.ingest_text(body.filename, body.content)
    raw = await request.body()
    filename = request.headers.get("x-filename", "upload.txt")
    return store.ingest_text(filename, raw.decode("utf-8", errors="ignore"))


@app.post("/api/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest):
    store = get_knowledge_store()
    return {"query": req.query, "hits": store.search(req.query, req.top_k), "status": store.status}


@app.delete("/api/knowledge/documents/{doc_id}")
def delete_knowledge_document(doc_id: str):
    deleted = get_knowledge_store().delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {"deleted": True, "doc_id": doc_id}
