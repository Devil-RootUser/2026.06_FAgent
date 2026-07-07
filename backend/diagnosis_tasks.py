from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.agent import RuleDiagnosisAgent
from backend.config.settings import OUTPUT_ROOT


TASK_ROOT = OUTPUT_ROOT / "diagnosis_tasks"
TASK_ROOT.mkdir(parents=True, exist_ok=True)

STAGES = [
    "created",
    "detecting",
    "evidence_collecting",
    "retrieving_knowledge",
    "reasoning",
    "reporting",
    "completed",
]


@dataclass
class DiagnosisTask:
    task_id: str
    dataset: str
    event_id: int | None
    question: str
    use_llm: bool = False
    status: str = "created"
    stage: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    thinking_chunks: list[str] = field(default_factory=list)
    report_chunks: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


TASKS: dict[str, DiagnosisTask] = {}
TASK_QUEUES: dict[str, "queue.Queue[dict[str, Any]]"] = {}
LOCK = threading.Lock()


def _task_path(task_id: str) -> Path:
    return TASK_ROOT / f"{task_id}.json"


def _save(task: DiagnosisTask) -> None:
    _task_path(task.task_id).write_text(json.dumps(asdict(task), ensure_ascii=False, indent=2), encoding="utf-8")


def _load(task_id: str) -> DiagnosisTask | None:
    path = _task_path(task_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return DiagnosisTask(**data)


def _publish(task_id: str, event_type: str, text: str, payload: dict[str, Any] | None = None) -> None:
    q = TASK_QUEUES.get(task_id)
    event = {"type": event_type, "text": text, "payload": payload or {}, "time": time.time()}
    if q:
        q.put(event)


def _set_stage(task: DiagnosisTask, stage: str, message: str) -> None:
    task.stage = stage
    task.status = "running" if stage not in {"completed", "failed"} else stage
    task.updated_at = time.time()
    task.thinking_chunks.append(message)
    _publish(task.task_id, "thinking", message, {"stage": stage})
    _save(task)


def _run_task(task_id: str) -> None:
    with LOCK:
        task = TASKS[task_id]
    try:
        task.stage = "detecting"
        task.status = "running"
        task.updated_at = time.time()
        _save(task)

        def emit(kind: str, text: str, payload: dict[str, Any] | None = None) -> None:
            if kind == "tool" and payload:
                task.tool_calls.append(payload)
                # 根据工具名称推进阶段
                tool = payload.get("name", "")
                if tool == "get_event_summary":
                    task.stage = "detecting"
                elif tool in ("rank_root_causes", "inspect_edge_degradation", "inspect_sensor_window"):
                    task.stage = "evidence_collecting"
                elif tool == "retrieve_maintenance_knowledge":
                    task.stage = "retrieving_knowledge"
                elif tool == "generate_report":
                    task.stage = "reporting"
                elif tool in ("dashscope_chat", "erniebot_reasoning"):
                    task.stage = "reasoning"
                task.thinking_chunks.append(text)
            elif kind == "report":
                task.report_chunks.append(text)
                task.stage = "completed"
            else:
                task.thinking_chunks.append(text)
            task.updated_at = time.time()
            _publish(task.task_id, kind, text, payload)
            _save(task)

        result = RuleDiagnosisAgent().execute(task.dataset, task.event_id, task.question, emit=emit, use_llm=task.use_llm)
        task.result = result
        task.tool_calls = result.get("tool_calls", task.tool_calls)
        task.status = "completed"
        task.updated_at = time.time()
        _save(task)
    except Exception as exc:
        task.status = "failed"
        task.stage = "failed"
        task.error = str(exc)
        task.updated_at = time.time()
        _publish(task.task_id, "error", str(exc), {"stage": "failed"})
        _save(task)
    finally:
        q = TASK_QUEUES.get(task_id)
        if q:
            q.put({"type": "done", "text": task.status, "payload": {}, "time": time.time()})


def create_task(dataset: str, event_id: int | None, question: str, use_llm: bool = False) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    task = DiagnosisTask(task_id=task_id, dataset=dataset, event_id=event_id, question=question, use_llm=use_llm)
    with LOCK:
        TASKS[task_id] = task
        TASK_QUEUES[task_id] = queue.Queue()
    _save(task)
    thread = threading.Thread(target=_run_task, args=(task_id,), daemon=True)
    thread.start()
    return task_summary(task_id)


def get_task(task_id: str) -> DiagnosisTask:
    with LOCK:
        task = TASKS.get(task_id)
    if task:
        return task
    loaded = _load(task_id)
    if loaded:
        with LOCK:
            TASKS[task_id] = loaded
        return loaded
    raise KeyError(task_id)


def task_summary(task_id: str) -> dict[str, Any]:
    task = get_task(task_id)
    return asdict(task)


def task_tool_calls(task_id: str) -> dict[str, Any]:
    task = get_task(task_id)
    return {"task_id": task_id, "tool_calls": task.tool_calls}


def stream_events(task_id: str, channel: str):
    get_task(task_id)
    q = TASK_QUEUES.setdefault(task_id, queue.Queue())
    snapshot = get_task(task_id)
    chunks = snapshot.report_chunks if channel == "report" else snapshot.thinking_chunks
    for text in chunks:
        yield {"type": channel, "text": text, "payload": {"snapshot": True}, "time": time.time()}
    while True:
        try:
            event = q.get(timeout=20)
        except queue.Empty:
            yield {"type": "heartbeat", "text": "", "payload": {}, "time": time.time()}
            continue
        if channel == "report" and event["type"] not in {"report", "done", "error"}:
            continue
        if channel == "thinking" and event["type"] not in {"thinking", "tool", "done", "error"}:
            continue
        yield event
        if event["type"] in {"done", "error"}:
            break


def format_sse(events):
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def list_history(limit: int = 30) -> list[dict[str, Any]]:
    """列出已完成的诊断任务历史（按时间倒序）。"""
    items = []
    for path in sorted(TASK_ROOT.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            answer = ""
            if data.get("result") and data["result"].get("answer"):
                answer = data["result"]["answer"]
            elif data.get("report_chunks"):
                answer = "".join(data["report_chunks"])
            items.append({
                "task_id": data["task_id"],
                "dataset": data.get("dataset", ""),
                "event_id": data.get("event_id"),
                "question": data.get("question", ""),
                "status": data.get("status", ""),
                "stage": data.get("stage", ""),
                "answer": answer[:800],
                "created_at": data.get("created_at", 0),
                "use_llm": data.get("use_llm", False),
            })
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        if len(items) >= limit:
            break
    return items
