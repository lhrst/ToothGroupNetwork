import os
import sys
import json
import time
import uuid
import shutil
import traceback
import asyncio
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List

# —— 无显示环境避免 OpenGL 报错（如需）——
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

sys.path.append(os.getcwd())

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import aiofiles

from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from predict_utils import ScanSegmentation

# ==================== 配置 ====================
UPLOAD_DIR = Path("temp_uploads"); UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path("results");      RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "tgnet"
DEFAULT_CHECKPOINT_PATH = "ckpts/tgnet_fps"    # 不带 .h5
DEFAULT_CHECKPOINT_PATH_BDL = "ckpts/tgnet_bdl"

VALID_MODELS = ["tsegnet", "tgnet", "pointnet", "pointnetpp", "dgcnn", "pointtransformer"]

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", str(24*3600)))

# 限流信号量（GPU 并发）
gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# 模型缓存：{ model_name: ScanSegmentation(...) }
pipeline_cache: Dict[str, ScanSegmentation] = {}

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskInfo:
    def __init__(self, task_id: str, object_id: str):
        self.task_id = task_id
        self.object_id = object_id
        self.status = TaskStatus.PENDING
        self.created_at: float = time.time()
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.result_file: Optional[Path] = None
        self.visualization_file: Optional[Path] = None
        self.work_dir: Optional[Path] = None
        self.result_dir: Optional[Path] = None

tasks: Dict[str, TaskInfo] = {}

# ==================== 应用 ====================
app = FastAPI(title="Tooth Segmentation API", version="1.0.0")

# 如需跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ========== 工具函数 ==========
def _check_and_resolve_ckpts(model_name: str):
    """返回 list[str] 的 ckpt 路径（带 .h5）"""
    if model_name == "tgnet":
        ck1 = f"{DEFAULT_CHECKPOINT_PATH}.h5"
        ck2 = f"{DEFAULT_CHECKPOINT_PATH_BDL}.h5"
        if not os.path.exists(ck1):
            raise FileNotFoundError(f"缺少权重文件: {ck1}")
        if not os.path.exists(ck2):
            raise FileNotFoundError(f"缺少权重文件: {ck2}")
        return [ck1, ck2]
    else:
        ck = f"{DEFAULT_CHECKPOINT_PATH}.h5"
        if not os.path.exists(ck):
            raise FileNotFoundError(f"缺少权重文件: {ck}")
        return [ck]

def _get_or_make_pipeline(model_name: str) -> ScanSegmentation:
    """懒加载 + 缓存模型实例（线程安全：FastAPI 单 worker 情况下即可）"""
    if model_name in pipeline_cache:
        return pipeline_cache[model_name]
    ckpts = _check_and_resolve_ckpts(model_name)
    pipeline = make_inference_pipeline(model_name, ckpts)
    pipeline_cache[model_name] = ScanSegmentation(pipeline)
    return pipeline_cache[model_name]

async def _save_upload_obj(upload: UploadFile, dst: Path):
    # 基础校验
    if not upload.filename.lower().endswith(".obj"):
        raise HTTPException(status_code=400, detail="只支持 .obj 文件")
    # 保存
    async with aiofiles.open(dst, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)
    # 空文件校验
    if dst.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="空文件")

def _safe_json_read(p: Path) -> dict:
    with open(p, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("结果 JSON 不是对象")
    if "labels" not in data or "instances" not in data:
        raise ValueError("结果 JSON 缺少必要字段")
    return data

async def _cleanup_expired_tasks():
    """后台清理超时任务（简单实现）"""
    now = time.time()
    for tid, t in list(tasks.items()):
        if now - t.created_at > TASK_TTL_SECONDS:
            # 移除磁盘文件
            if t.work_dir and t.work_dir.exists():
                shutil.rmtree(t.work_dir, ignore_errors=True)
            if t.result_dir and t.result_dir.exists():
                shutil.rmtree(t.result_dir, ignore_errors=True)
            tasks.pop(tid, None)

# ========== 路由 ==========
@app.get("/healthz")
async def healthz():
    import torch
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "loaded_models": list(pipeline_cache.keys()),
        "tasks": len(tasks),
        "max_concurrency": MAX_CONCURRENCY
    }

@app.get("/models")
async def get_models():
    return {
        "supported": VALID_MODELS,
        "loaded": list(pipeline_cache.keys()),
        "default": DEFAULT_MODEL
    }

@app.post("/api/teeth/segment", response_model=Dict)
async def segment_teeth(
    object_id: str = Form(...),
    model_name: str = Form(DEFAULT_MODEL),
    obj_file: UploadFile = File(...)
):
    if model_name not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"模型名称无效，支持: {', '.join(VALID_MODELS)}")

    task_id = str(uuid.uuid4())
    task = TaskInfo(task_id, object_id)
    tasks[task_id] = task

    work_dir = UPLOAD_DIR / task_id; work_dir.mkdir(parents=True, exist_ok=True)
    result_dir = RESULT_DIR / task_id; result_dir.mkdir(parents=True, exist_ok=True)
    task.work_dir = work_dir
    task.result_dir = result_dir

    # 保存上传文件
    obj_path = work_dir / obj_file.filename
    await _save_upload_obj(obj_file, obj_path)

    # 异步后台处理
    asyncio.create_task(process_segmentation_task(task_id, obj_path, model_name, result_dir))

    return {
        "task_id": task_id,
        "object_id": object_id,
        "status": task.status,
        "message": "分割任务已提交，正在排队/处理中"
    }

async def process_segmentation_task(task_id: str, obj_path: Path, model_name: str, result_dir: Path):
    task = tasks[task_id]
    task.status = TaskStatus.PROCESSING
    err_log = result_dir / "error.log"

    try:
        if not obj_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {obj_path}")

        # 先确保模型可加载（不占用 semaphore，快速失败）
        _ = _get_or_make_pipeline(model_name)

        # GPU 限流（避免 OOM）
        async with gpu_semaphore:
            # 组织输入路径：要求目录名=object_id
            patient_dir = obj_path.parent / task.object_id
            patient_dir.mkdir(exist_ok=True)
            new_obj_path = patient_dir / obj_path.name
            shutil.copy2(obj_path, new_obj_path)

            result_json_path = result_dir / f"{task.object_id}.json"

            def run_on_cpu_thread():
                try:
                    predictor = _get_or_make_pipeline(model_name)
                    predictor.process(str(new_obj_path), str(result_json_path))
                    if not result_json_path.exists():
                        raise FileNotFoundError("未生成结果文件")
                    _ = _safe_json_read(result_json_path)
                except Exception as e:
                    # 写入详细错误
                    with open(err_log, "w") as ef:
                        ef.write(traceback.format_exc())
                    raise

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_on_cpu_thread)

            task.result_file = result_json_path
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
    finally:
        # 尝试清理超时任务（轻量做法）
        await _cleanup_expired_tasks()

@app.get("/api/teeth/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="找不到任务")
    t = tasks[task_id]
    resp = {
        "task_id": t.task_id,
        "object_id": t.object_id,
        "status": t.status,
        "created_at": t.created_at,
        "completed_at": t.completed_at,
    }
    if t.status == TaskStatus.COMPLETED and t.result_file:
        resp["download_url"] = f"/api/teeth/download/{task_id}/json"
    if t.status == TaskStatus.FAILED and t.error:
        resp["error"] = t.error
        if t.result_dir and (t.result_dir / "error.log").exists():
            resp["error_log"] = f"/api/teeth/download/{task_id}/error"
    return resp

@app.get("/api/teeth/download/{task_id}/json")
async def download_json_result(task_id: str):
    if task_id not in tasks: raise HTTPException(status_code=404, detail="任务不存在")
    t = tasks[task_id]
    if t.status != TaskStatus.COMPLETED: raise HTTPException(status_code=400, detail=f"任务尚未完成: {t.status}")
    if not t.result_file or not t.result_file.exists(): raise HTTPException(status_code=404, detail="结果文件不存在")
    return FileResponse(path=t.result_file, filename=f"{t.object_id}_segmentation.json", media_type="application/json")

@app.get("/api/teeth/download/{task_id}/error")
async def download_error_log(task_id: str):
    if task_id not in tasks: raise HTTPException(status_code=404, detail="任务不存在")
    t = tasks[task_id]
    p = (t.result_dir / "error.log") if t.result_dir else None
    if not p or not p.exists(): raise HTTPException(status_code=404, detail="没有找到错误日志")
    return FileResponse(path=p, filename=f"{task_id}_error.log", media_type="text/plain")

@app.delete("/api/teeth/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks: raise HTTPException(status_code=404, detail="任务不存在")
    t = tasks.pop(task_id)
    if t.work_dir and t.work_dir.exists(): shutil.rmtree(t.work_dir, ignore_errors=True)
    if t.result_dir and t.result_dir.exists(): shutil.rmtree(t.result_dir, ignore_errors=True)
    return {"message": "任务及相关文件已删除"}

# 直接运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9096)
