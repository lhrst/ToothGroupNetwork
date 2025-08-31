#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

# GPU 并发限流：同一时刻最多多少个任务在跑
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
# 任务自动清理（秒）
export TASK_TTL_SECONDS=${TASK_TTL_SECONDS:-86400}

echo "========== 环境信息 =========="
python - <<'PY'
import torch, os, platform
print("Python:", platform.python_version())
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU[0]型号:", torch.cuda.get_device_name(0))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MAX_CONCURRENCY:", os.environ.get("MAX_CONCURRENCY"))
PY

echo "========== 启动牙齿分割服务 =========="
# 单 worker 即可；GPU 任务不建议多进程
exec python -m uvicorn service:app --host 0.0.0.0 --port 9096 --workers 1 --timeout-keep-alive 120
