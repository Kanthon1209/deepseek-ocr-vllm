import os
import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

import uuid

import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from process_pipeline import process_pdf_file

import asyncio

BASE_OUTPUT_DIR = '/ai/teacher/dkc/militech/DeepSeek-OCR/output/'

app = FastAPI(title="DeepSeek OCR Server")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传PDF文件并异步执行OCR解析"""

    task_id = str(uuid.uuid4())
    r = app.state.redis

    # 写入初始任务信息
    r.hset(f"task:{task_id}", mapping={
        "status": "pending",
        "filename": file.filename,
    })

    if not file.filename.lower().endswith(".pdf"):
        r.hset(f"task:{task_id}", "status", "error")
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    # 保存临时 PDF 到磁盘
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, file.filename)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 将任务丢到后台执行
    asyncio.create_task(process_pdf_file(pdf_path, BASE_OUTPUT_DIR, r, task_id))

    # **立即返回，不等待解析完成**
    return JSONResponse({
        "message": "上传成功，任务已进入队列",
        "task_id": task_id
    })

@app.get("/download")
async def download_file(filename: str = Query(..., description="要下载的文件名")):
    """
    下载指定的处理结果文件（例如 .mmd 或 .pdf）
    """
    file_path = os.path.join(BASE_OUTPUT_DIR, filename)

    # 防止目录穿越攻击
    abs_base = os.path.abspath(BASE_OUTPUT_DIR)
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_base):
        raise HTTPException(status_code=403, detail="非法路径")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(file_path, filename=os.path.basename(file_path))

@app.get("/task/{task_id}")
async def query_task_status(task_id: str):
    """
    查询任务状态
    """

    key = f"task:{task_id}"
    r = app.state.redis
    print(f'app.state: {app.state}')
    if not r.exists(key):
        raise HTTPException(status_code=404, detail="Task not found")

    filename, status = r.hmget(key, "filename", "status")

    return JSONResponse({"task_id": task_id, "filename": filename, "status": status})