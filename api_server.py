"""Flask API 封装，用于在 Coze 等平台以 HTTP 方式触发现有工作流。"""
from __future__ import annotations

import os
import shlex
import re
import uuid
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

import app as typhoon_cli
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp" / "api_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# 确保 Julia 使用项目内的环境
os.environ.setdefault("JULIA_PROJECT", str(PROJECT_ROOT))

app = Flask(__name__)

# 限制上传文件最大100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB


class WorkflowRuntime:
    """惰性初始化 Julia 运行时并暴露 Workflows 模块。"""

    def __init__(self) -> None:
        self._initialized = False
        self._workflows = None

    def ensure(self) -> None:
        if self._initialized:
            return
        from julia.api import Julia  # pylint: disable=import-error
        from julia import Main  # pylint: disable=import-error

        # compiled_modules=False 可以降低首次加载时间，兼容更多环境
        Julia(compiled_modules=False)
        Main.include(str(PROJECT_ROOT / "src" / "workflows.jl"))
        self._workflows = Main.Workflows
        self._initialized = True

    @property
    def workflows(self):
        self.ensure()
        return self._workflows


runtime = WorkflowRuntime()


def _normalize_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return str(_download_remote_file(text))
    return str(Path(text).expanduser().resolve())


def _download_remote_file(url: str) -> Path:
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    parsed = urlsplit(url)
    path_name = Path(parsed.path).name or "remote_file"
    suffix = _infer_suffix(path_name)
    safe_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix) else ".dat"
    temp_name = f"upload_{uuid.uuid4().hex}{safe_suffix}"
    temp_path = TEMP_UPLOAD_DIR / temp_name
    with temp_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                file_obj.write(chunk)
    return temp_path


def _infer_suffix(path_name: str) -> str:
    lowered = path_name.lower()
    for ext in ('.xlsx', '.xls', '.csv', '.json', '.txt'):
        if ext in lowered:
            return ext
    suffix = Path(path_name).suffix
    return suffix if suffix else '.dat'


def _extract_payload() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return payload


def _call_workflow(func_name: str, arguments: Dict[str, Any]) -> Any:
    workflows = runtime.workflows
    func = getattr(workflows, func_name)
    kwargs = {key: value for key, value in arguments.items() if value is not None}
    try:
        return func(**kwargs)
    except SystemExit as exc:  # noqa: PERF203 -- 防止工作流意外退出服务器
        if exc.code not in (None, 0):
            raise RuntimeError(f"工作流 {func_name} 提前退出: {exc.code}") from exc
        return None


def _success(message: str, **data: Any):
    return jsonify({"status": "success", "message": message, **data})


def _error(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


def _parse_command(command: Any) -> List[str]:
    if command is None:
        return []
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, (list, tuple)):
        return [str(part) for part in command]
    raise ValueError("command 字段必须是字符串或字符串数组")


def _stringify_command(command: Any) -> Optional[str]:
    if command is None:
        return None
    if isinstance(command, str):
        stripped = command.strip()
        return stripped or None
    if isinstance(command, (list, tuple)):
        parts = [str(part).strip() for part in command if str(part).strip()]
        return " ".join(parts) if parts else None
    raise ValueError("command 字段必须是字符串或字符串数组")


# ==================== 文件上传端点 ====================

@app.route("/api/upload", methods=["POST"])
def upload_file_route():
    """
    上传文件到API服务器并返回临时路径
    
    用法：
    客户端发送 multipart/form-data 请求，包含文件字段：
    - file: 要上传的文件
    
    支持的文件类型：.xlsx, .xls, .csv, .json, .txt
    文件大小限制：100MB
    
    返回：
    {
        "status": "success",
        "message": "文件上传成功",
        "file_path": "/path/to/temp/upload_xxx.xlsx",
        "file_name": "TowerSeg.xlsx",
        "file_size": 12345
    }
    """
    # 检查是否有文件
    if 'file' not in request.files:
        return _error("未提供文件，请使用 'file' 字段上传", status=400)
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return _error("文件名为空", status=400)
    
    filename = file.filename
    
    # 验证文件类型
    allowed_extensions = {'.xlsx', '.xls', '.csv', '.json', '.txt', '.dat'}
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return _error(
            f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(allowed_extensions)}",
            status=400
        )
    
    # 检查文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        return _error(f"文件过大，最大允许 {max_size // (1024*1024)}MB", status=400)
    
    # 生成唯一文件名
    temp_name = f"upload_{uuid.uuid4().hex}{file_ext}"
    temp_path = TEMP_UPLOAD_DIR / temp_name
    
    # 保存文件
    try:
        file.save(str(temp_path))
    except Exception as e:
        return _error(f"文件保存失败: {str(e)}", status=500)
    
    print(f"[upload] 文件已上传: {filename} -> {temp_path} ({file_size} bytes)")
    
    return _success(
        "文件上传成功",
        file_path=str(temp_path),
        file_name=filename,
        file_size=file_size
    )


@app.route("/api/upload-multiple", methods=["POST"])
def upload_multiple_files_route():
    """
    批量上传多个文件到API服务器
    
    用法：
    客户端发送 multipart/form-data 请求，包含多个文件字段：
    - tower-excel: 配电网结构文件
    - case-file: 案例文件
    - fault-file: 故障文件
    - 等等...
    
    返回：
    {
        "status": "success",
        "message": "文件上传成功",
        "files": {
            "tower-excel": {"path": "/path/...", "name": "...", "size": 123},
            "case-file": {"path": "/path/...", "name": "...", "size": 456}
        }
    }
    """
    if not request.files:
        return _error("未提供任何文件", status=400)
    
    uploaded_files = {}
    
    for field_name in request.files:
        file = request.files[field_name]
        if file.filename == '':
            continue
        
        # 验证文件类型
        file_ext = Path(file.filename).suffix.lower()
        allowed_extensions = {'.xlsx', '.xls', '.csv', '.json', '.txt', '.dat'}
        if file_ext not in allowed_extensions:
            continue
        
        # 检查文件大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            continue
        
        # 保存文件
        temp_name = f"upload_{uuid.uuid4().hex}{file_ext}"
        temp_path = TEMP_UPLOAD_DIR / temp_name
        file.save(str(temp_path))
        
        uploaded_files[field_name] = {
            "path": str(temp_path),
            "name": file.filename,
            "size": file_size
        }
    
    if not uploaded_files:
        return _error("没有有效的文件被上传", status=400)
    
    print(f"[upload-multiple] 批量上传完成: {len(uploaded_files)} 个文件")
    
    return _success(
        f"成功上传 {len(uploaded_files)} 个文件",
        files=uploaded_files
    )


# ==================== 原有API端点 ====================

@app.route("/api/classify-phases", methods=["POST"])
def classify_phases_route():
    try:
        payload = _extract_payload()
        args = {
            "input_path": _normalize_path(payload.get("input_path")),
            "output_path": _normalize_path(payload.get("output_path")),
            "lines_per_scenario": payload.get("lines_per_scenario"),
            "minutes_per_step": payload.get("minutes_per_step"),
            "stage2_minutes": payload.get("stage2_minutes"),
            "sheet_name": payload.get("sheet_name"),
        }
        _call_workflow("run_classify_phases", args)
        return _success("场景阶段分类已完成", requested_args=args)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/rolling-reconfig", methods=["POST"])
def rolling_reconfig_route():
    try:
        payload = _extract_payload()
        args = {
            "case_file": _normalize_path(payload.get("case_file")),
            "fault_file": _normalize_path(payload.get("fault_file")),
            "stage_file": _normalize_path(payload.get("stage_file")),
            "output_file": _normalize_path(payload.get("output_file")),
            "fault_sheet": payload.get("fault_sheet"),
            "stage_sheet": payload.get("stage_sheet"),
            "lines_per_scenario": payload.get("lines_per_scenario"),
        }
        _call_workflow("run_rolling_reconfig", args)
        return _success("滚动拓扑重构已完成", requested_args=args)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/mess-dispatch", methods=["POST"])
def mess_dispatch_route():
    try:
        payload = _extract_payload()
        args = {
            "case_path": _normalize_path(payload.get("case_path")),
            "topology_path": _normalize_path(payload.get("topology_path")),
            "fallback_topology": _normalize_path(payload.get("fallback_topology")),
            "output_file": _normalize_path(payload.get("output_file")),
        }
        _call_workflow("run_mess_dispatch", args)
        return _success("MESS 协同调度已完成", requested_args=args)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/full-pipeline", methods=["POST"])
def full_pipeline_route():
    try:
        payload = _extract_payload()
        classify_args = payload.get("classify", {})
        reconfig_args = payload.get("reconfig", {})
        dispatch_args = payload.get("dispatch", {})
        if not isinstance(classify_args, dict) or not isinstance(reconfig_args, dict) or not isinstance(dispatch_args, dict):
            raise ValueError("classify/reconfig/dispatch 字段必须是对象")

        _call_workflow(
            "run_classify_phases",
            {
                "input_path": _normalize_path(classify_args.get("input_path")),
                "output_path": _normalize_path(classify_args.get("output_path")),
                "lines_per_scenario": classify_args.get("lines_per_scenario"),
                "minutes_per_step": classify_args.get("minutes_per_step"),
                "stage2_minutes": classify_args.get("stage2_minutes"),
                "sheet_name": classify_args.get("sheet_name"),
            },
        )
        _call_workflow(
            "run_rolling_reconfig",
            {
                "case_file": _normalize_path(reconfig_args.get("case_file")),
                "fault_file": _normalize_path(reconfig_args.get("fault_file")),
                "stage_file": _normalize_path(reconfig_args.get("stage_file")),
                "output_file": _normalize_path(reconfig_args.get("output_file")),
                "fault_sheet": reconfig_args.get("fault_sheet"),
                "stage_sheet": reconfig_args.get("stage_sheet"),
                "lines_per_scenario": reconfig_args.get("lines_per_scenario"),
            },
        )
        _call_workflow(
            "run_mess_dispatch",
            {
                "case_path": _normalize_path(dispatch_args.get("case_path")),
                "topology_path": _normalize_path(dispatch_args.get("topology_path")),
                "fallback_topology": _normalize_path(dispatch_args.get("fallback_topology")),
                "output_file": _normalize_path(dispatch_args.get("output_file")),
            },
        )
        return _success("完整流程已成功执行")
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/typhoon-one-click", methods=["POST"])
def typhoon_full_generation_route():
    """触发 main.jl 功能4（台风场景生成工作流）并对外暴露。"""
    try:
        if request.is_json:
            payload = _extract_payload()
        else:
            payload = {**request.form}

        command_value = _stringify_command(payload.get("command"))
        args = {
            "command": command_value,
            "tower_excel": _normalize_path(payload.get("tower_excel")),
            "final_output": _normalize_path(payload.get("final_output")),
        }
        _call_workflow("run_typhoon_workflow", args)

        return _success(
            "台风场景全流程生成已启动",
            requested_args={k: v for k, v in args.items() if v is not None},
        )
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/typhoon-workflow", methods=["POST"])
def typhoon_workflow_route():
    """
    执行台风工作流
    
    请求格式（支持两种方式）：
    
    方式1：直接上传文件并执行（推荐）
    POST /api/typhoon-workflow
    Content-Type: multipart/form-data
    file: <TowerSeg.xlsx>
    command: --one-click
    
    方式2：使用已上传文件的路径
    POST /api/typhoon-workflow
    Content-Type: application/json
    {
        "command": ["--one-click", "--tower-excel", "/path/to/upload_xxx.xlsx"]
    }
    """
    try:
        # 处理文件上传和命令混合的情况
        command = None
        uploaded_file_path = None
        
        # 检查是否有文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # 上传文件
                file_ext = Path(file.filename).suffix.lower()
                allowed_extensions = {'.xlsx', '.xls', '.csv', '.json', '.txt', '.dat'}
                if file_ext not in allowed_extensions:
                    return _error(f"不支持的文件类型: {file_ext}", status=400)
                
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                max_size = 100 * 1024 * 1024
                if file_size > max_size:
                    return _error(f"文件过大，最大允许 {max_size // (1024*1024)}MB", status=400)
                
                temp_name = f"upload_{uuid.uuid4().hex}{file_ext}"
                temp_path = TEMP_UPLOAD_DIR / temp_name
                file.save(str(temp_path))
                uploaded_file_path = str(temp_path)
                print(f"[typhoon-workflow] 文件已上传: {file.filename} -> {uploaded_file_path}")
        
        # 解析命令
        if request.is_json:
            payload = _extract_payload()
            command = _parse_command(payload.get("command"))
        else:
            # 从表单数据获取命令
            command_str = request.form.get('command', '')
            command = _parse_command(command_str) if command_str else []
        
        if not command:
            raise ValueError("command 字段不能为空")
        
        # 如果上传了文件，自动替换命令中的文件路径
        if uploaded_file_path:
            # 查找命令中的文件路径参数并替换
            new_command = []
            for i, part in enumerate(command):
                if part.startswith('--') and i + 1 < len(command):
                    # 这是一个参数，检查下一个值是否是相对路径
                    next_val = command[i + 1]
                    if not next_val.startswith('/') and not next_val.startswith('http'):
                        # 这是一个相对路径，替换为上传的文件路径
                        new_command.extend([part, uploaded_file_path])
                    else:
                        new_command.append(part)
                else:
                    # 检查这个part本身是否是相对路径
                    if not part.startswith('-') and not part.startswith('/') and not part.startswith('http'):
                        if i > 0 and command[i-1].startswith('--'):
                            # 这是上一个参数的值，如果还没处理过
                            if not new_command or new_command[-1] != command[i-1]:
                                new_command.append(part)
                        elif '.' in part:
                            # 可能是文件路径，但不确定是哪个参数
                            new_command.append(part)
                        else:
                            new_command.append(part)
                    else:
                        new_command.append(part)
            command = new_command
        
        # 执行工作流
        print(f"[typhoon-workflow] 执行命令: {command}")
        try:
            typhoon_cli.main(command)
        except SystemExit as exc:  # 捕获 argparse 内部的 sys.exit
            if exc.code not in (None, 0):
                raise RuntimeError(f"台风工作流提前退出: {exc.code}") from exc
        
        return _success("台风工作流已执行", command=command, uploaded_file=uploaded_file_path)
    except Exception as exc:  # pylint: disable=broad-except
        import traceback
        error_detail = traceback.format_exc()
        print(f"[typhoon-workflow] 错误: {error_detail}")
        return _error(str(exc))


@app.route("/api/health", methods=["GET"])
def health_route():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    debug = os.environ.get("API_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)