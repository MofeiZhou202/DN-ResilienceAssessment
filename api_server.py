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
    return func(**kwargs)


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
            },
        )
        return _success("完整流程已成功执行")
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/typhoon-workflow", methods=["POST"])
def typhoon_workflow_route():
    try:
        payload = _extract_payload()
        command = _parse_command(payload.get("command"))
        if not command:
            raise ValueError("command 字段不能为空")
        typhoon_cli.main(command)
        return _success("台风工作流已执行", command=command)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/health", methods=["GET"])
def health_route():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    debug = os.environ.get("API_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, threaded=False, use_reloader=False)
