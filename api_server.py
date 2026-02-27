"""Flask API 封装，用于在 Coze 等平台以 HTTP 方式触发现有工作流。"""
from __future__ import annotations

# 禁用Tkinter，避免与Julia GC冲突导致内存访问违规
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import shlex
import re
import shutil
import subprocess
import sys
import uuid
import tempfile
import time
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not _VENV_PYTHON.exists():
    _VENV_PYTHON = Path("C:/Python311/python.exe")
if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

# 强制无缓冲输出，确保所有print立即显示 (放在os.execv之后)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'


def _ensure_julia_dll_search_path() -> None:
    """确保能够找到 Julia 打包的 DLL（如 libamd.dll）。"""

    def _probe_bindir(executable: Path) -> Optional[Path]:
        try:
            output = subprocess.check_output(
                [str(executable), "--startup-file=no", "-e", "print(Sys.BINDIR)"],
                timeout=30,
                text=True,
            ).strip()
            candidate = Path(output)
            if candidate.is_dir():
                return candidate
        except Exception:
            resolved = executable.resolve().parent
            if resolved.is_dir():
                return resolved
        return None

    candidate = os.environ.get("JULIA_BINDIR")
    path_obj = Path(candidate) if candidate else None
    if not path_obj or not path_obj.is_dir():
        julia_exe = shutil.which("julia")
        path_obj = _probe_bindir(Path(julia_exe)) if julia_exe else None
    if not path_obj or not path_obj.is_dir():
        juliaup_root = Path.home() / ".julia" / "juliaup"
        for option in juliaup_root.glob("julia-*/bin"):
            if option.is_dir():
                path_obj = option.resolve()
                break
    if not path_obj or not path_obj.is_dir():
        return

    bindir = str(path_obj)
    os.environ["JULIA_BINDIR"] = bindir
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(bindir)
        except OSError:
            pass
    existing_path = os.environ.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if bindir not in path_entries:
        os.environ["PATH"] = bindir if not existing_path else f"{bindir}{os.pathsep}{existing_path}"


_ensure_julia_dll_search_path()

from flask import Flask, jsonify, request

import app as typhoon_cli
import requests
from validate_inference import load_data
from run_ml_inference import extract_features_for_all_lines, infer_loss_improvement_ml
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp" / "api_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# 确保 Julia 使用项目内的环境
os.environ.setdefault("JULIA_PROJECT", str(PROJECT_ROOT))

DEFAULT_TYPHOON_WORKBOOK = Path(
    r"D:\DistributionPowerFlow-runze\ExtremeScenarioGeneration-DN\Topology_Reconfiguration\Topology_Reconfiguration\DN-ResilienceAssessment\data\05_STY.xlsx"
)

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
JULIA_LOCK = threading.Lock()

# ==================== 任务状态管理（异步模式）====================
TASK_STATUS = {}  # task_id -> {"status": "pending"|"running"|"completed"|"failed", "result": {...}, "error": "...}
TASK_LOCK = threading.Lock()

def _create_task(task_type: str, **metadata) -> str:
    """创建新任务并返回任务ID"""
    task_id = f"{task_type}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    with TASK_LOCK:
        TASK_STATUS[task_id] = {
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "completed_at": None,
            "task_type": task_type,
            "metadata": metadata,
            "result": None,
            "error": None
        }
    return task_id

def _update_task_status(task_id: str, status: str, **kwargs):
    """更新任务状态"""
    with TASK_LOCK:
        if task_id in TASK_STATUS:
            TASK_STATUS[task_id]["status"] = status
            if kwargs:
                TASK_STATUS[task_id].update(kwargs)
            if status == "running" and TASK_STATUS[task_id]["started_at"] is None:
                TASK_STATUS[task_id]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            elif status in ("completed", "failed"):
                TASK_STATUS[task_id]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

def _run_task_in_background(task_id: str, func, *args, **kwargs):
    """在后台线程中执行任务"""
    try:
        _update_task_status(task_id, "running")
        result = func(*args, **kwargs)
        _update_task_status(task_id, "completed", result=result)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        _update_task_status(task_id, "failed", error=str(e), error_detail=error_detail)
# ====================================================================

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


def _save_uploaded_file(upload_field: str) -> Optional[Path]:
    file_obj = request.files.get(upload_field)
    if file_obj is None or not file_obj.filename:
        return None
    suffix = _infer_suffix(file_obj.filename)
    safe_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix) else ".dat"
    temp_name = f"upload_{uuid.uuid4().hex}{safe_suffix}"
    target = TEMP_UPLOAD_DIR / temp_name
    file_obj.save(target)
    return target


def _call_workflow_direct(julia_code: str) -> None:
    """
    直接执行Julia代码，不返回任何值。
    这避免了PyJulia在处理大型返回值（如DataFrame）时可能出现的问题。
    """
    import sys
    print(f"[DEBUG] 直接执行Julia代码...")
    sys.stdout.flush()
    
    with JULIA_LOCK:
        try:
            from julia import Main
            # 使用eval执行代码，结果赋值给_discard变量并返回nothing
            Main.eval(f"begin; {julia_code}; nothing; end")
            print(f"[DEBUG] Julia代码执行完成")
            sys.stdout.flush()
        except Exception as e:
            print(f"[DEBUG] Julia代码执行出错: {type(e).__name__}: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise


def _call_julia_subprocess(julia_code: str, timeout: int = 3600) -> None:
    """
    使用子进程执行Julia代码，完全绕过PyJulia的问题。
    这是最可靠的方式，类似于main.jl直接运行Julia。
    
    Args:
        julia_code: 要执行的Julia代码字符串
        timeout: 超时时间（秒），默认1小时
    """
    import subprocess
    import sys
    
    print(f"[DEBUG] 使用子进程执行Julia代码...")
    sys.stdout.flush()
    
    # 构建完整的Julia脚本 - 使用绝对路径以避免路径解析问题
    project_root_julia = str(PROJECT_ROOT).replace(chr(92), '/')
    workflows_path = f"{project_root_julia}/src/workflows.jl"
    
    full_script = f'''
# 设置项目环境
cd("{project_root_julia}")
import Pkg
Pkg.activate(".")

# 加载Workflows模块 - 使用绝对路径
include("{workflows_path}")
using .Workflows

# 执行用户代码
{julia_code}

println("[JULIA_SUBPROCESS] 执行完成")
'''
    
    try:
        # 创建临时Julia脚本文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False, encoding='utf-8') as f:
            f.write(full_script)
            temp_script = f.name
        
        print(f"[DEBUG] 临时脚本: {temp_script}")
        sys.stdout.flush()
        
        # 执行Julia脚本
        result = subprocess.run(
            ['julia', '--project=.', temp_script],
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # 让输出直接显示在终端
            timeout=timeout,
            check=True
        )
        
        print(f"[DEBUG] Julia子进程执行完成，返回码: {result.returncode}")
        sys.stdout.flush()
        
    except subprocess.TimeoutExpired:
        print(f"[DEBUG] Julia子进程超时 ({timeout}秒)")
        sys.stdout.flush()
        raise RuntimeError(f"Julia执行超时（{timeout}秒）")
    except subprocess.CalledProcessError as e:
        print(f"[DEBUG] Julia子进程执行失败，返回码: {e.returncode}")
        sys.stdout.flush()
        raise RuntimeError(f"Julia执行失败: {e}")
    except Exception as e:
        print(f"[DEBUG] Julia子进程出错: {type(e).__name__}: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_script)
        except:
            pass


def _call_workflow(func_name: str, arguments: Dict[str, Any]) -> Any:
    import sys
    kwargs = {key: value for key, value in arguments.items() if value is not None}
    print(f"[DEBUG] 准备调用工作流: {func_name}")
    print(f"[DEBUG] 参数: {kwargs}")
    sys.stdout.flush()
    
    with JULIA_LOCK:  # 防止 PyJulia 并发调用导致内存访问错误
        print(f"[DEBUG] 已获取JULIA_LOCK，正在访问workflows模块...")
        sys.stdout.flush()
        
        try:
            workflows = runtime.workflows
            print(f"[DEBUG] 已获取workflows模块，正在获取函数 {func_name}...")
            sys.stdout.flush()
            
            func = getattr(workflows, func_name)
            print(f"[DEBUG] 已获取函数 {func_name}，正在执行...")
            sys.stdout.flush()
            
            # 执行Julia函数，不保留返回值避免PyJulia内存问题
            _ = func(**kwargs)
            print(f"[DEBUG] 函数 {func_name} 执行完成，返回结果已丢弃以避免内存问题")
            sys.stdout.flush()
            return None  # 返回None而不是Julia对象
        except SystemExit as exc:  # noqa: PERF203 -- 防止工作流意外退出服务器
            print(f"[DEBUG] 工作流 {func_name} 发生SystemExit: {exc.code}")
            sys.stdout.flush()
            if exc.code not in (None, 0):
                raise RuntimeError(f"工作流 {func_name} 提前退出: {exc.code}") from exc
            return None
        except Exception as e:
            print(f"[DEBUG] 工作流 {func_name} 执行出错: {type(e).__name__}: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise


def _build_line_display_name_map(topo_features: Dict[str, Any]) -> Dict[str, str]:
    """根据拓扑信息构建线路ID到真实名称(起止母线)的映射。"""
    network = topo_features.get("_network", {}) if isinstance(topo_features, dict) else {}
    all_line_info = network.get("all_line_info", {}) if isinstance(network, dict) else {}
    mapping: Dict[str, str] = {}
    for info in all_line_info.values():
        if not isinstance(info, dict):
            continue
        line_id = str(info.get("line_id", "")).strip()
        from_bus = str(info.get("from", "")).strip()
        to_bus = str(info.get("to", "")).strip()
        if not line_id or not from_bus or not to_bus:
            continue
        mapping[line_id] = f"{from_bus} - {to_bus}"
    return mapping


def _copy_to_data_if_needed(src: Path, dst: Path) -> None:
    """将文件复制到 data 目录目标路径；若源目标相同则跳过。"""
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_resolved, dst_resolved)


def _sync_dn_resilience_outputs_to_data(
    *,
    tower_seg_path: Path,
    case_path: Path,
    cluster_output: Path,
    phase_output: Path,
    topology_output: Path,
    dispatch_output: Path,
) -> Dict[str, str]:
    """将 DN-RESILIENCE 本次结果同步到 data/ 固定文件名，供后续推理层直接读取。"""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sync_targets = {
        case_path: data_dir / "ac_dc_real_case.xlsx",
        tower_seg_path: data_dir / "TowerSeg.xlsx",
        cluster_output: data_dir / "mc_simulation_results_k100_clusters.xlsx",
        phase_output: data_dir / "scenario_phase_classification.xlsx",
        topology_output: data_dir / "topology_reconfiguration_results.xlsx",
        dispatch_output: data_dir / "mess_dispatch_results.xlsx",
    }

    synced: Dict[str, str] = {}
    for src, dst in sync_targets.items():
        if src.exists():
            _copy_to_data_if_needed(src, dst)
            synced[dst.name] = str(dst)

    # 统一同步关键指标，兼容不同读取逻辑
    src_km = dispatch_output.with_name(dispatch_output.stem + "_key_metrics.json")
    if src_km.exists():
        dst_km_results = data_dir / "mess_dispatch_results_key_metrics.json"
        dst_km_report = data_dir / "mess_dispatch_report_key_metrics.json"
        _copy_to_data_if_needed(src_km, dst_km_results)
        _copy_to_data_if_needed(src_km, dst_km_report)
        synced[dst_km_results.name] = str(dst_km_results)
        synced[dst_km_report.name] = str(dst_km_report)

    return synced


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

        classify_input_path = classify_args.get("input_path") or str(DEFAULT_TYPHOON_WORKBOOK)
        _call_workflow(
            "run_classify_phases",
            {
                "input_path": _normalize_path(classify_input_path),
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
        import traceback
        error_msg = str(exc) or "未知错误"
        error_detail = traceback.format_exc()
        print(f"[full-pipeline] 错误: {error_detail}")
        return _error(error_msg)


@app.route("/api/typhoon-workflow", methods=["POST"])
def typhoon_workflow_route():
    try:
        payload = _extract_payload()
        command = _parse_command(payload.get("command"))
        if not command:
            raise ValueError("command 字段不能为空")
        try:
            typhoon_cli.main(command)
        except SystemExit as exc:  # 捕获 argparse 内部的 sys.exit
            if exc.code not in (None, 0):
                raise RuntimeError(f"台风工作流提前退出: {exc.code}") from exc
        return _success("台风工作流已执行", command=command)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(str(exc))


@app.route("/api/cluster-analysis", methods=["POST"])
def cluster_analysis_route():
    """
    执行聚类分析（使用仓库中main.jl的one-click流程）
    接收TowerSeg.xlsx文件，执行完整的台风评估流程并返回聚类结果
    """
    try:
        # 1. 获取上传的文件
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '未提供文件'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '文件名为空'
            }), 400
        
        # 2. 验证文件格式
        allowed_extensions = {'.xlsx', '.xls', '.csv'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'status': 'error',
                'message': f'不支持的文件格式: {file_ext}，请上传 .xlsx, .xls 或 .csv 文件'
            }), 400
        
        # 3. 保存上传的文件到临时目录
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        # 设置输出文件路径
        output_file_path = os.path.join(temp_dir, "mc_simulation_results_k100_clusters.xlsx")
        
        print(f"[cluster-analysis] 收到文件: {file.filename}")
        print(f"[cluster-analysis] 保存路径: {temp_file_path}")
        print(f"[cluster-analysis] 输出路径: {output_file_path}")
        
        # 4. 调用Python的main函数执行one-click流程
        start_time = time.time()
        
        try:
            # 切换到项目根目录
            original_cwd = os.getcwd()
            os.chdir(str(PROJECT_ROOT))
            
            # 调用main函数执行one-click流程
            typhoon_cli.main([
                "one-click",
                "--tower-excel", temp_file_path,
                "--final-output", output_file_path
            ])
            
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            execution_time = time.time() - start_time
            
            print(f"[cluster-analysis] one-click流程完成，耗时: {execution_time:.2f}秒")
            
            # 5. 检查输出文件是否存在
            if not os.path.exists(output_file_path):
                return jsonify({
                    'status': 'error',
                    'message': '聚类分析完成，但输出文件未生成'
                }), 500
            
            # 6. 构造响应
            return jsonify({
                'status': 'success',
                'message': '聚类分析完成（使用main.jl的one-click流程）',
                'uploaded_file': temp_file_path,
                'output_file': output_file_path,
                'execution_time': f'{execution_time:.2f}s'
            })
            
        except Exception as workflow_error:
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            import traceback
            workflow_error_detail = traceback.format_exc()
            print(f"[cluster-analysis] one-click流程执行错误: {workflow_error_detail}")
            
            return jsonify({
                'status': 'error',
                'message': f'one-click流程执行失败: {str(workflow_error)}',
                'error_detail': workflow_error_detail
            }), 500
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[cluster-analysis] 服务器错误: {error_detail}")
        
        return jsonify({
            'status': 'error',
            'message': f'聚类分析失败: {str(e)}',
            'error_detail': error_detail
        }), 500



@app.route("/api/health", methods=["GET"])
def health_route():
    return jsonify({"status": "ok"})


# ==================== DN-RESILIENCE 核心执行函数 ====================
def _execute_dn_resilience(tower_seg_path, case_path, output_dir=None):
    """
    执行DN-RESILIENCE配电网弹性评估流程的核心逻辑
    
    返回:
        dict: 执行结果，包含输出文件路径和各阶段耗时
    """
    start_time = time.time()
    
    tower_seg_path = Path(tower_seg_path)
    case_path = Path(case_path)
    
    # 设置输出目录
    if output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "data" / "auto_eval_runs" / f"dn_resilience_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义输出文件路径
    cluster_output = output_dir / "mc_simulation_results_k100_clusters.xlsx"
    phase_output = output_dir / "scenario_phase_classification.xlsx"
    topology_output = output_dir / "topology_reconfiguration_results.xlsx"
    dispatch_output = output_dir / "mess_dispatch_results.xlsx"
    
    print(f"\n{'='*60}")
    print(f"[DN-RESILIENCE] 开始配电网弹性评估流程")
    print(f"{'='*60}")
    print(f"[DN-RESILIENCE] TowerSeg文件: {tower_seg_path}")
    print(f"[DN-RESILIENCE] 算例文件: {case_path}")
    print(f"[DN-RESILIENCE] 输出目录: {output_dir}")
    sys.stdout.flush()
    
    # 预初始化Julia运行时
    print(f"\n[DN-RESILIENCE] 预初始化Julia运行时...")
    with JULIA_LOCK:
        runtime.ensure()
    print(f"[DN-RESILIENCE] Julia运行时初始化完成")
    sys.stdout.flush()
    
    # Step 1: 台风场景生成
    step1_start = time.time()
    print(f"\n[Step 1/4] 执行台风场景生成...")
    sys.stdout.flush()
    
    original_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT))
        typhoon_cli.main([
            "one-click",
            "--tower-excel", str(tower_seg_path),
            "--final-output", str(cluster_output)
        ])
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(f"台风场景生成失败 (exit code: {exc.code})") from exc
    finally:
        os.chdir(original_cwd)
    
    step1_time = time.time() - step1_start
    print(f"[Step 1/4] ✓ 完成，耗时: {step1_time:.2f}秒")
    sys.stdout.flush()
    
    if not cluster_output.exists():
        raise RuntimeError("台风场景生成失败：聚类结果文件未生成")
    
    import gc
    gc.collect()
    
    # Step 2: 场景阶段分类
    step2_start = time.time()
    print(f"\n[Step 2/4] 执行场景阶段分类...")
    sys.stdout.flush()
    
    cluster_output_escaped = str(cluster_output).replace("\\", "/")
    phase_output_escaped = str(phase_output).replace("\\", "/")
    
    _call_julia_subprocess(f'''
run_classify_phases(
    input_path = "{cluster_output_escaped}",
    output_path = "{phase_output_escaped}"
)
    ''')
    step2_time = time.time() - step2_start
    print(f"[Step 2/4] ✓ 完成，耗时: {step2_time:.2f}秒")
    sys.stdout.flush()
    
    case_path_escaped = str(case_path).replace("\\", "/")
    topology_output_escaped = str(topology_output).replace("\\", "/")
    dispatch_output_escaped = str(dispatch_output).replace("\\", "/")
    
    # Step 3: 滚动拓扑重构
    step3_start = time.time()
    print(f"\n[Step 3/4] 执行滚动拓扑重构...")
    sys.stdout.flush()
    
    _call_julia_subprocess(f'''
run_rolling_reconfiguration(
    case_file = "{case_path_escaped}",
    fault_file = "{cluster_output_escaped}",
    stage_file = "{phase_output_escaped}",
    output_file = "{topology_output_escaped}"
)
    ''')
    step3_time = time.time() - step3_start
    print(f"[Step 3/4] ✓ 完成，耗时: {step3_time:.2f}秒")
    sys.stdout.flush()
    
    gc.collect()
    
    # Step 4: MESS协同调度
    step4_start = time.time()
    print(f"\n[Step 4/4] 执行MESS协同调度 (移动储能求解)...")
    sys.stdout.flush()
    
    # 使用子进程方式调用（与其他步骤一致，避免内存访问违规）
    _call_julia_subprocess(f'''
run_mess_dispatch_julia(
    case_path = "{case_path_escaped}",
    topology_path = "{topology_output_escaped}",
    fallback_topology = "{cluster_output_escaped}",
    output_file = "{dispatch_output_escaped}"
)
    ''')
    
    # 从临时JSON文件读取key_metrics
    key_metrics_file = str(dispatch_output).replace(".xlsx", "_key_metrics.json")
    import json
    if os.path.exists(key_metrics_file):
        print(f"[DEBUG] 找到key_metrics文件: {key_metrics_file}")
        with open(key_metrics_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[DEBUG] JSON文件内容: {content[:500]}...")  # 打印前500字符
            key_metrics = json.loads(content)
        print(f"[DEBUG] 已从 {key_metrics_file} 读取关键指标")
        print(f"[DEBUG] expected_load_shed_total: {key_metrics.get('expected_load_shed_total')}")
        print(f"[DEBUG] expected_supply_ratio: {key_metrics.get('expected_supply_ratio')}")
        print(f"[DEBUG] objective_value: {key_metrics.get('objective_value')}")
        print(f"[DEBUG] violations count: {len(key_metrics.get('violations', []))}")
    else:
        print(f"[WARNING] 未找到关键指标文件 {key_metrics_file}，使用默认值")
        key_metrics = {
            "expected_load_shed_total": 0.0,
            "expected_supply_ratio": 1.0,
            "objective_value": 0.0,
            "violations": []
        }
    
    step4_time = time.time() - step4_start
    print(f"[Step 4/4] ✓ 完成，耗时: {step4_time:.2f}秒")
    sys.stdout.flush()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"[DN-RESILIENCE] ✓ 配电网弹性评估流程完成")
    print(f"[DN-RESILIENCE] 总耗时: {total_time:.2f}秒")
    print(f"[DN-RESILIENCE] 移动储能求解结果: {dispatch_output}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    if not dispatch_output.exists():
        raise RuntimeError("MESS调度完成，但移动储能求解结果文件未生成")

    # 关键：每次执行后同步到 data/ 固定文件名，确保推理层直接使用最新文件
    synced_data_files = _sync_dn_resilience_outputs_to_data(
        tower_seg_path=tower_seg_path,
        case_path=case_path,
        cluster_output=cluster_output,
        phase_output=phase_output,
        topology_output=topology_output,
        dispatch_output=dispatch_output,
    )
    
    return {
        "status": "success",
        "message": "DN-RESILIENCE 配电网弹性评估完成",
        "input_files": {
            "tower_seg_file": str(tower_seg_path),
            "case_file": str(case_path),
        },
        "output_files": {
            "cluster_output": str(cluster_output),
            "phase_output": str(phase_output),
            "topology_output": str(topology_output),
            "dispatch_output": str(dispatch_output),
        },
        "synced_data_files": synced_data_files,
        "execution_time": {
            "step1_typhoon_generation": f"{step1_time:.2f}s",
            "step2_phase_classification": f"{step2_time:.2f}s",
            "step3_topology_reconfig": f"{step3_time:.2f}s",
            "step4_mess_dispatch": f"{step4_time:.2f}s",
            "total": f"{total_time:.2f}s",
        },
        "dispatch_result_path": str(dispatch_output),
        "key_metrics": {
            "expected_load_shed_total": key_metrics.get("expected_load_shed_total"),
            "expected_supply_ratio": key_metrics.get("expected_supply_ratio"),
            "objective_value": key_metrics.get("objective_value"),
            "violations": key_metrics.get("violations", []),
        }
    }
# ======================================================================


@app.route("/api/DN-RESILIENCE", methods=["POST"])
def dn_resilience_route():
    """
    DN-RESILIENCE 配电网弹性评估API (异步模式)
    
    用户需提供两个Excel文件：
    1. tower_seg_file: TowerSeg.xlsx - 配电网塔杆分段结构文件
    2. case_file: ac_dc_real_case.xlsx - 混合交直流配电网算例文件
    
    执行流程（异步执行）：
    Step 1: 台风场景生成 - 生成 mc_simulation_results_k100_clusters.xlsx
    Step 2: 场景阶段分类 - 生成 scenario_phase_classification.xlsx
    Step 3: 滚动拓扑重构 - 生成 topology_reconfiguration_results.xlsx
    Step 4: MESS协同调度 - 生成 mess_dispatch_results.xlsx (最终移动储能求解结果)
    
    输入方式：
    1. 文件上传模式 (multipart/form-data):
       - tower_seg_file: 上传的TowerSeg.xlsx文件
       - case_file: 上传的ac_dc_real_case.xlsx文件
       - output_dir: (可选) 输出目录路径
    
    2. JSON参数模式 (application/json):
       {
         "tower_seg_file": "TowerSeg.xlsx文件路径或URL",
         "case_file": "ac_dc_real_case.xlsx文件路径或URL",
         "output_dir": "(可选) 输出目录路径"
       }
    
    返回（异步）：
    {
        "status": "started",
        "task_id": "DN-RESILIENCE_20260121_140316_abc12345",
        "message": "任务已启动，请使用 /api/TASK/{task_id} 查询进度"
    }
    
    查询任务状态：
    GET /api/TASK/{task_id}
    返回：
    {
        "status": "running" | "completed" | "failed",
        "result": {...},  // 仅当status为completed时
        "error": "..."    // 仅当status为failed时
    }
    """
    try:
        # 1. 处理输入文件
        tower_seg_path = None
        case_path = None
        output_dir = None
        
        # 检查是否有文件上传 (multipart/form-data)
        if 'tower_seg_file' in request.files and 'case_file' in request.files:
            # 文件上传模式
            tower_seg_file = request.files['tower_seg_file']
            case_file = request.files['case_file']
            
            if tower_seg_file.filename == '' or case_file.filename == '':
                return _error("文件名为空，请确保两个Excel文件都已正确上传")
            
            # 验证文件格式
            for f, name in [(tower_seg_file, "tower_seg_file"), (case_file, "case_file")]:
                ext = os.path.splitext(f.filename)[1].lower()
                if ext not in {'.xlsx', '.xls'}:
                    return _error(f"{name} 必须是Excel文件 (.xlsx 或 .xls)，当前格式: {ext}")
            
            # 保存上传的文件
            tower_seg_path = TEMP_UPLOAD_DIR / f"dn_resilience_{uuid.uuid4().hex}_TowerSeg.xlsx"
            case_path = TEMP_UPLOAD_DIR / f"dn_resilience_{uuid.uuid4().hex}_ac_dc_real_case.xlsx"
            tower_seg_file.save(str(tower_seg_path))
            case_file.save(str(case_path))
            
            # 获取可选的输出目录
            output_dir = request.form.get('output_dir')
        else:
            # JSON参数模式
            payload = _extract_payload()
            tower_seg_path = _normalize_path(payload.get("tower_seg_file"))
            case_path = _normalize_path(payload.get("case_file"))
            output_dir = _normalize_path(payload.get("output_dir"))
        
        # 验证必要参数
        if not tower_seg_path or not case_path:
            return _error(
                "必须提供两个Excel文件:\n"
                "  1. tower_seg_file: TowerSeg.xlsx (配电网塔杆分段结构文件)\n"
                "  2. case_file: ac_dc_real_case.xlsx (混合交直流配电网算例文件)"
            )
        
        tower_seg_path = Path(tower_seg_path)
        case_path = Path(case_path)
        
        # 检查文件是否存在
        if not tower_seg_path.exists():
            return _error(f"TowerSeg文件不存在: {tower_seg_path}")
        if not case_path.exists():
            return _error(f"算例文件不存在: {case_path}")
        
        # 2. 创建任务
        task_id = _create_task("DN-RESILIENCE", 
                              tower_seg_file=str(tower_seg_path),
                              case_file=str(case_path),
                              output_dir=output_dir)
        
        # 3. 在后台线程中执行任务
        thread = threading.Thread(
            target=_run_task_in_background,
            args=(task_id, _execute_dn_resilience),
            kwargs={"tower_seg_path": str(tower_seg_path), 
                    "case_path": str(case_path), 
                    "output_dir": output_dir}
        )
        thread.daemon = True
        thread.start()
        
        # 4. 立即返回任务ID
        return jsonify({
            "status": "started",
            "task_id": task_id,
            "message": f"任务已启动，请使用 GET /api/TASK/{task_id} 查询进度",
            "query_url": f"/api/TASK/{task_id}"
        })
        
    except Exception as exc:
        import traceback
        error_detail = traceback.format_exc()
        return jsonify({
            "status": "error",
            "message": f"任务创建失败: {str(exc)}",
            "error_detail": error_detail,
        }), 500


@app.route("/api/TASK/<task_id>", methods=["GET"])
def task_status_route(task_id: str):
    """
    查询任务状态
    
    返回：
    {
        "status": "pending" | "running" | "completed" | "failed",
        "task_type": "DN-RESILIENCE",
        "created_at": "2026-01-21 14:03:16",
        "started_at": "2026-01-21 14:03:16",
        "completed_at": "2026-01-21 14:13:41",  // 仅当status为completed或failed时
        "result": {...},  // 仅当status为completed时
        "error": "...",   // 仅当status为failed时
        "metadata": {...}
    }
    """
    with TASK_LOCK:
        if task_id not in TASK_STATUS:
            return _error(f"任务不存在: {task_id}", status=404)
        
        task_info = TASK_STATUS[task_id].copy()
        
        # 移除敏感信息
        if "error_detail" in task_info:
            del task_info["error_detail"]
        
        return jsonify(task_info)


@app.route("/api/resilience-assessment", methods=["POST"])
def resilience_assessment_route():
    """
    完整弹性评估流程API
    
    用户只需提供两个文件：
    1. TowerSeg.xlsx - 配电网塔杆分段结构文件
    2. ac_dc_real_case.xlsx - 混合交直流配电网算例文件
    
    执行流程：
    Step 1: 台风场景生成 (功能4) - 生成 mc_simulation_results_k100_clusters.xlsx
    Step 2: 场景阶段分类 - 生成 scenario_phase_classification.xlsx
    Step 3: 滚动拓扑重构 - 生成 topology_reconfiguration_results.xlsx
    Step 4: MESS协同调度 - 生成 mess_dispatch_results.xlsx
    
    输入参数 (JSON body 或 form-data):
    - tower_seg_file: TowerSeg.xlsx文件路径或上传的文件
    - case_file: ac_dc_real_case.xlsx文件路径或上传的文件
    - output_dir: (可选) 输出目录路径
    
    返回：
    - mess_dispatch_results.xlsx 的路径及各中间文件路径
    """
    try:
        start_time = time.time()
        
        # 1. 处理输入文件
        tower_seg_path = None
        case_path = None
        output_dir = None
        
        # 检查是否有文件上传
        if 'tower_seg_file' in request.files and 'case_file' in request.files:
            # 文件上传模式
            tower_seg_file = request.files['tower_seg_file']
            case_file = request.files['case_file']
            
            if tower_seg_file.filename == '' or case_file.filename == '':
                return _error("文件名为空")
            
            # 保存上传的文件
            tower_seg_path = TEMP_UPLOAD_DIR / f"upload_{uuid.uuid4().hex}_TowerSeg.xlsx"
            case_path = TEMP_UPLOAD_DIR / f"upload_{uuid.uuid4().hex}_ac_dc_real_case.xlsx"
            tower_seg_file.save(str(tower_seg_path))
            case_file.save(str(case_path))
            
            # 获取可选的输出目录
            output_dir = request.form.get('output_dir')
        else:
            # JSON参数模式
            payload = _extract_payload()
            tower_seg_path = _normalize_path(payload.get("tower_seg_file"))
            case_path = _normalize_path(payload.get("case_file"))
            output_dir = _normalize_path(payload.get("output_dir"))
        
        if not tower_seg_path or not case_path:
            return _error("必须提供 tower_seg_file (TowerSeg.xlsx) 和 case_file (ac_dc_real_case.xlsx)")
        
        tower_seg_path = Path(tower_seg_path)
        case_path = Path(case_path)
        
        if not tower_seg_path.exists():
            return _error(f"TowerSeg文件不存在: {tower_seg_path}")
        if not case_path.exists():
            return _error(f"算例文件不存在: {case_path}")
        
        # 设置输出路径
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = PROJECT_ROOT / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cluster_output = output_dir / "mc_simulation_results_k100_clusters.xlsx"
        phase_output = output_dir / "scenario_phase_classification.xlsx"
        topology_output = output_dir / "topology_reconfiguration_results.xlsx"
        dispatch_output = output_dir / "mess_dispatch_results.xlsx"
        
        print(f"[resilience-assessment] 开始完整弹性评估流程")
        print(f"[resilience-assessment] TowerSeg: {tower_seg_path}")
        print(f"[resilience-assessment] Case: {case_path}")
        print(f"[resilience-assessment] 输出目录: {output_dir}")
        
        # ===========================================================
        # 预初始化Julia运行时（关键！必须在Step 1之前初始化）
        # ===========================================================
        print(f"\n[resilience-assessment] 预初始化Julia运行时...")
        with JULIA_LOCK:
            runtime.ensure()
        print(f"[resilience-assessment] Julia运行时初始化完成")
        
        # ===========================================================
        # Step 1: 台风场景生成 (功能4)
        # ===========================================================
        step1_start = time.time()
        print(f"\n[Step 1/4] 执行台风场景生成...")
        
        original_cwd = os.getcwd()
        try:
            os.chdir(str(PROJECT_ROOT))
            typhoon_cli.main([
                "one-click",
                "--tower-excel", str(tower_seg_path),
                "--final-output", str(cluster_output)
            ])
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError(f"台风场景生成失败: {exc.code}") from exc
        finally:
            os.chdir(original_cwd)
        
        step1_time = time.time() - step1_start
        print(f"[Step 1/4] 完成，耗时: {step1_time:.2f}秒")
        
        if not cluster_output.exists():
            return _error("台风场景生成失败：聚类结果文件未生成")
        
        # 强制Python GC以清理Step 1产生的大量临时对象
        import gc
        gc.collect()
        
        # ===========================================================
        # Step 2-4: 执行 full-pipeline (功能5)
        # ===========================================================
        print(f"\n[Step 2/4] 执行场景阶段分类...")
        step2_start = time.time()
        _call_workflow(
            "run_classify_phases",
            {
                "input_path": str(cluster_output),
                "output_path": str(phase_output),
            },
        )
        step2_time = time.time() - step2_start
        print(f"[Step 2/4] 完成，耗时: {step2_time:.2f}秒")
        
        print(f"\n[Step 3/4] 执行滚动拓扑重构...")
        step3_start = time.time()
        _call_workflow(
            "run_rolling_reconfig",
            {
                "case_file": str(case_path),
                "fault_file": str(cluster_output),
                "stage_file": str(phase_output),
                "output_file": str(topology_output),
            },
        )
        step3_time = time.time() - step3_start
        print(f"[Step 3/4] 完成，耗时: {step3_time:.2f}秒")
        
        print(f"\n[Step 4/4] 执行MESS协同调度...")
        step4_start = time.time()
        _call_workflow(
            "run_mess_dispatch",
            {
                "case_path": str(case_path),
                "topology_path": str(topology_output),
                "fallback_topology": str(cluster_output),
                "output_file": str(dispatch_output),
            },
        )
        step4_time = time.time() - step4_start
        print(f"[Step 4/4] 完成，耗时: {step4_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"\n[resilience-assessment] 完整弹性评估流程完成，总耗时: {total_time:.2f}秒")
        
        # 检查最终输出文件
        if not dispatch_output.exists():
            return _error("MESS调度完成，但输出文件未生成")
        
        return jsonify({
            "status": "success",
            "message": "完整弹性评估流程已成功执行",
            "input_files": {
                "tower_seg_file": str(tower_seg_path),
                "case_file": str(case_path),
            },
            "output_files": {
                "cluster_output": str(cluster_output),
                "phase_output": str(phase_output),
                "topology_output": str(topology_output),
                "dispatch_output": str(dispatch_output),
            },
            "execution_time": {
                "step1_typhoon_generation": f"{step1_time:.2f}s",
                "step2_phase_classification": f"{step2_time:.2f}s",
                "step3_topology_reconfig": f"{step3_time:.2f}s",
                "step4_mess_dispatch": f"{step4_time:.2f}s",
                "total": f"{total_time:.2f}s",
            },
            "final_result": str(dispatch_output),
        })
        
    except Exception as exc:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[resilience-assessment] 错误: {error_detail}")
        return jsonify({
            "status": "error",
            "message": f"弹性评估流程执行失败: {str(exc)}",
            "error_detail": error_detail,
        }), 500


# ==================== 推理层 API ====================

@app.route("/api/inference/analyze", methods=["POST"])
def api_inference_analyze():
    """
    推理层API: 执行配电网弹性推理分析
    
    基于蒙特卡洛仿真数据，执行"统计评估 -> 归因诊断 -> 策略推演"的三层分析流程。
    
    请求体参数:
        data_file (str): 输入数据文件路径（Excel/CSV），可以是本地路径或URL
        output_format (str, optional): 输出格式 "json" | "markdown"，默认 "json"
        top_n_diagnosis (int, optional): 诊断时识别的Top N薄弱线路数量，默认5
        top_n_prescriptions (int, optional): 生成策略建议的数量，默认3
        line_prefix (str, optional): 线路状态列前缀，默认 "Line_"
        line_suffix (str, optional): 线路状态列后缀，默认 "_Status"
        target_column (str, optional): 目标列名，默认 "Total_Load_Loss"
        
    返回:
        成功时返回推理分析报告
        失败时返回错误信息
    """
    try:
        # 导入推理模块
        from src.inference import ResilienceInferenceSystem
        
        payload = _extract_payload()
        
        # 获取输入文件路径
        data_file = _normalize_path(payload.get("data_file"))
        if not data_file:
            return jsonify({
                "status": "error",
                "message": "缺少必需参数: data_file"
            }), 400
        
        data_path = Path(data_file)
        if not data_path.exists():
            return jsonify({
                "status": "error",
                "message": f"数据文件不存在: {data_file}"
            }), 400
        
        # 获取可选参数
        output_format = payload.get("output_format", "json")
        top_n_diagnosis = int(payload.get("top_n_diagnosis", 5))
        top_n_prescriptions = int(payload.get("top_n_prescriptions", 3))
        line_prefix = payload.get("line_prefix", "Line_")
        line_suffix = payload.get("line_suffix", "_Status")
        target_column = payload.get("target_column", "Total_Load_Loss")
        supply_rate_column = payload.get("supply_rate_column", "Supply_Rate")
        
        print(f"[inference] 开始推理分析: {data_file}")
        
        # 加载数据并创建推理系统
        if str(data_path).endswith('.csv'):
            system = ResilienceInferenceSystem.from_csv(
                str(data_path),
                line_prefix=line_prefix,
                line_suffix=line_suffix,
                target_column=target_column,
                supply_rate_column=supply_rate_column,
            )
        else:
            system = ResilienceInferenceSystem.from_excel(
                str(data_path),
                line_prefix=line_prefix,
                line_suffix=line_suffix,
                target_column=target_column,
                supply_rate_column=supply_rate_column,
            )
        
        # 执行推理分析
        report = system.run_full_inference(
            top_n_diagnosis=top_n_diagnosis,
            top_n_prescriptions=top_n_prescriptions,
        )
        
        print(f"[inference] 分析完成")
        
        # 根据输出格式返回结果
        if output_format == "markdown":
            return jsonify({
                "status": "success",
                "format": "markdown",
                "report": report.to_markdown(),
            })
        else:
            return jsonify({
                "status": "success",
                "format": "json",
                "report": report.to_dict(),
            })
            
    except ImportError as e:
        return jsonify({
            "status": "error",
            "message": f"推理模块依赖缺失: {str(e)}。请运行: pip install xgboost shap",
        }), 500
    except Exception as exc:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[inference] 错误: {error_detail}")
        return jsonify({
            "status": "error",
            "message": f"推理分析失败: {str(exc)}",
            "error_detail": error_detail,
        }), 500


@app.route("/api/inference/quick-stats", methods=["POST"])
def api_inference_quick_stats():
    """
    推理层API: 快速统计分析（仅执行统计评估模块）
    
    请求体参数:
        data_file (str): 输入数据文件路径
        
    返回:
        统计评估结果
    """
    try:
        from src.inference import ResilienceInferenceSystem
        
        payload = _extract_payload()
        data_file = _normalize_path(payload.get("data_file"))
        
        if not data_file:
            return jsonify({
                "status": "error",
                "message": "缺少必需参数: data_file"
            }), 400
        
        data_path = Path(data_file)
        if not data_path.exists():
            return jsonify({
                "status": "error",
                "message": f"数据文件不存在: {data_file}"
            }), 400
        
        # 加载数据
        if str(data_path).endswith('.csv'):
            system = ResilienceInferenceSystem.from_csv(str(data_path))
        else:
            system = ResilienceInferenceSystem.from_excel(str(data_path))
        
        # 仅执行统计分析
        stats = system.compute_statistics()
        
        return jsonify({
            "status": "success",
            "statistics": {
                "epsr": stats.epsr,
                "mean_load_loss": stats.mean_load_loss,
                "std_load_loss": stats.std_load_loss,
                "high_risk_nodes": stats.high_risk_nodes,
                "node_risk_profile": stats.node_risk_profile,
                "sample_count": stats.sample_count,
            }
        })
        
    except Exception as exc:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"统计分析失败: {str(exc)}",
            "error_detail": traceback.format_exc(),
        }), 500


@app.route("/api/ml-inference/ranking", methods=["POST"])
def api_ml_inference_ranking():
    """
    机器学习推理层API（无监督、失负荷单目标）

    说明:
      - 仅替换配网规划文件 `ac_dc_real_case.xlsx`
      - 其余输入文件沿用 data/ 默认文件
      - 返回加固线路排序和预测失负荷削减比例
      - 不做真实值对比

    输入（JSON或multipart/form-data）:
      - acdc_real_case_file: 本地路径或URL（JSON）
      - acdc_real_case_upload: 直接上传文件（multipart）
      - top_n: 返回前N条（默认35）
    """
    uploaded_temp: Optional[Path] = None
    try:
        payload = _extract_payload() if request.is_json else {}
        top_n = int((payload.get("top_n") if payload else request.form.get("top_n", 35)) or 35)
        top_n = max(1, min(35, top_n))

        # 优先multipart上传，其次JSON路径/URL
        uploaded_temp = _save_uploaded_file("acdc_real_case_upload")
        case_path_str = None
        if uploaded_temp is not None:
            case_path = uploaded_temp.resolve()
        else:
            case_path_str = (payload.get("acdc_real_case_file") if payload else request.form.get("acdc_real_case_file"))
            if not case_path_str:
                return jsonify({
                    "status": "error",
                    "message": "缺少配网规划文件参数，请提供 acdc_real_case_file 或 acdc_real_case_upload",
                }), 400
            resolved = _normalize_path(case_path_str)
            if not resolved:
                return jsonify({
                    "status": "error",
                    "message": "acdc_real_case_file 解析失败",
                }), 400
            case_path = Path(resolved)

        if not case_path.exists():
            return jsonify({
                "status": "error",
                "message": f"配网规划文件不存在: {case_path}",
            }), 400

        if case_path.suffix.lower() not in {".xlsx", ".xls"}:
            return jsonify({
                "status": "error",
                "message": "配网规划文件必须为 Excel（.xlsx/.xls）",
            }), 400

        merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data(case_path_override=case_path)
        features_df = extract_features_for_all_lines(
            merged,
            line_cols,
            baseline,
            topo_features,
            gt_df=None,
            original_preds={},
        )
        pred, confidence, infer_meta = infer_loss_improvement_ml(features_df, baseline, return_meta=True)

        line_names = features_df.index.tolist()
        line_display_map = _build_line_display_name_map(topo_features)
        order = np.argsort(-pred)
        ranking = []
        for rank_i, idx in enumerate(order[:top_n], 1):
            line_id = line_names[idx]
            ranking.append({
                "rank": rank_i,
                "line_name": line_display_map.get(line_id, line_id.replace("_", " ")),
                "predicted_loss_reduction_ratio": float(pred[idx]),
                "predicted_loss_reduction_percent": float(pred[idx] * 100.0),
                "confidence": float(confidence[idx]),
            })

        return jsonify({
            "status": "success",
            "mode": "loss_only_no_supervised_training",
            "input": {
                "acdc_real_case_file": str(case_path),
                "dispatch_file": str(disp_path),
                "top_n": top_n,
            },
            "inference_meta": infer_meta,
            "ranking": ranking,
        })

    except Exception as exc:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"ML推理失败: {str(exc)}",
            "error_detail": traceback.format_exc(),
        }), 500
    finally:
        if uploaded_temp is not None:
            try:
                uploaded_temp.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "5000"))
    debug = os.environ.get("API_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
