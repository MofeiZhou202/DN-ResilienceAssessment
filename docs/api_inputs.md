# Flask API 输入清单

本文档梳理了 `api_server.py` 中每个路由需要的输入文件、主要参数及默认值，方便在 Coze 平台配置插件。

## 通用说明
- 所有路径参数都支持相对路径、绝对路径与 `~`，服务会自动展开并归一化。
- 若某一字段留空（或完全省略），系统会回退到 `src/workflows.jl` 中定义的默认文件。
- 返回结果均为 JSON。成功响应包含 `status`=`"success"` 与 `message`，失败时返回 `status`=`"error"` 及错误描述。

## 1. 场景阶段分类
- **方法**：`POST /api/classify-phases`
- **用途**：调用 `run_classify_phases`，读取 Monte Carlo 场景文件并生成阶段分类结果。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `input_path` | string | 否 | Monte Carlo 结果 Excel，默认 `data/mc_simulation_results_k100_clusters.xlsx`。 |
| `output_path` | string | 否 | 分类结果输出 Excel，默认 `data/scenario_phase_classification.xlsx`。 |
| `lines_per_scenario` | integer | 否 | 每个场景的线路数量，默认 35。 |
| `minutes_per_step` | number | 否 | 时间步长（分钟），默认 60。 |
| `stage2_minutes` | number | 否 | 二阶段累计分钟数，默认 120。 |
| `sheet_name` | string | 否 | 输入表格工作表名称，默认 `cluster_representatives`。 |

**示例请求**：
```json
{
  "input_path": "D:/data/mc_results.xlsx",
  "output_path": "D:/output/stage_classification.xlsx",
  "lines_per_scenario": 40
}
```

## 2. 滚动拓扑重构
- **方法**：`POST /api/rolling-reconfig`
- **用途**：加载电网算例、阶段分类与故障场景，生成拓扑重构方案。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `case_file` | string | 否 | 配电网算例 Excel，默认 `data/ac_dc_real_case.xlsx`。 |
| `fault_file` | string | 否 | Monte Carlo 场景 Excel，默认 `data/mc_simulation_results_k100_clusters.xlsx`。 |
| `stage_file` | string | 否 | 上一步分类结果 Excel，默认 `data/scenario_phase_classification.xlsx`。 |
| `output_file` | string | 否 | 拓扑重构输出 Excel，默认 `data/topology_reconfiguration_results.xlsx`。 |
| `fault_sheet` | string | 否 | 故障场景工作表，默认 `cluster_representatives`。 |
| `stage_sheet` | string | 否 | 阶段分类工作表，默认 `StageDetails`。 |
| `lines_per_scenario` | integer | 否 | 单场景线路数量，默认 35。 |

**示例请求**：
```json
{
  "case_file": "~/cases/ac_dc_case.xlsx",
  "output_file": "~/runs/reconfig.xlsx"
}
```

## 3. MESS 协同调度
- **方法**：`POST /api/mess-dispatch`
- **用途**：运行混合配电网 + 移动储能调度模型。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `case_path` | string | 否 | 配电网算例（与上一步一致），默认 `data/ac_dc_real_case.xlsx`。 |
| `topology_path` | string | 否 | 拓扑重构结果，默认 `data/topology_reconfiguration_results.xlsx`。 |
| `fallback_topology` | string | 否 | 回退的 Monte Carlo 拓扑文件（缺少结果时使用），默认 `data/mc_simulation_results_k100_clusters.xlsx`。 |

> 当前 API 没有暴露自定义 MESS 设备参数，保持与 Julia 中的默认设定一致。

**示例请求**：
```json
{
  "case_path": "D:/cases/ac_dc_real_case.xlsx",
  "topology_path": "D:/runs/reconfig.xlsx"
}
```

## 4. 台风工作流
- **方法**：`POST /api/typhoon-workflow`
- **用途**：直接调用 `app.py` 的 CLI 子命令，实现台风生命周期、传输线路影响、风电影响、伪 MC 等流程。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `command` | string or string array | 是 | 传递给 `python app.py` 的完整参数。可以是 shell 风格字符串（自动按空格解析），也可以是字符串数组。 |

**示例请求**：
```json
{
  "command": "typhoon --months 7 8 9 --samples 20"
}
```
或
```json
{
  "command": ["monte-carlo", "--hurricane-file", "D:/data/hurricane.xlsx"]
}
```

> 若命令需要额外输入文件，请参考 `app.py` 各子命令的 CLI 说明。

## 5. 完整流程（分类→重构→调度）
- **方法**：`POST /api/full-pipeline`
- **用途**：顺序执行前三个步骤，可分别传入配置。缺省将完全使用默认路径。

| 参数分组 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `classify` | object | 否 | 与 `/api/classify-phases` 相同的字段集合。 |
| `reconfig` | object | 否 | 与 `/api/rolling-reconfig` 相同的字段集合。 |
| `dispatch` | object | 否 | 与 `/api/mess-dispatch` 相同的字段集合。 |

**示例请求**：
```json
{
  "classify": {"input_path": "D:/data/mc.xlsx"},
  "reconfig": {"output_file": "D:/runs/reconfig.xlsx"},
  "dispatch": {"topology_path": "D:/runs/reconfig.xlsx"}
}
```

## 6. DN-RESILIENCE 配电网弹性评估 ★推荐★
- **方法**：`POST /api/DN-RESILIENCE`
- **用途**：一键执行完整配电网弹性评估流程，将台风场景生成与完整流程融合，返回移动储能求解结果文件路径。

**用户只需提供两个Excel文件即可获得最终移动储能调度结果！**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `tower_seg_file` | string/file | 是 | TowerSeg.xlsx - 配电网塔杆分段结构文件路径，或通过 form-data 上传。 |
| `case_file` | string/file | 是 | ac_dc_real_case.xlsx - 混合交直流配电网算例文件路径，或通过 form-data 上传。 |
| `output_dir` | string | 否 | 输出目录路径，默认自动创建带时间戳的目录。 |

### 执行流程
1. **Step 1**: 台风场景生成 → 生成 `mc_simulation_results_k100_clusters.xlsx`
2. **Step 2**: 场景阶段分类 → 生成 `scenario_phase_classification.xlsx`
3. **Step 3**: 滚动拓扑重构 → 生成 `topology_reconfiguration_results.xlsx`
4. **Step 4**: MESS协同调度 → 生成 `mess_dispatch_results.xlsx` ← **最终移动储能求解结果**

### 支持的调用方式

**方式1: JSON Body 方式（指定文件路径或URL）**
```json
{
  "tower_seg_file": "D:/path/to/TowerSeg.xlsx",
  "case_file": "D:/path/to/ac_dc_real_case.xlsx",
  "output_dir": "D:/output"
}
```

**方式2: Form-data 文件上传方式**
```bash
curl -X POST http://localhost:8000/api/DN-RESILIENCE \
  -F "tower_seg_file=@TowerSeg.xlsx" \
  -F "case_file=@ac_dc_real_case.xlsx" \
  -F "output_dir=D:/output"
```

### 返回结果
```json
{
  "status": "success",
  "message": "DN-RESILIENCE 配电网弹性评估完成",
  "input_files": {
    "tower_seg_file": "D:/.../TowerSeg.xlsx",
    "case_file": "D:/.../ac_dc_real_case.xlsx"
  },
  "output_files": {
    "cluster_output": "D:/.../mc_simulation_results_k100_clusters.xlsx",
    "phase_output": "D:/.../scenario_phase_classification.xlsx",
    "topology_output": "D:/.../topology_reconfiguration_results.xlsx",
    "dispatch_output": "D:/.../mess_dispatch_results.xlsx"
  },
  "execution_time": {
    "step1_typhoon_generation": "120.50s",
    "step2_phase_classification": "5.20s",
    "step3_topology_reconfig": "30.15s",
    "step4_mess_dispatch": "45.80s",
    "total": "201.65s"
  },
  "dispatch_result_path": "D:/.../mess_dispatch_results.xlsx"
}
```

> **注意**：`dispatch_result_path` 字段直接返回移动储能求解结果文件的完整路径。

## 7. 完整弹性评估（旧版兼容）
- **方法**：`POST /api/resilience-assessment`
- **用途**：与 `/api/DN-RESILIENCE` 功能相同，保留用于向后兼容。

参数与返回格式与 `/api/DN-RESILIENCE` 相同。

## 8. 聚类分析（仅台风场景生成）
- **方法**：`POST /api/cluster-analysis`
- **用途**：执行单独的聚类分析流程（仅功能4）。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 通过 form-data 上传的 TowerSeg.xlsx 文件。 |

## 9. 健康检查
- **方法**：`GET /api/health`
- **用途**：快速检查服务是否已经启动。返回 `{ "status": "ok" }`。

## 启动方式
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API（可在环境变量中调整端口/地址）
python api_server.py
```
- 环境变量 `API_HOST`、`API_PORT`、`API_DEBUG` 控制监听参数。
- 运行前需要确保本机已经正确配置 Julia 及其依赖包（Project.toml/Manifest.toml）。

---

## 快速入门：DN-RESILIENCE 配电网弹性评估

如果你是第一次使用本系统，建议直接使用 **DN-RESILIENCE API** (`/api/DN-RESILIENCE`)：

1. 准备两个输入Excel文件：
   - `TowerSeg.xlsx` - 描述配电网的塔杆分段结构
   - `ac_dc_real_case.xlsx` - 混合交直流配电网算例数据

2. 调用 API：
```bash
curl -X POST http://localhost:8000/api/DN-RESILIENCE \
  -H "Content-Type: application/json" \
  -d '{
    "tower_seg_file": "D:/path/to/TowerSeg.xlsx",
    "case_file": "D:/path/to/ac_dc_real_case.xlsx"
  }'
```

3. 等待执行完成，从响应中的 `dispatch_result_path` 字段获取移动储能求解结果文件路径。
