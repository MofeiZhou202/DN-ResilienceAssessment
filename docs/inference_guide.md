# 推理层使用指南

## 快速开始

### 前提条件

1. Python 3.10+，已安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   核心依赖：`pandas`, `numpy`, `networkx`, `xgboost`, `openpyxl`

2. 数据文件就位（在 `data/` 目录下）：
   - `ac_dc_real_case.xlsx` — 配电网拓扑数据（母线、线路、负荷、VSC、DC）
   - `topology_reconfiguration_results.xlsx` — Julia拓扑重构结果（100场景×48时间步×35线路状态）
   - `mess_dispatch_hourly.xlsx` — Julia MESS调度结果（失负荷、供电率等弹性指标）
   - `inference_baseline_real.json` — Julia基线统计（期望失负荷、超时节点等）

3. （可选）Julia + Gurobi — 仅在需要"Julia真实验证"时才需要

---

## 两种运行方式

### 方式一：35条线路全排序（推荐）

```bash
# 仅预测排序，不跑Julia验证（约2分钟）
python run_ranking_validation.py --predict-only

# 预测排序 + 验证Top10（约2小时）
python run_ranking_validation.py

# 只验证Top5
python run_ranking_validation.py --top 5
```

**输出**：
- `output/line_ranking_report.md` — Markdown排序报告
- `output/line_ranking_report.json` — JSON格式完整数据

**示例输出**：
```
  排名 线路         综合改善    失负荷    超时改善  关键
  ────────────────────────────────────────────────────
     1 AC_Line_20    8.92%    10.23%     6.95%   ★ ◄
     2 AC_Line_6     7.45%     8.12%     6.44%   ★ ◄
     3 DC_Line_1     5.21%     4.89%     5.69%      ◄
   ...
    35 VSC_Line_4    0.00%     0.00%     0.00%
```

### 方式二：指定线路验证

```bash
# 单条线路预测
python validate_inference.py --lines AC_Line_19 --predict-only

# 单条线路预测 + Julia验证
python validate_inference.py --lines AC_Line_19

# 多条线路逐个测试
python validate_inference.py --lines AC_Line_19 AC_Line_6 DC_Line_1

# 多条线路同时加固（组合效应）
python validate_inference.py --lines AC_Line_19 AC_Line_6 --multi

# 逐个 + 组合都测
python validate_inference.py --lines AC_Line_19 AC_Line_6 --both

# 批量测试（逐个 + 两两 + 全部）
python validate_inference.py --batch --lines AC_Line_19 AC_Line_6 AC_Line_16

# 只做预测不跑Julia
python validate_inference.py --lines AC_Line_19 AC_Line_6 --predict-only
```

---

## 线路命名规范

| 类型 | 命名格式 | 编号范围 | MC故障矩阵行号 |
|------|---------|---------|---------------|
| AC线路 | `AC_Line_1` ~ `AC_Line_26` | 1-26 | 1-26 |
| DC线路 | `DC_Line_1` ~ `DC_Line_2` | 1-2 | 27-28 |
| VSC换流器 | `VSC_Line_1` ~ `VSC_Line_7` | 1-7 | 29-35 |

常开线路（联络开关）：AC_Line_24, AC_Line_25, AC_Line_26, VSC_Line_1, VSC_Line_3

---

## 文件结构

```
DN-ResilienceAssessment/
├── run_ranking_validation.py   ← 主入口：35条线路全排序
├── validate_inference.py       ← 核心引擎：预测 + Julia验证
├── run_inference_pipeline_v2.py← Julia调用管道：反事实MC数据生成
├── data/
│   ├── ac_dc_real_case.xlsx          ← 配电网拓扑
│   ├── topology_reconfiguration_results.xlsx ← 拓扑重构结果
│   ├── mess_dispatch_hourly.xlsx     ← MESS调度结果
│   ├── inference_baseline_real.json  ← Julia基线
│   └── monte_carlo/                  ← MC仿真原始数据
├── output/
│   ├── line_ranking_report.md        ← 排序报告
│   └── line_ranking_report.json      ← 排序数据
└── docs/
    ├── algorithm_explanation.md      ← 算法说明（傻瓜式）
    └── inference_guide.md            ← 本使用指南
```

---

## 关键参数说明

### run_ranking_validation.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--predict-only` / `-p` | 仅预测排序，不跑Julia验证 | 否 |
| `--top N` / `-t N` | Julia验证排名前N条线路 | 10 |
| `--output PATH` / `-o PATH` | 输出报告路径 | `output/line_ranking_report.md` |

### validate_inference.py

| 参数 | 说明 |
|------|------|
| `--lines L1 L2 ...` | 要测试的线路名称（如 AC_Line_19 DC_Line_1） |
| `--multi` / `-m` | 将所有线路作为一组同时加固 |
| `--both` / `-b` | 逐个 + 组合都测 |
| `--batch` | 批量模式（逐个 + 两两 + 全部） |
| `--predict-only` / `-p` | 只做推理预测，不跑Julia |

---

## 常见问题

### Q: 运行报错 "未找到拓扑文件"
确保 `data/ac_dc_real_case.xlsx` 存在。这是配电网拓扑文件，包含 bus、cable、dcbus、dcimpedance、inverter 等工作表。

### Q: 运行报错 "未找到MC数据文件"
仅在 Julia 验证模式下需要 MC 数据。使用 `--predict-only` 可以跳过。

### Q: 预测结果全是0
检查 `topology_reconfiguration_results.xlsx` 中目标线路是否有故障记录（状态=0）。如果某条线路在所有场景中都保持正常（状态=1），则无法评估加固收益。

### Q: Julia验证失败
1. 确保 Julia 已安装且在 PATH 中
2. 确保 Gurobi 许可证有效
3. 检查 `Project.toml` 中的依赖是否已安装（运行 `julia setup.jl`）

### Q: 预测误差较大
推理层是近似方法，对于某些特殊拓扑可能误差较大。误差等级：
- ✓ 误差 < 2%（优秀）
- △ 误差 < 5%（良好）
- ✗ 误差 ≥ 5%（需改进）

---

## 算法原理

详见 [algorithm_explanation.md](algorithm_explanation.md)

核心公式：

**综合改善率** = 0.6 × 失负荷改善率 + 0.4 × 超时节点改善率

三层预测方法：
1. **拓扑加权比例归因**（下界）
2. **XGBoost分位数回归反事实**（数据驱动）
3. **连通性物理模型**（上界）

集成策略：
- 关键瓶颈线路（BC>0.08 或 孤立负荷>20%）→ 使用拓扑归因
- 普通分支线路 → 取 avg(拓扑归因, 连通性模型)
