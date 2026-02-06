# 配电网弹性推理层使用指南

## 一、概述

推理层是规划方案仿真验证智能体的第四层，负责基于蒙特卡洛仿真数据执行**"统计评估 → 归因诊断 → 策略推演"**的三层分析流程。

### 架构位置

```
用户交互层 (Coze)
      ↓
认知层 (豆包大模型)
      ↓
计算层 (api_server.py - DN-RESILIENCE) ← 计算每个场景的弹性指标
      ↓
推理层 (ResilienceInferenceSystem) ← 基于弹性指标进行归因和推演
```

### ⚠️ 重要：推理层的数据来源

**推理层不直接计算潮流或弹性指标！** 它的工作原理是：

1. **数据来源**：从计算层（弹性评估）获取每个场景的：
   - 线路状态（0=故障, 1=正常）
   - 弹性指标（失负荷量、供电率等）

2. **推理机制**：
   - 建立"线路状态 → 弹性指标"的映射关系（使用XGBoost代理模型）
   - 使用SHAP分析每条线路对弹性的影响
   - 通过反事实推理模拟加固效果

**如果缺少场景级别的弹性指标数据，推理结果将不准确！**

### 核心功能

1. **统计评估与代理建模**：计算EPSR、识别高风险节点、训练XGBoost代理模型
2. **SHAP归因诊断**：识别关键薄弱线路，量化各线路对系统弹性的影响
3. **反事实策略生成**：模拟加固效果，给出具体的改进建议和预期收益

---

## 二、完整工作流程

### 正确的流程（需要完整弹性评估数据）

```bash
# Step 1: 运行弹性评估生成场景数据
python api_server.py
# 调用 /api/resilience-assessment API

# Step 2: 运行推理分析
python run_inference_pipeline_v2.py
```

### 快速测试流程（使用现有数据估算）

如果你只想快速了解推理功能，可以使用简化版本：

```bash
python run_inference_pipeline.py
```

⚠️ **注意**：简化版本使用优化目标值（Objective）作为弹性指标的代理，结果是估算值，不是真实的潮流计算结果。

---

## 三、安装依赖

```bash
# 安装推理层额外依赖
pip install xgboost shap

# 或安装全部依赖
pip install -r requirements.txt
```

---

## 三、数据格式要求

### 输入数据规范

推理层接受包含以下列的 DataFrame/Excel/CSV 文件：

#### 1. 特征列（线路状态）

| 列名格式 | 说明 | 取值 |
|---------|------|-----|
| `Line_1_Status`, `Line_2_Status`, ... | 线路物理状态 | `0`=故障/断开, `1`=正常/闭合 |

#### 2. 目标列（仿真结果）

| 列名 | 说明 | 取值范围 |
|------|------|---------|
| `Total_Load_Loss` | 总失负荷量 | float ≥ 0 |
| `Supply_Rate` | 供电率 | 0~1 |
| `Node_1_Time`, `Node_2_Time`, ... | 节点复电时间 | float (小时) |

### 示例数据

```
Line_1_Status | Line_2_Status | Line_3_Status | Total_Load_Loss | Supply_Rate | Node_1_Time | Node_2_Time
       1      |       1       |       0       |      2.5        |    0.95     |     0.5     |     1.2
       0      |       1       |       1       |      5.8        |    0.88     |     2.1     |     0.8
       1      |       0       |       0       |      8.2        |    0.83     |     3.5     |     2.9
       ...
```

---

## 四、使用方法

### 方法1：Python代码调用

```python
# 导入模块
from src.inference import ResilienceInferenceSystem, analyze_resilience

# 方式A：使用类（完全控制）
system = ResilienceInferenceSystem.from_excel("data/mc_results.xlsx")
report = system.run_full_inference(
    top_n_diagnosis=5,      # 识别Top 5薄弱线路
    top_n_prescriptions=3,  # 生成3条加固建议
)

# 输出Markdown报告
print(report.to_markdown())

# 获取JSON格式结果
result_dict = report.to_dict()

# 方式B：使用便捷函数（一行代码）
report = analyze_resilience(
    data="data/mc_results.xlsx",
    output_file="output/report.md"  # 可选，自动保存报告
)
```

### 方法2：API接口调用

启动API服务后，可通过HTTP接口调用推理层：

```bash
# 启动API服务
python api_server.py
```

#### 完整推理分析

```bash
curl -X POST http://localhost:5000/api/inference/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "data_file": "D:/path/to/mc_results.xlsx",
    "output_format": "json",
    "top_n_diagnosis": 5,
    "top_n_prescriptions": 3
  }'
```

#### 快速统计分析

```bash
curl -X POST http://localhost:5000/api/inference/quick-stats \
  -H "Content-Type: application/json" \
  -d '{
    "data_file": "D:/path/to/mc_results.xlsx"
  }'
```

### 方法3：运行演示脚本

```bash
python inference_demo.py
```

---

## 五、输出结果说明

### 1. 统计评估结果

```json
{
  "epsr": 0.9234,           // 期望供电率 (Expected Power Supply Rate)
  "mean_load_loss": 3.45,   // 平均失负荷量
  "std_load_loss": 1.23,    // 失负荷标准差
  "high_risk_nodes": ["Node_5", "Node_12"],  // 高风险节点列表
  "sample_count": 500       // 分析样本数
}
```

### 2. 归因诊断结果

```json
{
  "top_vulnerable_lines": ["Line_7", "Line_3", "Line_15", "Line_2", "Line_9"],
  "global_sensitivity": [
    {"line": "Line_7", "sensitivity": 0.523},
    {"line": "Line_3", "sensitivity": 0.412},
    ...
  ]
}
```

### 3. 策略推演结果

```json
{
  "prescriptions": [
    {
      "target_line": "Line_7",
      "affected_samples": 87,
      "original_loss_mean": 6.82,
      "counterfactual_loss_mean": 3.15,
      "improvement_rate": 0.538,
      "expected_benefit": 3.67,
      "recommendation": "强烈建议加固Line_7。预计可减少平均失负荷53.8%，收益显著。"
    }
  ]
}
```

---

## 六、与Coze集成

### 在Coze中调用推理层

1. 在Coze中配置API插件，指向 `http://your-server:5000`
2. 创建工作流，在需要分析的节点调用 `/api/inference/analyze`
3. 将返回的报告传递给豆包大模型进行自然语言解读

### 示例Coze工作流

```
用户请求: "分析刚才仿真结果的弹性水平"
     ↓
认知层(豆包): 识别意图，调用推理API
     ↓
计算层: 执行推理分析
     ↓
推理层返回: JSON/Markdown报告
     ↓
认知层(豆包): 解读报告，生成用户友好的回复
     ↓
回复用户: "分析结果显示，系统期望供电率为92.34%...建议优先加固Line_7..."
```

---

## 七、高级配置

### 自定义列名识别

```python
system = ResilienceInferenceSystem(
    data=df,
    line_prefix="Branch_",        # 自定义线路列前缀
    line_suffix="_State",         # 自定义线路列后缀
    node_time_prefix="Bus_",      # 自定义节点列前缀
    node_time_suffix="_Recovery", # 自定义节点列后缀
    target_column="LoadShed",     # 自定义目标列名
    high_risk_threshold=0.85,     # 自定义高风险阈值
    recovery_time_threshold=3.0,  # 自定义复电时间阈值(小时)
)
```

### 调整代理模型参数

```python
report = system.run_full_inference(
    n_estimators=200,    # XGBoost树的数量
    max_depth=8,         # 树的最大深度
    learning_rate=0.05,  # 学习率
)
```

---

## 八、文件结构

```
DN-ResilienceAssessment/
├── src/
│   └── inference/
│       ├── __init__.py              # 模块导出
│       └── resilience_inference.py  # 核心推理系统类
├── inference_demo.py                # 使用演示脚本
├── api_server.py                    # API服务（已集成推理层接口）
├── requirements.txt                 # 依赖（已添加xgboost, shap）
└── docs/
    └── inference_guide.md           # 本使用指南
```

---

## 九、常见问题

### Q1: 报错 "XGBoost未安装"
```bash
pip install xgboost
```

### Q2: 报错 "SHAP未安装"
```bash
pip install shap
```

### Q3: 数据列识别不正确
检查数据列名是否符合要求，或使用自定义列名配置。

### Q4: SHAP计算很慢
SHAP计算需要遍历所有样本，数据量大时可能较慢。可以先对数据采样：
```python
df_sample = df.sample(n=1000, random_state=42)
system = ResilienceInferenceSystem(df_sample)
```

---

## 十、技术原理

### 代理模型 (Surrogate Model)
使用XGBoost回归器建立拓扑状态到失负荷量的映射：
$$f(X_{topology}) \rightarrow Y_{loss}$$

### SHAP归因
基于博弈论的Shapley值计算每个特征的贡献：
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [f(S \cup \{i\}) - f(S)]$$

全局敏感性：
$$S_i = \text{mean}(|\phi_i|)$$

### 反事实推理
模拟干预效果：
$$\Delta = \frac{Y_{original} - Y_{counterfactual}}{Y_{original}}$$

---

如有问题，请联系开发团队。
