# DN-ResilienceAssessment

## 配电网韧性评估系统

本项目整合了配电网韧性评估的完整工作流程，包括：
- 台风生成与仿真
- 输电线路故障概率计算
- 风电场输出估计
- 伪蒙特卡洛采样
- 聚类分析
- 场景阶段分类
- 滚动拓扑重构优化
- 交直流混合配电网+微电网+MESS协同调度

## 目录结构

```
DN-ResilienceAssessment/
├── main.jl                    # Julia主入口（统一调用所有功能）
├── app.py                     # Python台风仿真工作流
├── Project.toml               # Julia项目配置
├── requirements.txt           # Python依赖
├── README.md                  # 本文件
├── data/                      # 数据文件
├── dataset/                   # 数据集目录
├── checkpoints/               # 模型检查点
├── solvers/                   # Julia求解器
└── src/                       # 源代码
```

## 安装依赖

### 1. Python依赖
```bash
pip install -r requirements.txt
```

### 2. Julia依赖
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

注意：需要安装Gurobi优化器并获取许可证。

## 使用方法（全部通过Julia调用）

### 交互式菜单
```bash
julia main.jl
```

运行后会显示菜单：
```
============================================================
DN-ResilienceAssessment 配电网韧性评估系统
============================================================

请选择要运行的功能：

  1. 场景阶段分类
  2. 滚动拓扑重构
  3. 混合配电网+MESS协同调度
  4. 台风场景生成工作流
  5. 完整流程（1→2→3）
  q. 退出
```

### 命令行方式
```bash
# 场景阶段分类
julia main.jl --classify

# 滚动拓扑重构
julia main.jl --reconfig

# 混合配电网+MESS协同调度
julia main.jl --dispatch

# 台风场景生成工作流（交互式）
julia main.jl --typhoon

# 运行完整流程（分类→重构→调度）
julia main.jl --full

# 查看帮助
julia main.jl --help
```

### 在Julia REPL中使用
```julia
# 进入项目目录
cd("path/to/DN-ResilienceAssessment")

# 激活项目环境
using Pkg
Pkg.activate(".")

# 加载主模块
include("main.jl")

# 运行各功能
run_classify_phases()           # 场景阶段分类
run_rolling_reconfig()          # 滚动拓扑重构
run_mess_dispatch()             # MESS协同调度
run_typhoon_workflow()          # 台风场景生成
run_full_pipeline()             # 完整流程
```

## 功能说明

### 场景阶段分类
将蒙特卡洛故障轨迹分为四个阶段：
- 阶段0: 基准拓扑
- 阶段1: 故障聚集
- 阶段2: 前期恢复
- 阶段3: 恢复后

### 滚动拓扑重构
三阶段拓扑重构数学模型：
- 第一阶段: 故障隔离
- 第二阶段: 故障后重构
- 第三阶段: 修复后重构

### MESS协同调度
交直流混合配电网+微电网+移动储能系统(MESS)协同优化调度

### 台风场景生成
- 季节性台风模拟
- 神经网络台风生成
- 输电线路影响计算
- 风电场输出估计
- 伪蒙特卡洛采样
- 聚类分析

## 许可证

MIT License
