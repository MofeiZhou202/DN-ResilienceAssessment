"""
配电网弹性推理系统 (Resilience Inference System)
================================================
基于蒙特卡洛仿真数据，执行"统计评估 -> 归因诊断 -> 策略推演"的三层分析流程。

Author: DN-Resilience Assessment Team
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# 尝试导入可选依赖
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost未安装，代理模型功能将不可用。请运行: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP未安装，归因诊断功能将不可用。请运行: pip install shap")


@dataclass
class StatisticsResult:
    """统计评估结果"""
    epsr: float  # 期望供电率 (Expected Power Supply Rate)
    mean_load_loss: float  # 平均失负荷量
    std_load_loss: float  # 失负荷标准差
    high_risk_nodes: List[str]  # 高风险节点列表
    node_risk_profile: Dict[str, Dict[str, float]]  # 节点风险画像
    sample_count: int  # 样本数量


@dataclass
class DiagnosisResult:
    """归因诊断结果"""
    global_sensitivity: pd.DataFrame  # 全局敏感性排名
    top_vulnerable_lines: List[str]  # Top N 薄弱线路
    shap_values: Optional[np.ndarray] = None  # SHAP值矩阵
    feature_importance: Optional[Dict[str, float]] = None  # 特征重要性


@dataclass 
class PrescriptionResult:
    """策略推演结果"""
    target_line: str  # 目标线路
    affected_samples: int  # 受影响样本数
    original_loss_mean: float  # 原始平均损失
    counterfactual_loss_mean: float  # 反事实平均损失
    improvement_rate: float  # 改善率
    expected_benefit: float  # 预期收益
    recommendation: str  # 加固建议


@dataclass
class InferenceReport:
    """完整推理报告"""
    statistics: StatisticsResult
    diagnosis: DiagnosisResult
    prescriptions: List[PrescriptionResult]
    summary: str = ""
    
    @staticmethod
    def _convert_to_native(obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: InferenceReport._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [InferenceReport._convert_to_native(v) for v in obj]
        return obj
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "statistics": {
                "epsr": float(self.statistics.epsr),
                "mean_load_loss": float(self.statistics.mean_load_loss),
                "std_load_loss": float(self.statistics.std_load_loss),
                "high_risk_nodes": self.statistics.high_risk_nodes,
                "node_risk_profile": self._convert_to_native(self.statistics.node_risk_profile),
                "sample_count": int(self.statistics.sample_count),
            },
            "diagnosis": {
                "global_sensitivity": [
                    {k: self._convert_to_native(v) for k, v in row.items()}
                    for row in self.diagnosis.global_sensitivity.to_dict(orient='records')
                ],
                "top_vulnerable_lines": self.diagnosis.top_vulnerable_lines,
            },
            "prescriptions": [
                {
                    "target_line": p.target_line,
                    "affected_samples": int(p.affected_samples),
                    "original_loss_mean": float(p.original_loss_mean),
                    "counterfactual_loss_mean": float(p.counterfactual_loss_mean),
                    "improvement_rate": float(p.improvement_rate),
                    "expected_benefit": float(p.expected_benefit),
                    "recommendation": p.recommendation,
                }
                for p in self.prescriptions
            ],
            "summary": self.summary,
        }
        return result
    
    def to_markdown(self) -> str:
        """生成Markdown格式报告"""
        lines = [
            "# 配电网弹性推理分析报告",
            "",
            "## 一、统计评估结果",
            "",
            f"- **期望供电率 (EPSR)**: {self.statistics.epsr:.4f} ({self.statistics.epsr*100:.2f}%)",
            f"- **平均失负荷量**: {self.statistics.mean_load_loss:.4f}",
            f"- **失负荷标准差**: {self.statistics.std_load_loss:.4f}",
            f"- **分析样本数**: {self.statistics.sample_count}",
            "",
            "### 高风险节点",
            "",
        ]
        
        if self.statistics.high_risk_nodes:
            for node in self.statistics.high_risk_nodes:
                profile = self.statistics.node_risk_profile.get(node, {})
                prob = profile.get('recovery_prob_2h', 0)
                lines.append(f"- **{node}**: 2小时内复电概率 = {prob:.2%}")
        else:
            lines.append("- 无高风险节点（所有节点2小时内复电概率均 ≥ 80%）")
        
        lines.extend([
            "",
            "## 二、归因诊断结果",
            "",
            "### 线路敏感性排名（按全局重要性降序）",
            "",
            "| 排名 | 线路 | 敏感性指数 |",
            "|------|------|-----------|",
        ])
        
        for i, row in self.diagnosis.global_sensitivity.head(10).iterrows():
            lines.append(f"| {i+1} | {row['line']} | {row['sensitivity']:.6f} |")
        
        lines.extend([
            "",
            f"### Top {len(self.diagnosis.top_vulnerable_lines)} 关键薄弱线路",
            "",
        ])
        for i, line in enumerate(self.diagnosis.top_vulnerable_lines, 1):
            lines.append(f"{i}. {line}")
        
        lines.extend([
            "",
            "## 三、策略推演结果",
            "",
        ])
        
        for i, p in enumerate(self.prescriptions, 1):
            lines.extend([
                f"### 策略 {i}: 加固 {p.target_line}",
                "",
                f"- **受影响故障场景数**: {p.affected_samples}",
                f"- **原始平均失负荷**: {p.original_loss_mean:.4f}",
                f"- **加固后预测失负荷**: {p.counterfactual_loss_mean:.4f}",
                f"- **预期改善率**: {p.improvement_rate:.2%}",
                f"- **预期收益**: {p.expected_benefit:.4f}",
                "",
                f"**建议**: {p.recommendation}",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "## 总结",
            "",
            self.summary,
        ])
        
        return "\n".join(lines)


class ResilienceInferenceSystem:
    """
    配电网弹性推理系统
    
    基于蒙特卡洛仿真数据，执行三层分析流程：
    1. 统计评估与代理建模 (Statistics & Surrogate)
    2. 归因诊断 (Diagnosis via SHAP)
    3. 反事实策略生成 (Counterfactual Prescription)
    
    Attributes:
        data: 输入数据DataFrame
        line_columns: 线路状态列名列表
        node_time_columns: 节点复电时间列名列表
        target_column: 目标列名（默认为Total_Load_Loss）
        supply_rate_column: 供电率列名
        surrogate_model: 训练好的代理模型
        shap_explainer: SHAP解释器
    
    Example:
        >>> # 加载数据
        >>> df = pd.read_excel("mc_results.xlsx")
        >>> 
        >>> # 创建推理系统
        >>> system = ResilienceInferenceSystem(df)
        >>> 
        >>> # 执行完整推理流程
        >>> report = system.run_full_inference()
        >>> 
        >>> # 输出报告
        >>> print(report.to_markdown())
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        line_prefix: str = "Line_",
        line_suffix: str = "_Status",
        node_time_prefix: str = "Node_",
        node_time_suffix: str = "_Time",
        target_column: str = "Total_Load_Loss",
        supply_rate_column: str = "Supply_Rate",
        high_risk_threshold: float = 0.8,
        recovery_time_threshold: float = 2.0,
        random_state: int = 42,
    ):
        """
        初始化推理系统
        
        Args:
            data: 包含仿真结果的DataFrame
            line_prefix: 线路状态列的前缀
            line_suffix: 线路状态列的后缀
            node_time_prefix: 节点复电时间列的前缀
            node_time_suffix: 节点复电时间列的后缀
            target_column: 目标列名（失负荷量）
            supply_rate_column: 供电率列名
            high_risk_threshold: 高风险阈值（复电概率低于此值为高风险）
            recovery_time_threshold: 复电时间阈值（小时）
            random_state: 随机种子
        """
        self.data = data.copy()
        self.line_prefix = line_prefix
        self.line_suffix = line_suffix
        self.node_time_prefix = node_time_prefix
        self.node_time_suffix = node_time_suffix
        self.target_column = target_column
        self.supply_rate_column = supply_rate_column
        self.high_risk_threshold = high_risk_threshold
        self.recovery_time_threshold = recovery_time_threshold
        self.random_state = random_state
        
        # 自动识别列
        self._identify_columns()
        
        # 模型相关
        self.surrogate_model: Optional[xgb.XGBRegressor] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self._shap_values: Optional[np.ndarray] = None
        
    def _identify_columns(self) -> None:
        """自动识别特征列和目标列"""
        all_columns = self.data.columns.tolist()
        
        # 识别线路状态列
        self.line_columns = [
            col for col in all_columns 
            if col.startswith(self.line_prefix) and col.endswith(self.line_suffix)
        ]
        
        # 如果没有找到带后缀的列，尝试只用前缀
        if not self.line_columns:
            self.line_columns = [
                col for col in all_columns 
                if col.startswith(self.line_prefix)
            ]
        
        # 识别节点复电时间列
        self.node_time_columns = [
            col for col in all_columns
            if col.startswith(self.node_time_prefix) and col.endswith(self.node_time_suffix)
        ]
        
        # 如果没有找到带后缀的列，尝试只用前缀
        if not self.node_time_columns:
            self.node_time_columns = [
                col for col in all_columns
                if col.startswith(self.node_time_prefix) and self.node_time_suffix in col
            ]
        
        # 确保目标列存在
        if self.target_column not in self.data.columns:
            # 尝试查找类似的列
            candidate_cols = [col for col in all_columns if 'loss' in col.lower() or 'load' in col.lower()]
            if candidate_cols:
                self.target_column = candidate_cols[0]
                warnings.warn(f"未找到'{self.target_column}'列，使用'{candidate_cols[0]}'代替")
            else:
                raise ValueError(f"数据中未找到目标列'{self.target_column}'，也找不到类似的列")
        
        # 供电率列
        if self.supply_rate_column not in self.data.columns:
            candidate_cols = [col for col in all_columns if 'supply' in col.lower() or 'rate' in col.lower()]
            if candidate_cols:
                self.supply_rate_column = candidate_cols[0]
            else:
                self.supply_rate_column = None
                warnings.warn("未找到供电率列，将从失负荷量推算")
        
        print(f"[INFO] 识别到 {len(self.line_columns)} 条线路状态列")
        print(f"[INFO] 识别到 {len(self.node_time_columns)} 个节点复电时间列")
        print(f"[INFO] 目标列: {self.target_column}")
    
    def get_feature_matrix(self) -> pd.DataFrame:
        """获取特征矩阵 X（线路拓扑状态）"""
        if not self.line_columns:
            raise ValueError("未识别到线路状态列，请检查数据格式")
        return self.data[self.line_columns].copy()
    
    def get_target_vector(self) -> pd.Series:
        """获取目标向量 Y（失负荷量）"""
        return self.data[self.target_column].copy()
    
    # ==================== 模块一：统计评估与代理建模 ====================
    
    def compute_statistics(self) -> StatisticsResult:
        """
        模块一：统计评估
        
        计算系统期望供电率(EPSR)和节点风险画像
        
        Returns:
            StatisticsResult: 统计评估结果
        """
        print("\n[模块一] 开始统计评估...")
        
        # 计算期望供电率
        if self.supply_rate_column and self.supply_rate_column in self.data.columns:
            epsr = self.data[self.supply_rate_column].mean()
        else:
            # 从失负荷量推算（假设有总负荷信息）
            total_loss = self.data[self.target_column]
            if 'Total_Load' in self.data.columns:
                supply_rates = 1 - total_loss / self.data['Total_Load']
            else:
                # 假设最大损失时供电率为0
                max_loss = total_loss.max()
                if max_loss > 0:
                    supply_rates = 1 - total_loss / max_loss
                else:
                    supply_rates = pd.Series([1.0] * len(total_loss))
            epsr = supply_rates.mean()
        
        # 失负荷统计
        mean_loss = self.data[self.target_column].mean()
        std_loss = self.data[self.target_column].std()
        
        # 节点风险画像
        node_risk_profile = {}
        high_risk_nodes = []
        
        for col in self.node_time_columns:
            node_name = col.replace(self.node_time_prefix, "").replace(self.node_time_suffix, "")
            if col in self.data.columns:
                recovery_times = self.data[col]
                # 计算2小时内复电概率
                prob_2h = (recovery_times < self.recovery_time_threshold).mean()
                
                node_risk_profile[node_name] = {
                    'recovery_prob_2h': prob_2h,
                    'mean_recovery_time': recovery_times.mean(),
                    'max_recovery_time': recovery_times.max(),
                }
                
                # 判断是否为高风险节点
                if prob_2h < self.high_risk_threshold:
                    high_risk_nodes.append(node_name)
        
        result = StatisticsResult(
            epsr=epsr,
            mean_load_loss=mean_loss,
            std_load_loss=std_loss,
            high_risk_nodes=high_risk_nodes,
            node_risk_profile=node_risk_profile,
            sample_count=len(self.data),
        )
        
        print(f"  - 期望供电率 EPSR: {epsr:.4f}")
        print(f"  - 平均失负荷: {mean_loss:.4f}")
        print(f"  - 高风险节点数: {len(high_risk_nodes)}")
        
        return result
    
    def train_surrogate_model(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **xgb_params
    ) -> 'xgb.XGBRegressor':
        """
        训练XGBoost代理模型
        
        建立拓扑结构与系统损失的快速预测映射：f(X_topology) -> Y_loss
        
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            **xgb_params: 其他XGBoost参数
            
        Returns:
            训练好的XGBoost模型
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost未安装，请运行: pip install xgboost")
        
        print("\n[模块一] 训练代理模型...")
        
        X = self.get_feature_matrix()
        y = self.get_target_vector()
        
        # 创建并训练模型
        self.surrogate_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            **xgb_params
        )
        
        self.surrogate_model.fit(X, y)
        
        # 计算训练性能
        y_pred = self.surrogate_model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        
        print(f"  - 训练样本数: {len(X)}")
        print(f"  - 特征数: {X.shape[1]}")
        print(f"  - MSE: {mse:.6f}")
        print(f"  - R²: {r2:.4f}")
        
        return self.surrogate_model
    
    # ==================== 模块二：归因诊断 ====================
    
    def diagnose_with_shap(self, top_n: int = 5) -> DiagnosisResult:
        """
        模块二：SHAP归因诊断
        
        使用SHAP (TreeExplainer) 解释训练好的代理模型，
        计算每条线路的全局重要性，识别关键薄弱线路。
        
        Args:
            top_n: 返回Top N关键薄弱线路
            
        Returns:
            DiagnosisResult: 诊断结果
        """
        if not HAS_SHAP:
            raise ImportError("SHAP未安装，请运行: pip install shap")
        
        if self.surrogate_model is None:
            print("[INFO] 代理模型未训练，自动训练中...")
            self.train_surrogate_model()
        
        print("\n[模块二] SHAP归因诊断...")
        
        X = self.get_feature_matrix()
        
        # 创建SHAP解释器
        self.shap_explainer = shap.TreeExplainer(self.surrogate_model)
        
        # 计算SHAP值
        print("  - 计算SHAP值中...")
        self._shap_values = self.shap_explainer.shap_values(X)
        
        # 计算全局敏感性：S_i = mean(|φ_i|)
        global_sensitivity = np.mean(np.abs(self._shap_values), axis=0)
        
        # 创建敏感性DataFrame
        sensitivity_df = pd.DataFrame({
            'line': self.line_columns,
            'sensitivity': global_sensitivity
        }).sort_values('sensitivity', ascending=False).reset_index(drop=True)
        
        # 提取Top N薄弱线路
        top_vulnerable = sensitivity_df.head(top_n)['line'].tolist()
        
        result = DiagnosisResult(
            global_sensitivity=sensitivity_df,
            top_vulnerable_lines=top_vulnerable,
            shap_values=self._shap_values,
            feature_importance=dict(zip(self.line_columns, global_sensitivity))
        )
        
        print(f"  - 分析完成，识别出 Top {top_n} 薄弱线路:")
        for i, line in enumerate(top_vulnerable, 1):
            sens = sensitivity_df[sensitivity_df['line'] == line]['sensitivity'].values[0]
            print(f"    {i}. {line} (敏感性: {sens:.6f})")
        
        return result
    
    # ==================== 模块三：反事实策略生成 ====================
    
    def generate_counterfactual_prescription(
        self,
        target_line: Optional[str] = None,
        top_n_prescriptions: int = 3
    ) -> List[PrescriptionResult]:
        """
        模块三：反事实策略生成
        
        针对薄弱线路进行"反事实"模拟，评估加固效益。
        
        步骤：
        1. 选取该线路处于"故障(0)"状态的样本子集
        2. 干预：强制将该线路状态修改为"正常(1)"（模拟加固/修复）
        3. 重预测：使用代理模型预测修改后的系统损失
        4. 效益评估：计算平均改善率
        
        Args:
            target_line: 目标线路（可选，默认使用No.1薄弱线路）
            top_n_prescriptions: 生成建议数量
            
        Returns:
            List[PrescriptionResult]: 策略建议列表
        """
        print("\n[模块三] 反事实策略生成...")
        
        if self.surrogate_model is None:
            print("[INFO] 代理模型未训练，自动训练中...")
            self.train_surrogate_model()
        
        # 获取薄弱线路列表
        if target_line:
            target_lines = [target_line]
        else:
            # 先执行诊断获取Top N薄弱线路
            if self._shap_values is None:
                diagnosis = self.diagnose_with_shap(top_n=top_n_prescriptions)
                target_lines = diagnosis.top_vulnerable_lines
            else:
                # 使用已有的SHAP值计算
                global_sens = np.mean(np.abs(self._shap_values), axis=0)
                sens_df = pd.DataFrame({
                    'line': self.line_columns,
                    'sensitivity': global_sens
                }).sort_values('sensitivity', ascending=False)
                target_lines = sens_df.head(top_n_prescriptions)['line'].tolist()
        
        prescriptions = []
        X = self.get_feature_matrix()
        y = self.get_target_vector()
        
        for line in target_lines:
            if line not in X.columns:
                print(f"  [警告] 线路 {line} 不在数据中，跳过")
                continue
            
            print(f"\n  分析线路: {line}")
            
            # 步骤1: 选取该线路故障(0)的样本子集
            fault_mask = X[line] == 0
            if fault_mask.sum() == 0:
                print(f"    - 无故障样本，跳过")
                continue
            
            X_fault = X[fault_mask].copy()
            y_fault = y[fault_mask].values
            
            print(f"    - 故障样本数: {len(X_fault)}")
            
            # 步骤2: 干预 - 将该线路状态改为正常(1)
            X_counterfactual = X_fault.copy()
            X_counterfactual[line] = 1
            
            # 步骤3: 重预测
            y_original = self.surrogate_model.predict(X_fault)
            y_counterfactual = self.surrogate_model.predict(X_counterfactual)
            
            # 步骤4: 效益评估
            original_mean = y_original.mean()
            counterfactual_mean = y_counterfactual.mean()
            
            if original_mean > 0:
                improvement_rate = (original_mean - counterfactual_mean) / original_mean
            else:
                improvement_rate = 0.0
            
            expected_benefit = original_mean - counterfactual_mean
            
            # 生成建议
            if improvement_rate > 0.2:
                recommendation = f"强烈建议加固{line}。预计可减少平均失负荷{improvement_rate*100:.1f}%，收益显著。"
            elif improvement_rate > 0.1:
                recommendation = f"建议考虑加固{line}。预计可减少平均失负荷{improvement_rate*100:.1f}%。"
            elif improvement_rate > 0:
                recommendation = f"加固{line}有一定效果，预计改善{improvement_rate*100:.1f}%，可作为备选方案。"
            else:
                recommendation = f"加固{line}对系统弹性提升有限，建议优先考虑其他线路。"
            
            prescription = PrescriptionResult(
                target_line=line,
                affected_samples=len(X_fault),
                original_loss_mean=original_mean,
                counterfactual_loss_mean=counterfactual_mean,
                improvement_rate=improvement_rate,
                expected_benefit=expected_benefit,
                recommendation=recommendation,
            )
            prescriptions.append(prescription)
            
            print(f"    - 原始平均损失: {original_mean:.4f}")
            print(f"    - 反事实平均损失: {counterfactual_mean:.4f}")
            print(f"    - 改善率: {improvement_rate:.2%}")
        
        return prescriptions
    
    # ==================== 完整推理流程 ====================
    
    def run_full_inference(
        self,
        top_n_diagnosis: int = 5,
        top_n_prescriptions: int = 3,
        **surrogate_params
    ) -> InferenceReport:
        """
        执行完整的三层推理流程
        
        Args:
            top_n_diagnosis: 诊断时识别的Top N薄弱线路数量
            top_n_prescriptions: 生成策略建议的数量
            **surrogate_params: 代理模型训练参数
            
        Returns:
            InferenceReport: 完整推理报告
        """
        print("=" * 60)
        print("配电网弹性推理系统 - 开始分析")
        print("=" * 60)
        
        # 1. 统计评估
        statistics = self.compute_statistics()
        
        # 2. 训练代理模型
        self.train_surrogate_model(**surrogate_params)
        
        # 3. 归因诊断
        diagnosis = self.diagnose_with_shap(top_n=top_n_diagnosis)
        
        # 4. 策略推演
        prescriptions = self.generate_counterfactual_prescription(
            top_n_prescriptions=top_n_prescriptions
        )
        
        # 生成总结
        summary = self._generate_summary(statistics, diagnosis, prescriptions)
        
        report = InferenceReport(
            statistics=statistics,
            diagnosis=diagnosis,
            prescriptions=prescriptions,
            summary=summary,
        )
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
        return report
    
    def _generate_summary(
        self,
        stats: StatisticsResult,
        diag: DiagnosisResult,
        presc: List[PrescriptionResult]
    ) -> str:
        """生成分析总结"""
        lines = []
        
        # 系统状态评价
        if stats.epsr >= 0.95:
            lines.append("系统整体弹性水平优秀，期望供电率达到95%以上。")
        elif stats.epsr >= 0.85:
            lines.append("系统弹性水平良好，但仍有改进空间。")
        elif stats.epsr >= 0.7:
            lines.append("系统弹性水平一般，建议重点关注薄弱环节。")
        else:
            lines.append("系统弹性水平较低，亟需采取加固措施。")
        
        # 高风险节点
        if stats.high_risk_nodes:
            lines.append(f"存在{len(stats.high_risk_nodes)}个高风险节点，2小时内复电概率不足80%。")
        
        # 薄弱线路
        lines.append(f"Top {len(diag.top_vulnerable_lines)} 关键薄弱线路: {', '.join(diag.top_vulnerable_lines)}。")
        
        # 策略建议
        if presc:
            best = max(presc, key=lambda x: x.improvement_rate)
            if best.improvement_rate > 0:
                lines.append(f"优先建议加固 {best.target_line}，预期可提升系统弹性约{best.improvement_rate*100:.1f}%。")
        
        return " ".join(lines)
    
    # ==================== 数据加载辅助方法 ====================
    
    @classmethod
    def from_excel(
        cls,
        filepath: str,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> 'ResilienceInferenceSystem':
        """
        从Excel文件加载数据并创建推理系统
        
        Args:
            filepath: Excel文件路径
            sheet_name: 工作表名或索引
            **kwargs: 传递给__init__的其他参数
            
        Returns:
            ResilienceInferenceSystem实例
        """
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        return cls(df, **kwargs)
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        **kwargs
    ) -> 'ResilienceInferenceSystem':
        """
        从CSV文件加载数据并创建推理系统
        
        Args:
            filepath: CSV文件路径
            **kwargs: 传递给__init__的其他参数
            
        Returns:
            ResilienceInferenceSystem实例
        """
        df = pd.read_csv(filepath)
        return cls(df, **kwargs)
    
    @classmethod
    def from_monte_carlo_results(
        cls,
        mc_results_path: str,
        line_status_sheet: str = "line_status",
        node_recovery_sheet: str = "node_recovery",
        metrics_sheet: str = "metrics",
        **kwargs
    ) -> 'ResilienceInferenceSystem':
        """
        从蒙特卡洛仿真结果文件加载数据
        
        Args:
            mc_results_path: 蒙特卡洛结果Excel文件路径
            line_status_sheet: 线路状态工作表名
            node_recovery_sheet: 节点复电时间工作表名
            metrics_sheet: 指标工作表名
            **kwargs: 其他参数
            
        Returns:
            ResilienceInferenceSystem实例
        """
        try:
            xl = pd.ExcelFile(mc_results_path)
            
            # 读取各工作表
            dfs = []
            
            if line_status_sheet in xl.sheet_names:
                line_df = pd.read_excel(xl, sheet_name=line_status_sheet)
                dfs.append(line_df)
            
            if node_recovery_sheet in xl.sheet_names:
                node_df = pd.read_excel(xl, sheet_name=node_recovery_sheet)
                dfs.append(node_df)
            
            if metrics_sheet in xl.sheet_names:
                metrics_df = pd.read_excel(xl, sheet_name=metrics_sheet)
                dfs.append(metrics_df)
            
            if not dfs:
                # 尝试读取第一个工作表
                df = pd.read_excel(mc_results_path)
            else:
                # 合并数据
                df = pd.concat(dfs, axis=1)
                # 去除重复列
                df = df.loc[:, ~df.columns.duplicated()]
            
            return cls(df, **kwargs)
            
        except Exception as e:
            warnings.warn(f"加载蒙特卡洛结果时出错: {e}，尝试直接读取...")
            df = pd.read_excel(mc_results_path)
            return cls(df, **kwargs)


# ==================== 便捷函数 ====================

def analyze_resilience(
    data: Union[str, pd.DataFrame],
    output_file: Optional[str] = None,
    **kwargs
) -> InferenceReport:
    """
    便捷函数：一键执行弹性分析
    
    Args:
        data: DataFrame或数据文件路径（支持Excel/CSV）
        output_file: 输出文件路径（可选，支持.md/.json/.html）
        **kwargs: 传递给ResilienceInferenceSystem的参数
        
    Returns:
        InferenceReport: 分析报告
    
    Example:
        >>> report = analyze_resilience("mc_results.xlsx")
        >>> print(report.to_markdown())
    """
    # 加载数据
    if isinstance(data, str):
        if data.endswith('.csv'):
            system = ResilienceInferenceSystem.from_csv(data, **kwargs)
        else:
            system = ResilienceInferenceSystem.from_excel(data, **kwargs)
    else:
        system = ResilienceInferenceSystem(data, **kwargs)
    
    # 执行分析
    report = system.run_full_inference()
    
    # 输出报告
    if output_file:
        if output_file.endswith('.md'):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report.to_markdown())
            print(f"[INFO] 报告已保存至: {output_file}")
        elif output_file.endswith('.json'):
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"[INFO] 报告已保存至: {output_file}")
    
    return report
