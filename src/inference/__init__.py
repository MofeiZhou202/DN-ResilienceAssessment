"""
配电网弹性推理模块 (Inference Module)
====================================
提供基于蒙特卡洛仿真数据的弹性分析能力。

主要组件：
- ResilienceInferenceSystem: 核心推理系统类
- analyze_resilience: 便捷分析函数
- InferenceReport: 推理报告数据类

使用示例：
    >>> from src.inference import ResilienceInferenceSystem, analyze_resilience
    >>> 
    >>> # 方式1: 使用类
    >>> system = ResilienceInferenceSystem.from_excel("data.xlsx")
    >>> report = system.run_full_inference()
    >>> 
    >>> # 方式2: 使用便捷函数
    >>> report = analyze_resilience("data.xlsx", output_file="report.md")
"""

from .resilience_inference import (
    ResilienceInferenceSystem,
    analyze_resilience,
    StatisticsResult,
    DiagnosisResult,
    PrescriptionResult,
    InferenceReport,
)

__all__ = [
    'ResilienceInferenceSystem',
    'analyze_resilience',
    'StatisticsResult',
    'DiagnosisResult', 
    'PrescriptionResult',
    'InferenceReport',
]

__version__ = '1.0.0'
