
# 设置项目根目录
const PROJECT_ROOT = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment"

# 加载main.jl来设置环境
include(joinpath(PROJECT_ROOT, "main.jl"))

println("=" ^ 60)
println("反事实弹性评估 - 完整流程")
println("=" ^ 60)

# Step 1: 场景阶段分类
println("\n[Step 1/3] 场景阶段分类...")
Workflows.run_classify_phases(
    input_path = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_mc_VSC7_reinforced.xlsx",
    output_path = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_pipeline/scenario_phase_classification.xlsx"
)

# Step 2: 滚动拓扑重构
println("\n[Step 2/3] 滚动拓扑重构...")
Workflows.run_rolling_reconfig(
    case_file = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/data/ac_dc_real_case.xlsx",
    fault_file = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_mc_VSC7_reinforced.xlsx",
    stage_file = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_pipeline/scenario_phase_classification.xlsx",
    output_file = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_pipeline/topology_reconfiguration_results.xlsx"
)

# Step 3: MESS协同调度
println("\n[Step 3/3] MESS协同调度...")
Workflows.run_mess_dispatch(
    case_path = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/data/ac_dc_real_case.xlsx",
    topology_path = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_pipeline/topology_reconfiguration_results.xlsx",
    fallback_topology = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_mc_VSC7_reinforced.xlsx",
    output_file = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_pipeline/mess_dispatch_results.xlsx"
)

println("\n" * "=" ^ 60)
println("✓ 反事实弹性评估完成!")
println("=" ^ 60)
