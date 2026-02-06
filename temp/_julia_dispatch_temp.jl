
# 设置项目根目录
const PROJECT_ROOT = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment"

# 加载main.jl来设置环境
include(joinpath(PROJECT_ROOT, "main.jl"))

# 调用MESS调度
Workflows.run_mess_dispatch(
    case_path = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/data/ac_dc_real_case.xlsx",
    topology_path = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_AC_Line_20_status1.xlsx",
    output_file = raw"D:/DistributionPowerFlow-runze/ExtremeScenarioGeneration-DN/Topology_Reconfiguration/Topology_Reconfiguration/DN-ResilienceAssessment/output/counterfactual_dispatch_AC_Line_20_Status.xlsx"
)
println("Julia计算完成!")
