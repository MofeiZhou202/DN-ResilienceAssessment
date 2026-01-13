# 检查数据文件
using XLSX

data_dir = joinpath(@__DIR__, "data")
println("数据目录: $data_dir")
println()

# 检查mc_simulation_results_k100_clusters.xlsx
mc_file = joinpath(data_dir, "mc_simulation_results_k100_clusters.xlsx")
if isfile(mc_file)
    println("✓ mc_simulation_results_k100_clusters.xlsx 存在")
    xf = XLSX.readxlsx(mc_file)
    println("  Sheets: $(XLSX.sheetnames(xf))")
    for sn in XLSX.sheetnames(xf)
        sheet = xf[sn]
        println("    - $sn: $(XLSX.dim(sheet))")
    end
else
    println("✗ mc_simulation_results_k100_clusters.xlsx 不存在")
end

println()

# 检查scenario_phase_classification.xlsx
phase_file = joinpath(data_dir, "scenario_phase_classification.xlsx")
if isfile(phase_file)
    println("✓ scenario_phase_classification.xlsx 存在")
    xf = XLSX.readxlsx(phase_file)
    println("  Sheets: $(XLSX.sheetnames(xf))")
    for sn in XLSX.sheetnames(xf)
        sheet = xf[sn]
        println("    - $sn: $(XLSX.dim(sheet))")
    end
else
    println("✗ scenario_phase_classification.xlsx 不存在 (需要运行 --classify)")
end

println()

# 检查topology_reconfiguration_results.xlsx
topo_file = joinpath(data_dir, "topology_reconfiguration_results.xlsx")
if isfile(topo_file)
    println("✓ topology_reconfiguration_results.xlsx 存在")
    xf = XLSX.readxlsx(topo_file)
    println("  Sheets: $(XLSX.sheetnames(xf))")
else
    println("✗ topology_reconfiguration_results.xlsx 不存在 (需要运行 --reconfig)")
end
