# 检查 pvarray 工作表的数据
using XLSX
using DataFrames

cd(dirname(@__DIR__))

xf = XLSX.readxlsx("data/ac_dc_real_case.xlsx")

# 检查加载到 case 中的数据
println("检查 case.pvarray 数据:")
include(joinpath(@__DIR__, "..", "src", "workflows.jl"))
using .Workflows

case = Workflows.load_julia_power_data("data/ac_dc_real_case.xlsx")

println("\n\nDC母线列表 (case.busesDC):")
for bus in case.busesDC
    println("  DC母线 $(bus.index): $(bus.name)")
end

println("\n\nAC母线列表 (case.busesAC):")
for bus in case.busesAC
    println("  AC母线 $(bus.index): $(bus.name)")
end

println("\n\n检查 pvarray 表中的所有行:")
df = DataFrame(XLSX.readtable("data/ac_dc_real_case.xlsx", "pvarray"))
for (i, row) in enumerate(eachrow(df))
    bus_name = row.Bus
    pva_power = row.PVAPower
    inv = row.InverterIncluded
    
    in_ac = haskey(case.bus_name_to_id, bus_name)
    in_dc = haskey(case.busdc_name_to_id, bus_name)
    
    println("行 $i: Bus=$(bus_name)")
    println("       PVAPower=$(pva_power) MW, InverterIncluded=$(inv)")
    println("       在AC母线字典: $in_ac, 在DC母线字典: $in_dc")
    println()
end
