# 检查 inverter 表的数据
project_root = dirname(@__DIR__)
cd(project_root)

using XLSX
using DataFrames

# 读取 inverter 表
inverter_df = DataFrame(XLSX.readtable("data/ac_dc_real_case.xlsx", "inverter"))
println("inverter 表的列名:")
println(names(inverter_df))

println("\ninverter 表的数据:")
for (i, row) in enumerate(eachrow(inverter_df))
    id = haskey(row, :ID) ? row[:ID] : "N/A"
    parent = haskey(row, :ParentName) ? row[:ParentName] : "N/A"
    println("行 $i: ID=$id, ParentName=$parent")
end

# 读取 pvarray 表
pvarray_df = DataFrame(XLSX.readtable("data/ac_dc_real_case.xlsx", "pvarray"))
println("\n\npvarray 表的 ID 列:")
for (i, row) in enumerate(eachrow(pvarray_df))
    id = row.ID
    bus = row.Bus
    inv_inc = row.InverterIncluded
    println("行 $i: ID=$id, Bus=$bus, InverterIncluded=$inv_inc")
end
