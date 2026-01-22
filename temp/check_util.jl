# 检查Excel文件中各表的关键列
using Pkg
Pkg.activate(".")
using DataFrames, XLSX

case_path = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")

# util表
println("=" ^ 60)
println("util表:")
df = DataFrame(XLSX.readtable(case_path, "util"))
println("关键列: OpMW, OpMvar")
for row in eachrow(df)
    println("  ID=$(row.ID), OpMW=$(row.OpMW), OpMvar=$(row.OpMvar)")
end

# lumpedload表
println("\n" * "=" ^ 60)
println("lumpedload表:")
df = DataFrame(XLSX.readtable(case_path, "lumpedload"))
println("列名: ", names(df))
println("前5行的MVA列:")
for (i, row) in enumerate(eachrow(df))
    if i <= 5
        mva = haskey(row, :MVA) ? row.MVA : "N/A"
        println("  ID=$(row.ID), Bus=$(row.Bus), MVA=$mva")
    end
end

# inverter表
println("\n" * "=" ^ 60)
println("inverter表:")
df = DataFrame(XLSX.readtable(case_path, "inverter"))
println("关键列: DckW, DcPercentEFF")
for row in eachrow(df)
    dckw = haskey(row, :DckW) ? row.DckW : "N/A"
    eff = haskey(row, :DcPercentEFF) ? row.DcPercentEFF : "N/A"
    println("  ID=$(row.ID), DckW=$dckw, DcPercentEFF=$eff")
end

# dclumpload表
println("\n" * "=" ^ 60)
println("dclumpload表:")
df = DataFrame(XLSX.readtable(case_path, "dclumpload"))
println("列名: ", names(df))
for row in eachrow(df)
    kw = haskey(row, :KW) ? row.KW : "N/A"
    println("  ID=$(row.ID), Bus=$(row.Bus), KW=$kw")
end

