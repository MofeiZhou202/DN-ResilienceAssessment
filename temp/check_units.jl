# 检查Excel数据中实际的负荷、发电等数值单位
using XLSX
using DataFrames

xlsx_path = "data/ac_dc_real_case.xlsx"
println("读取文件: $xlsx_path")

xf = XLSX.readxlsx(xlsx_path)
sheets = XLSX.sheetnames(xf)
println("\n所有工作表: ", sheets)

# 检查负荷表 lumpedload
if "lumpedload" in sheets
    println("\n=== 负荷数据 (lumpedload) ===")
    df = XLSX.gettable(xf["lumpedload"]) |> DataFrame
    println("列名: ", names(df))
    println("\n前5行:")
    show(stdout, first(df, 5); allcols=true)
    println()
end

# 检查光伏表
if "pvarray" in sheets
    println("\n=== 光伏数据 (pvarray) ===")
    df = XLSX.gettable(xf["pvarray"]) |> DataFrame
    println("列名: ", names(df))
    println("\n前5行:")
    show(stdout, first(df, 5); allcols=true)
    println()
end

# 检查储能表
if "battery" in sheets
    println("\n=== 储能数据 (battery) ===")
    df = XLSX.gettable(xf["battery"]) |> DataFrame
    println("列名: ", names(df))
    println("\n前5行:")
    show(stdout, first(df, 5); allcols=true)
    println()
end

# 检查直流负荷表
if "dclumpload" in sheets
    println("\n=== 直流负荷数据 (dclumpload) ===")
    df = XLSX.gettable(xf["dclumpload"]) |> DataFrame
    println("列名: ", names(df))
    println("\n前5行:")
    show(stdout, first(df, 5); allcols=true)
    println()
end

# 检查发电机表
if "syngen" in sheets
    println("\n=== 发电机数据 (syngen) ===")
    df = XLSX.gettable(xf["syngen"]) |> DataFrame
    println("列名: ", names(df))
    println("\n前5行:")
    show(stdout, first(df, 5); allcols=true)
    println()
end

# 检查逆变器表
if "inverter" in sheets
    println("\n=== 逆变器数据 (inverter) ===")
    df = XLSX.gettable(xf["inverter"]) |> DataFrame
    println("列名: ", names(df))
    println("\n所有行:")
    show(stdout, df; allcols=true)
    println()
end

# 检查总线电压等级
if "bus" in sheets
    println("\n=== 总线数据 (bus) ===")
    df = XLSX.gettable(xf["bus"]) |> DataFrame
    println("列名: ", names(df))
    if "NomkV" in names(df)
        unique_kv = unique(df[!, :NomkV])
        println("电压等级 (NomkV): ", unique_kv)
    end
    println("\n前5行:")
    show(stdout, first(df, 5); allcols=true)
    println()
end
