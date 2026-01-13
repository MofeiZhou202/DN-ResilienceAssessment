# 安装依赖脚本
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
println("依赖安装完成！")
