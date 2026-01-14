# 运行Julia版本的MESS调度
println("启动MESS调度...")

# 首先加载所有工具文件
include("src/utils/idx.jl")
include("src/utils/ComponentStructs.jl")
include("src/utils/Types.jl")
include("src/utils/ETAPImporter.jl")
include("src/utils/juliapowercase2jpc_tp.jl")

# 加载调度模块
include("src/dispatch_main.jl")

# 运行主函数
main()
