"""
交通网络定义，对应 Python 中的 transportation_network.py。
用于移动储能系统 (MESS) 的路径规划。
"""

module TransportationNetwork

export transportation_network, TransportNetworkData

"""
交通网络数据结构
"""
struct TransportNetworkData
    bus::Matrix{Float64}       # 节点信息 [node_id, capacity, grid_mapping]
    branch::Matrix{Float64}    # 支路信息 [from, to, travel_time]
    initial_status::Vector{Float64}
    end_status::Vector{Float64}
end

"""
    transportation_network() -> TransportNetworkData

返回交通网络数据，包含6个节点和9条支路。
节点对应电网节点: 1->5, 2->10, 3->15, 4->20, 5->25, 6->30
支路包含双向旅行时间（小时）。
"""
function transportation_network()::TransportNetworkData
    # 节点数据: [node_id, capacity, grid_node_mapping]
    bus = [
        1.0 100.0 1.0;
        2.0 100.0 3.0;
        3.0 100.0 6.0;
        4.0 100.0 10.0;
        5.0 100.0 14.0;
        6.0 100.0 24.0
    ]

    # 支路数据: [from_node, to_node, travel_time_hours]
    branch = [
        1.0 2.0 1.0;
        1.0 3.0 4.0;
        1.0 4.0 2.0;
        2.0 4.0 3.0;
        2.0 5.0 2.0;
        3.0 4.0 5.0;
        4.0 5.0 3.0;
        5.0 6.0 3.0;
        3.0 6.0 2.0
    ]

    # 初始/终止状态
    initial_status = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    end_status = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return TransportNetworkData(bus, branch, initial_status, end_status)
end

end # module
