"""
    Branch and Bound Algorithm for Mixed Integer Linear Programming (MILP)
    Uses interior point method to solve LP relaxations
"""

using DataStructures
using LinearAlgebra
using SparseArrays
include("interior_point_method.jl")

"""
    solve_milp_bb(milp::MILP, bb_params::BBSearchParams, ipm_params::IPM)

Solve MILP using branch and bound with interior point method for LP relaxations.
"""
function solve_milp_bb(milp::MILP, bb_params::BBSearchParams, ipm_params::IPM)
    start_time = time()
    
    # Initialize branch and bound
    best_objective = Inf
    best_solution = zeros(length(milp.c))
    best_bound = -Inf
    nodes_explored = 0
    
    # Priority queue for nodes (min-heap based on LP objective)
    node_queue = PriorityQueue{BBNode, Float64}()
    
    # Create root node using the BBNode structure from types.jl
    root_node = BBNode(
        copy(milp.lb),             # lower_bounds
        copy(milp.ub),             # upper_bounds
        Inf,                       # relaxation_value
        zeros(length(milp.c)),     # solution
        false,                     # is_integer_feasible
        0                          # depth
    )
    
    # Solve root LP relaxation
    solve_node_lp!(root_node, milp, ipm_params)
    
    if root_node.relaxation_value < Inf
        best_bound = root_node.relaxation_value
        enqueue!(node_queue, root_node, root_node.relaxation_value)
        
        if bb_params.verbose
            println("Root LP objective: $(root_node.relaxation_value)")
        end
    else
        return MINLP_Solution(
            best_solution, Inf, "Infeasible", 0, Inf, time() - start_time, -Inf
        )
    end
    
    # Main branch and bound loop
    while !isempty(node_queue) && nodes_explored < bb_params.max_nodes
        elapsed_time = time() - start_time
        if elapsed_time > bb_params.time_limit
            break
        end
        
        # Get node with best (lowest) bound
        current_node = dequeue!(node_queue)
        nodes_explored += 1
        
        if bb_params.verbose && nodes_explored % 100 == 0
            gap = best_objective < Inf ? (best_objective - best_bound) / max(abs(best_objective), 1.0) : Inf
            println("Nodes: $nodes_explored, Best obj: $best_objective, Best bound: $best_bound, Gap: $(gap*100)%")
        end
        
        # Prune if bound is worse than current best
        if current_node.relaxation_value >= best_objective - bb_params.gap_tolerance
            continue
        end
        
        # Update best bound
        best_bound = max(best_bound, current_node.relaxation_value)
        
        # Check if integer feasible
        if check_integer_feasibility(current_node, milp, bb_params.integer_tolerance)
            current_node.is_integer_feasible = true
            if current_node.relaxation_value < best_objective
                best_objective = current_node.relaxation_value
                best_solution = current_node.solution[1:length(milp.c)]
                
                if bb_params.verbose
                    println("New best integer solution found: $best_objective")
                end
            end
            continue
        end
        
        # Branch on most fractional integer variable
        branching_var = select_branching_variable(current_node, milp, bb_params.integer_tolerance)
        
        if branching_var == -1
            continue  # No fractional integer variables found
        end
        
        branching_value = current_node.solution[branching_var]
        
        # Create left child (var <= floor(value))
        left_child = create_child_node(current_node, branching_var, floor(branching_value), true)
        solve_node_lp!(left_child, milp, ipm_params)
        if left_child.relaxation_value < best_objective - bb_params.gap_tolerance
            enqueue!(node_queue, left_child, left_child.relaxation_value)
        end
        
        # Create right child (var >= ceil(value))
        right_child = create_child_node(current_node, branching_var, ceil(branching_value), false)
        solve_node_lp!(right_child, milp, ipm_params)
        if right_child.relaxation_value < best_objective - bb_params.gap_tolerance
            enqueue!(node_queue, right_child, right_child.relaxation_value)
        end
        
        # Check termination criteria
        if best_objective < Inf
            gap = (best_objective - best_bound) / max(abs(best_objective), 1.0)
            if gap <= bb_params.gap_tolerance
                break
            end
        end
    end
    
    # Determine final status
    final_gap = best_objective < Inf ? (best_objective - best_bound) / max(abs(best_objective), 1.0) : Inf
    elapsed_time = time() - start_time
    
    status = if best_objective < Inf
        if final_gap <= bb_params.gap_tolerance
            "Optimal"
        elseif elapsed_time > bb_params.time_limit
            "Time limit reached"
        elseif nodes_explored >= bb_params.max_nodes
            "Node limit reached"
        else
            "Feasible"
        end
    else
        "No integer solution found"
    end
    
    return MINLP_Solution(
        best_solution, best_objective, status, nodes_explored, 
        final_gap, elapsed_time, best_bound
    )
end

"""
    solve_node_lp!(node::BBNode, milp::MILP, ipm_params::IPM)

Solve LP relaxation for a branch and bound node.
"""
function solve_node_lp!(node::BBNode, milp::MILP, ipm_params::IPM)
    # Create LP with node bounds
    lp = LP(milp.c, milp.A, milp.b, milp.Aeq, milp.beq, node.lower_bounds, node.upper_bounds)
    
    try
        # Solve using interior point method
        solution = interior_point_method(lp, ipm_params)
        
        if solution.eflag && isfinite(solution.obj)
            node.solution = solution.x
            node.relaxation_value = solution.obj
        else
            node.relaxation_value = Inf
        end
        
    catch e
        node.relaxation_value = Inf
        if ipm_params.verbose
            println("LP solve failed for node: $e")
        end
    end
end

"""
    check_integer_feasibility(node::BBNode, milp::MILP, tolerance::Float64)

Check if the LP solution satisfies integer constraints.
"""
function check_integer_feasibility(node::BBNode, milp::MILP, tolerance::Float64)
    if node.relaxation_value == Inf
        return false
    end
    
    x = node.solution
    
    # Check integer variables
    for i in milp.integer_vars
        if i <= length(x) && abs(x[i] - round(x[i])) > tolerance
            return false
        end
    end
    
    # Check binary variables
    for i in milp.binary_vars
        if i <= length(x) && (x[i] < -tolerance || x[i] > 1.0 + tolerance)
            return false
        end
        if abs(x[i] - round(x[i])) > tolerance
            return false
        end
    end
    
    return true
end

"""
    select_branching_variable(node::BBNode, milp::MILP, tolerance::Float64)

Select the most fractional integer variable for branching.
"""
function select_branching_variable(node::BBNode, milp::MILP, tolerance::Float64)
    if node.relaxation_value == Inf
        return -1
    end
    
    x = node.solution
    max_fractionality = 0.0
    branching_var = -1
    
    # Check integer variables
    for i in milp.integer_vars
        if i <= length(x)
            fractionality = abs(x[i] - round(x[i]))
            if fractionality > tolerance && fractionality > max_fractionality
                max_fractionality = fractionality
                branching_var = i
            end
        end
    end
    
    return branching_var
end

"""
    create_child_node(parent::BBNode, branching_var::Int, branching_value::Float64, is_left::Bool)

Create a child node with updated bounds.
"""
function create_child_node(parent::BBNode, branching_var::Int, branching_value::Float64, is_left::Bool)
    # Copy parent bounds
    new_lb = copy(parent.lower_bounds)
    new_ub = copy(parent.upper_bounds)
    
    # Update bounds based on branching
    if is_left
        # Left child: var <= branching_value
        new_ub[branching_var] = min(new_ub[branching_var], branching_value)
    else
        # Right child: var >= branching_value
        new_lb[branching_var] = max(new_lb[branching_var], branching_value)
    end
    
    return BBNode(
        new_lb, new_ub, Inf, zeros(length(parent.solution)), false, parent.depth + 1
    )
end
function create_child_node(parent::BBNode, id::Int, branching_var::Int, 
                          branching_value::Float64, is_left::Bool, milp::MILP)
    # Copy parent bounds
    new_lb = copy(parent.lb)
    new_ub = copy(parent.ub)
    
    # Update bounds based on branching
    if is_left
        # Left child: var <= branching_value
        new_ub[branching_var] = min(new_ub[branching_var], branching_value)
    else
        # Right child: var >= branching_value
        new_lb[branching_var] = max(new_lb[branching_var], branching_value)
    end
    
    return BBNode(
        id, new_lb, new_ub, parent.id, branching_var, branching_value,
        is_left, nothing, Inf, false, false, false, parent.depth + 1
    )
end

