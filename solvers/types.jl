# =============================================================================
# LINEAR PROGRAMMING STRUCTURES
# =============================================================================
using SparseArrays
"""
Standard form of a Linear Programming (LP) problem:
    minimize    c'x
    subject to  Aeq * x = beq     (equality constraints)
                A * x <= b        (inequality constraints)  
                lb <= x <= ub     (variable bounds)
                
where:
    - c: objective coefficient vector (n×1)
    - Aeq: equality constraint matrix (meq×n)
    - beq: equality constraint RHS vector (meq×1)
    - A: inequality constraint matrix (m×n)
    - b: inequality constraint RHS vector (m×1)
    - lb: variable lower bounds (n×1)
    - ub: variable upper bounds (n×1)
    - x: decision variables (n×1)
"""
struct LP
    c::Vector{Float64}                      # Objective coefficients
    A::SparseMatrixCSC{Float64}            # Inequality constraint matrix (m×n)
    b::Vector{Float64}                      # Inequality constraint RHS
    Aeq::SparseMatrixCSC{Float64}          # Equality constraint matrix (meq×n)
    beq::Vector{Float64}                    # Equality constraint RHS
    lb::Vector{Float64}                     # Variable lower bounds
    ub::Vector{Float64}                     # Variable upper bounds
end

# =============================================================================
# NONLINEAR PROGRAMMING STRUCTURES
# =============================================================================

"""
Standard form of a Nonconvex Optimization (NLP) problem:
    minimize    f(x)
    subject to  g(x) = 0          (equality constraints)
                h(x) <= 0         (inequality constraints)
                lb <= x <= ub     (variable bounds)
                
where:
    - f: objective function R^n → R
    - ∇f: gradient of objective function R^n → R^n
    - ∇²f: Hessian of objective function R^n → R^(n×n)
    - g: equality constraint function R^n → R^meq
    - ∇g: Jacobian of equality constraints R^n → R^(meq×n)
    - h: inequality constraint function R^n → R^m
    - ∇h: Jacobian of inequality constraints R^n → R^(m×n)
    - Lx: Lagrangian gradient ∇L(x,λ,μ) = ∇f + ∇g'λ + ∇h'μ
    - Lxx: Lagrangian Hessian ∇²L(x,λ,μ) = ∇²f + Σλᵢ∇²gᵢ + Σμⱼ∇²hⱼ
    - x0: initial point for optimization
"""
struct NonConvexOPT
    f::Function                             # Objective function f: R^n → R
    ∇f::Function                           # Gradient ∇f: R^n → R^n
    ∇2f::Function                          # Hessian ∇²f: R^n → R^(n×n)
    g::Function                             # Equality constraints g: R^n → R^meq
    ∇g::Function                           # Equality Jacobian ∇g: R^n → R^(meq×n)
    h::Function                             # Inequality constraints h: R^n → R^m
    ∇h::Function                           # Inequality Jacobian ∇h: R^n → R^(m×n)
    lb::Vector{Float64}                     # Variable lower bounds
    ub::Vector{Float64}                     # Variable upper bounds
    Lx::Function                            # Lagrangian gradient: (x,λ,μ) → R^n
    Lxx::Function                           # Lagrangian Hessian: (x,λ,μ) → R^(n×n)
    x0::Vector{Float64}                     # Initial point
end

# =============================================================================
# MIXED INTEGER PROGRAMMING STRUCTURES
# =============================================================================

"""
Mixed Integer Linear Programming (MILP) problem - immutable structure.
"""
struct MILP
    c::Vector{Float64}                      # Objective coefficients
    A::SparseMatrixCSC{Float64}            # Inequality constraint matrix (m×n)
    b::Vector{Float64}                      # Inequality constraint RHS
    Aeq::SparseMatrixCSC{Float64}          # Equality constraint matrix (meq×n)
    beq::Vector{Float64}                    # Equality constraint RHS
    lb::Vector{Float64}                     # Variable lower bounds
    ub::Vector{Float64}                     # Variable upper bounds
    integer_vars::Vector{Int}               # Indices of integer variables
    binary_vars::Vector{Int}                # Indices of binary variables (subset of integer_vars)
end

"""
Mixed Integer Nonlinear Programming (MINLP) problem - immutable structure.
"""
struct MINLP
    problem::NonConvexOPT                   # Base nonconvex optimization problem
    integer_vars::Vector{Int}               # Indices of integer variables
    binary_vars::Vector{Int}                # Indices of binary variables (subset of integer_vars)
end

# =============================================================================
# ALGORITHM PARAMETER STRUCTURES
# =============================================================================

"""
Interior Point Method parameters - immutable configuration.
"""
struct IPM_LP
    tol::Float64                            # Convergence tolerance
    max_iter::Int                           # Maximum iterations
    positive_tol::Float64                   # Positivity tolerance for slack variables
    feasible_tol::Float64                   # Feasibility tolerance
    initial_point_projection::Bool          # Project initial point to feasible region
end

"""
Interior Point Method parameters for Nonlinear Programming - immutable configuration.
"""
struct IPM
    tol::Float64                            # Convergence tolerance
    max_iter::Int                           # Maximum iterations
    positive_tol::Float64                   # Positivity tolerance for slack variables
    feasible_tol::Float64                   # Feasibility tolerance
    initial_point_projection::Bool          # Project initial point to feasible region
    verbose::Bool                           # Print iteration information
    ξ::Float64                              # Barrier parameter reduction factor
    precision_cleanup_tol::Float64          # Numerical precision cleanup tolerance
    linear_solver::String                   # Linear solver: "direct", "bicgstab", "ldl", "qr"
    linear_solver_tol::Float64              # Tolerance for iterative linear solvers
    linear_solver_max_iter::Int             # Maximum iterations for iterative solvers
end

"""
Branch and Bound search parameters - immutable configuration.
"""
struct BBSearchParams
    tolerance::Float64                      # Optimality tolerance
    integer_tolerance::Float64              # Integer feasibility tolerance
    max_iterations::Int                     # Max IPM iterations per node
    max_nodes::Int                          # Maximum number of nodes to explore
    node_limit::Int                         # Alias for max_nodes (backward compatibility)
    time_limit::Float64                     # Maximum time in seconds
    gap_tolerance::Float64                  # Relative gap tolerance for termination
    verbose::Bool                           # Print detailed progress information
    branching_strategy::String              # "most_fractional", "strong_branching"
    node_selection::String                  # "best_first", "depth_first", "breadth_first"
end

"""
Cutting plane parameters - immutable configuration.
"""
struct CutParams
    max_cut_iterations::Int                 # Maximum cutting plane iterations per node
    max_cuts_per_iteration::Int             # Maximum cuts to add per iteration
    min_improvement::Float64                # Minimum objective improvement to continue cuts
    integer_tolerance::Float64              # Tolerance for integer feasibility
    use_gomory_cuts::Bool                   # Enable Gomory fractional cuts
    use_knapsack_cuts::Bool                 # Enable knapsack cover cuts
    use_cover_cuts::Bool                    # Enable clique cuts
    verbose::Bool                           # Print cutting plane information
end

# =============================================================================
# MUTABLE STRUCTURES (Need to be modified during algorithms)
# =============================================================================

"""
Iteration history for Interior Point Method - mutable because it's built incrementally.
"""
mutable struct History
    x_record::Matrix{Float64}               # Primal iterates (n × iterations)
    λ_record::Matrix{Float64}               # Equality multipliers (meq × iterations)
    μ_record::Matrix{Float64}               # Inequality multipliers (m × iterations)
    z_record::Matrix{Float64}               # Slack variables (m × iterations)
    obj_record::Vector{Float64}             # Objective values
    feascond_record::Vector{Float64}        # Feasibility condition values
    gradcond_record::Vector{Float64}        # Gradient condition values
    compcond_record::Vector{Float64}        # Complementarity condition values
    costcond_record::Vector{Float64}        # Cost reduction condition values
end

"""
Solution structure for Linear Programming - immutable result.
"""
struct Solution
    x::Vector{Float64}                      # Primal solution
    λ::Vector{Float64}                      # Dual multipliers (equality constraints)
    s::Vector{Float64}                      # Slack variables (inequality constraints)
    obj::Float64                            # Objective function value
end

"""
Solution structure for Interior Point Method - immutable result.
"""
struct IPM_Solution
    x::Vector{Float64}                      # Primal solution
    λ::Vector{Float64}                      # Dual multipliers (equality constraints)
    μ::Vector{Float64}                      # Dual multipliers (inequality constraints)
    obj::Float64                            # Objective function value
    eflag::Bool                             # Convergence flag (true if converged)
    hist::History                           # Iteration history
end

"""
Solution structure for Mixed Integer Programming - immutable result.
"""
struct MINLP_Solution
    x::Vector{Float64}                      # Optimal solution
    objective::Float64                      # Optimal objective value
    status::String                          # Solution status: "optimal", "infeasible", "unbounded", "time_limit"
    nodes_explored::Int                     # Number of branch-and-bound nodes explored
    gap::Float64                            # Final optimality gap
    solve_time::Float64                     # Total solution time in seconds
    best_bound::Float64                     # Best dual bound found
end

"""
Branch and Bound tree node - mutable because node state changes during search.
"""
mutable struct BBNode
    # Bounds (with aliases for compatibility)
    lower_bounds::Vector{Float64}           # Variable lower bounds for this node
    upper_bounds::Vector{Float64}           # Variable upper bounds for this node
    lb::Vector{Float64}                     # Alias for lower_bounds
    ub::Vector{Float64}                     # Alias for upper_bounds
    
    # Solution information
    relaxation_value::Float64               # Objective value of relaxation solution
    solution::Vector{Float64}               # Relaxation solution
    is_integer_feasible::Bool               # Whether solution satisfies integer constraints
    depth::Int                              # Depth in the branch-and-bound tree
    
    # Extended fields (for advanced algorithms)
    id::Int                                 # Node identifier
    parent_id::Int                         # Parent node ID
    branching_var::Int                     # Variable used for branching
    branching_value::Float64               # Value used for branching
    is_left_child::Bool                    # True if left child, false if right
    lp_solution::Union{Nothing, Any}       # LP solution (can be LP_Solution type)
    lp_objective::Float64                  # LP objective value
    is_feasible::Bool                      # LP feasibility status
    is_pruned::Bool                        # Pruning status
    
    # Primary constructor with all fields (13 parameters - for branch_bound_algorithm.jl)
    function BBNode(id::Int, lb::Vector{Float64}, ub::Vector{Float64}, 
                   parent_id::Int, branching_var::Int, branching_value::Float64,
                   is_left_child::Bool, lp_solution, lp_objective::Float64,
                   is_feasible::Bool, is_integer_feasible::Bool, is_pruned::Bool, depth::Int)
        lb_copy = copy(lb)
        ub_copy = copy(ub)
        new(lb_copy, ub_copy, lb_copy, ub_copy,
            lp_objective, Float64[], is_integer_feasible, depth,
            id, parent_id, branching_var, branching_value, is_left_child, lp_solution,
            lp_objective, is_feasible, is_pruned)
    end
    
    # Backward compatibility constructor (6 parameters)
    function BBNode(lb::Vector{Float64}, ub::Vector{Float64}, relaxation_value::Float64,
                   solution::Vector{Float64}, is_integer_feasible::Bool, depth::Int)
        lb_copy = copy(lb)
        ub_copy = copy(ub)
        new(lb_copy, ub_copy, lb_copy, ub_copy,
            relaxation_value, copy(solution), is_integer_feasible, depth,
            0, -1, -1, 0.0, true, nothing, relaxation_value, true, false)
    end
end

"""
Branch and Bound history tracking - mutable because it's continuously updated.
"""
mutable struct BBHistory
    node_id::Int                            # Counter for node IDs
    nodes_created::Vector{Int}              # List of created node IDs
    node_bounds::Vector{Tuple{Vector{Float64}, Vector{Float64}}}  # (lb, ub) for each node
    node_objectives::Vector{Float64}        # Relaxation objective values
    node_solutions::Vector{Vector{Float64}} # Relaxation solutions
    node_parents::Vector{Int}               # Parent node ID for each node
    node_depths::Vector{Int}                # Tree depth for each node
    branching_variables::Vector{Int}        # Variable branched on for each node
    branching_values::Vector{Float64}       # Value at which branching occurred
    branch_directions::Vector{String}       # "left", "right", or "root"
    incumbent_updates::Vector{Tuple{Int, Float64, Vector{Float64}}}  # (node_id, obj, sol)
    pruned_nodes::Vector{Tuple{Int, String}} # (node_id, reason) for pruned nodes
    processing_order::Vector{Int}           # Order in which nodes were processed
    queue_sizes::Vector{Int}                # Queue size when each node was processed
    timestamps::Vector{Float64}             # Timestamp for each event
end

"""
Linear operator wrapper - immutable utility structure.
"""
struct LinearOperator
    matvec::Function                        # Matrix-vector multiplication function
    size::Tuple{Int, Int}                   # Dimensions (m, n)
end

# =============================================================================
# CONSTRUCTOR FUNCTIONS
# =============================================================================

function BBHistory()
    return BBHistory(
        0, Int[], Tuple{Vector{Float64}, Vector{Float64}}[], Float64[], Vector{Vector{Float64}}(),
        Int[], Int[], Int[], Float64[], String[], 
        Tuple{Int, Float64, Vector{Float64}}[], Tuple{Int, String}[], 
        Int[], Int[], Float64[]
    )
end

function BBSearchParams(; tolerance=1e-6, integer_tolerance=1e-6, max_iterations=100, 
                        max_nodes=1000, time_limit=3600.0, gap_tolerance=1e-4, 
                        verbose=true, branching_strategy="most_fractional", 
                        node_selection="best_first")
    return BBSearchParams(tolerance, integer_tolerance, max_iterations, max_nodes, 
                         max_nodes, time_limit, gap_tolerance, verbose, branching_strategy, node_selection)
end

function CutParams(; max_cut_iterations=5, max_cuts_per_iteration=10, min_improvement=1e-6,
                   integer_tolerance=1e-6, use_gomory_cuts=true, use_knapsack_cuts=true,
                   use_cover_cuts=true, verbose=false)
    return CutParams(max_cut_iterations, max_cuts_per_iteration, min_improvement,
                    integer_tolerance, use_gomory_cuts, use_knapsack_cuts, use_cover_cuts, verbose)
end

function IPM(tol, max_iter, positive_tol, feasible_tol, initial_point_projection, verbose, ξ; 
             precision_cleanup_tol=1e-10, linear_solver="direct", linear_solver_tol=1e-12, 
             linear_solver_max_iter=1000)
    return IPM(tol, max_iter, positive_tol, feasible_tol, initial_point_projection, verbose, ξ, 
               precision_cleanup_tol, linear_solver, linear_solver_tol, linear_solver_max_iter)
end

# =============================================================================
# LOGGING FUNCTIONS FOR BRANCH AND BOUND
# =============================================================================

function log_node_creation!(hist::BBHistory, parent_id::Int, lb::Vector{Float64}, ub::Vector{Float64}, 
                           branch_var::Int, branch_val::Float64, direction::String, depth::Int)
    hist.node_id += 1
    push!(hist.nodes_created, hist.node_id)
    push!(hist.node_bounds, (copy(lb), copy(ub)))
    push!(hist.node_parents, parent_id)
    push!(hist.node_depths, depth)
    push!(hist.branching_variables, branch_var)
    push!(hist.branching_values, branch_val)
    push!(hist.branch_directions, direction)
    push!(hist.timestamps, time())
    return hist.node_id
end

function log_node_solution!(hist::BBHistory, node_id::Int, obj_val::Float64, solution::Vector{Float64})
    while length(hist.node_objectives) < node_id
        push!(hist.node_objectives, 0.0)
        push!(hist.node_solutions, Float64[])
    end
    hist.node_objectives[node_id] = obj_val
    hist.node_solutions[node_id] = copy(solution)
end

function log_incumbent_update!(hist::BBHistory, node_id::Int, obj_val::Float64, solution::Vector{Float64})
    push!(hist.incumbent_updates, (node_id, obj_val, copy(solution)))
end

function log_node_pruning!(hist::BBHistory, node_id::Int, reason::String)
    push!(hist.pruned_nodes, (node_id, reason))
end

function log_processing!(hist::BBHistory, node_id::Int, queue_size::Int)
    push!(hist.processing_order, node_id)
    push!(hist.queue_sizes, queue_size)
end

function print_bb_history(hist::BBHistory, integer_vars::Vector{Int})
    println("\n" * "="^80)
    println("BRANCH AND BOUND SEARCH HISTORY")
    println("="^80)
    
    println("\n📊 SEARCH STATISTICS:")
    println("   Total nodes created: $(length(hist.nodes_created))")
    println("   Nodes processed: $(length(hist.processing_order))")
    println("   Incumbent updates: $(length(hist.incumbent_updates))")
    println("   Nodes pruned: $(length(hist.pruned_nodes))")
    
    println("\n🌳 NODE CREATION TREE:")
    for i in 1:length(hist.nodes_created)
        node_id = hist.nodes_created[i]
        parent_id = hist.node_parents[i]
        depth = hist.node_depths[i]
        branch_var = hist.branching_variables[i]
        branch_val = hist.branch_directions[i]
        direction = hist.branch_directions[i]
        
        indent = "  " ^ depth
        if parent_id == 0
            println("$(indent)📍 Node $node_id (ROOT)")
        else
            var_name = "x[$branch_var]"
            constraint = direction == "left" ? "$var_name ≤ $(floor(branch_val))" : "$var_name ≥ $(ceil(branch_val))"
            println("$(indent)├─ Node $node_id ($direction child of $parent_id): $constraint")
        end
        
        # Show bounds for integer variables
        if i <= length(hist.node_bounds)
            lb, ub = hist.node_bounds[i]
            int_bounds = []
            for var in integer_vars
                if var <= length(lb) && var <= length(ub)
                    push!(int_bounds, "x[$var] ∈ [$(lb[var]), $(ub[var])]")
                end
            end
            if !isempty(int_bounds)
                println("$(indent)   Integer bounds: $(join(int_bounds, ", "))")
            end
        end
    end
    
    println("\n🔍 PROCESSING ORDER:")
    for i in 1:length(hist.processing_order)
        node_id = hist.processing_order[i]
        queue_size = hist.queue_sizes[i]
        
        if node_id <= length(hist.node_objectives) && hist.node_objectives[node_id] != 0.0
            obj_val = hist.node_objectives[node_id]
            solution = hist.node_solutions[node_id]
            if !isempty(solution) && length(solution) >= maximum(integer_vars)
                int_vals = [solution[var] for var in integer_vars]
                is_integer = all(abs(val - round(val)) < 1e-6 for val in int_vals)
                status = is_integer ? "✅ INTEGER" : "🔄 FRACTIONAL"
                
                println("   Step $i: Process Node $node_id (obj = $(round(obj_val, digits=4))) $status")
                println("            Integer vars: $(join(["x[$var]=$(round(solution[var], digits=3))" for var in integer_vars], ", "))")
                println("            Queue size: $queue_size")
            else
                println("   Step $i: Process Node $node_id (obj = $(round(obj_val, digits=4))) - Queue size: $queue_size")
            end
        else
            println("   Step $i: Process Node $node_id (INFEASIBLE) - Queue size: $queue_size")
        end
    end
    
    println("\n🎯 INCUMBENT UPDATES:")
    if isempty(hist.incumbent_updates)
        println("   No incumbent found")
    else
        for (i, (node_id, obj_val, solution)) in enumerate(hist.incumbent_updates)
            int_vals = join(["x[$var]=$(round(solution[var]))" for var in integer_vars if var <= length(solution)], ", ")
            println("   Update $i: Node $node_id found incumbent = $(round(obj_val, digits=4))")
            println("              Solution: $int_vals")
        end
    end
    
    println("\n✂️ PRUNED NODES:")
    if isempty(hist.pruned_nodes)
        println("   No nodes were explicitly pruned")
    else
        for (node_id, reason) in hist.pruned_nodes
            println("   Node $node_id: $reason")
        end
    end
    
    println("\n" * "="^80)
end

# Add type aliases to match the branch_bound_algorithm.jl expectations
const BBParams = BBSearchParams
const MILP_Solution = MINLP_Solution