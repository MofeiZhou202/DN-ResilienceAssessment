using SparseArrays
using Gurobi
using Base: error

const QUIET_GUROBI_IO = devnull

quiet_gurobi(f) = redirect_stdout(QUIET_GUROBI_IO) do
    redirect_stderr(QUIET_GUROBI_IO) do
        return f()
    end
end
#=
    Standard mixed-integer linear programming (MILP) solver using Gurobi.
        The programming is to solve the following type of MILP problems:
        ⩟ min/max c'x
        ⩟ s.t. Ax ≤ b (inequality constraints)
        ⩟ Aeqx = beq (equality constraints)
        ⩟ lb ≤ x ≤ ub (bounds)
        ⩟ x ∈ {C, I, B} (continuous, integer, binary)
    
=#
function solve_milp_sparse(
    c::Vector{Float64},
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    Aeq::SparseMatrixCSC{Float64},
    beq::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    vtype::Vector{Char};
    modelsense::String="min",
    verbose::Bool=true,
    debug::Bool=false
)
    # Validate input dimensions
    n = length(c)
    if size(A, 2) != n && !isempty(A)
        error("Inequality constraint matrix A has $(size(A, 2)) columns, but expected $n")
    end
    if size(Aeq, 2) != n && !isempty(Aeq)
        error("Equality constraint matrix Aeq has $(size(Aeq, 2)) columns, but expected $n")
    end
    if length(lb) != n
        error("Lower bounds vector has $(length(lb)) elements, but expected $n")
    end
    if length(ub) != n
        error("Upper bounds vector has $(length(ub)) elements, but expected $n")
    end
    if length(vtype) != n
        error("Variable type vector has $(length(vtype)) elements, but expected $n")
    end
    
    # Initialize Gurobi environment
    env_p = Ref{Ptr{Cvoid}}()
    error = quiet_gurobi(() -> GRBloadenv(env_p, C_NULL))  # Use C_NULL instead of empty string
    env = env_p[]
    # Disable any initial console output immediately
    error = quiet_gurobi(() -> GRBsetparam(env, "LogToConsole", "0"))
    # Disable academic license output message
    error = quiet_gurobi(() -> GRBsetparam(env, "DisplayInterval", "0"))
    
    # Set verbosity
    error = quiet_gurobi(() -> GRBsetparam(env, "OutputFlag", verbose ? "1" : "0"))
    # Disable additional logging parameters
    error = quiet_gurobi(() -> GRBsetparam(env, "LogFile", ""))
    # Set constraint handling precision
    error = quiet_gurobi(() -> GRBsetparam(env, "FeasibilityTol", "1e-9"))
    error = quiet_gurobi(() -> GRBsetparam(env, "IntFeasTol", "1e-9"))
    error = quiet_gurobi(() -> GRBsetparam(env, "OptimalityTol", "1e-9"))

    # Create a new model
    model_p = Ref{Ptr{Cvoid}}()
    error = GRBnewmodel(env, model_p, "milp", 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL)
    model = model_p[]
    
    # Add variables
    vtype_int8 = map(x -> Int8(x), vtype)
    error = GRBaddvars(
        model,    # model
        n,        # numvars
        0,        # numnz
        C_NULL,   # *vbeg
        C_NULL,   # *vind
        C_NULL,   # *vval
        c,        # *obj
        lb,       # *lb
        ub,       # *ub
        vtype_int8, # *vtype
        C_NULL    # **varnames
    )
    
    # Add inequality constraints (A*x <= b) using sparse format
    if !isempty(A) && !isempty(b)
        m_ineq = size(A, 1)
        # Prepare data for GRBaddconstrs
        row_starts = Vector{Cint}(undef, m_ineq + 1)
        row_indices = Vector{Cint}()
        row_values = Vector{Float64}()
        row_senses = Vector{UInt8}(undef, m_ineq)
        row_rhs = Vector{Float64}(undef, m_ineq)
        
        row_starts[1] = 0
        for i in 1:m_ineq
            # Get the row data directly from the CSC format
            row_vals = Vector{Float64}()
            row_inds = Vector{Int}()
            
            # Manually extract non-zero elements from this row
            for j in 1:n
                val = A[i, j]
                if abs(val) > 1e-10  # Use a small threshold to avoid numerical issues
                    push!(row_vals, val)
                    push!(row_inds, j)
                end
            end
            
            if !isempty(row_inds)
                # Convert to 0-based indexing for Gurobi
                indices_0based = convert(Vector{Cint}, row_inds .- 1)
                
                append!(row_indices, indices_0based)
                append!(row_values, row_vals)
                row_starts[i + 1] = length(row_indices)
                row_senses[i] = GRB_LESS_EQUAL
                row_rhs[i] = b[i]
            end
        end
        
        error = GRBaddconstrs(
            model,         # *model
            m_ineq,        # numconstrs
            length(row_indices), # numnz
            row_starts,    # *cbeg
            row_indices,   # *cind
            row_values,    # *cval
            row_senses,    # *sense
            row_rhs,       # *rhs
            C_NULL         # **constrname
        )
        if error != 0
            println("Error adding inequality constraints: $error")
        end
    end
    
    # Add equality constraints (Aeq*x = beq) using sparse format
    if !isempty(Aeq) && !isempty(beq)
        m_eq = size(Aeq, 1)
        for i in 1:m_eq
            # Get the row data directly from the CSC format
            row_vals = Vector{Float64}()
            row_inds = Vector{Int}()
            
            # Manually extract non-zero elements from this row
            for j in 1:n
                val = Aeq[i, j]
                if abs(val) > 1e-10  # Use a small threshold to avoid numerical issues
                    push!(row_vals, val)
                    push!(row_inds, j)
                end
            end
            
            if !isempty(row_inds)
                # Convert to 0-based indexing for Gurobi
                indices_0based = convert(Vector{Cint}, row_inds .- 1)
                
                error = GRBaddconstr(
                    model,         # *model
                    length(row_inds), # numnz
                    indices_0based, # *cind
                    row_vals,      # *cval
                    GRB_EQUAL,     # sense
                    beq[i],        # rhs
                    C_NULL         # *constrname
                )
                if error != 0
                    println("Error adding equality constraint $i: $error")
                end
            end
        end
    end
    
    # Set optimization direction
    if lowercase(modelsense) == "min"
        error = GRBsetintattr(model, "ModelSense", GRB_MINIMIZE)
    elseif lowercase(modelsense) == "max"
        error = GRBsetintattr(model, "ModelSense", GRB_MAXIMIZE)
    else
        error("Invalid optimization sense: $sense. Use 'min' or 'max'")
    end
    
    # Update model and optimize
    error = GRBupdatemodel(model)
    if error != 0
        println("Error updating model: $error")
    end
    
    error = GRBoptimize(model)
    if error != 0
        println("Error optimizing model: $error")
    end
    
    # Get results
    optimstatus = Ref{Cint}()
    objval = Ref{Cdouble}()
    runtime = Ref{Cdouble}()
    mip_gap = Ref{Cdouble}()
    
    # Get solution status
    error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, optimstatus)
    
    # Get runtime
    error = GRBgetdblattr(model, GRB_DBL_ATTR_RUNTIME, runtime)
    
    # Prepare solution array
    sol = zeros(n)
    
    # Map status code to string
    status_map = Dict(
        GRB_OPTIMAL => "Optimal solution found",
        GRB_INFEASIBLE => "Problem is infeasible",
        GRB_UNBOUNDED => "Problem is unbounded",
        GRB_INF_OR_UNBD => "Problem is infeasible or unbounded",
        GRB_ITERATION_LIMIT => "Iteration limit reached",
        GRB_NODE_LIMIT => "Node limit reached",
        GRB_TIME_LIMIT => "Time limit reached",
        GRB_SOLUTION_LIMIT => "Solution limit reached",
        GRB_INTERRUPTED => "Optimization was interrupted",
        GRB_NUMERIC => "Numerical issues encountered",
        GRB_SUBOPTIMAL => "Suboptimal solution found",
        GRB_INPROGRESS => "Optimization in progress",
        GRB_USER_OBJ_LIMIT => "User objective limit reached"
    )
    
    status_name = get(status_map, optimstatus[], "Unknown status ($(optimstatus[]))")
    
    # Check if optimal solution was found
    if optimstatus[] == GRB_OPTIMAL || optimstatus[] == GRB_SUBOPTIMAL
        error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, objval)
        error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, n, sol)
        
        # Try to get MIP gap if it's a MIP problem
        has_discrete = any(vtype .!= 'C')
        if has_discrete
            try
                error = GRBgetdblattr(model, GRB_DBL_ATTR_MIPGAP, mip_gap)
            catch
                mip_gap[] = 0.0
            end
        else
            mip_gap[] = 0.0
        end
        
        # Verify constraints if verbose
        if debug
            println("\nSolution verification:")
            
            # Check inequality constraints
            if !isempty(A) && !isempty(b)
                max_violation = 0.0
                violation_count = 0
                
                for i in axes(A, 1)
                    lhs = 0.0
                    # Manually compute dot product to avoid potential sparse matrix issues
                    for j in 1:n
                        lhs += A[i, j] * sol[j]
                    end
                    
                    violation = max(0.0, lhs - b[i])
                    max_violation = max(max_violation, violation)
                    if violation > 1e-6
                        violation_count += 1
                        if violation_count <= 10  # Limit output to first 10 violations
                            println("Inequality constraint $i violated: $lhs > $(b[i]), violation = $violation")
                        end
                    end
                end
                
                if violation_count > 10
                    println("... and $(violation_count - 10) more inequality violations")
                end
                
                println("Max inequality violation: $max_violation (total violations: $violation_count)")
            end
            
            # Check equality constraints
            if !isempty(Aeq) && !isempty(beq)
                max_violation = 0.0
                violation_count = 0
                
                for i in axes(Aeq, 1)
                    lhs = 0.0
                    # Manually compute dot product to avoid potential sparse matrix issues
                    for j in 1:n
                        lhs += Aeq[i, j] * sol[j]
                    end
                    
                    violation = abs(lhs - beq[i])
                    max_violation = max(max_violation, violation)
                    if violation > 1e-6
                        violation_count += 1
                        if violation_count <= 10  # Limit output to first 10 violations
                            println("Equality constraint $i violated: $lhs != $(beq[i]), violation = $violation")
                        end
                    end
                end
                
                if violation_count > 10
                    println("... and $(violation_count - 10) more equality violations")
                end
                
                println("Max equality violation: $max_violation (total violations: $violation_count)")
            end
        end
    else
        objval[] = NaN
        mip_gap[] = NaN
        
        # Print a message about the non-optimal status
        if verbose
            println("No optimal solution found. Status: $(status_name)")
        end
    end
    
    # Free resources
    GRBfreemodel(model)
    GRBfreeenv(env)
    
    # Return results
    return Dict(
        :x => sol, 
        :objval => objval[], 
        :runtime => runtime[], 
        :mip_gap => mip_gap[],
        :status => optimstatus[],
        :status_name => status_name
    )
end

function solve_milp_sparse(params::Dict)
    # Extract required parameters
    c = params[:c]
    n = length(c)
    
    # Extract inequality constraints (required)
    A = params[:A]
    b = params[:b]
    
    # Extract equality constraints (optional)
    Aeq = get(params, :Aeq, spzeros(0, n))
    beq = get(params, :beq, Float64[])
    
    # Extract bounds (required)
    lb = params[:lb]
    ub = params[:ub]
    
    # Extract variable types (required)
    vtype = params[:vtype]
    
    # Extract optional parameters
    modelsense = get(params, :sense, "min")
    verbose = get(params, :verbose, false)
    debug = get(params, :debug, false)
    
    # Call the original function with the extracted parameters
    return solve_milp_sparse(
        c, A, b, Aeq, beq, lb, ub, vtype;
        modelsense=modelsense, verbose=verbose, debug=debug
    )
end


#=
    Standard mixed-integer linear programming (MILP) solver using Gurobi.
        The programming is to solve the following type of MILP problems:
        ⩟ min/max c'x
        ⩟ s.t. Ax >= b (inequality constraints)
        ⩟ Aeqx = beq (equality constraints)
        ⩟ lb ≤ x ≤ ub (bounds)
        ⩟ x ∈ {C, I, B} (continuous, integer, binary)
    
=#
function solve_milp_dual(
    c::Vector{Float64},
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    Aeq::SparseMatrixCSC{Float64},
    beq::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    vtype::Vector{Char};
    modelsense::String="min",
    verbose::Bool=true,
    debug::Bool=false
)
    # Validate input dimensions
    n = length(c)
    if size(A, 2) != n && !isempty(A)
        error("Inequality constraint matrix A has $(size(A, 2)) columns, but expected $n")
    end
    if size(Aeq, 2) != n && !isempty(Aeq)
        error("Equality constraint matrix Aeq has $(size(Aeq, 2)) columns, but expected $n")
    end
    if length(lb) != n
        error("Lower bounds vector has $(length(lb)) elements, but expected $n")
    end
    if length(ub) != n
        error("Upper bounds vector has $(length(ub)) elements, but expected $n")
    end
    if length(vtype) != n
        error("Variable type vector has $(length(vtype)) elements, but expected $n")
    end
    
    # Initialize Gurobi environment
    env_p = Ref{Ptr{Cvoid}}()
    error = quiet_gurobi(() -> GRBloadenv(env_p, C_NULL))  # Use C_NULL instead of empty string
    env = env_p[]
    # Disable any initial console output immediately
    error = quiet_gurobi(() -> GRBsetparam(env, "LogToConsole", "0"))
    # Disable academic license output message
    error = quiet_gurobi(() -> GRBsetparam(env, "DisplayInterval", "0"))
    
    # Set verbosity
    error = quiet_gurobi(() -> GRBsetparam(env, "OutputFlag", verbose ? "1" : "0"))
    # Disable additional logging parameters
    error = quiet_gurobi(() -> GRBsetparam(env, "LogFile", ""))
    # Set constraint handling precision
    error = quiet_gurobi(() -> GRBsetparam(env, "FeasibilityTol", "1e-9"))
    error = quiet_gurobi(() -> GRBsetparam(env, "IntFeasTol", "1e-9"))
    error = quiet_gurobi(() -> GRBsetparam(env, "OptimalityTol", "1e-9"))

    # Create a new model
    model_p = Ref{Ptr{Cvoid}}()
    error = GRBnewmodel(env, model_p, "milp", 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL)
    model = model_p[]
    
    # Add variables
    vtype_int8 = map(x -> Int8(x), vtype)
    error = GRBaddvars(
        model,    # model
        n,        # numvars
        0,        # numnz
        C_NULL,   # *vbeg
        C_NULL,   # *vind
        C_NULL,   # *vval
        c,        # *obj
        lb,       # *lb
        ub,       # *ub
        vtype_int8, # *vtype
        C_NULL    # **varnames
    )
    
    # Add inequality constraints (A*x <= b) using sparse format
    if !isempty(A) && !isempty(b)
        m_ineq = size(A, 1)
        # Prepare data for GRBaddconstrs
        row_starts = Vector{Cint}(undef, m_ineq + 1)
        row_indices = Vector{Cint}()
        row_values = Vector{Float64}()
        row_senses = Vector{UInt8}(undef, m_ineq)
        row_rhs = Vector{Float64}(undef, m_ineq)
        
        row_starts[1] = 0
        for i in 1:m_ineq
            # Get the row data directly from the CSC format
            row_vals = Vector{Float64}()
            row_inds = Vector{Int}()
            
            # Manually extract non-zero elements from this row
            for j in 1:n
                val = A[i, j]
                if abs(val) > 1e-10  # Use a small threshold to avoid numerical issues
                    push!(row_vals, val)
                    push!(row_inds, j)
                end
            end
            
            if !isempty(row_inds)
                # Convert to 0-based indexing for Gurobi
                indices_0based = convert(Vector{Cint}, row_inds .- 1)
                
                append!(row_indices, indices_0based)
                append!(row_values, row_vals)
                row_starts[i + 1] = length(row_indices)
                row_senses[i] = GRB_GREATER_EQUAL
                row_rhs[i] = b[i]
            end
        end
        
        error = GRBaddconstrs(
            model,         # *model
            m_ineq,        # numconstrs
            length(row_indices), # numnz
            row_starts,    # *cbeg
            row_indices,   # *cind
            row_values,    # *cval
            row_senses,    # *sense
            row_rhs,       # *rhs
            C_NULL         # **constrname
        )
        if error != 0
            println("Error adding inequality constraints: $error")
        end
    end
    
    # Add equality constraints (Aeq*x = beq) using sparse format
    if !isempty(Aeq) && !isempty(beq)
        m_eq = size(Aeq, 1)
        for i in 1:m_eq
            # Get the row data directly from the CSC format
            row_vals = Vector{Float64}()
            row_inds = Vector{Int}()
            
            # Manually extract non-zero elements from this row
            for j in 1:n
                val = Aeq[i, j]
                if abs(val) > 1e-10  # Use a small threshold to avoid numerical issues
                    push!(row_vals, val)
                    push!(row_inds, j)
                end
            end
            
            if !isempty(row_inds)
                # Convert to 0-based indexing for Gurobi
                indices_0based = convert(Vector{Cint}, row_inds .- 1)
                
                error = GRBaddconstr(
                    model,         # *model
                    length(row_inds), # numnz
                    indices_0based, # *cind
                    row_vals,      # *cval
                    GRB_EQUAL,     # sense
                    beq[i],        # rhs
                    C_NULL         # *constrname
                )
                if error != 0
                    println("Error adding equality constraint $i: $error")
                end
            end
        end
    end
    
    # Set optimization direction
    if lowercase(modelsense) == "min"
        error = GRBsetintattr(model, "ModelSense", GRB_MINIMIZE)
    elseif lowercase(modelsense) == "max"
        error = GRBsetintattr(model, "ModelSense", GRB_MAXIMIZE)
    else
        error("Invalid optimization sense: $sense. Use 'min' or 'max'")
    end
    
    # Update model and optimize
    error = GRBupdatemodel(model)
    if error != 0
        println("Error updating model: $error")
    end
    
    error = GRBoptimize(model)
    if error != 0
        println("Error optimizing model: $error")
    end
    
    # Get results
    optimstatus = Ref{Cint}()
    objval = Ref{Cdouble}()
    runtime = Ref{Cdouble}()
    mip_gap = Ref{Cdouble}()
    
    # Get solution status
    error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, optimstatus)
    
    # Get runtime
    error = GRBgetdblattr(model, GRB_DBL_ATTR_RUNTIME, runtime)
    
    # Prepare solution array
    sol = zeros(n)
    
    # Map status code to string
    status_map = Dict(
        GRB_OPTIMAL => "Optimal solution found",
        GRB_INFEASIBLE => "Problem is infeasible",
        GRB_UNBOUNDED => "Problem is unbounded",
        GRB_INF_OR_UNBD => "Problem is infeasible or unbounded",
        GRB_ITERATION_LIMIT => "Iteration limit reached",
        GRB_NODE_LIMIT => "Node limit reached",
        GRB_TIME_LIMIT => "Time limit reached",
        GRB_SOLUTION_LIMIT => "Solution limit reached",
        GRB_INTERRUPTED => "Optimization was interrupted",
        GRB_NUMERIC => "Numerical issues encountered",
        GRB_SUBOPTIMAL => "Suboptimal solution found",
        GRB_INPROGRESS => "Optimization in progress",
        GRB_USER_OBJ_LIMIT => "User objective limit reached"
    )
    
    status_name = get(status_map, optimstatus[], "Unknown status ($(optimstatus[]))")
    
    # Check if optimal solution was found
    if optimstatus[] == GRB_OPTIMAL || optimstatus[] == GRB_SUBOPTIMAL
        error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, objval)
        error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, n, sol)
        
        # Try to get MIP gap if it's a MIP problem
        has_discrete = any(vtype .!= 'C')
        if has_discrete
            try
                error = GRBgetdblattr(model, GRB_DBL_ATTR_MIPGAP, mip_gap)
            catch
                mip_gap[] = 0.0
            end
        else
            mip_gap[] = 0.0
        end
        
        # Verify constraints if verbose
        if debug
            println("\nSolution verification:")
            
            # Check inequality constraints
            if !isempty(A) && !isempty(b)
                max_violation = 0.0
                violation_count = 0
                
                for i in axes(A, 1)
                    lhs = 0.0
                    # Manually compute dot product to avoid potential sparse matrix issues
                    for j in 1:n
                        lhs += A[i, j] * sol[j]
                    end
                    
                    violation = max(0.0, lhs - b[i])
                    max_violation = max(max_violation, violation)
                    if violation > 1e-6
                        violation_count += 1
                        if violation_count <= 10  # Limit output to first 10 violations
                            println("Inequality constraint $i violated: $lhs > $(b[i]), violation = $violation")
                        end
                    end
                end
                
                if violation_count > 10
                    println("... and $(violation_count - 10) more inequality violations")
                end
                
                println("Max inequality violation: $max_violation (total violations: $violation_count)")
            end
            
            # Check equality constraints
            if !isempty(Aeq) && !isempty(beq)
                max_violation = 0.0
                violation_count = 0
                
                for i in axes(Aeq, 1)
                    lhs = 0.0
                    # Manually compute dot product to avoid potential sparse matrix issues
                    for j in 1:n
                        lhs += Aeq[i, j] * sol[j]
                    end
                    
                    violation = abs(lhs - beq[i])
                    max_violation = max(max_violation, violation)
                    if violation > 1e-6
                        violation_count += 1
                        if violation_count <= 10  # Limit output to first 10 violations
                            println("Equality constraint $i violated: $lhs != $(beq[i]), violation = $violation")
                        end
                    end
                end
                
                if violation_count > 10
                    println("... and $(violation_count - 10) more equality violations")
                end
                
                println("Max equality violation: $max_violation (total violations: $violation_count)")
            end
        end
    else
        objval[] = NaN
        mip_gap[] = NaN
        
        # Print a message about the non-optimal status
        if verbose
            println("No optimal solution found. Status: $(status_name)")
        end
    end
    
    # Free resources
    GRBfreemodel(model)
    GRBfreeenv(env)
    
    # Return results
    return Dict(
        :x => sol, 
        :objval => objval[], 
        :runtime => runtime[], 
        :mip_gap => mip_gap[],
        :status => optimstatus[],
        :status_name => status_name
    )
end

function solve_milp_dual(params::Dict)
    # Extract required parameters
    c = params[:c]
    n = length(c)
    
    # Extract inequality constraints (required)
    A = params[:A]
    b = params[:b]
    
    # Extract equality constraints (optional)
    Aeq = get(params, :Aeq, spzeros(0, n))
    beq = get(params, :beq, Float64[])
    
    # Extract bounds (required)
    lb = params[:lb]
    ub = params[:ub]
    
    # Extract variable types (required)
    vtype = params[:vtype]
    
    # Extract optional parameters
    modelsense = get(params, :sense, "min")
    verbose = get(params, :verbose, false)
    debug = get(params, :debug, false)
    
    # Call the original function with the extracted parameters
    return solve_milp_sparse(
        c, A, b, Aeq, beq, lb, ub, vtype;
        modelsense=modelsense, verbose=verbose, debug=debug
    )
end


# function test_mixed_integer()
#     println("=== Test Case 2: Mixed Integer Problem ===")
#     # max 2x1 + 3x2
#     # s.t. x1 + x2 <= 3.5
#     #      2x1 + x2 <= 5
#     #      x1 integer, x2 continuous
#     #      x1, x2 >= 0
    
#     c = [2.0, 3.0]
#     A = sparse([1.0 1.0; 2.0 1.0])
#     b = [3.5, 5.0]
#     Aeq = spzeros(0, 2)
#     beq = Float64[]
#     lb = [0.0, 0.0]
#     ub = [10.0, 10.0]
#     vtype = ['I', 'C']
    
#     result = solve_milp_sparse(c, A, b, Aeq, beq, lb, ub, vtype, sense="max")
    
#     println("Solution: ", result["x"])
#     println("Objective: ", result["objval"])
#     println("Expected solution: [0.0, 3.5]")
#     println("Expected objective: 10.5")
    
# end

# @time test_mixed_integer()

# # # Example usage
# params = Dict(
#     :c => [2.0, 3.0],
#     :A => sparse([1.0 1.0; 2.0 1.0]),
#     :b => [3.5, 5.0],
#     :lb => [0.0, 0.0],
#     :ub => [10.0, 10.0],
#     :vtype => ['I', 'C'],
#     :sense => "max"
# )

# # Call the function with the dictionary
# result = solve_milp_sparse(params)
