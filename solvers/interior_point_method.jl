# It is an implementation of the interior point method for linear programming
# The Mehrotra's Predictor-Corrector Method is adopted in this implementation

include("bicgstab.jl")  # Include BiCGStab solver

# Define the objective function
function interior_point_method(problem::LP, ipm::IPM)
    # define the initial point, we have four types of initial points: x, z, λ, μ
    c = problem.c
    n = length(c)
    
    # Extract constraints directly from LP struct fields
    A = problem.A
    b = problem.b
    Aeq = problem.Aeq
    beq = problem.beq
    
    # Extract bounds (required)
    # Extend A and b to include the bounds
    lb = problem.lb
    ub = problem.ub
    if length(lb) > 0 && length(ub) > 0
        A = vcat(A, sparse(I, n, n))
        b = vcat(b, ub)
        A = vcat(A, -sparse(I, n, n))
        b = vcat(b, -lb)
    end
    # evaluate the cost, constraints, derivaties
    x = (ub + lb) / 2.0  # Initial guess: midpoint of bounds
    obj = c' * x  # Initial objective value
    rg = Aeq * x - beq  # Equality constraints
    rh = A* x - b  # Inequality constraints
    # get the size of the problem
    niq = length(rh) # number of inequality constraints
    neq = length(rg) # number of equality constraints
    # initialize the multipliers
    z = ones(niq)
    λ = zeros(neq)
    μ = ones(niq)
    z0 = 1.0
    eflag = 0
    γ = 1.0
    σ = 0.1
    # update z and μ
    for i in 1:niq
        if rh[i] < -z0
            z[i] = -rh[i]
        end
        μ[i] = γ / z0
    end
    e = ones(niq)
    obj0 = obj
    Lx = c + Aeq' * λ + A' * μ  # Lagrangian gradient
    Lxx = zeros(n, n)  # Hessian of the Lagrangian
    # initializate
    converged = false
    maxh = maximum(rh)
    feascond = maximum([norm(rg, Inf), maxh]) / (1 + maximum([norm(x, Inf), norm(z, Inf)]))
    gradcond = norm(Lx, Inf) / (1 + maximum([norm(μ, Inf),norm(λ, Inf)]))
    compcond = (z'*μ) / (1 + norm(x, Inf))
    costcond = abs(obj - obj0) / (1 + abs(obj0))
    # record the results
    hist = History(zeros(n, ipm.max_iter), zeros(neq, ipm.max_iter), zeros(niq, ipm.max_iter), zeros(niq, ipm.max_iter), zeros(ipm.max_iter), zeros(ipm.max_iter), zeros(ipm.max_iter), zeros(ipm.max_iter),zeros(ipm.max_iter))
    rg = Aeq * x - beq  # Equality constraints
    rh = A * x - b  # Inequality constraints
    dg = Aeq'
    dh = A'
    df = c
    Lx = c + Aeq' * λ + A' * μ  # Lagrangian gradient
    # start the iteration using the while funciton
    iter = 1
    
    while iter < ipm.max_iter
        # calculate the KKT step
        Δx, Δλ, Δμ, Δz = KKT_step(μ, z, e, rg, rh, dg, dh, Lx, Lxx, γ, n, neq, ipm)
        # calculate the step length
        ξ = ipm.ξ
        k = findall(<(0), Δz)
        if length(k) > 0
            α_primal = minimum([ξ*minimum(-z[k]./Δz[k]), 1.0])
        else
            α_primal = 1.0
        end
        k = findall(<(0), Δμ)
        if length(k) > 0
            α_dual = minimum([ξ*minimum(-μ[k]./Δμ[k]), 1.0])
        else
            α_dual = 1.0
        end
        # update the point
        x = x + α_primal*Δx
        z = z + α_primal*Δz
        λ = λ + α_dual*Δλ
        μ = μ + α_dual*Δμ

        if niq > 0
            γ = σ * z'*μ / niq
        end

        # update the cost, constraints, derivaties   
        obj = c'* x 
        rg = Aeq * x - beq
        rh = A*x - b
        dg = Aeq'
        dh = A'
        df = c
        Lx = c + Aeq' * λ + A' * μ  # Lagrangian gradient

        maxh = maximum(rh)
        feascond = maximum([norm(rg, Inf), maxh]) / (1 + maximum([norm(x, Inf), norm(z, Inf)]))
        gradcond = norm(Lx, Inf) / (1 + maximum([norm(μ, Inf),norm(λ, Inf)]))
        compcond = (z'*μ) / (1 + norm(x, Inf))
        costcond = abs(obj - obj0) / (1 + abs(obj0))
        # record the results
        hist.x_record[:, iter] = x
        hist.λ_record[:, iter] = λ
        hist.μ_record[:, iter] = μ
        hist.z_record[:, iter] = z
        hist.obj_record[iter] = obj
        hist.feascond_record[iter] = feascond
        hist.gradcond_record[iter] = gradcond
        hist.compcond_record[iter] = compcond
        hist.costcond_record[iter] = costcond
        obj0 = obj

        if feascond < ipm.feasible_tol && gradcond < ipm.feasible_tol && compcond < ipm.feasible_tol && costcond < ipm.feasible_tol
            eflag = true
            break
        end
        iter += 1
    end
    # reduce the size of the record
    hist.x_record = hist.x_record[:, 1:iter]
    hist.λ_record = hist.λ_record[:, 1:iter]
    hist.μ_record = hist.μ_record[:, 1:iter]
    hist.z_record = hist.z_record[:, 1:iter]
    hist.obj_record = hist.obj_record[1:iter]
    hist.feascond_record = hist.feascond_record[1:iter]
    hist.gradcond_record = hist.gradcond_record[1:iter]
    hist.compcond_record = hist.compcond_record[1:iter]
    hist.costcond_record = hist.costcond_record[1:iter]
    
    # Clean up small numerical values to improve solution readability
    precision_tol = ipm.precision_cleanup_tol
    
    # Clean solution vector with improved rounding logic
    for i in eachindex(x)
        # First check if it's essentially zero
        if abs(x[i]) < precision_tol
            x[i] = 0.0
        else
            # Check if it's close to an integer (for problems with integer-like solutions)
            rounded_val = round(x[i])
            if abs(x[i] - rounded_val) < precision_tol * 10  # Use larger tolerance for integer rounding
                x[i] = rounded_val
            else
                # For non-integer values, round to reasonable precision
                x[i] = round(x[i], digits=10)
            end
        end
    end
    
    # Clean multipliers with similar logic
    for i in eachindex(λ)
        if abs(λ[i]) < precision_tol
            λ[i] = 0.0
        else
            λ[i] = round(λ[i], digits=10)
        end
    end
    
    for i in eachindex(μ)
        if abs(μ[i]) < precision_tol
            μ[i] = 0.0
        else
            μ[i] = round(μ[i], digits=10)
        end
    end
    
    # Clean up the objective value as well
    obj_clean = c' * x
    if abs(obj_clean - round(obj_clean)) < precision_tol * 10
        obj_clean = round(obj_clean)
    else
        obj_clean = round(obj_clean, digits=10)
    end
    
    return IPM_Solution(x, λ, μ, obj_clean, eflag, hist)
end


# define the KKT iteration with configurable linear solver
function KKT_step(μ, z, e, rg, rh, dg, dh, Lx, Lxx, γ, n, neq, ipm::IPM)
    zinvdiag = Diagonal(1.0 ./ z)
    mudiag = Diagonal(μ)
    dh_zinv = dh * zinvdiag
    
    # 3.39: Form the KKT matrix
    M = Lxx + dh_zinv * mudiag * dh'
    
    # 3.42: Form the right-hand side
    N = Lx + dh_zinv * (mudiag*rh + γ*e)
    
    # Form the full KKT system: [M dg; dg' 0] * [Δx; Δλ] = -[N; rg]
    KKT_matrix = [M dg; dg' spzeros(neq, neq)]
    KKT_rhs = -[N; rg]
    
    # Solve the KKT system using the specified linear solver
    Δ = solve_linear_system(KKT_matrix, KKT_rhs, ipm)
    
    Δx = Δ[1:n]
    Δλ = Δ[n+1:n+neq]
    
    # 3.36: Compute Δz
    Δz = -rh - z - dh' * Δx
    
    # 3.35: Compute Δμ
    Δμ = -μ + zinvdiag*(γ.*e - mudiag * Δz)
    
    return Δx, Δλ, Δμ, Δz
end

"""
    solve_linear_system(A, b, ipm::IPM)

Solve the linear system Ax = b using the specified solver in ipm parameters.

Available solvers:
- "direct": Direct factorization (LU or Cholesky)
- "bicgstab": Biconjugate Gradient Stabilized method
- "ldl": LDL factorization for symmetric systems
- "qr": QR factorization
"""
function solve_linear_system(A, b, ipm::IPM)
    solver = lowercase(ipm.linear_solver)
    
    if solver == "direct"
        return solve_direct(A, b, ipm)
    elseif solver == "bicgstab"
        return solve_bicgstab(A, b, ipm)
    elseif solver == "ldl"
        return solve_ldl(A, b, ipm)
    elseif solver == "qr"
        return solve_qr(A, b, ipm)
    elseif solver == "cholesky"
        return solve_cholesky(A, b, ipm)
    else
        if ipm.verbose
            println("Warning: Unknown linear solver '$(ipm.linear_solver)'. Using direct solver.")
        end
        return solve_direct(A, b, ipm)
    end
end

"""
    solve_direct(A, b, ipm::IPM)

Solve using Julia's direct solver (backslash operator).
"""
function solve_direct(A, b, ipm::IPM)
    try
        return A \ b
    catch e
        if ipm.verbose
            println("Direct solver failed: $e")
            println("Attempting regularized solve...")
        end
        
        # Try regularized system if direct solve fails
        n = size(A, 1)
        regularization = 1e-12
        A_reg = A + regularization * sparse(I, n, n)
        
        try
            return A_reg \ b
        catch e2
            if ipm.verbose
                println("Regularized direct solver also failed: $e2")
            end
            throw(e2)
        end
    end
end

"""
    solve_bicgstab(A, b, ipm::IPM)

Solve using BiCGStab iterative method.
"""
function solve_bicgstab(A, b, ipm::IPM)
    x, flag, relres, iter = bicgstab(A, b; 
                                    tol=ipm.linear_solver_tol, 
                                    max_iter=ipm.linear_solver_max_iter,
                                    verbose=false)
    
    if flag != 0 && ipm.verbose
        println("BiCGStab warning: flag=$flag, relative residual=$relres, iterations=$iter")
    end
    
    # If BiCGStab fails, fall back to direct method
    if flag == 2  # Breakdown occurred
        if ipm.verbose
            println("BiCGStab breakdown, falling back to direct solver")
        end
        return solve_direct(A, b, ipm)
    end
    
    return x
end

"""
    solve_ldl(A, b, ipm::IPM)

Solve using LDL factorization for symmetric systems.
"""
function solve_ldl(A, b, ipm::IPM)
    try
        # Check if matrix is approximately symmetric
        if norm(A - A', Inf) > 1e-10
            if ipm.verbose
                println("Warning: Matrix is not symmetric for LDL. Using direct solver.")
            end
            return solve_direct(A, b, ipm)
        end
        
        # Use LDL factorization
        F = ldlt(Matrix(A))
        return F \ b
    catch e
        if ipm.verbose
            println("LDL factorization failed: $e. Using direct solver.")
        end
        return solve_direct(A, b, ipm)
    end
end

"""
    solve_qr(A, b, ipm::IPM)

Solve using QR factorization.
"""
function solve_qr(A, b, ipm::IPM)
    try
        F = qr(Matrix(A))
        return F \ b
    catch e
        if ipm.verbose
            println("QR factorization failed: $e. Using direct solver.")
        end
        return solve_direct(A, b, ipm)
    end
end

"""
    solve_cholesky(A, b, ipm::IPM)

Solve using Cholesky factorization for positive definite systems.
"""
function solve_cholesky(A, b, ipm::IPM)
    try
        # Check if matrix is approximately symmetric
        if norm(A - A', Inf) > 1e-10
            if ipm.verbose
                println("Warning: Matrix is not symmetric for Cholesky. Using direct solver.")
            end
            return solve_direct(A, b, ipm)
        end
        
        # Use Cholesky factorization
        F = cholesky(Matrix(A) + 1e-12*I)  # Add small regularization
        return F \ b
    catch e
        if ipm.verbose
            println("Cholesky factorization failed: $e. Using direct solver.")
        end
        return solve_direct(A, b, ipm)
    end
end