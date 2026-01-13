using LinearAlgebra
using SparseArrays

"""
    BiCGStab method for solving linear systems of equations.
    
    BiCGStab (Biconjugate Gradient Stabilized) is an iterative method for solving
    linear systems Ax = b, particularly effective for non-symmetric matrices.
    
    The method combines BiCG with stabilization to avoid the irregular convergence
    patterns that can occur with the standard BiCG method.
"""

Base.:*(op::LinearOperator, x) = op.matvec(x)
Base.size(op::LinearOperator) = op.size

"""
    bicgstab(A, b, x0=nothing; tol=1e-6, max_iter=1000, verbose=false)

Solve the linear system Ax = b using the BiCGStab method.

# Arguments
- `A`: Coefficient matrix (can be sparse or dense)
- `b`: Right-hand side vector
- `x0`: Initial guess (if nothing, uses zero vector)
- `tol`: Convergence tolerance (default: 1e-6)
- `max_iter`: Maximum number of iterations (default: 1000)
- `verbose`: Print convergence information (default: false)

# Returns
- `x`: Solution vector
- `flag`: Convergence flag (0: converged, 1: max iterations, 2: breakdown)
- `relres`: Relative residual at termination
- `iter`: Number of iterations performed
- `resvec`: Vector of residual norms (if verbose=true)

# Example
```julia
A = [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 4.0]
b = [1.0, 2.0, 3.0]
x, flag, relres, iter = bicgstab(A, b)
```
"""
function bicgstab(A, b, x0 = nothing; tol::Float64 = 1e-6, max_iter::Int = 1000, verbose::Bool = false)
    n = length(b)
    
    # Initialize solution vector
    if x0 === nothing
        x = zeros(n)
    else
        x = copy(x0)
    end
    
    # Compute initial residual
    r = b - A * x
    r_norm0 = norm(r)
    
    # Check if already converged
    if r_norm0 < tol
        return x, 0, 0.0, 0, verbose ? [r_norm0] : Float64[]
    end
    
    # Choose arbitrary vector r̂ (often r̂ = r₀)
    r_hat = copy(r)
    
    # Initialize vectors
    p = copy(r)
    v = zeros(n)
    
    # Scalars
    rho = 1.0
    alpha = 1.0
    omega = 1.0
    
    # Storage for convergence history
    resvec = verbose ? [r_norm0] : Float64[]
    
    # BiCGStab iteration
    for iter = 1:max_iter
        # Compute rho_i = (r̂, r_{i-1})
        rho_new = dot(r_hat, r)
        
        # Check for breakdown
        if abs(rho_new) < eps(Float64)
            if verbose
                println("BiCGStab breakdown: rho = $rho_new at iteration $iter")
            end
            return x, 2, norm(r) / r_norm0, iter, resvec
        end
        
        # Compute beta and update p
        if iter == 1
            p = copy(r)
        else
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        end
        
        # Compute v = A * p
        v = A * p
        
        # Compute alpha
        alpha_denom = dot(r_hat, v)
        if abs(alpha_denom) < eps(Float64)
            if verbose
                println("BiCGStab breakdown: alpha denominator = $alpha_denom at iteration $iter")
            end
            return x, 2, norm(r) / r_norm0, iter, resvec
        end
        alpha = rho_new / alpha_denom
        
        # Compute s = r - alpha * v
        s = r - alpha * v
        
        # Check if we can terminate early
        s_norm = norm(s)
        if s_norm / r_norm0 < tol
            x = x + alpha * p
            if verbose
                push!(resvec, s_norm)
                println("BiCGStab converged at iteration $iter with relative residual $(s_norm / r_norm0)")
            end
            return x, 0, s_norm / r_norm0, iter, resvec
        end
        
        # Compute t = A * s
        t = A * s
        
        # Compute omega
        omega_denom = dot(t, t)
        if omega_denom < eps(Float64)
            if verbose
                println("BiCGStab breakdown: omega denominator = $omega_denom at iteration $iter")
            end
            return x, 2, norm(r) / r_norm0, iter, resvec
        end
        omega = dot(t, s) / omega_denom
        
        # Update solution
        x = x + alpha * p + omega * s
        
        # Update residual
        r = s - omega * t
        r_norm = norm(r)
        
        # Store residual norm
        if verbose
            push!(resvec, r_norm)
        end
        
        # Check convergence
        relres = r_norm / r_norm0
        if relres < tol
            if verbose
                println("BiCGStab converged at iteration $iter with relative residual $relres")
            end
            return x, 0, relres, iter, resvec
        end
        
        # Check for breakdown
        if abs(omega) < eps(Float64)
            if verbose
                println("BiCGStab breakdown: omega = $omega at iteration $iter")
            end
            return x, 2, relres, iter, resvec
        end
        
        # Update rho for next iteration
        rho = rho_new
        
        # Print progress
        if verbose && (iter % 10 == 0 || iter <= 5)
            println("Iteration $iter: relative residual = $relres")
        end
    end
    
    # Maximum iterations reached
    final_relres = norm(r) / r_norm0
    if verbose
        println("BiCGStab reached maximum iterations ($max_iter) with relative residual $final_relres")
    end
    
    return x, 1, final_relres, max_iter, resvec
end

"""
    preconditioned_bicgstab(A, b, M; kwargs...)

Solve Ax = b using preconditioned BiCGStab with preconditioner M.

The preconditioner M should approximate A⁻¹, so that M⁻¹ ≈ A.
Common choices include incomplete LU factorization, diagonal preconditioning, etc.

# Arguments
- `A`: Coefficient matrix
- `b`: Right-hand side vector  
- `M`: Preconditioner matrix or function
- `kwargs...`: Additional arguments passed to bicgstab

# Example
```julia
# Diagonal preconditioning
A = sparse([4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 4.0])
b = [1.0, 2.0, 3.0]
M = Diagonal(diag(A))  # Diagonal preconditioner
x, flag, relres, iter = preconditioned_bicgstab(A, b, M)
```
"""
function preconditioned_bicgstab(A, b, M; kwargs...)
    # Apply left preconditioning: solve M⁻¹Ax = M⁻¹b
    if isa(M, Function)
        # M is a function that applies M⁻¹
        A_precond = x -> M(A * x)
        b_precond = M(b)
    else
        # M is a matrix, compute M⁻¹
        M_inv = inv(Matrix(M))
        A_precond = x -> M_inv * (A * x)
        b_precond = M_inv * b
    end
    
    # Create linear operator for preconditioned system
    preconditioned_matvec = x -> A_precond(x)
    
    return bicgstab_operator(preconditioned_matvec, b_precond, length(b); kwargs...)
end

"""
    bicgstab_operator(matvec, b, n; kwargs...)

BiCGStab for linear operators defined by matrix-vector product function.

# Arguments
- `matvec`: Function that computes A*x for given x
- `b`: Right-hand side vector
- `n`: Size of the system
- `kwargs...`: Additional arguments passed to bicgstab
"""
function bicgstab_operator(matvec::Function, b, n::Int; kwargs...)
    # Create a linear operator wrapper
    A_op = LinearOperator(matvec, (n, n))
    return bicgstab(A_op, b; kwargs...)
end

"""
    bicgstab_with_restart(A, b; restart_iter=20, max_restarts=5, kwargs...)

BiCGStab with restart capability to improve robustness.

Restarts the algorithm every `restart_iter` iterations if convergence is slow.
This can help when the method stagnates or encounters numerical difficulties.
"""
function bicgstab_with_restart(A, b; restart_iter::Int = 20, max_restarts::Int = 5, kwargs...)
    n = length(b)
    x = zeros(n)
    total_iter = 0
    
    for restart = 1:max_restarts
        # Solve with current initial guess
        x_new, flag, relres, iter, resvec = bicgstab(A, b, x; max_iter=restart_iter, kwargs...)
        
        total_iter += iter
        
        # Check if converged
        if flag == 0
            return x_new, flag, relres, total_iter, resvec
        end
        
        # Update initial guess for next restart
        x = x_new
        
        # Check if breakdown occurred (don't restart in this case)
        if flag == 2
            return x_new, flag, relres, total_iter, resvec
        end
    end
    
    # Maximum restarts reached
    final_residual = norm(b - A * x) / norm(b)
    return x, 1, final_residual, total_iter, Float64[]
end

"""
    test_bicgstab()

Test the BiCGStab implementation with various test cases.
"""
function test_bicgstab()
    println("Testing BiCGStab Implementation")
    println("="^40)
    
    # Test 1: Simple 3x3 system
    println("\nTest 1: Simple 3x3 system")
    A1 = [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 4.0]
    b1 = [1.0, 2.0, 3.0]
    x_exact1 = A1 \ b1
    
    x1, flag1, relres1, iter1 = bicgstab(A1, b1, verbose=true)
    error1 = norm(x1 - x_exact1)
    
    println("Exact solution: $x_exact1")
    println("BiCGStab solution: $x1")
    println("Error: $error1")
    println("Flag: $flag1, Iterations: $iter1, Relative residual: $relres1")
    
    # Test 2: Larger sparse system
    println("\nTest 2: Sparse tridiagonal system (n=100)")
    n = 100
    A2 = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))
    b2 = ones(n)
    
    x2, flag2, relres2, iter2 = bicgstab(A2, b2, tol=1e-8)
    residual2 = norm(A2 * x2 - b2) / norm(b2)
    
    println("Problem size: $(size(A2))")
    println("Flag: $flag2, Iterations: $iter2, Relative residual: $relres2")
    println("Actual residual: $residual2")
    
    # Test 3: Ill-conditioned system
    println("\nTest 3: Ill-conditioned system")
    A3 = [1.0 1.0; 1.0 1.0001]
    b3 = [2.0, 2.0001]
    
    x3, flag3, relres3, iter3 = bicgstab(A3, b3, verbose=true)
    residual3 = norm(A3 * x3 - b3) / norm(b3)
    
    println("Condition number: $(cond(A3))")
    println("BiCGStab solution: $x3")
    println("Flag: $flag3, Iterations: $iter3, Relative residual: $relres3")
    
    # Test 4: Preconditioned BiCGStab
    println("\nTest 4: Preconditioned BiCGStab")
    A4 = A2  # Use the same sparse system
    b4 = b2
    M4 = Diagonal(diag(A4))  # Diagonal preconditioner
    
    x4, flag4, relres4, iter4 = preconditioned_bicgstab(A4, b4, M4, tol=1e-8)
    residual4 = norm(A4 * x4 - b4) / norm(b4)
    
    println("Preconditioned - Flag: $flag4, Iterations: $iter4")
    println("Unpreconditioned iterations: $iter2 vs Preconditioned: $iter4")
    
    println("\nAll tests completed!")
end

# Good - clearly defines what users should use
export bicgstab, preconditioned_bicgstab
