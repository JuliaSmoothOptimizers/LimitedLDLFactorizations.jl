using LinearAlgebra, SparseArrays, Test

using AMD, Metis, LimitedLDLFactorizations

@testset "SQD no shift" begin
  # this matrix possesses an LDLᵀ factorization without pivoting
  A = [
    1.7 0 0 0 0 0 0 0 0.13 0
    0 1.0 0 0 0.02 0 0 0 0 0.01
    0 0 1.5 0 0 0 0 0 0 0
    0 0 0 1.1 0 0 0 0 0 0
    0 0.02 0 0 2.6 0 0.16 0.09 0.52 0.53
    0 0 0 0 0 1.2 0 0 0 0
    0 0 0 0 0.16 0 1.3 0 0 0.56
    0 0 0 0 0.09 0 0 1.6 0.11 0
    0.13 0 0 0 0.52 0 0 0.11 1.4 0
    0 0.01 0 0 0.53 0 0.56 0 0 3.1
  ]
  A = sparse(A)

  for perm ∈ (1:(A.n), amd(A), Metis.permutation(A)[1])
    LLDL = lldl(A, P = perm, memory = 0)
    nnzl0 = nnz(LLDL)
    @test nnzl0 == nnz(tril(A))
    @test LLDL.α_out == 0

    LLDL = lldl(Symmetric(A, :L), P = perm, memory = 5) # test symmetric lldl
    nnzl5 = nnz(LLDL)
    @test nnzl5 ≥ nnzl0
    @test LLDL.α_out == 0

    LLDL = lldl(A, P = perm, memory = 10)
    @test nnz(LLDL) ≥ nnzl5
    @test LLDL.α_out == 0
    L = LLDL.L + I
    @test norm(L * diagm(0 => LLDL.D) * L' - A[perm, perm]) ≤ sqrt(eps()) * norm(A)

    for sol in (ones(A.n), ones(A.n, 3)) # testing Vector{Float64} and Matrix{Float64} RHS
      b = A * sol
      x = LLDL \ b
      @test x ≈ sol

      y = similar(b)
      ldiv!(y, LLDL, b)
      @test y ≈ sol

      ldiv!(LLDL, b)
      @test b ≈ sol

    end

  end
end

@testset "with shift" begin
  # this matrix requires a shift
  A = [
    1.0 1.0
    1.0 0.0
  ]
  LLDL = lldl(A)
  @test LLDL.α_out ≥ sqrt(eps())
  @test LLDL.D[1] > 0
  @test LLDL.D[2] < 0

  # specify our own shift
  LLDL = lldl(A, α = 1.0e-2)
  @test LLDL.α_out ≥ 1.0e-2
  @test LLDL.D[1] > 0
  @test LLDL.D[2] < 0
end

@testset "SQD no shift lower triangle" begin
  # Lower triangle only
  A = [
    1.7 0 0 0 0 0 0 0 0.13 0
    0 1.0 0 0 0.02 0 0 0 0 0.01
    0 0 1.5 0 0 0 0 0 0 0
    0 0 0 1.1 0 0 0 0 0 0
    0 0.02 0 0 2.6 0 0.16 0.09 0.52 0.53
    0 0 0 0 0 1.2 0 0 0 0
    0 0 0 0 0.16 0 1.3 0 0 0.56
    0 0 0 0 0.09 0 0 1.6 0.11 0
    0.13 0 0 0 0.52 0 0 0.11 1.4 0
    0 0.01 0 0 0.53 0 0.56 0 0 3.1
  ]
  A = sparse(A)
  Al = tril(A)

  for perm ∈ (1:(A.n), amd(A), Metis.permutation(A)[1])
    LLDL = lldl(Al, P = perm, memory = 0)
    nnzl0 = nnz(LLDL)
    @test nnzl0 == nnz(tril(A))
    @test LLDL.α_out == 0

    LLDL = lldl(Al, P = perm, memory = 5)
    nnzl5 = nnz(LLDL)
    @test nnzl5 ≥ nnzl0
    @test LLDL.α_out == 0

    LLDL = lldl(Al, P = perm, memory = 10)
    @test nnz(LLDL) ≥ nnzl5
    @test LLDL.α_out == 0
    L = LLDL.L + I
    @test norm(L * diagm(0 => LLDL.D) * L' - A[perm, perm]) ≤ sqrt(eps()) * norm(A)

    sol = ones(A.n)
    Sol = rand(A.n, 4)
    b = A * sol
    B = A * Sol # test matrix rhs
    x = LLDL \ b
    X = LLDL \ B
    @test x ≈ sol
    @test isapprox(X, Sol, atol = sqrt(eps()))

    y = similar(b)
    Y = similar(B)
    ldiv!(y, LLDL, b)
    ldiv!(Y, LLDL, B)
    @test y ≈ sol
    @test isapprox(Y, Sol, atol = sqrt(eps()))

    ldiv!(LLDL, b)
    ldiv!(LLDL, B)
    @test b ≈ sol
    @test isapprox(B, Sol, atol = sqrt(eps()))
  end
end

@testset "test in-place version" begin
  # Lower triangle only
  A = [
    1.7 0 0 0 0 0 0 0 0.13 0
    0 1.0 0 0 0.02 0 0 0 0 0.01
    0 0 1.5 0 0 0 0 0 0 0
    0 0 0 1.1 0 0 0 0 0 0
    0 0.02 0 0 2.6 0 0.16 0.09 0.52 0.53
    0 0 0 0 0 1.2 0 0 0 0
    0 0 0 0 0.16 0 1.3 0 0 0.56
    0 0 0 0 0.09 0 0 1.6 0.11 0
    0.13 0 0 0 0.52 0 0 0.11 1.4 0
    0 0.01 0 0 0.53 0 0.56 0 0 3.1
  ]
  A = sparse(A)
  Alow = tril(A)
  perm = amd(A)
  LLDL = LimitedLDLFactorization(Alow, P = perm, memory = 10)
  @test !factorized(LLDL)
  @test_throws LimitedLDLFactorizations.LLDLException LLDL \ rand(size(A, 1))

  lldl_factorize!(LLDL, Alow)
  @test factorized(LLDL)
  @test LLDL.α_out == 0
  L = LLDL.L + I
  @test norm(L * diagm(0 => LLDL.D) * L' - A[perm, perm]) ≤ sqrt(eps()) * norm(A)

  sol = ones(A.n)
  b = A * sol
  x = LLDL \ b
  @test x ≈ sol

  y = similar(b)
  ldiv!(y, LLDL, b)
  @test y ≈ sol

  ldiv!(LLDL, b)
  @test b ≈ sol

  A2 = [
    10.7 0 0 0 0 0 0 0 0.33 0
    0 1.0 0 0 0.02 0 0 0 0 0.01
    0 0 1.5 0 0 0 0 0 0 0
    0 0 0 3.1 0 0 0 0 0 0
    0 0.02 0 0 2.6 0 0.16 0.9 0.52 0.53
    0 0 0 0 0 1.2 0 0 0 0
    0 0 0 0 0.16 0 1.3 0 0 1.56
    0 0 0 0 0.9 0 0 1.6 0.11 0
    0.33 0 0 0 0.52 0 0 0.11 1.4 0
    0 0.01 0 0 0.53 0 1.56 0 0 30.1
  ]
  A2 = sparse(A2)
  A2low = tril(A2)
  lldl_factorize!(LLDL, A2low)
  allocs = @allocated lldl_factorize!(LLDL, A2low)
  @test allocs == 0
  @test LLDL.α_out == 0
  L = LLDL.L + I
  @test norm(L * diagm(0 => LLDL.D) * L' - A2[perm, perm]) ≤ sqrt(eps()) * norm(A2)

  sol = ones(A2.n)
  b = A2 * sol
  x = LLDL \ b
  @test x ≈ sol

  y = similar(b)
  ldiv!(y, LLDL, b)
  @test y ≈ sol

  allocs = @allocated ldiv!(y, LLDL, b)
  @test allocs == 0

  ldiv!(LLDL, b)
  @test b ≈ sol

  allocs = @allocated ldiv!(LLDL, b)
  @test allocs == 0
end

@testset "with shift lower triangle" begin
  # this matrix requires a shift
  A = sparse([
    1.0 0.0
    1.0 0.0
  ])
  LLDL = lldl(A)
  @test LLDL.α_out ≥ sqrt(eps())
  @test LLDL.D[1] > 0
  @test LLDL.D[2] < 0

  # specify our own shift
  update_shift!(LLDL, 1.0e-2)
  update_shift_increase_factor!(LLDL, 5)
  lldl_factorize!(LLDL, A)
  @test LLDL.α_out ≥ 1.0e-2
  @test LLDL.D[1] > 0
  @test LLDL.D[2] < 0
end

@testset "with int32 indices" begin
  A = [
    1.7 0 0 0 0 0 0 0 0.13 0
    0 1.0 0 0 0.02 0 0 0 0 0.01
    0 0 1.5 0 0 0 0 0 0 0
    0 0 0 1.1 0 0 0 0 0 0
    0 0.02 0 0 2.6 0 0.16 0.09 0.52 0.53
    0 0 0 0 0 1.2 0 0 0 0
    0 0 0 0 0.16 0 1.3 0 0 0.56
    0 0 0 0 0.09 0 0 1.6 0.11 0
    0.13 0 0 0 0.52 0 0 0.11 1.4 0
    0 0.01 0 0 0.53 0 0.56 0 0 3.1
  ]
  A = convert(SparseMatrixCSC{Float64, Int32}, sparse(A))
  LLDL = lldl(A, memory = 5)
  @test LLDL.α_out == 0
end

@testset "test lower-precision factorization" begin
  # Lower triangle only
  Tf = Float32
  A = [
    1.7 0 0 0 0 0 0 0 0.13 0
    0 1.0 0 0 0.02 0 0 0 0 0.01
    0 0 1.5 0 0 0 0 0 0 0
    0 0 0 1.1 0 0 0 0 0 0
    0 0.02 0 0 2.6 0 0.16 0.09 0.52 0.53
    0 0 0 0 0 1.2 0 0 0 0
    0 0 0 0 0.16 0 1.3 0 0 0.56
    0 0 0 0 0.09 0 0 1.6 0.11 0
    0.13 0 0 0 0.52 0 0 0.11 1.4 0
    0 0.01 0 0 0.53 0 0.56 0 0 3.1
  ]
  A = sparse(A)
  Alow = tril(A)
  perm = amd(A)
  LLDL = lldl(Alow, Tf, P = perm, memory = 10)
  @test eltype(LLDL.D) == Tf

  L = LLDL.L + I
  @test norm(L * diagm(0 => LLDL.D) * L' - A[perm, perm]) ≤ sqrt(eps(Tf)) * norm(A)

  sol = ones(Tf, A.n)
  b = similar(sol)
  mul!(b, A, sol)
  x = LLDL \ b
  @test x ≈ sol
  @test eltype(x) == Tf

  y = similar(b)
  ldiv!(y, LLDL, b)
  @test y ≈ sol

  ldiv!(LLDL, b)
  @test b ≈ sol
end
