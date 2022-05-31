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
    LLDL = lldl(A, perm, memory = 0)
    nnzl0 = nnz(LLDL)
    @test nnzl0 == nnz(tril(A))
    @test LLDL.α == 0

    LLDL = lldl(A, perm, memory = 5)
    nnzl5 = nnz(LLDL)
    @test nnzl5 ≥ nnzl0
    @test LLDL.α == 0

    LLDL = lldl(A, perm, memory = 10)
    @test nnz(LLDL) ≥ nnzl5
    @test LLDL.α == 0
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
  end
end

@testset "with shift" begin
  # this matrix requires a shift
  A = [
    1.0 1.0
    1.0 0.0
  ]
  LLDL = lldl(A)
  @test LLDL.α ≥ 1.0e-3
  @test LLDL.D[1] > 0
  @test LLDL.D[2] < 0

  # specify our own shift
  LLDL = lldl(A, α = 1.0e-2)
  @test LLDL.α ≥ 1.0e-2
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
  B = tril(A)

  for perm ∈ (1:(A.n), amd(A), Metis.permutation(A)[1])
    LLDL = lldl(B, perm, memory = 0)
    nnzl0 = nnz(LLDL)
    @test nnzl0 == nnz(tril(A))
    @test LLDL.α == 0

    LLDL = lldl(B, perm, memory = 5)
    nnzl5 = nnz(LLDL)
    @test nnzl5 ≥ nnzl0
    @test LLDL.α == 0

    LLDL = lldl(B, perm, memory = 10)
    @test nnz(LLDL) ≥ nnzl5
    @test LLDL.α == 0
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
  Alow, adiag = tril(A, -1), diag(A)
  perm = amd(A)
  LLDL = lldl_allocate(Alow, adiag, perm, memory = 10)
  lldl_factorize!(Alow, adiag, LLDL)
  @test LLDL.α == 0
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
  A2low, a2diag = tril(A2, -1), diag(A2)
  lldl_factorize!(A2low, a2diag, LLDL)
  @test LLDL.α == 0
  L = LLDL.L + I
  @test norm(L * diagm(0 => LLDL.D) * L' - A2[perm, perm]) ≤ sqrt(eps()) * norm(A2)

  sol = ones(A2.n)
  b = A2 * sol
  x = LLDL \ b
  @test x ≈ sol

  y = similar(b)
  ldiv!(y, LLDL, b)
  @test y ≈ sol

  ldiv!(LLDL, b)
  @test b ≈ sol
end

@testset "with shift lower triangle" begin
  # this matrix requires a shift
  A = [
    1.0 0.0
    1.0 0.0
  ]
  LLDL = lldl(A)
  @test LLDL.α ≥ 1.0e-3
  @test LLDL.D[1] > 0
  @test LLDL.D[2] < 0

  # specify our own shift
  LLDL = lldl(A, α = 1.0e-2)
  @test LLDL.α ≥ 1.0e-2
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
  @test LLDL.α == 0
end
