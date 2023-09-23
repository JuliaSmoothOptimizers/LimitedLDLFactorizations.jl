using LinearAlgebra, SparseArrays, Test

using AMD, Metis, LimitedLDLFactorizations

@testset "SQD no shift" begin
  # this matrix possesses an LDLᵀ factorization without pivoting
  #! format: off
  A = [
    1.7     0       0   0   0       0   0       0       0.13+im 0
    0       1.0     0   0   0.02+im 0   0       0       0       0.01+im
    0       0       1.5 0   0       0   0       0       0       0
    0       0       0   1.1 0       0   0       0       0       0
    0       0.02-im 0   0   2.6     0   0.16+im 0.09+im 0.52+im 0.53+im
    0       0       0   0   0       1.2 0       0       0       0
    0       0       0   0   0.16-im 0   1.3     0       0       0.56+im
    0       0       0   0   0.09-im 0   0       1.6     0.11+im 0
    0.13-im 0       0   0   0.52-im 0   0       0.11-im 1.4     0
    0       0.01-im 0   0   0.53-im 0   0.56-im 0       0       3.1
  ]
  #! format: on
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

@testset "SQD no shift lower triangle" begin
  # Lower triangle only
  #! format: off
  A = [
    1.7     0       0   0   0       0   0       0       0.13+im 0
    0       1.0     0   0   0.02+im 0   0       0       0       0.01+im
    0       0       1.5 0   0       0   0       0       0       0
    0       0       0   1.1 0       0   0       0       0       0
    0       0.02-im 0   0   2.6     0   0.16+im 0.09+im 0.52+im 0.53+im
    0       0       0   0   0       1.2 0       0       0       0
    0       0       0   0   0.16-im 0   1.3     0       0       0.56+im
    0       0       0   0   0.09-im 0   0       1.6     0.11+im 0
    0.13-im 0       0   0   0.52-im 0   0       0.11-im 1.4     0
    0       0.01-im 0   0   0.53-im 0   0.56-im 0       0       3.1
  ]
  #! format: on
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
  #! format: off
  A = [
    1.7     0       0   0   0       0   0       0       0.13+im 0
    0       1.0     0   0   0.02+im 0   0       0       0       0.01+im
    0       0       1.5 0   0       0   0       0       0       0
    0       0       0   1.1 0       0   0       0       0       0
    0       0.02-im 0   0   2.6     0   0.16+im 0.09+im 0.52+im 0.53+im
    0       0       0   0   0       1.2 0       0       0       0
    0       0       0   0   0.16-im 0   1.3     0       0       0.56+im
    0       0       0   0   0.09-im 0   0       1.6     0.11+im 0
    0.13-im 0       0   0   0.52-im 0   0       0.11-im 1.4     0
    0       0.01-im 0   0   0.53-im 0   0.56-im 0       0       3.1
  ]
  #! format: on
  A = sparse(A)
  Alow, adiag = tril(A, -1), diag(A)
  perm = amd(A)
  LLDL = LimitedLDLFactorization(Alow, adiag, perm, memory = 10)
  lldl_factorize!(LLDL, Alow, adiag)
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

  #! format: off
  A2 = [
    10.7    0       0   0   0       0   0       0       0.33+im 0
    0       1.0     0   0   0.02+im 0   0       0       0       0.01+im
    0       0       1.5 0   0       0   0       0       0       0
    0       0       0   3.1 0       0   0       0       0       0
    0       0.02-im 0   0   2.6     0   0.16+im 0.09+im 0.52+im 0.53+im
    0       0       0   0   0       1.2 0       0       0       0
    0       0       0   0   0.16-im 0   1.3     0       0       1.56+im
    0       0       0   0   0.09-im 0   0       1.6     0.11+im 0
    0.33-im 0       0   0   0.52-im 0   0       0.11-im 1.4     0
    0       0.01-im 0   0   0.53-im 0   1.56-im 0       0       30.1
  ]
  #! format: on
  A2 = sparse(A2)
  A2low, a2diag = tril(A2, -1), diag(A2)
  lldl_factorize!(LLDL, A2low, a2diag)
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

@testset "with int32 indices" begin
  #! format: off
  A = [
    1.7     0       0   0   0       0   0       0       0.13+im 0
    0       1.0     0   0   0.02+im 0   0       0       0       0.01+im
    0       0       1.5 0   0       0   0       0       0       0
    0       0       0   1.1 0       0   0       0       0       0
    0       0.02-im 0   0   2.6     0   0.16+im 0.09+im 0.52+im 0.53+im
    0       0       0   0   0       1.2 0       0       0       0
    0       0       0   0   0.16-im 0   1.3     0       0       0.56+im
    0       0       0   0   0.09-im 0   0       1.6     0.11+im 0
    0.13-im 0       0   0   0.52-im 0   0       0.11-im 1.4     0
    0       0.01-im 0   0   0.53-im 0   0.56-im 0       0       3.1
  ]
  #! format: on
  A = convert(SparseMatrixCSC{ComplexF64, Int32}, sparse(A))
  LLDL = lldl(A, memory = 5)
  @test LLDL.α == 0
end
