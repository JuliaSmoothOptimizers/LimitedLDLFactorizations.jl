using LimitedLDLFactorizations, Test, LinearAlgebra, SparseArrays

# this matrix possesses an LDL factorization without pivoting
A = [ 1.7     0     0     0     0     0     0     0   .13     0
        0    1.     0     0   .02     0     0     0     0   .01
        0     0   1.5     0     0     0     0     0     0     0
        0     0     0   1.1     0     0     0     0     0     0
        0   .02     0     0   2.6     0   .16   .09   .52   .53
        0     0     0     0     0   1.2     0     0     0     0
        0     0     0     0   .16     0   1.3     0     0   .56
        0     0     0     0   .09     0     0   1.6   .11     0
      .13     0     0     0   .52     0     0   .11   1.4     0
        0   .01     0     0   .53     0   .56     0     0   3.1 ]
A = sparse(A)

L, d, α = lldl(A, memory=0)
nnzl0 = nnz(L)
@test nnzl0 == nnz(tril(A, -1))
@test α == 0

L, d, α = lldl(A, memory=5)
nnzl5 = nnz(L)
@test nnzl5 ≥ nnzl0
@test α == 0

L, d, α = lldl(A, memory=10)  # should be the exact factorization
@test nnz(L) ≥ nnzl5
@test α == 0
L = L + I
@test norm(L * diagm(0 => d) * L' - A) ≤ sqrt(eps()) * norm(A)

# this matrix requires a shift
A = [ 1.  1.
      1.  0. ]
L, d, α = lldl(A)
@test α ≥ 1.0e-3
@test d[1] > 0
@test d[2] < 0

# specify our own shift
L, d, α = lldl(A, α=1.0e-2)
@test α ≥ 1.0e-2
@test d[1] > 0
@test d[2] < 0
