using LimitedLDLFactorizations
using AMD, MatrixMarket, Printf

# Warmup
A = sprand(10, 10, .5); A = A * A' + spdiagm(0 => rand(10))
LLDL = lldl(A, memory=5)

# obtain matrix from
# https://www.cise.ufl.edu/research/sparse/matrices/HB/bcsstk09.html
K = MatrixMarket.mmread("bcsstk09.mtx")
# make K lower triangular and extract diagonal here
# so we don't time tril() and diag().
Kdiag = diag(K)
K1 = tril(K, -1)
P = amd(K1)
nnzK1 = nnz(K1)
n = size(K, 1)

@printf("%2s %6s %6s %8s %7s %8s\n",
        "p", "nnz(K)", "nnz(L)", "‖LDLᵀ-K‖", "α", "time")
for p = 0 : 5 : 130
  LLDL, t, b, g, m = @timed lldl(K1, Kdiag, P, memory=p)
  L = LLDL.L + I
  @printf("%2d %6d %6d %8.2e %7.1e %8.2e\n",
          p, nnzK1, nnz(L) - n, norm(L*diagm(0 => LLDL.D)*L' - K[LLDL.P,LLDL.P], 1) / norm(K, 1), LLDL.α, t)
end
