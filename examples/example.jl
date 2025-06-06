using LimitedLDLFactorizations
using MatrixMarket
using LinearAlgebra, Printf, SparseArrays
using AMD, Metis

# Warmup
A = sprand(10, 10, 0.5);
A = A * A' + spdiagm(0 => rand(10));
_ = lldl(A, memory = 5)

K = MatrixMarket.mmread("bcsstk09.mtx")
nnzK = nnz(K)
n = size(K, 1)
normK = norm(K, 1)

# AMD and METIS orderings
AMD_P = amd(K)
METIS_P, METIS_invP = Metis.permutation(K)

println("AMD ordering")
@printf("%3s  %6s  %9s  %8s  %7s  %8s\n", "p", "nnz(K)", "nnz(L)", "‖LDLᵀ-K‖", "α", "time")
for p = 0:5:130
  LLDL, t, b, g, m = @timed lldl(K, P = AMD_P, memory = p)
  L = LLDL.L + I
  @printf(
    "%3d  %6d  %9d  %8.2e  %7.1e  %8.2e\n",
    p,
    nnzK,
    nnz(LLDL),
    norm(L * spdiagm(0 => LLDL.D) * L' - K[LLDL.P, LLDL.P], 1) / normK,
    LLDL.α,
    t
  )
end

println()

println("Metis ordering")
@printf("%3s  %6s  %9s  %8s  %7s  %8s\n", "p", "nnz(K)", "nnz(LDLᵀ)", "‖LDLᵀ-K‖", "α", "time")
for p = 0:5:130
  LLDL, t, b, g, m = @timed lldl(K, P = METIS_P, memory = p)
  L = LLDL.L + I
  @printf(
    "%3d  %6d  %9d  %8.2e  %7.1e  %8.2e\n",
    p,
    nnzK,
    nnz(LLDL),
    norm(L * spdiagm(0 => LLDL.D) * L' - K[LLDL.P, LLDL.P], 1) / normK,
    LLDL.α,
    t
  )
end
