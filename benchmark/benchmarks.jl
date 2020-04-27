using BenchmarkTools, MatrixMarket
using LinearAlgebra, SparseArrays, DelimitedFiles
using LimitedLDLFactorizations

# download from https://github.com/optimizers/sqd-collection
# run(`git clone https://github.com/optimizers/sqd-collection.git`)

const sqd_path = joinpath(dirname(pathof(LimitedLDLFactorizations)), "..", "benchmark", "sqd-collection")
subdirs = readdir(sqd_path)
const formulations = ("2x2", "3x3")
const iters = (0, 5, 10)

const SUITE = BenchmarkGroup()

SUITE["LLDL"] = BenchmarkGroup()
SUITE["LDIV"] = BenchmarkGroup()

for subdir ∈ subdirs
  subdir == ".git" && continue
  isdir(joinpath(sqd_path, subdir)) || continue  # ignore regular files
  for formulation ∈ formulations
    for iter ∈ iters
      iterpath = joinpath(sqd_path, subdir, formulation, "iter_$(iter)")
      isdir(iterpath) || continue
      name = "$(subdir)_$(formulation)_$(iter)"
      A = MatrixMarket.mmread(joinpath(iterpath, "K_$(iter).mtx"))
      b = readdlm(joinpath(iterpath, "rhs_$(iter).rhs"), Float64)[:]
      n = size(A, 1)
      L = tril(A, -1)
      D = diag(A)
      P = collect(1:n)
      SUITE["LLDL"][name] = @benchmarkable lldl($L, $D, $P)
      lldlt = lldl(L, D, P)
      y = similar(b)
      SUITE["LDIV"][name] = @benchmarkable ldiv!($y, $lldlt, $b)
    end
  end
end
