using BenchmarkTools, MatrixMarket
using LinearAlgebra, SparseArrays, DelimitedFiles
using LimitedLDLFactorizations
using Pkg.Artifacts

# obtain path to SQD collection
const artifact_toml = joinpath(@__DIR__, "Artifacts.toml")
ensure_artifact_installed("sqdcollection", artifact_toml)
const sqd_hash = artifact_hash("sqdcollection", artifact_toml)
@assert artifact_exists(sqd_hash)
const sqd_path = joinpath(artifact_path(sqd_hash), "sqd-collection-0.1")

subdirs = readdir(sqd_path)
const formulations = ("2x2",)  # "3x3")
const iters = (0,)  # 5, 10)

const SUITE = BenchmarkGroup()

SUITE["LLDL"] = BenchmarkGroup()
# SUITE["LDIV"] = BenchmarkGroup()

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
            L = tril(A)
            P = 1:n
            SUITE["LLDL"][name] = @benchmarkable lldl($L, P=$P)
            # lldlt = lldl(L, P = P)
            # y = similar(b)
            # SUITE["LDIV"][name] = @benchmarkable ldiv!($y, $lldlt, $b)
        end
    end
end
