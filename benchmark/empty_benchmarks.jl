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
const formulations = ("2x2", "3x3")
const iters = (0, 5, 10)

const SUITE = BenchmarkGroup()
