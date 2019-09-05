using BenchmarkTools
using MatrixMarket

using LinearAlgebra
using SparseArrays

include("problems.jl")

SUITE = BenchmarkGroup()

SUITE["LLDL"] = BenchmarkGroup()

for mem in [0,5,10]
for p in problems
    
  A = MatrixMarket.mmread("../sqd-collection/$p/3x3/iter_0/K_0.mtx")
  A_upper = triu(MatrixMarket.mmread("../sqd-collection/$p/3x3/iter_0/K_0.mtx"))
  
  SUITE["FULL"]["LOADING"][p] = @benchmarkable MatrixMarket.mmread("../sqd-collection/" * string($p) *"/3x3/iter_0/K_0.mtx") samples = 10
  SUITE["UPPER"]["LOADING"][p] = @benchmarkable triu(MatrixMarket.mmread("../sqd-collection/" * string($p) *"/3x3/iter_0/K_0.mtx")) samples = 10
    
  SUITE["FULL"]["LDL"][p] = @benchmarkable ldl($A) samples = 10
  SUITE["UPPER"]["LDL"][p] = @benchmarkable ldl($A_upper, upper = $true) samples = 10
end
