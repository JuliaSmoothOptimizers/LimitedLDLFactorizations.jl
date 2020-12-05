# Limited-Memory LDLᵀ Factorization

![CI](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl/workflows/CI/badge.svg?branch=master)
[![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl)

A Port of LLDL to Julia.
See https://github.com/optimizers/lldl.

LimitedLDLFactorizations is a limited-memory LDLᵀ factorization for symmetric matrices.
Given a symmetric matrix A, we search for a unit lower triangular L, a
diagonal D and a diagonal ∆ such that LDLᵀ is an incomplete factorization
of A+∆. The elements of the diagonal matrix ∆ have the form ±α, where α ≥ 0
is a shift.

It is possible to only supply the lower triangle of A and/or a prescribed permutation that attempts to diminish fill-in.
[AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl) and [Metis.jl](https://github.com/JuliaSparse/Metis.jl) are recommended packages for computing fill-reducing orderings of sparse matrices.

## Installing

```julia
julia> ]
pkg> add LimitedLDLFactorizations
```

## Brief Description

The only functions exported are `lldl`, `\`, `ldiv!` and `nnz`.
Supply a dense array or sparse matrix to `lldl`.
Dense arrays will be converted to sparse.
The strict lower triangle and diagonal of sparse matrices will be extracted.

Optionally, supply
* a memory parameter to allow more fill in the L factor;
* a drop tolerance to discard small elements in the L factor;
* an initial shift to speed up the process in case the factorization does not exist without shift.

Using a memory parameter larger than or equal to the size of A will yield an
exact factorization provided one exists with the permutation supplied.
In particular, the full factorization exists for any symmetric permutation of a symmetric quasi-definite matrix.

`lldl` returns a factorization in the form of a `LimitedLDLFactorization` object.
The `\` and `ldiv!` methods are implemented for objects of type `LimitedLDLFactorization`

## More Examples

See `examples/example.jl` and `tests/runtest.jl`.

## Complete Description

[1] C.-J. Lin and J. J. Moré. Incomplete Cholesky factorizations with limited
    memory. SIAM Journal on Scientific Computing, 21(1):24--45, 1999.
    DOI [10.1137/S1064827597327334](https://doi.org/10.1137/S1064827597327334).
<br>
[2] http://www.mcs.anl.gov/~more/icfs
<br>
[3] D. Orban. Limited-Memory LDLᵀ Factorization of Symmetric Quasi-Definite
    Matrices with Application to Constrained Optimization. Numerical Algorithms
    70(1):9--41, 2015. DOI [10.1007/s11075-014-9933-x](https://doi.org/10.1007/s11075-014-9933-x).
<br>
[4] https://github.com/optimizers/lldl
