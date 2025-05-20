"""
A Pure Julia Version of limited-memory LDLᵀ factorization.
A left-looking implementation of the sparse LDLᵀ factorization
of a symmetric matrix with the possibility to compute a
limited-memory incomplete factorization.

Dominique Orban <dominique.orban@gmail.com>
Montreal, April 2015, December 2015, July 2017.

This code is strongly inspired by Lin and Moré's ICFS [1,2].
The modified version is described in [3,4].

# References
[1] C.-J. Lin and J. J. Moré. Incomplete Cholesky factorizations with limited
    memory. SIAM Journal on Scientific Computing, 21(1):24--45, 1999.
[2] http://www.mcs.anl.gov/~more/icfs
[3] D. Orban. Limited-Memory LDLᵀ Factorization of Symmetric Quasi-Definite
    Matrices with Application to Constrained Optimization. Numerical Algorithms
    70(1):9--41, 2015. DOI 10.1007/s11075-014-9933-x
[4] https://github.com/optimizers/lldl
"""
module LimitedLDLFactorizations

export lldl,
  lldl_factorize!,
  \,
  ldiv!,
  nnz,
  LimitedLDLFactorization,
  factorized,
  update_shift!,
  update_shift_increase_factor!

using AMD, LinearAlgebra, SparseArrays

mutable struct LLDLException <: Exception
  msg::String
end

const error_string = "LLDL factorization was not computed or failed"

mutable struct LimitedLDLFactorization{
  T <: Real,
  Ti <: Integer,
  V1 <: AbstractVector,
  V2 <: AbstractVector,
}
  __factorized::Bool

  n::Int
  colptr::Vector{Ti}
  rowind::Vector{Ti}
  Lrowind::SubArray{Ti, 1, Vector{Ti}, Tuple{UnitRange{Int}}, true}
  lvals::Vector{T}
  Lnzvals::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}

  nnz_diag::Int
  adiag::Vector{T}

  D::Vector{T}
  P::V1
  α::T
  α_increase_factor::T
  α_out::T # α value after factorization (may be different of α if attempt_lldl! had failures)

  memory::Int

  Pinv::V2
  wa1::Vector{T}
  s::Vector{T}
  w::Vector{T}
  indr::Vector{Ti}
  indf::Vector{Ti}
  list::Vector{Ti}
  pos::Vector{Int}
  neg::Vector{Int}

  computed_posneg::Bool # true if pos and neg are computed (becomes false after factorization)
end

"""
    isfact = factorized(LLDL)

Returns true if the most recent factorization stored in `LLDL` [`LimitedLDLFactorization`](@ref) succeeded.
"""
factorized(LLDL::LimitedLDLFactorization) = LLDL.__factorized

"""
    update_shift!(LLDL, α)

Updates the shift `α` of the `LimitedLDLFactorization` object `LLDL`.
"""
function update_shift!(LLDL::LimitedLDLFactorization{T}, α::T) where {T <: Real}
  LLDL.α = α
  LLDL
end

"""
    update_shift_increase_factor!(LLDL, α_increase_factor)

Updates the shift increase value `α_increase_factor` of the `LimitedLDLFactorization` object `LLDL`
by which the shift `α` will be increased each time a `attempt_lldl!` fails.
"""
function update_shift_increase_factor!(
  LLDL::LimitedLDLFactorization{T},
  α_increase_factor::Number,
) where {T <: Real}
  LLDL.α_increase_factor = T(α_increase_factor)
  LLDL
end

function LimitedLDLFactorization(
  T::SparseMatrixCSC{Tv, Ti},
  P::AbstractVector{<:Integer},
  memory::Int,
  α::Tf,
  α_increase_factor::Tf,
  n::Int,
  nnzT::Int,
  ::Type{Tf},
) where {Tv <: Number, Ti, Tf <: Real}
  np = n * memory
  Pinv = similar(P)

  nnz_diag = 0
  adiag = Vector{Tf}(undef, n)
  for col = 1:n
    k = T.colptr[col]
    row = (k ≤ nnzT) ? T.rowval[k] : 0
    if row == col
      nnz_diag += 1
      adiag[col] = Tf(T.nzval[k])
    else
      adiag[col] = zero(Tf)
    end
  end

  # Make room to store L.
  nnzLmax = nnzT + np - nnz_diag
  d = Vector{Tf}(undef, n)  # Diagonal matrix D.
  lvals = Vector{Tf}(undef, nnzLmax)  # Strict lower triangle of L.
  rowind = Vector{Ti}(undef, nnzLmax)
  colptr = Vector{Ti}(undef, n + 1)
  wa1 = Vector{Tf}(undef, n)
  s = Vector{Tf}(undef, n)

  w = Vector{Tf}(undef, n)     # contents of the current column of A.
  indr = Vector{Ti}(undef, n)  # row indices of the nonzeros in the current column after it's been loaded into w.
  indf = Vector{Ti}(undef, n)  # indf[col] = position in w of the next entry in column col to be used during the factorization.
  list = zeros(Ti, n)  # list[col] = linked list of columns that will update column col.

  pos = findall(adiag[P] .> Tf(0))
  neg = findall(adiag[P] .≤ Tf(0))

  Lrowind = view(rowind, 1:nnzLmax)
  Lnzvals = view(lvals, 1:nnzLmax)

  return LimitedLDLFactorization(
    false,
    n,
    colptr,
    rowind,
    Lrowind,
    lvals,
    Lnzvals,
    nnz_diag,
    adiag,
    d,
    P,
    α,
    α_increase_factor,
    zero(Tf),
    memory,
    Pinv,
    wa1,
    s,
    w,
    indr,
    indf,
    list,
    pos,
    neg,
    true,
  )
end

"""
    LLDL = LimitedLDLFactorization(T; P = amd(T), memory = 0, α = 0, α_increase_factor = 10)
    LLDL = LimitedLDLFactorization(T, ::Type{Tf}; P = amd(T), memory = 0, α = 0, α_increase_factor = 10)

Perform the allocations for the LLDL factorization of symmetric matrix whose lower triangle is `T` 
with the permutation vector `P`.

# Arguments
- `T::SparseMatrixCSC{Tv,Ti}`: lower triangle of the matrix to factorize;
- `::Type{Tf}`: type used for the factorization, by default the type of the elements of `A`.

# Keyword arguments
- `P::AbstractVector{<:Integer} = amd(T)`: permutation vector;
- `memory::Int=0`: extra amount of memory to allocate for the incomplete factor `L`.
                   The total memory allocated is nnz(T) + n * `memory`, where
                   `T` is the strict lower triangle of A and `n` is the size of `A`;
- `α::Number=0`: initial value of the shift in case the incomplete LDLᵀ
                 factorization of `A` is found to not exist. The shift will be
                 gradually increased from this initial value until success;
- `α_increase_factor::Number=10`: value by which the shift will be increased after 
                                  the incomplete LDLᵀ factorization of `T` is found
                                  to not exist.

# Example
    A = sprand(Float64, 10, 10, 0.2)
    T = tril(A * A' + I)
    LLDL = LimitedLDLFactorization(T) # Float64 factorization
    LLDL = LimitedLDLFactorization(T, Float32) # Float32 factorization
"""
function LimitedLDLFactorization(
  T::SparseMatrixCSC{Tv, Ti},
  ::Type{Tf};
  P::AbstractVector{<:Integer} = amd(T),
  memory::Int = 0,
  α::Number = 0,
  α_increase_factor::Number = 10,
) where {Tv <: Number, Ti <: Integer, Tf <: Real}
  memory < 0 && error("limited-memory parameter must be nonnegative")
  n = size(T, 1)
  n != size(T, 2) && error("input matrix must be square")
  return LimitedLDLFactorization(T, P, memory, Tf(α), Tf(α_increase_factor), n, nnz(T), Tf)
end

LimitedLDLFactorization(T::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv <: Number, Ti <: Integer} =
  LimitedLDLFactorization(T, Tv; kwargs...)

# Here T is the lower triangle of A.
"""
    lldl_factorize!(S, T; droptol = 0.0)

Perform the in-place factorization of a symmetric matrix whose lower triangle is `T` 
with the permutation vector.

# Arguments
- `S::LimitedLDLFactorization{Tf, Ti}`;
- `T::SparseMatrixCSC{Tv,Ti}`: lower triangle of the matrix to factorize.
`T` should keep the same nonzero pattern and the sign of its diagonal elements.

# Keyword arguments
- `droptol::Tf=Tf(0)`: to further sparsify `L`, all elements with magnitude smaller
                       than `droptol` are dropped.
"""
function lldl_factorize!(
  S::LimitedLDLFactorization{Tf, Ti},
  T::SparseMatrixCSC{Tv, Ti};
  droptol::Tf = Tf(0),
) where {Tf <: Number, Tv <: Number, Ti <: Integer}
  n = size(T, 1)
  n != size(T, 2) && error("input matrix must be square")

  colptr = S.colptr

  nnzT = nnz(T)
  nnzT_nodiag = nnzT - S.nnz_diag
  memory = S.memory
  α = S.α
  α_increase_factor = S.α_increase_factor

  P = S.P
  Pinv = S.Pinv
  # Compute inverse permutation
  @inbounds for k = 1:n
    Pinv[P[k]] = k
  end

  adiag = S.adiag
  for col = 1:n
    k = T.colptr[col]
    row = (k ≤ nnzT) ? T.rowval[k] : 0
    adiag[col] = (row == col) ? Tf(T.nzval[k]) : zero(Tf)
  end

  d = S.D  # Diagonal matrix D.
  lvals = S.lvals  # Strict lower triangle of L.
  rowind = S.rowind
  colptr = S.colptr

  # Compute the 2-norm of columns of A
  # and the diagonal scaling matrix.
  wa1 = S.wa1
  wa1 .= zero(Tf)
  s = S.s
  @inbounds for col = 1:n
    s[col] = Tf(1) # Initialization
    @inbounds for k = T.colptr[col]:(T.colptr[col + 1] - 1)
      row = T.rowval[k]
      (row == col) && continue
      val = Tf(T.nzval[k])
      val2 = val * val
      wa1[Pinv[col]] += val2  # Contribution to column Pinv[col].
      wa1[Pinv[row]] += val2  # Contribution to column Pinv[row].
    end
  end

  @inbounds @simd for col = 1:n
    dpcol = adiag[P[col]]
    wa1[col] += dpcol * dpcol
    wa1[col] = sqrt(wa1[col])
    wa1[col] > 0 && (s[col] = 1 / sqrt(wa1[col]))
  end

  # Set initial shift. Keep it at zero if possible.
  α_min = sqrt(eps(Tv))
  if α > 0
    α = max(α, α_min)
  end
  max_increase_α = 3
  nb_increase_α = 0
  if any(x -> x == 0, adiag)
    α = max(α, α_min)
  end

  factorized = false
  tired = false

  # Work arrays.
  w = S.w    # contents of the current column of A.
  indr = S.indr  # row indices of the nonzeros in the current column after it's been loaded into w.
  indf = S.indf  # indf[col] = position in w of the next entry in column col to be used during the factorization.
  list = S.list  # list[col] = linked list of columns that will update column col.

  pos = S.pos
  neg = S.neg
  cpos = 0
  cneg = 0
  if !(S.computed_posneg)
    for i = 1:n
      adiagPi = adiag[P[i]]
      if adiagPi > Tf(0)
        cpos += 1
        pos[cpos] = i
      else
        cneg += 1
        neg[cneg] = i
      end
    end
  end
  S.computed_posneg = false

  while !(factorized || tired)

    # Copy the sparsity structure of A into L.
    # We use indr as a work array to save memory
    fill!(indr, 0)
    @inbounds for col = 1:n
      @inbounds for k = T.colptr[col]:(T.colptr[col + 1] - 1)
        row = T.rowval[k]
        (row == col) && continue
        indr[min(Pinv[row], Pinv[col])] += one(Ti)
      end
    end
    # cumulative sum
    colptr[1] = one(Ti)
    @inbounds for col = 1:n
      colptr[col + 1] = indr[col] + colptr[col]
      indr[col] = colptr[col]
    end

    # Store the scaled A into L.
    # Make room to store the computed factors.
    # We could save lots of storage here by "undoing" the scaling
    # at every attempt, at the price of introducing small errors
    # every time.

    @inbounds for col = 1:n
      pinvcol = Pinv[col]
      scol = s[pinvcol]
      d[pinvcol] = adiag[col] * scol * scol
      @inbounds for k = T.colptr[col]:(T.colptr[col + 1] - 1)
        row = T.rowval[k]
        (row == col) && continue
        pinvrow = Pinv[row]
        q = indr[min(pinvrow, pinvcol)]
        rowind[q] = max(pinvrow, pinvcol)
        lvals[q] = T.nzval[k] * scol * s[pinvrow]
        indr[min(pinvrow, pinvcol)] += one(Ti)
      end
    end

    @inbounds @simd for col in pos
      d[col] += α
    end
    @inbounds @simd for col in neg
      d[col] -= α
    end

    # Attempt a factorization.
    factorized = attempt_lldl!(
      nnzT_nodiag,
      d,
      lvals,
      rowind,
      colptr,
      w,
      indr,
      indf,
      list,
      memory = Ti(memory),
      droptol = droptol,
    )

    # Increase shift if the factorization didn't succeed.
    if !factorized
      nb_increase_α += 1
      α = (α == 0) ? α_min : α_increase_factor * α
      tired = nb_increase_α > max_increase_α
    end
  end

  S.__factorized = factorized

  # Unscale L.
  if factorized
    @inbounds @simd for col = 1:n
      scol = s[col]
      d[col] /= scol * scol
      @inbounds @simd for k = colptr[col]:(colptr[col + 1] - 1)
        lvals[k] *= scol / s[rowind[k]]
      end
    end
  end

  S.α_out = α
  nz = colptr[end] - 1
  S.Lrowind = view(rowind, 1:nz)
  S.Lnzvals = view(lvals, 1:nz)
  return S
end

"""
    lldl(A; P = amd(A), memory = 0, α = 0, droptol = 0, check_tril = true)
    lldl(A, ::Type{Tf}; P = amd(A), memory = 0, α = 0, droptol = 0, check_tril = true)

Compute the limited-memory LDLᵀ factorization of `A`.
`A` should be a lower triangular matrix.

# Arguments
- `A::SparseMatrixCSC{Tv,Ti}`: matrix to factorize (its strict lower triangle and
                               diagonal will be extracted);
- `::Type{Tf}`: type used for the factorization, by default the type of the elements of `A`.

# Keyword arguments
- `P::AbstractVector{<:Integer} = amd(A)`: permutation vector.
- `memory::Int=0`: extra amount of memory to allocate for the incomplete factor `L`.
                   The total memory allocated is nnz(T) + n * `memory`, where
                   `T` is the strict lower triangle of A and `n` is the size of `A`;
- `α::Number=0`: initial value of the shift in case the incomplete LDLᵀ
                 factorization of `A` is found to not exist. The shift will be
                 gradually increased from this initial value until success;
- `α_increase_factor::Number = 10`: value by which the shift will be increased after 
                                    the incomplete LDLᵀ factorization of `T` is found
                                    to not exist.
- `droptol::Tv=Tv(0)`: to further sparsify `L`, all elements with magnitude smaller
                       than `droptol` are dropped;
- `check_tril::Bool = true`: check if `A` is a lower triangular matrix.

# Example
    A = sprand(Float64, 10, 10, 0.2)
    As = A * A' + I
    LLDL = lldl(As) # lower triangle is extracted
    T = tril(As)
    LLDL = lldl(T) # Float64 factorization
    LLDL = lldl(T, Float32) # Float32 factorization
"""
function lldl(
  A::SparseMatrixCSC{Tv, Ti},
  ::Type{Tf};
  P::AbstractVector{<:Integer} = amd(A),
  memory::Int = 0,
  droptol::Real = Tv(0),
  α::Number = 0,
  α_increase_factor::Number = 10,
  check_tril::Bool = true,
) where {Tv <: Number, Ti <: Integer, Tf <: Real}
  T = (!check_tril || istril(A)) ? A : tril(A)
  S = LimitedLDLFactorization(
    T,
    Tf;
    P = P,
    memory = memory,
    α = α,
    α_increase_factor = α_increase_factor,
  )
  lldl_factorize!(S, T, droptol = Tf(droptol))
end

lldl(A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv <: Number, Ti <: Integer} =
  lldl(A, Tv; kwargs...)

lldl(A::Matrix{Tv}; kwargs...) where {Tv <: Number} = lldl(sparse(A); kwargs...)

# symmetric matrix input
function lldl(
  sA::Union{Symmetric{T, SparseMatrixCSC{T, Ti}}, Hermitian{T, SparseMatrixCSC{T, Ti}}};
  kwargs...,
) where {T <: Real, Ti <: Integer}
  sA.uplo == 'U' && error("matrix must contain the lower triangle")
  A = sA.data
  lldl(A; kwargs...)
end

function attempt_lldl!(
  nnzT::Int,
  d::Vector{Tv},
  lvals::Vector{Tv},
  rowind::Vector{Ti},
  colptr::Vector{Ti},
  w::Vector{Tv},
  indr::Vector{Ti},
  indf::Vector{Ti},
  list::Vector{Ti};
  memory::Ti = 0,
  droptol::Tv = Tv(0),
) where {Ti <: Integer, Tv <: Number}
  fill!(list, 0)
  fill!(indf, 0)
  n = size(d, 1)
  np = n * memory
  droptol = max(0, droptol)

  # Make room for L.
  @inbounds @simd for col = 1:(n + 1)
    colptr[col] += np
  end

  @inbounds @simd for k = nnzT:-1:1
    rowind[np + k] = rowind[k]
    lvals[np + k] = lvals[k]
  end

  # Attempt an incomplete LDL factorization.
  col_start = colptr[1]
  colptr[1] = one(Ti)

  # Scan each column in turn.
  for col = 1:n

    # The factorization fails if the current pivot is zero.
    dcol = d[col]
    dcol == 0 && return false

    # Load column col of A into w.
    col_end = colptr[col + 1] - one(Ti)
    nzcol = zero(Ti)
    @inbounds for k = col_start:col_end
      row = rowind[k]
      w[row] = lvals[k]  # w[row] = A[row, col] for each col in turn.
      nzcol += one(Ti)
      indr[nzcol] = row
      indf[row] = one(Ti)
    end

    # nzcol = number of nonzeros in current column.
    # (indr, w): sparse representation of current column:
    # - the nonzero element A[row, col] is in w[row], and
    # - the k-th nonzero element of A[:, col] is in w[indr[k]], (k = 1, ..., nzcol).

    # Update column col using previous columns.
    k = list[col]

    while k != 0
      kth_col_start = indf[k]
      kth_col_end = colptr[k + 1] - one(Ti)
      lval = lvals[kth_col_start]  # lval = L[col, k].
      dl = -d[k] * lval

      newk = list[k]
      kth_col_start += one(Ti)
      if kth_col_start < kth_col_end
        row = rowind[kth_col_start]
        indf[k] = kth_col_start
        list[k] = list[row]
        list[row] = k
      end

      # Perform the update L[row, col] <- L[row, col] - D[k, k] * L[col, k] * L[row, k].
      @inbounds for i = kth_col_start:kth_col_end
        row = rowind[i]
        dli = dl * lvals[i]
        if indf[row] != 0
          w[row] += dli  # w[row] = L[row, col], lval = L[col, k], lvals[i] = L[row, k].
        else
          indf[row] = one(Ti)
          nzcol += one(Ti)
          indr[nzcol] = row
          w[row] = dli
        end
      end

      k = newk
    end

    # Compute (incomplete) column col of L.
    @inbounds @simd for k = 1:nzcol
      w[indr[k]] /= dcol
      # d[row] -= d[col] * w[row] * w[row];  # Variant I.
    end

    nz_to_keep = min(col_end - col_start + one(Ti) + memory, nzcol)
    kth = nzcol - nz_to_keep + one(Ti)

    if nzcol ≥ 1
      # Determine the kth smallest elements in current column
      abspermute!(w, view(indr, 1:nzcol), kth)
      # At this point, w[indr[1:nzcol]] is partially sorted in increasing order of absolute
      # values. The kth smallest element of w in absolute value is in w[indr[kth]].

      # Sort the row indices of the nz_to_keep largest elements
      # so we can later retrieve L[i,k] from indf[k].
      sort!(indr, kth, nzcol, nz_to_keep ≤ 50 ? InsertionSort : MergeSort, Base.Order.Forward)
    end

    new_col_start = colptr[col]
    new_col_end = new_col_start + nz_to_keep - one(Ti)
    l = new_col_start
    @inbounds @simd for k = new_col_start:new_col_end
      k1 = indr[kth + k - new_col_start]
      val = w[k1]
      # record element unless it should be dropped
      if abs(val) > droptol
        lvals[l] = val
        rowind[l] = k1
        l += one(Ti)
      end
    end
    new_col_end = l - one(Ti)

    # Variant II of diagonal elements update.
    @inbounds @simd for k = kth:nzcol
      k1 = indr[k]
      wk1 = w[k1]
      d[k1] -= dcol * wk1 * wk1
    end

    if new_col_start < new_col_end
      indf[col] = new_col_start
      row1 = rowind[new_col_start]
      list[col] = list[row1]
      list[row1] = col
    end

    @inbounds @simd for k = 1:nzcol
      indf[indr[k]] = zero(Ti)
    end
    col_start = colptr[col + 1]
    colptr[col + 1] = new_col_end + one(Ti)
  end

  return true
end

"""Permute the elements of `keys` in place so that
    abs(x[keys[i]]) ≤ abs(x[keys[k]])  for i = 1, ..., k
    abs(x[keys[k]]) ≤ abs(x[keys[i]])  for i = k, ..., n,
where `n` is the length of `keys`. The length of `x` should be
at least `n`. Only `keys` is modified.
From the MINPACK2 function `dsel2` by Kastak, Lin and Moré.
"""
function abspermute!(
  x::Vector{Tv},
  keys::AbstractVector{Ti},
  k::Ti,
) where {Tv <: Number, Ti <: Integer}
  n = size(keys, 1)
  (n <= 1 || k < 1 || k > n) && return

  l = one(Ti)
  u = Ti(n)
  lc = Ti(n)
  lp = Ti(2 * n)

  while l < u
    p1 = div(u + 3 * l, 4)
    p2 = div(u + l, 2)
    p3 = div(3 * u + l, 4)

    if abs(x[keys[l]]) > abs(x[keys[p1]])
      swap = keys[l]
      keys[l] = keys[p1]
      keys[p1] = swap
    end

    if abs(x[keys[p2]]) > abs(x[keys[p3]])
      swap = keys[p2]
      keys[p2] = keys[p3]
      keys[p3] = swap
    end

    if abs(x[keys[p3]]) > abs(x[keys[p1]])
      swap = keys[p3]
      keys[p3] = keys[u]
      keys[u] = swap
      if abs(x[keys[p2]]) > abs(x[keys[p3]])
        swap = keys[p2]
        keys[p2] = keys[p3]
        keys[p3] = swap
      end
    else
      swap = keys[p1]
      keys[p1] = keys[u]
      keys[u] = swap
      if abs(x[keys[l]]) > abs(x[keys[p1]])
        swap = keys[l]
        keys[l] = keys[p1]
        keys[p1] = swap
      end
    end

    if abs(x[keys[p1]]) > abs(x[keys[p3]])
      if abs(x[keys[l]]) ≤ abs(x[keys[p3]])
        swap = keys[l]
        keys[l] = keys[p3]
        keys[p3] = swap
      end
    else
      if abs(x[keys[p2]]) ≤ abs(x[keys[p1]])
        swap = keys[l]
        keys[l] = keys[p1]
        keys[p1] = swap
      else
        swap = keys[l]
        keys[l] = keys[p2]
        keys[p2] = swap
      end
    end

    m = l
    absxl = abs(x[keys[l]])
    @inbounds for i = (l + 1):u
      if abs(x[keys[i]]) < absxl
        m = m + one(Ti)
        swap = keys[m]
        keys[m] = keys[i]
        keys[i] = swap
      end
    end

    swap = keys[l]
    keys[l] = keys[m]
    keys[m] = swap

    k ≥ m && (l = m + one(Ti))
    k ≤ m && (u = m - one(Ti))

    if (3 * (u - l) > 2 * lp) && (k > m)
      p = m
      absxm = abs(x[keys[m]])
      @inbounds for i = (m + 1):u
        if abs(x[keys[i]]) == absxm
          p = p + one(Ti)
          swap = keys[p]
          keys[p] = keys[i]
          keys[i] = swap
        end
      end

      l = p + one(Ti)
      k ≤ p && (u = p - one(Ti))
    end

    lp = lc
    lc = u - l
  end
end

function lldl_lsolve!(n, x::AbstractVector, Lp, Li, Lx)
  @inbounds for j = 1:n
    xj = x[j]
    @inbounds for p = Lp[j]:(Lp[j + 1] - 1)
      x[Li[p]] -= Lx[p] * xj
    end
  end
  return x
end

function lldl_dsolve!(n, x::AbstractVector, D)
  @inbounds for j = 1:n
    x[j] /= D[j]
  end
  return x
end

function lldl_ltsolve!(n, x::AbstractVector, Lp, Li, Lx)
  @inbounds for j = n:-1:1
    xj = x[j]
    @inbounds for p = Lp[j]:(Lp[j + 1] - 1)
      xj -= Lx[p] * x[Li[p]]
    end
    x[j] = xj
  end
  return x
end

function lldl_solve!(n, b::AbstractVector, Lp, Li, Lx, D, P)
  @views y = b[P]
  lldl_lsolve!(n, y, Lp, Li, Lx)
  lldl_dsolve!(n, y, D)
  lldl_ltsolve!(n, y, Lp, Li, Lx)
  return b
end

# solve functions for multiple rhs
function lldl_lsolve!(n, X::AbstractMatrix, Lp, Li, Lx)
  @inbounds for j = 1:n
    @inbounds for p = Lp[j]:(Lp[j + 1] - 1)
      for k ∈ axes(X, 2)
        X[Li[p], k] -= Lx[p] * X[j, k]
      end
    end
  end
  return X
end

function lldl_dsolve!(n, X::AbstractMatrix, D)
  @inbounds for j = 1:n
    for k ∈ axes(X, 2)
      X[j, k] /= D[j]
    end
  end
  return X
end

function lldl_ltsolve!(n, X::AbstractMatrix, Lp, Li, Lx)
  @inbounds for j = n:-1:1
    @inbounds for p = Lp[j]:(Lp[j + 1] - 1)
      for k ∈ axes(X, 2)
        X[j, k] -= conj(Lx[p]) * X[Li[p], k]
      end
    end
  end
  return X
end

function lldl_solve!(n, B::AbstractMatrix, Lp, Li, Lx, D, P)
  @views Y = B[P, :]
  lldl_lsolve!(n, Y, Lp, Li, Lx)
  lldl_dsolve!(n, Y, D)
  lldl_ltsolve!(n, Y, Lp, Li, Lx)
  return B
end

function _ldiv!(LLDL::LimitedLDLFactorization, b)
  factorized(LLDL) || throw(LLDLException(error_string))
  lldl_solve!(LLDL.n, b, LLDL.colptr, LLDL.Lrowind, LLDL.Lnzvals, LLDL.D, LLDL.P)
end
LinearAlgebra.ldiv!(LLDL::LimitedLDLFactorization, b) = _ldiv!(LLDL, b)

function _ldiv!(y, LLDL::LimitedLDLFactorization, b)
  y .= b
  ldiv!(LLDL, y)
end
LinearAlgebra.ldiv!(y, LLDL::LimitedLDLFactorization, b) = _ldiv!(y, LLDL, b)

Base.:\(LLDL::LimitedLDLFactorization, b) = ldiv!(LLDL, copy(b))

SparseArrays.nnz(LLDL::LimitedLDLFactorization) = length(LLDL.Lrowind) + length(LLDL.D)

function Base.getproperty(LLDL::LimitedLDLFactorization, prop::Symbol)
  if prop == :L
    nz = LLDL.colptr[end] - 1
    return SparseMatrixCSC(LLDL.n, LLDL.n, LLDL.colptr, LLDL.Lrowind[1:nz], LLDL.Lnzvals[1:nz])
  else
    getfield(LLDL, prop)
  end
end

end  # Module.
