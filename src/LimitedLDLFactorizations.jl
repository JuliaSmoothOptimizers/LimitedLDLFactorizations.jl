"""
A Pure Julia Version of LLDL.
A left-looking implementation of the sparse LDL factorization
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
[3] D. Orban. Limited-Memory LDLT Factorization of Symmetric Quasi-Definite
    Matrices with Application to Constrained Optimization. Numerical Algorithms
    70(1):9--41, 2015. DOI 10.1007/s11075-014-9933-x
[4] https://github.com/optimizers/lldl
"""
module LimitedLDLFactorizations

export lldl

using AMD, LinearAlgebra, SparseArrays

lldl(A::Array{Tv,2}; kwargs...) where Tv<:Number = lldl(sparse(A); kwargs...)

"""
    lldl(A)

Compute the limited-memory LDLᵀ factorization of A without pivoting.

# Arguments
- `A::SparseMatrixCSC{Tv,Ti}`: matrix to factorize (its strict lower triangle and
                               diagonal will be extracted)

# Keyword arguments
- `memory::Int=0`: extra amount of memory to allocate for the incomplete factor `L`.
                   The total memory allocated is nnz(T) + n * `memory`, where
                   `T` is the strict lower triangle of A and `n` is the size of `A`.
- `α::Tv=Tv(0)`: initial value of the shift in case the incomplete LDLᵀ
                 factorization of `A` is found to not exist. The shift will be
                 gradually increased from this initial value until success.
- `droptol::Tv=Tv(0)`: to further sparsify `L`, all elements with magnitude smaller
                       than `droptol` are dropped.
"""

function lldl(A::SparseMatrixCSC{Tv,Ti}; kwargs...) where {Tv<:Number, Ti<:Integer}
  lldl(tril(A, -1), diag(A), amd(A); kwargs...)
end

function lldl(A::SparseMatrixCSC{Tv,Ti}, P::Vector{Ti}; kwargs...) where {Tv<:Number, Ti<:Integer}
  lldl(tril(A, -1), diag(A), P; kwargs...)
end

# symmetric matrix input
function lldl(sA::Symmetric{T,SparseMatrixCSC{T,Ti}}, args...; kwargs...) where {T<:Real,Ti<:Integer}
  sA.uplo == 'U' && error("matrix must contain the lower triangle")
  A = sA.data
  lldl(A, args...; kwargs...)
end

# Here T is the strict lower triangle of A.
function lldl(T::SparseMatrixCSC{Tv,Ti},
              adiag::AbstractVector{Tv},
              P::Vector{Ti};
              memory::Int=0,
              α::Tv=Tv(0),
              droptol::Tv=Tv(0)) where {Tv<:Number, Ti<:Integer}

  memory < 0 && error("limited-memory parameter must be nonnegative")
  n = size(T, 1)
  n != size(T, 2) && error("input matrix must be square")
  n != length(adiag) && error("inconsistent size of diagonal")

  nnzT = nnz(T)
  np = n * memory

  Pinv = Vector{Ti}(undef, n)
  # Compute inverse permutation
  @inbounds for k = 1 : n
    Pinv[P[k]] = k
  end

  # Make room to store L.
  nnzLmax = nnzT + np
  d = Vector{Tv}(undef, n)  # Diagonal matrix D.
  lvals = Vector{Tv}(undef, nnzLmax)  # Strict lower triangle of L.
  rowind = Vector{Ti}(undef, nnzLmax)
  colptr = Vector{Ti}(undef, n+1)

  # Compute the 2-norm of columns of A
  # and the diagonal scaling matrix.
  wa1 = zeros(Tv, n)
  s = Vector{Tv}(undef, n)
  @inbounds @simd for col = 1 : n
    s[col] = Tv(1) # Initialization
    @inbounds @simd for k = T.colptr[col] : T.colptr[col+1] - 1
      val = T.nzval[k]
      val2 = val * val
      wa1[Pinv[col]] += val2  # Contribution to column Pinv[col].
      wa1[Pinv[T.rowval[k]]] += val2  # Contribution to column Pinv[T.rowval[k]].
    end
  end

  @inbounds @simd for col = 1 : n
    dpcol = adiag[P[col]]
    wa1[col] += dpcol * dpcol
    wa1[col] = sqrt(wa1[col])
    wa1[col] > 0 && (s[col] = 1 / sqrt(wa1[col]))
  end

  # Set initial shift. Keep it at zero if possible.
  α_min = 1.0e-3
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
  w = Vector{Tv}(undef, n)     # contents of the current column of A.
  indr = Vector{Ti}(undef, n)  # row indices of the nonzeros in the current column after it's been loaded into w.
  indf = Vector{Ti}(undef, n)  # indf[col] = position in w of the next entry in column col to be used during the factorization.
  list = zeros(Ti, n)  # list[col] = linked list of columns that will update column col.

  pos = findall(adiag[P] .> Tv(0))
  neg = findall(adiag[P] .≤ Tv(0))

  while !(factorized || tired)

    # Copy the sparsity structure of A into L.
    # We use indr as a work array to save memory
    fill!(indr, 0)
    @inbounds for col = 1 : n
      @inbounds for k = T.colptr[col] : T.colptr[col+1]-1
        row = Pinv[T.rowval[k]]
        indr[min(row, Pinv[col])] += 1
      end
    end
    # cumulative sum
    colptr[1] = 1
    @inbounds for col = 1 : n
      colptr[col+1] = indr[col] + colptr[col]
      indr[col] = colptr[col]
    end

    # Store the scaled A into L.
    # Make room to store the computed factors.
    # We could save lots of storage here by "undoing" the scaling
    # at every attempt, at the price of introducing small errors
    # every time.

    @inbounds for col = 1 : n
      pinvcol = Pinv[col]
      scol = s[pinvcol]
      d[pinvcol] = adiag[col] * scol * scol
      @inbounds for k = T.colptr[col] : T.colptr[col+1]-1
        row = Pinv[T.rowval[k]]
        q = indr[min(row,pinvcol)]
        rowind[q] = max(row,pinvcol)
        lvals[q] = T.nzval[k]*scol*s[row]
        indr[min(row,pinvcol)] += 1
      end
    end

    @inbounds @simd for col in pos
      d[col] += α
    end
    @inbounds @simd for col in neg
      d[col] -= α
    end

    # Attempt a factorization.
    factorized = attempt_lldl!(nnzT, d, lvals, rowind, colptr, w, indr, indf, list, memory=memory, droptol=droptol)

    # Increase shift if the factorization didn't succeed.
    if !factorized
      nb_increase_α += 1
      α *= 2
      tired = nb_increase_α > max_increase_α
    end
  end

  # Unscale L.
  @inbounds @simd for col = 1 : n
    scol = s[col]
    d[col] /= scol * scol
    @inbounds @simd for k = colptr[col] : colptr[col+1] - 1
      lvals[k] *= scol / s[rowind[k]]
    end
  end

  L = SparseMatrixCSC{Tv,Ti}(n, n, colptr, rowind, lvals)
  return (L, d, α, P)
end

function attempt_lldl!(nnzT::Int, d::Vector{Tv}, lvals::Vector{Tv},
                       rowind::Vector{Ti}, colptr::Vector{Ti},
                       w::Vector{Tv}, indr::Vector{Ti},
                       indf::Vector{Ti}, list::Vector{Ti};
                       memory::Int=0,
                       droptol::Tv=Tv(0)) where {Ti<:Integer, Tv<:Number}

  n = size(d, 1)
  np = n * memory
  droptol = max(0, droptol)

  # Make room for L.
  @inbounds @simd for col = 1 : n+1
    colptr[col] += np
  end

  @inbounds @simd for k = nnzT : -1 : 1
    rowind[np + k] = rowind[k]
    lvals[np + k] = lvals[k]
  end

  # Attempt an incomplete LDL factorization.
  col_start = colptr[1]
  colptr[1] = 1

  # Scan each column in turn.
  for col = 1 : n

    # The factorization fails if the current pivot is zero.
    dcol = d[col]
    dcol == 0 && return false

    # Load column col of A into w.
    col_end = colptr[col + 1] - 1
    nzcol = 0
    @inbounds for k = col_start : col_end
      row = rowind[k]
      w[row] = lvals[k]  # w[row] = A[row, col] for each col in turn.
      nzcol += 1
      indr[nzcol] = row
      indf[row] = 1
    end

    # nzcol = number of nonzeros in current column.
    # (indr, w): sparse representation of current column:
    # - the nonzero element A[row, col] is in w[row], and
    # - the k-th nonzero element of A[:, col] is in w[indr[k]], (k = 1, ..., nzcol).

    # Update column col using previous columns.
    k = list[col]

    while k != 0
      kth_col_start = indf[k]
      kth_col_end = colptr[k + 1] - 1
      lval = lvals[kth_col_start]  # lval = L[col, k].
      dl = -d[k] * lval

      newk = list[k]
      kth_col_start += 1
      if kth_col_start < kth_col_end
        row = rowind[kth_col_start]
        indf[k] = kth_col_start
        list[k] = list[row]
        list[row] = k
      end

      # Perform the update L[row, col] <- L[row, col] - D[k, k] * L[col, k] * L[row, k].
      @inbounds for i = kth_col_start : kth_col_end
        row = rowind[i]
        dli = dl * lvals[i]
        if indf[row] != 0
          w[row] += dli  # w[row] = L[row, col], lval = L[col, k], lvals[i] = L[row, k].
        else
          indf[row] = 1
          nzcol += 1
          indr[nzcol] = row
          w[row] = dli
        end
      end

      k = newk
    end

    # Compute (incomplete) column col of L.
    @inbounds @simd for k = 1 : nzcol
      w[indr[k]] /= dcol
      # d[row] -= d[col] * w[row] * w[row];  # Variant I.
    end

    nz_to_keep = min(col_end - col_start + 1 + memory, nzcol)
    kth = nzcol - nz_to_keep + 1

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
    new_col_end = new_col_start + nz_to_keep - 1
    l = new_col_start
    @inbounds @simd for k = new_col_start : new_col_end
      k1 = indr[kth + k - new_col_start]
      val = w[k1]
      # record element unless it should be dropped
      if abs(val) > droptol
        lvals[l] = val
        rowind[l] = k1
        l += 1
      end
    end
    new_col_end = l - 1

    # Variant II of diagonal elements update.
    @inbounds @simd for k = kth : nzcol
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

    @inbounds @simd for k = 1 : nzcol
      indf[indr[k]] = 0
    end
    col_start = colptr[col + 1]
    colptr[col + 1] = new_col_end + 1
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
function abspermute!(x::Vector{Tv}, keys::AbstractVector{Ti}, k::Ti) where {Tv<:Number, Ti<:Integer}

  n = size(keys, 1)
  (n <= 1 || k < 1 || k > n) && return

  l = Ti(1)
  u = Ti(n)
  lc = Ti(n)
  lp = Ti(2 * n)

  while l < u
    p1 = div(u + 3 * l,  4)
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
    @inbounds for i = l + 1 : u
      if abs(x[keys[i]]) < absxl
        m = m + 1
        swap = keys[m]
        keys[m] = keys[i]
        keys[i] = swap
      end
    end

    swap = keys[l]
    keys[l] = keys[m]
    keys[m] = swap

    k ≥ m && (l = m + 1)
    k ≤ m && (u = m - 1)

    if (3 * (u - l) > 2 * lp) && (k > m)
      p = m
      absxm = abs(x[keys[m]])
      @inbounds for i = m + 1 : u
        if abs(x[keys[i]]) == absxm
          p = p + 1
          swap = keys[p]
          keys[p] = keys[i]
          keys[i] = swap
        end
      end

      l = p + 1
      k ≤ p && (u = p - 1)
    end

    lp = lc
    lc = u - l
  end
end

end  # Module.
