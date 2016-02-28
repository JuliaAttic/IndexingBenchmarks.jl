# simple_fft.jl

# Precompute complex exponentials exp(-2πik/N)
twiddles(T,N) = T[exp(-2im*T(pi)*T(k)/N) for k in 0:N>>1-1]

function fft_copy!(y, x, Nratio = 1, tw = twiddles(eltype(y),length(y)))
    N = length(x)
    if N == 2
        y[1] = x[1] + x[2]
        y[2] = x[1] - x[2]
    else
        Nhalf = N>>1
        y_even = y[1:Nhalf]
        y_odd = y[Nhalf+1:end]
        fft_copy!(y_even, x[1:2:end], 2*Nratio, tw)
        fft_copy!(y_odd, x[2:2:end], 2*Nratio, tw)
        @inbounds for k = 1:Nhalf
            t = tw[(k-1)*Nratio + 1]
            y[k] = y_even[k] + t * y_odd[k]
            y[Nhalf+k] = y_even[k] - t * y_odd[k]
        end
    end
    y
end


function fft_sub!(y, x, Nratio = 1, tw = twiddles(eltype(y),length(y)))
    N = length(x)
    if N == 2
        y[1] = x[1] + x[2]
        y[2] = x[1] - x[2]
    else
        Nhalf = N>>1
        y_even = sub(y, 1:Nhalf)
        y_odd = sub(y, Nhalf+1:N)
        fft_sub!(y_even, sub(x, 1:2:N-1), 2*Nratio, tw)
        fft_sub!(y_odd, sub(x, 2:2:N), 2*Nratio, tw)
        @inbounds for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y_even[k]
            t = tw[(k-1)*Nratio + 1]
            # t = exp(-2im*pi*(k-1)/N)
            y[k] = y_even_k + t * y_odd[k]
            y[Nhalf+k] = y_even_k - t * y_odd[k]
        end
    end
    y
end

using ArrayViews

function fft_view!(y, x, Nratio = 1, tw = twiddles(eltype(y),length(y)))
    N = length(x)
    if N == 2
        y[1] = x[1] + x[2]
        y[2] = x[1] - x[2]
    else
        Nhalf = N>>1
        y_even = view(y, 1:Nhalf)
        y_odd = view(y, Nhalf+1:N)
        fft_view!(y_even, view(x, 1:2:N-1), 2*Nratio, tw)
        fft_view!(y_odd, view(x, 2:2:N), 2*Nratio, tw)
        @inbounds for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y_even[k]
            t = tw[(k-1)*Nratio + 1]
            y[k] = y_even_k + t * y_odd[k]
            y[Nhalf+k] = y_even_k - t * y_odd[k]
        end
    end
    y
end


fft_strides!(y, x, Nratio = 1, tw = twiddles(eltype(y),length(y))) =
    fft_strides_rec!(y, x, 1, length(y), 1, 1, length(x), Nratio, tw)

# Compute the fft of x[j1:stride:j1+(N-1)*stride] and store the result in y[i1:i2]
function fft_strides_rec!(y, x, i1, i2, j1, stride, N, Nratio, tw)
    if N == 2
        y[i1] = x[j1] + x[j1+stride]
        y[i1+1] = x[j1] - x[j1+stride]
    else
        Nhalf = N>>1
        fft_strides_rec!(y, x, i1, i1+Nhalf-1, j1, 2*stride, Nhalf, 2*Nratio, tw)
        fft_strides_rec!(y, x, i1+Nhalf, i1+N-1, j1+stride, 2*stride, Nhalf, 2*Nratio, tw)
        @inbounds for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y[i1+k-1]
            t = tw[(k-1)*Nratio + 1]
            y[i1+k-1] = y_even_k + t * y[i1+Nhalf+k-1]
            y[i1+Nhalf+k-1] = y_even_k - t * y[i1+Nhalf+k-1]
        end
    end
    y
end

fft_ranges!(y, x, Nratio = 1, tw = twiddles(eltype(y),length(y))) =
    fft_ranges_rec!(y, x, 1:length(y), 1:length(x), Nratio, tw)

# Compute the fft of x[xrange] and store the result in y[yrange]
function fft_ranges_rec!(y, x, yrange, xrange, Nratio, tw)
    N = length(xrange)
    if N == 2
        y[yrange[1]] = x[xrange[1]] + x[xrange[2]]
        y[yrange[2]] = x[xrange[1]] - x[xrange[2]]
    else
        Nhalf = N>>1
        fft_ranges_rec!(y, x, yrange[1:Nhalf], xrange[1:2:end], 2*Nratio, tw)
        fft_ranges_rec!(y, x, yrange[Nhalf+1:end], xrange[2:2:end], 2*Nratio, tw)
        @inbounds for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y[yrange[k]]
            t = tw[(k-1)*Nratio + 1]
            y[yrange[k]] = y_even_k + t * y[yrange[Nhalf+k]]
            y[yrange[Nhalf+k]] = y_even_k - t * y[yrange[Nhalf+k]]
        end
    end
    y
end


function run_single_test(T, N)
    x = zeros(T, N)
    for i in eachindex(x)
        x[i] = rand()
    end
    y_copy = zeros(Complex{T}, size(x))
    y_sub = zeros(Complex{T}, size(x))
    y_view = zeros(Complex{T}, size(x))
    y_strides = zeros(Complex{T}, size(x))
    y_ranges = zeros(Complex{T}, size(x))

    tw = twiddles(Complex{T}, N)

    # Number of times the computations are performed when timing
    ITER = 10
    println("N: 2^", Int(log2(N)))
    print("    copy: ")
    @time for i in 1:ITER
        fft_copy!(y_copy, x, 1, tw)
    end
    print("    sub: ")
    @time for i in 1:ITER
        fft_sub!(y_sub, x, 1, tw)
    end
    print("    view: ")
    @time for i in 1:ITER
        fft_view!(y_view, x, 1, tw)
    end
    print("    strides: ")
    @time for i in 1:ITER
        fft_strides!(y_strides, x, 1, tw)
    end
    print("    ranges: ")
    @time for i in 1:ITER
        fft_ranges!(y_ranges, x, 1, tw)
    end
    println("Maximal difference: ", maximum(abs(y_copy-y_sub)) + 
        maximum(abs(y_copy-y_view)) + maximum(abs(y_copy-y_strides)) + maximum(abs(y_copy-y_ranges)))
end

function run_several_tests(T, Nvalues)
    # Warmup
    x = zeros(T, 2^5)
    y_copy = zeros(Complex{T}, length(x))
    y_sub = similar(y_copy)
    y_view = similar(y_copy)
    y_strides = similar(y_copy)
    y_ranges = similar(y_copy)
    fft_copy!(y_copy, x)
    fft_sub!(y_sub, x)
    fft_view!(y_view, x)
    fft_strides!(y_strides, x)
    fft_ranges!(y_ranges, x)

    println("#####################")
    println("T is ", T)
    println("#####################")
    for N in Nvalues
        run_single_test(T, N)
    end
    println()
end

function runtests()
    println(test_explanation)
    run_several_tests(Float64, (2^5, 2^10, 2^15, 2^17))
    run_several_tests(BigFloat, (2^5, 2^10))
end


test_explanation = 
"""
We test a naieve implementation of a butterfly fft in various ways.

The butterfly algorithm is recursive: the FFT of a vector of length N is computed by recombining
the FFT of two subvectors of length N/2. The subvectors contain the 'even' and 'odd' samples of x.
We restrict N to be a power of 2 for simplicity, and we test five different ways
of recursively passing the subvectors:
(1) by copying
(2) using sub
(3) using view
(4) by manually passing indices and strides
(5) by manually passing ranges.

Recursion is performed until N==2. As a result, a very large number of views is being created.
The actual computations being performed are exactly the same in all cases: the computational difference
is exactly zero. No optimization has been applied, except for the precomputation of the complex exponentials.

Preliminary results at the time of writing (end of February 2016, on an old Macbook Air):
- The timings between making copies and creating views (sub/view) are very similar, with a small edge for the
views (a factor less than 2).
- Manually passing indices and strides is 3-4 times faster than creating views (on an old Macbook Air) and does
not allocate memory.
- Manually passing ranges does not allocate memory either, but it is the slowest approach.
- A lot of memory is being allocated for views and subs. The amount of memory is O(N), because so many views
are being created.
- Results are comparable between Julia 0.4.3 and 0.5 master.
- The timings for BigFloats are all very similar. They seem dominated by memory allocation for the BigFloat's.
"""

runtests()

