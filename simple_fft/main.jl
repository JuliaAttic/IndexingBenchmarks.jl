# simple_fft.jl


function fft_copy!(y, x)
    N = length(x)
    if N == 1
        y[1] = x[1]
    else
        Nhalf = N>>1
        y_even = y[1:Nhalf]
        y_odd = y[Nhalf+1:end]
        fft_copy!(y_even, x[1:2:end])
        fft_copy!(y_odd, x[2:2:end])
        for k = 1:Nhalf
            y[k] = y_even[k] + exp(-2im*pi*(k-1)/N) * y_odd[k]
            y[Nhalf+k] = y_even[k] - exp(-2im*pi*(k-1)/N) * y_odd[k]
        end
    end
    y
end


function fft_sub!(y, x)
    N = length(x)
    if N == 2
        y[1] = x[1]+x[2]
        y[2] = x[1]-x[2]
    else
        Nhalf = N>>1
        y_even = sub(y, 1:Nhalf)
        y_odd = sub(y, Nhalf+1:N)
        fft_sub!(y_even, sub(x, 1:2:N-1))
        fft_sub!(y_odd, sub(x, 2:2:N))
        for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y_even[k]
            y[k] = y_even_k + exp(-2im*pi*(k-1)/N) * y_odd[k]
            y[Nhalf+k] = y_even_k - exp(-2im*pi*(k-1)/N) * y_odd[k]
        end
    end
    y
end

using ArrayViews

function fft_view!(y, x)
    N = length(x)
    if N == 1
        y[1] = x[1]
    else
        Nhalf = N>>1
        y_even = view(y, 1:Nhalf)
        y_odd = view(y, Nhalf+1:N)
        fft_view!(y_even, view(x, 1:2:N-1))
        fft_view!(y_odd, view(x, 2:2:N))
        for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y_even[k]
            y[k] = y_even_k + exp(-2im*pi*(k-1)/N) * y_odd[k]
            y[Nhalf+k] = y_even_k - exp(-2im*pi*(k-1)/N) * y_odd[k]
        end
    end
    y
end


fft_strides!(y, x) = fft_strides_rec!(y, x, 1, length(y), 1, 1, length(x))

# Compute the fft of x[j1:stride:j1+(N-1)*stride] and store the result in y[i1:i2]
function fft_strides_rec!(y, x, i1, i2, j1, stride, N)
    if N == 1
        y[i1] = x[j1]
    else
        Nhalf = N>>1
        fft_strides_rec!(y, x, i1, i1+Nhalf-1, j1, 2*stride, Nhalf)
        fft_strides_rec!(y, x, i1+Nhalf, i1+N-1, j1+stride, 2*stride, Nhalf)
        for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y[i1+k-1]
            y[i1+k-1] = y_even_k + exp(-2im*pi*(k-1)/N) * y[i1+Nhalf+k-1]
            y[i1+Nhalf+k-1] = y_even_k - exp(-2im*pi*(k-1)/N) * y[i1+Nhalf+k-1]
        end
    end
    y
end

fft_ranges!(y, x) = fft_ranges_rec!(y, x, 1:length(y), 1:length(x))

# Compute the fft of x[j1:stride:j1+(N-1)*stride] and store the result in y[i1:i2]
function fft_ranges_rec!(y, x, yrange, xrange)
    N = length(xrange)
    if N == 1
        y[yrange[1]] = x[xrange[1]]
    else
        Nhalf = N>>1
        fft_ranges_rec!(y, x, yrange[1:Nhalf], xrange[1:2:end])
        fft_ranges_rec!(y, x, yrange[Nhalf+1:end], xrange[2:2:end])
        for k = 1:Nhalf
            # Store y_even[k] because we will overwrite it
            y_even_k = y[yrange[k]]
            y[yrange[k]] = y_even_k + exp(-2im*pi*(k-1)/N) * y[yrange[Nhalf+k]]
            y[yrange[Nhalf+k]] = y_even_k - exp(-2im*pi*(k-1)/N) * y[yrange[Nhalf+k]]
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

    println("N: 2^", Int(log2(N)))
    print("    copy: ")
    @time fft_copy!(y_copy, x)
    print("    sub: ")
    @time fft_sub!(y_sub, x)
    print("    view: ")
    @time fft_view!(y_view, x)
    print("    strides: ")
    @time fft_strides!(y_strides, x)
    print("    ranges: ")
    @time fft_ranges!(y_ranges, x)
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
    run_several_tests(Float64, (2^5, 2^10, 2^15, 2^18))
    run_several_tests(BigFloat, (2^5, 2^10, 2^13))
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

Recursion is performed until N==1. As a result, a very large number of views is being created.
The actual computations being performed are exactly the same in all cases: the computational difference
is exactly zero. No optimization has been applied.

Preliminary results at the time of writing (end of February 2016):
- Timings are comparable for all implementations. Manually passing indices and strides is fastest
for Float64. For BigFloat there is no difference: the overhead of memory allocations seems dominant.
- A lot of memory is being allocated for views and subs. The amount of memory is O(N), because so many views
are being created.
- Very little memory is allocated for the manual methods. Yet, they are not much faster. That remains so
even when N is larger than the values used in the tests. The cost of the actual computations seems dominant.
- Results are comparable between Julia 0.4.3 and 0.5 master.
"""

runtests()

