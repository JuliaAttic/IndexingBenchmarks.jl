include("lucompletepiv.jl")
timings = zeros(0, 9);
for n = [10, 20, 50, 100, 200, 500, 1000, 2000]
    A = randn(n, n);
    t = zeros(5);
    for i = 1:5
        t[i] = @elapsed lucompletepiv!(A);
    end
    newtimings = [n, t..., minimum(t), mean(t), std(t)]
    println(newtimings)
    timings = [timings; newtimings']
end
writecsv("../julia.csv", timings)
