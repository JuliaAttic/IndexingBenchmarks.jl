# This file tests several different ways of computing the Euler flux over the
# mesh
# It tests various ways of taking small pieces of an array and passing them 
# to another function, as well as a double for loop for comparison

include("funcs1.jl")
using ArrayViews




function runtest()
n = 3000000  # number of elements
nnodes = 3  # number of nodes per element


q = rand(4, nnodes, n)  # input array
F = Array(Float64, 4, nnodes, n)  # output array

@time func1(q, F)
println("double loop @time printed above")

gc()


@time func2(q, F)
println("unsafe ArrayView @time printed above")

gc()

@time func2a(q, F)
println("safe ArrayView @time printed above")

gc()



@time func3(q, F)
println("slice @time printed above")

gc()

if VERSION >= v"0.4.3-pre+6"
  @time func4(q, F)
  println("unsafe slice @time printed above")

  gc()

end

@time func5(q, F)
println("single copy @time printed above")



gc()
return nothing

end

println("warming up")
runtest()
println("\nFinal testing: \n")
gc()
runtest()

