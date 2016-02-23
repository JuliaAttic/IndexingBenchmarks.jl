function eulermaruyama1!(yy, dww, B)
    N = size(dww, 2)
    for i in 1:N-1
        yy[:, i+1] += B*yy[:, i] + dww[:,i]
    end
    yy
end

function eulermaruyama2!(yy, dww, B)
    N = size(dww, 2)
    for i in 1:N-1
        yy[:, i+1] += B*sub(yy, :, i) + sub(dww,:,i)
    end
    yy
end

function eulermaruyamatest(problemparam, iter)
    d, n, eulermaruyamai = problemparam # d > 1, n > 0
    B = -eye(d); B[1:2,1:2] = [-1. -0.1; 0.2  -1.2]
    dW = randn(d, n)
    Y = zeros(d, n)
    for i in 1:iter
        Y[:] = 0
        eulermaruyamai(Y, dW, B)
    end
    Y 
end



