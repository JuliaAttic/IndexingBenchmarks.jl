# different ways to calculate the Euler flux


# double for loop
function func1(q, F_xi)
    getEulerFlux(q, F_xi)
  return nothing
end

# unsafe view
# note that there has been a performacne regression in 0.4 release
# see ArrayViews issue #41
# before, this was as fast as the double for loop
function func2(q, F_xi)

   (ncomp, nnodes,  nel) = size(q)
   for i=1:nel
      for j=1:nnodes
         q_j = unsafe_view(q, :, j, i)
         F_j = unsafe_view(q, :, j, i)

         getEulerFlux(q_j, F_j)
      end
   end

   return nothing

end

# safe view
function func2a(q, F_xi)

   (ncomp, nnodes,  nel) = size(q)
   for i=1:nel
      for j=1:nnodes
         q_j = view(q, :, j, i)
         F_j = view(q, :, j, i)

         getEulerFlux(q_j, F_j)
      end
   end

   return nothing

end


# slice
function func3(q, F_xi)

    (ncomp, nnodes,  nel) = size(q)
   for i=1:nel
      for j=1:nnodes
         q_j = slice(q, :, j, i)
         F_j = slice(q, :, j, i)
         getEulerFlux(q_j, F_j)
      end
   end

  return nothing
end

# unsafe slice
function func4(q, F_xi)

    (ncomp, nnodes,  nel) = size(q)
   for i=1:nel
      for j=1:nnodes
         q_j = Base._slice_unsafe(q, (:, j, i))
         F_j = Base._slice_unsafe(q, (:, j, i))
         getEulerFlux(q_j, F_j)
      end
   end

  return nothing
end

# copy the q_j, use view of F_j because it is necessary to get correct
# results
function func5(q, F_xi)

    (ncomp, nnodes,  nel) = size(q)
   for i=1:nel
      for j=1:nnodes
         q_j = q[:, j, i]
         F_j = unsafe_view(q, :, j, i)
         getEulerFlux(q_j, F_j)
      end
   end

  return nothing
end


#------------------------------------------------------------------------------
# Functions that do all the computation
#------------------------------------------------------------------------------

function getEulerFlux{T}( q::AbstractArray{T,3},  F_xi::AbstractArray{T,3})
# loop over big arrays to calculate euler flux

(ncomp, nnodes, nel) = size(q)  # get sizes of things

  for i=1:nel  # loop over elements
    for j=1:nnodes  # loop over nodes within element
      # get direction vector components (xi direction)
      nx = 1
      ny = 0
      # calculate pressure 
      press = (1.4-1)*(q[4, j, i] - 0.5*(q[2, j, i]^2 + q[3, j, i]^2)/q[1, j, i])

      # calculate flux in xi direction
      # hopefully elements of q get stored in a register for reuse in eta direction
      U = (q[2, j, i]*nx + q[3, j, i]*ny)/q[1, j, i]
      F_xi[1, j, i] = q[1, j, i]*U
      F_xi[2, j, i] = q[2, j, i]*U + nx*press
      F_xi[3, j, i] = q[3, j, i]*U + ny*press
      F_xi[4, j, i] = (q[4, j, i] + press)*U
    end
  end

  return nothing

end

function getEulerFlux{T}( q::AbstractArray{T,1}, F_xi::AbstractArray{T,1})
# get flux at single node 

      # get direction vector components (xi direction)
      nx = 1
      ny = 0
      # calculate pressure 
      press = (1.4-1)*(q[4] - 0.5*(q[2]^2 + q[3]^2)/q[1])

      # calculate flux in xi direction
      # hopefully elements of q get stored in a register for reuse in eta direction
      U = (q[2]*nx + q[3]*ny)/q[1]
      F_xi[1] = q[1]*U
      F_xi[2] = q[2]*U + nx*press
      F_xi[3] = q[3]*U + ny*press
      F_xi[4] = (q[4] + press)*U

     return nothing
end
 


