

#Spectral decomposition
"""

     eigenK(K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true)

Returns a 3d-array (or a matrix) of eigen vectors (orthogonal matrix) and the corresponding matrix (or vector) of eigen values if `LOCO` is true (false).

# Arguments

- `K` : a 3d-array of a symmetric positive definite kinship, `size(K) = (n,n, # of Chromosomes)` if `LOCO = true` (default).  Or a n x n matrix if false.

## Keyword Arguments

- `LOCO` : Boolean.  Default is true. 

# Output

- `T` : a 3d-array (or matrix) of eigen vectors 
- `S` : a matrix (or vector) of eigen values


"""
function eigenK(K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true,δ::Float64=0.001)
    
   if (LOCO)
       chr = size(K,3); 
       S = zeros(n,nchr);
       T = zeros(n,n,chr);
    
        for j =1:nchr ## add parallelization later
           if(!isposdef(K[:,:,j])) # check pdf
                K[:,:,j]=K[:,:,j]+ (abs(eigmin(K[:,:,j]))+δ)*I
           end
                
       #spectral decomposition
         F=eigen(K[:,:,j])
         T[:,:,j], S[:,j] = F.vectors, F.values
        end
       
        return  T, S
    else
          if(!isposdef(K)) # check pdf
                K=K+ (abs(eigmin(K))+δ)*I
           end
        F=eigen(K)
        return F.vectors, F.values
        
    end
        
end




"""
  
    rotate(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},T::Matrix{Float64})
    rotateX(X::Matrix{Float64},T::Matrix{Float64})
    rotateY(y::Vector{Float64},T::Matrix{Float64})

Returns transformed data rotated by eigenvectors.  `rotateX` transforms a matrix, `rotateY` does a vector, and `rotate` does both.

# Arguments

- `y` : a n x 1 vector of binary trait
- `X` : a n x p matrix of genetic markers selected from QTL analysis (per Chromosome for LOCO)
- `X₀` : a n x c matrix of covariates including intercepts
- `T` : a n x n matrix of orthogonal matrix (eigen vectors from kinship, `K`).  See also [`eigenK`](@ref).

# Output

- `yt` : a tranformed `y`
- `Xt` : a transformed `X`
- `Xt₀` : a transformed `X₀`


"""
function rotate(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},T::Matrix{Float64})

    Xt= rotateX(X,T)
    Xt₀=rotateX(X₀,T)
    yt= rotateY(y,T)
    
    return Xt, Xt₀, yt


end

function rotateX(X::Matrix{Float64},T::Matrix{Float64})
    
    
    #U'X, U'X₀,U'(y-1/2)
    
    Xt=BLAS.gemm('T','N',T,X)
       
    return Xt
    
end


function rotateY(y::Vector{Float64},T::Matrix{Float64})
    
    n=length(y)
    
    #U'X, U'X₀, U'(y-1/2)
    yt= BLAS.gemv('T',T,(y-0.5*ones(n)))
       
    return yt
    
end




#compute X'y : 't': 'T' for transpose, 'N' for no transpose
function getXy(Xt::Matrix{Float64},yt::Vector{Float64},t::Char)
    
    y1= BLAS.gemv(t,Xt,yt)
    
    return y1



"""

    logistic(ξ::Float64)
    
A logistic function computing the probablity of an event.  
    
"""
function logistic(ξ::Float64)
    
    return 1.0/(1.0+ exp(-ξ))
    
end

# λ(ξ)
function Lambda(ξ::Float64)
   
    return 0.25*tanh(ξ)/ξ    
end


"""    

    Seed(M::Int64,replace=::Bool=false)
    
Reseed random number generators to workers (or processes) by sampling different number of seeds between 1 and M for distributed computing.  

    
# Arguments

- `M` : an integer.  Set the maximum number of seeds.  
- `replace` : Bool.  Default is false. Sampling with/without replacement.

# Examples
    
```julia
    
julia> using Distributed, Random
julia> addprocs(10)    # generate 10 workers (or processes)
julia> @everywhere using SuSiEGLMM  # access the pkg to all workers
julia> Seed(20)
    
```

"""    
function Seed(M::Int64,replace=::Bool=false)
        
        np=nprocs(); pid=procs()
        if (M < np) && (!replace)
            println("Error. The random number generator should be greater than the number of nprocs, $(np).")
        seeds = sample(1:M, np, replace=replace)
        
        for j =1:np
            remotecall_fetch(()->Random.seed!(seeds[i]),pid[i])
        end
        
end
    