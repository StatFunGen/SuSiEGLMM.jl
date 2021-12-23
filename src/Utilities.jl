export svdK,rotate,logistic,Lambda,Seed, dropCol

#Spectral decomposition
"""

svdK(K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true)

Returns a 3d-array (or a matrix) of orthogonal vectors and the corresponding matrix (or vector) of singluar values if `LOCO` is true (false).

# Arguments

- `K` : a 3d-array of a symmetric positive definite kinship, `size(K) = (n,n, # of Chromosomes)` if `LOCO = true` (default).  Or a n x n matrix if false.

## Keyword Arguments

- `LOCO` : Boolean.  Default is true. 

# Output

- `T` : a 3d-array (or matrix) of orthogonal vectors
- `S` : a matrix (or vector) of singluar values in descending order


"""
function svdK(K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true,δ::Float64=0.001)
    
    if (LOCO)
        nchr = size(K,3);  n = size(K,1);
        S = zeros(n,nchr);
        T = zeros(n,n,nchr);
     
         for j =1:nchr 
            # may consider to svd from the genotype data
            if(!isposdef(K[:,:,j])) # check pdf
                 K[:,:,j]=K[:,:,j]+ (abs(eigmin(K[:,:,j]))+δ)*I
            end
                 
        #cholesky & svd for numerical stability
        #   C=cholesky(K[:,:,j])
          F=svd(K[:,:,j])
          T[:,:,j], S[:,j] = F.Vt, F.S
         end
        
         return  T, S
     else
           if(!isposdef(K)) # check pdf
                 K=K+ (abs(eigmin(K))+δ)*I
            end
        #  C=cholesky(K)   
         F=svd(K)
         return F.Vt, F.S
         
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
    
    Xt=BLAS.gemm('N','N',T,X)
       
    return Xt
    
end


function rotateY(y::Vector{Float64},T::Matrix{Float64})
    
    n=length(y)
    
    #U'X, U'X₀, U'(y-1/2)
    yt= BLAS.gemv('N',T,(y.-0.5))
       
    return yt
    
end




#compute X'y : 't': 'T' for transpose, 'N' for no transpose
function getXy(t::Char,Xt::Matrix{Float64},yt::Vector{Float64})
    
    y1= BLAS.gemv(t,Xt,yt)
    
    return y1
end


#matrix multiplication AB: 'tA', 'tB': 'T' for transpose, 'N' for no transpose
function getXX(tA::Char,A::Matrix{Float64},tB::Char,B::Matrix)
    
     X = BLAS.gemm(tA,tB,A,B)
    
    return X
    
end

# compute symmetric square matrix, t ='T' for X'X, t='N' for XX'
function symXX(t::Char,X::Array{Float64,2})
    
   XtX = Symmetric(BLAS.syrk('U',t,1.0,X))
    
    return XtX
    
end


function dropCol(B::Matrix{Float64},l::Int64)

    return B[:,1:end.!= l]
  end
  

"""

    logistic(ξ::Float64)
    
A logistic function computing the probablity of an event.  
    
"""
function logistic(ξ::Float64)
    
    return 1.0/(1.0+ exp(-ξ))
    
end

# 2λ(ξ)
function Lambda(ξ::Float64)
        if(ξ==0.0) #adjust 0/0 case by L'Hopital
            return 0.25
        else
            return 0.5*tanh(ξ/2)/ξ    
        end
end


"""

    GenoInfo(snp::Array{String,1},chr::Array{Any,1},pos::Array{Float64,1})

A type struct includes information on genotype data

# Arguements

- `snp` : a vector of strings for genetic marker (or SNP) names
- `chr` : a vector of Chromosome information corresponding to `snp`
- `pos` : a vector of `snp` position in `cM`

"""
struct GenoInfo
    snp::Array{String,1}
    chr::Array{Any,1}
    pos::Array{Float64,1} #positon 
end



"""    

    Seed(M::Int64)
    
Reseed random number generators to assign workers (or processes) different number of seeds (increased by 1) for reproducible distributed computing.  

    
# Arguments

- `M` : an integer.  Set the seed number.  

# Examples
    
```julia
    
julia> using Distributed, Random
julia> addprocs(10)    # generate 10 workers (or processes)
julia> @everywhere using SuSiEGLMM  # access the pkg to all workers
julia> Seed(20)
    
```

"""    
function Seed(M::Int64)
        
        np=nprocs(); pid=procs()
        
        seeds = collect(M:1:M+np-1)
        
        for i =1:np
            remotecall_fetch(()->Random.seed!(seeds[i]),pid[i])
        end
        
end
    