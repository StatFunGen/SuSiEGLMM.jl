
#kinship, rotate, random seeds,

#Spectral decomposition
function eigenK(K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true,ρ::Float64=0.001)
    
   if (LOCO)
       chr = size(K,3); 
       S = zeros(n,nchr);
       T = zeros(n,n,chr);
    
        for j =1:nchr ## add parallelization later
           if(!isposdef(K[:,:,j])) # check pdf
                K[:,:,j]=K[:,:,j]+ (abs(eigmin(K[:,:,j]))+ρ)
           end
                
       #spectral decomposition
         F=eigen(K[:,:,j])
         T[:,:,j], S[:,j] = F.vectors, F.values
        end
       
        return  T, S
    else
          if(!isposdef(K)) # check pdf
                K=K+ (abs(eigmin(K))+ρ)
           end
        F=eigen(K)
        return F.vectors, F.values
        
    end
        
end




"""
  
    rotate(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},K::Matrix{Float64})

Returns eigenvalues of a kinship matrix, and transformed data rotated by eigenvectors.

# Arguments

- `y` : a vector of n x 1 binary trait
- `X` : a n x p matrix of genetic markers selected from QTL analysis (per Chromosome for LOCO)
- `X₀` : a matrix of n x c covariates including intercepts
- `K` : a matrix of n x n genetic kinship (symmetric pdf)

# Output

- `yt` : 
- `Xt` :
- `Xt₀` :
- `Eigen values` 

"""
function rotate(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},T::Matrix{Float64})
    
    n=length(y)
    
    #U'X, U'X₀, U'(y-1/2)
    yt= BLAS.gemv('T',T,(y-0.5*ones(n)))
    Xt=BLAS.gemm('T','N',T,X)
    Xt₀=BLAS.gemm('T','N',T,X₀)
       
    return yt, Xt, Xt₀
    
end


function rotate(y::Vector{Float64},X₀::Matrix{Float64},T::Matrix{Float64})
    
    n=length(y)
    
    #U'X, U'X₀, U'(y-1/2)
    yt= BLAS.gemv('T',T,(y-0.5*ones(n)))
    # Xt=BLAS.gemm('T','N',T,X)
    Xt₀=BLAS.gemm('T','N',T,X₀)
       
    return yt, Xt₀
    
end




#compute X'y : 't': 'T' for transpose, 'N' for no transpose
function getXy(Xt::Matrix{Float64},yt::Vector{Float64},t::Char)
    
    y1= BLAS.gemv(t,Xt,yt)
    
    return y1







function logistic(ξ::Float64)
    
    return 1.0/(1.0+ exp(-ξ))
    
end

# λ(ξ)
function Lambda(ξ::Float64)
   
    return 0.25*tanh(ξ)/ξ    
end

