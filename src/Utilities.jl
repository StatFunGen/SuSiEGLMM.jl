
#kinship, rotate, initialization +L, random seeds,


"""
  
    rotate(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},K::Matrix{Float64})

Returns eigenvalues of a kinship matrix, and transformed data rotated by eigenvectors.

# Arguments

- `y` : a binary trait
- `X` : genetic markers selected from QTL analysis
- `X₀` : covariates
- `K` : a genetic kinship (symmetric pdf)

# Output

- `yt` :
- `Xt` :
- `Xt₀` :
- `Eigen values` 

"""
function rotate(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},K::Matrix{Float64})
    
    n=length(y)
    #spectral decomposition
    F=eigen(K)
    
    #U'X, U'X₀, U'(y-1/2)
    yt= BLAS.gemv('T',F.vectors,(y-0.5*ones(n)))
    Xt=BLAS.gemm('T','U',F.vectors,X)
    Xt₀=BLAS.gemm('T','U',F.vectors,X₀)
       
    return yt, Xt, Xt₀, F.values
    
end

#compute X'y 
function getXy(Xt::Matrix{Float64},yt::Vector{Float64})
    
    y1= BLAS.gemv('T',Xt,yt)
    
    return y1


function ELBO()
    
   # log.(logistic.(ξ))- 0.5*ξ+ yX*β
        
 
    
end




function logistic(ξ::Float64)
    
    return 1.0/(1.0+ exp(-ξ))
    
end

# λ(ξ)
function Lambda(ξ::Float64)
   
    return 0.25*tanh(ξ)/ξ    
end

