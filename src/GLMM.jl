

function init(yt::Vector{Float64},Xt₀::Matrix{Float64},S::Vector{Float64};tol=1e-4)
    
     τ2 = rand(1)*0.5; #arbitray
    # may need to change
     β = zeros(axes(Xt₀,2)) 
     ξ = rand(n)
      
     res= emGLMM(yt,Xt₀,S,τ2,β,ξ;tol=tol)
    
    return res
        
end

struct GenoInfo
    snp::Array{String,1}
    chr::Array{Any,1}
    pos::Array{Float64,1} #positon 
end


"""

    initialization(y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1),T::Matrix{Float64},
        S::Vector{Float64};tol=1e-4)

Returns initial values for parameters `τ2, β, ξ` to run fine-mapping for SuSiEGLMM.

# Arguments

- `y` : a vector of n x 1 binary trait
- `X₀` : a matrix of n x c covariates.  The intercept is default if no covariates is added.
- `T` : a matrix of eigen-vectors by eigen decomposition to K (kinship)
- `S` : a vecor of eigen-values by eigen decomposition to K (kinship)

## Keyword Arguments

- `tol` : tolerance. Default value is `1e-4`.


# Output

- `Xt₀`: a transformed covariate matrix
- `yt`: a transformed trait vector
- `init_est` : Type `Null_est`, i.e. initial values of parameters `τ2,β,ξ` obtained by EM algorithm

"""
function initialization(y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1),T::Matrix{Float64},
        S::Vector{Float64};tol=1e-4)
    
    # check if covariates are added as input and include the intercept. 
    if(X₀!= ones(length(y),1))
        X₀ = hcat(ones(length(y)),X₀)
    end
    
        
                   Xt₀ = rotateX(X₀,T)
                   yt = rotateY(y,T)
                   init_est= init(yt,Xt₀,S;tol=tol)
       
    return Xt₀, yt, init_est
end


# X₀: check if it includes intercept
function SuSiEGLMM(L::Int64,Π::Vector{Float64},yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},
        S::Vector{Float64},est0::Null_est;tol=1e-4)
    
    n, p = size(Xt)
    #initialization :
     σ0 = 0.1*ones(L);
     
        result = emGLMM(L,yt,Xt,Xt₀,S,est0.τ2,est0.β,est0.ξ,σ0,Π;tol::Float64=1e-4)
            
    return result
    
end 

function fineMapping_GLMM(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1),
        T::Union{Array{Float64,3},Matrix{Float64}},S::Union{Matrix{Float64},Vector{Float64}};LOCO::Bool=true,tol=1e-4)
    
    
     Chr=sort(unique(G.chr));
     if(LOCO)     
            
   est= @distributed (vcat) for j= eachindex(Chr)
                midx= findall(G.chr.== Chr[j])
                Xt₀, yt, init0 = initialization(y,X₀,T[:,:,j],S[:,j];tol=tol)
                Xt= rotateX(X[:,midx],T[:,:,j])
                res0= SuSiEGLMM(L,Π,yt,Xt,Xt₀,S[:,j],init0;tol=tol)
                res0
                           end
           
        else #no loco
            
            if(X₀!= ones(length(y),1))
               X₀ = hcat(ones(length(y)),X₀)
            end
            
              Xt, Xt₀, yt = rotate(y,X,X₀,T)    
              init0= init(yt,Xt₀,S;tol=tol)
              
              est= @distributed (vcat) for j in eachindex(Chr)
                      est0 = SuSiEGLMM(L,Π,yt,Xt,Xt₀,S,init0;tol=tol)
                      est0
                            end
            
    end # loco
 
    # need to add credile sets
    return est
    
end


function fineMapping(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1);
        K::Union{Array{Float64,3},Matrix{Float64}}=Matrix(1.0I,1,1),
        model=["susieglmm","susie","mvsusie"],LOCO::Bool=true,tol=1e-4)
    
    #need to work more   
    
    if(model=="susieglmm")
        
        T, S = eigenK(K;LOCO=LOCO,δ=0.001)
        println("Eigen-decomposition is completed.")
        
        est = fineMapping_GLMM(G,y,X,X₀,T,S;LOCO=LOCO,tol=tol)
            println("SuSiEGLMM is completed.")  
        
         return est
        
        
    elseif(model="susie")
        
        
    else #model=mvsusie
        
        
        
    end
        
        
    
        
        
        
        
        
        
        
     
    
    
    
    
end




function fineMapping1(f::Function,args...;kwargs...)                   
        
  #need to work more : using splats to give more freedom gets slow in performance.     
        res = f(args...;kwargs...)
    
    
    return res
        
end