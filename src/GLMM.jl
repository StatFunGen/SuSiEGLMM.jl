

include("Utilities.jl")
include("VEM.jl")
include("GLM.jl")


function init(yt::Vector{Float64},Xt₀::Matrix{Float64},S::Vector{Float64};tol=1e-4)
   
     #initialization
    #  τ0 = rand(1)[1]/sqrt(n); #arbitray
    τ0=0.4
    
     β0 = glm(Xt₀,yt,Binomial()) |> coef
     ξ0 =sqrt.(getXy('N',Xt₀,β0).^2+ τ2*S)
      
     res= emGLMM(yt,Xt₀,S,τ0,β0,ξ0;tol=tol)
    
    return res
        
end



"""

    initialization(y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},
        S::Vector{Float64};tol=1e-4)

Returns initial values for parameters `τ2, β, ξ` to run fine-mapping for SuSiEGLMM.


# Arguments

- `y` : a n x 1 vector of binary trait
- `X₀` : a n x c matrix of covariates.  The intercept is default if no covariates is added.
- `T` : a matrix of eigen-vectors by eigen decomposition to K (kinship)
- `S` : a vecor of eigen-values by eigen decomposition to K (kinship)

## Keyword Arguments

- `tol` : tolerance. Default value is `1e-4`.


# Output

- `Xt₀`: a n x c transformed covariate matrix
- `yt`: a transformed trait vector
-  Type of struct, `Null_est` :  initial values of parameters obtained by EM algorithm, or a null model of the SuSiE-GLMM excluding SuSiE implementation.  It includes following estimates: 

    - `τ2` : a constant variance in the variance component of genetic relatedness, `τ²K`, where K is a kinship matrix
    -  `β` : a c x 1 vector of fixed effects for covariates
    -  `ξ` : a n x 1 vector of variational parameters to fit a mixed logitstic function

"""
function initialization(y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},
        S::Vector{Float64};tol=1e-4)
    
    # check if covariates are added as input and include the intercept. 
    if(X₀!= ones(length(y),1)) #&&(size(X₀,2)>1)
        X₀ = hcat(ones(length(y)),X₀)
    end
    
        
#                    Xt₀ = rotateX(X₀,T)
#                    yt = rotateY(y,T)
                   Xt, Xt₀, yt = rotate(y,X,X₀,T)   
                   init_est= init(yt,Xt₀,S;tol=tol)
       
    return Xt, Xt₀, yt, init_est
end


# X₀: check if it includes intercept
function susieGLMM(L::Int64,Π::Vector{Float64},yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},
        S::Vector{Float64},est0::Null_est;tol=1e-4)
    
    # n, p = size(Xt)
    #initialization :
     σ0 = 0.1*ones(L);
   
     result = emGLMM(L,yt,Xt,Xt₀,S,est0.τ2,est0.β,est0.ξ,σ0,Π;tol=tol)
            
    return result
    
end 

#try with random small initial values to test H1 model
function susieGLMM(L::Int64,Π::Vector{Float64},yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},
    S::Vector{Float64};tol=1e-4)

n, c = size(Xt₀)
#initialization :
 σ0 = 0.1*ones(L);
 τ2 = rand(1)[1]*0.001; #arbitray
 β = rand(c)*0.0001
 ξ = rand(n)*0.001

    result = emGLMM(L,yt,Xt,Xt₀,S,τ2,β,ξ,σ0,Π;tol=tol)
        
return result

end 

# compute score test statistic : need to check again
function computeT(init0::Null_est,yt::Vector{Float64},Xt₀::Matrix{Float64},Xt::Matrix{Float64})
    
        p=axes(Xt,2)
        r₀ =  2*yt.*(getXy('N',Xt₀,init0.β)+init0.μ)  
        p̂ = logistic.(r₀)
        Γ  = p̂.*(1.0.-p̂)
        
        proj= I - Xt₀*symXX('T',sqrt.(Γ).*Xt₀)\(Xt₀'Diagonal(Γ)) 
        G̃ = getXX('N',proj,'N',Xt)
    
        Tstat= zeros(p)
        
         ĝ = getXy('T',G̃,yt-p̂).^2
    
         @views for j = p
           
              Tstat[j] = ĝ[j]/(G̃[:,j]'*(Γ.*G̃[:,j]))
           end
    
    return Tstat
end




"""

    scoreTest(G::GenoInfo,y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},X::Matrix{Float64},
        K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true,tol=1e-4)

Returns 1-df score test statistic for case-control association tests that follows Chi-square distribution, `T²∼Χ²₍₁₎` under `H₀: β=0`.

# Arguments

- `G` : a Type of struct, `GenoInfo`. See [`GenoInfo`](@ref).
- `y` : a n x 1 vector of  binary trait
- `X₀`: a n x c matrix of covariates.  The intercept is default if no covariates is added.
- `X` : a n x p matrix of genetic markers (SNPs)
- `K` : a 3d-array of a symmetric positive definite kinship, `size(K) = (n,n, # of Chromosomes)` if `LOCO = true` (default).  Or a n x n matrix if false.

## Keyword Arguments

- `LOCO`: Boolean. Default is `true` performs score test according to the Leave One Chromosome Out (LOCO) scheme.
- `tol` : tolerance. Default is `1e-4`. 

# Output

- `T_stat` : a p x 1 vector of test statistics for case-control association tests.  The  test statistic is computed based on [reginie](https://www.nature.com/articles/s41588-021-00870-7).

"""
function scoreTest(G::GenoInfo,y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},X::Matrix{Float64},
        K::Union{Array{Float64,3},Matrix{Float64}};LOCO::Bool=true,tol=1e-4)
    
    
    if (LOCO)

        Chr=sort(unique(G.chr));

        # T, S = eigenK(K;LOCO=LOCO,δ=0.001)
        T, S = svdK(K;LOCO=LOCO)
        println("Eigen-decomposition is completed.")

        # T_stat=zeros(size(X,2))

         T_stat = @distributed (vcat) for j= eachindex(Chr)
                       midx= findall(G.chr.== Chr[j])
                       Xt, Xt₀, yt, init0 = initialization(y,X[:,midx],X₀,T[:,:,j],S[:,j];tol=tol) 
                       tstat = computeT(init0,yt,Xt₀,Xt)
                        tstat
                     end


    else #no loco

         
        #  T, S = eigenK(K;LOCO=LOCO,δ=0.001)
         T, S = svdK(K;LOCO=LOCO)
         println("Eigen-decomposition is completed.")

         Xt, Xt₀, yt, init0 = initialization(y,X,X₀,T,S;tol=tol) 
         T_stat = computeT(init0,yt,Xt₀,Xt)
        
    end
    
    p_value = ccdf.(Chisq(1),T_stat) #check

    return T_stat, p_value
end





"""

     fineQTL_glmm(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},
        X₀::Union{Matrix{Float64},Vector{Float64}},
        T::Union{Array{Float64,3},Matrix{Float64}},S::Union{Matrix{Float64},Vector{Float64}};
        LOCO::Bool=true,L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],tol=1e-4)


Performs SuSiE (Sum of Single Effects model) GLMM fine-mapping analysis for a binary trait (logistic mixed model).



# Arguments

- `G` : a Type of struct, `GenoInfo`. See [`GenoInfo`](@ref).
- `y` : a n x 1 vector of  binary trait
- `X` : a n x p matrix of genetic markers selected from QTL analysis (per Chromosome for LOCO)
- `X₀`: a n x 1 vector or n x c matrix of covariates.  The intercept is default if no covariates is added.
- `T` : a matrix of eigen-vectors by eigen decomposition to K (kinship)
- `S` : a vecor of eigen-values by eigen decomposition to K (kinship)

## Keyword Arguments

- `LOCO`: Boolean. Default is `true` indicates fine-mapping performs according to the Leave One Chromosome Out (LOCO) scheme.
- `L` : an integer. The number of single effects for SuSiE implementation. Default is `10`.
- `Π` : a p x 1 vector of prior inclusion probabilities for SuSiE.  Default is `1/p`, where `p = size(X,2)`. If different probabilities are added to SNPs, the length of Π should be `p`.
- `tol`: tolerance. Default is `1e-4`. 

# Output

 Returns a Type of struct, `Result`, which includes 

-  `ξ` : a n x 1 vector of variational parameters to fit a mixed logitstic function
-  `β` : a c x 1 vector of fixed effects for covariates
-  `σ0` : a L x 1 vector of hyper-parameters for prior variances for SuSiE
-  `τ2` : a constant variance in the variance component of genetic relatedness, `τ²K`, where K is a kinship matrix
-  `α` : p x L matrix of posterior inclusion probabilities for SuSiE
-  `ν` : p x L matrix of posterior mean of SuSiE
-  `Σ` : p x L matrix of posterior variances of SuSiE


"""
function fineQTL_glmm(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},
        X₀::Union{Matrix{Float64},Vector{Float64}},
        T::Union{Array{Float64,3},Matrix{Float64}},S::Union{Matrix{Float64},Vector{Float64}};
        LOCO::Bool=true,L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],tol=1e-4)
    
    
    
     if(LOCO)     
             Chr=sort(unique(G.chr));
   est= @distributed (vcat) for j= eachindex(Chr)
                midx= findall(G.chr.== Chr[j])
                Xt, Xt₀, yt, init0 = initialization(y,X[:,midx],X₀,T[:,:,j],S[:,j];tol=tol)
                           #check size of Π
                           
                          if (Π==[1/size(X,2)]) #default value
                              m=length(midx)
                              Π1 =repeat(1/m,m) #adjusting πⱼ
                             elseif (length(Π)!= size(X,2))
                                println("Error. The length of Π should match $(size(X,2)) SNPs!")
                             else
                              Π1 = Π[midx]
                           end
                     
                est0= susieGLMM(L,Π1,yt,Xt,Xt₀,S[:,j],init0;tol=tol)
                        est0
                           end
           
    else #no loco for one genomic region
            
            
                  if(length(Π)==1)
                     Π =repeat(Π,size(X,2))
                  elseif (length(Π)!= size(X,2))
                    println("Error. The length of Π should match $(size(X,2)) SNPs!")
                  end
                 
                 Xt, Xt₀, yt, init0 = initialization(y,X,X₀,T,S;tol=tol) 
                 est = susieGLMM(L,Π,yt,Xt,Xt₀,S,init0;tol=tol)
                     
                           
            
    end # loco
 
 
    return est
    
end



function randomizeData!(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},
    T::Array{Float64,3},S::Matrix{Float64})

    #   Random.seed!(seed)
    ivec = Vector(1:length(yt))
    #randomize minibatches
    shuffle!(MersenneTwister(123), ivec) # shuffle index
    y[:] = y[ivec]
    X[:,:] = X[ivec,:]
    X₀[:,:] = X₀[ivec,:]
    T[:,:,:] = T[ivec,:,:]
    S[:,:] = S[ivec,:]    
    
end

function randomizeData!(y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},
    T::Array{Float64,2},S::Vector{Float64})

    #   Random.seed!(seed)
    ivec = Vector(1:length(yt))
    #randomize minibatches
    shuffle!(MersenneTwister(123), ivec) # shuffle index
    y[:] = y[ivec]
    X[:,:] = X[ivec,:]
    X₀[:,:] = X₀[ivec,:]
    T[:,:] = T[ivec,:]
    S[:] = S[ivec]    
    
end


# function fineMapping_miniBatch(y,X,X₀,K;bsize::Int64=256,LOCO::Bool=true,L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],tol=1e-5)
   
#     M= length(y)
#     num_epoch = floor(M/bsize)
    

#     T,S = svdK(K;LOCO=LOCO)
#     println("Eigen-decomposition is completed.")
#     randomizeData!(y,X,X₀,T,S)


# end

function fineMapping(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1);
        K::Union{Array{Float64,3},Matrix{Float64}}=Matrix(1.0I,1,1),L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],LOCO::Bool=true,
        model=["susieglmm","susie","mvsusie"],tol=1e-4)
    
    #need to work more   
    
    if(model=="susieglmm")
        
        T, S = svdK(K;LOCO=LOCO)
       
        println("Eigen-decomposition is completed.")
        
        est = fineQTL_glmm(G,y,X,X₀,T,S;L=L,Π=Π,LOCO=LOCO,tol=tol)
            println("SuSiEGLMM is completed.")  
        
         return est
        
        
    elseif(model="susie")
        
        
    else #model=mvsusie
        
        
        
    end #model
        
end
        
    
        
        
        
        
        
        
        
     
    
    
    
    





function fineMapping1(f::Function,args...;kwargs...)                   
        
  #need to work more : using splats to give more freedom gets slow in performance.     
        res = f(args...;kwargs...)
    
    
    return res
        
end


export init, initialization, fineQTL_glmm, susieGLMM, computeT, scoreTest, GenoInfo