

export susieGLM, fineQTL_glm
using GLM

function susieGLM(n::Int64,L::Int64,Π::Vector{Float64},y::Vector{Float64},X::Matrix{Float64},
    X₀::Union{Matrix{Float64},Vector{Float64}};tol=1e-4,maxitr=1000)
  
    # n =size(X,1)
    y1= zeros(n)
  
    if(X₀!= ones(n,1)) #&&(size(X₀,2)>1)
        X₀ = hcat(ones(n),X₀)
    end
         
    y1 =y.-0.5 # centered y
#initialization :
 σ0 = ones(L);
#  β = rand(c)/sqrt(c)
#  ξ = rand(n)/sqrt(n)
 
 β0 = glm(X₀,y,Binomial()) |> coef
 ν0 =sum(repeat(Π,outer=(1,L)).*σ0',dims=2)[:,1] ; #ν²0
 ξ0 =sqrt.(getXy('N',X.^2.0,ν0)+getXy('N',X₀,β0).^2)
 

    result = emGLM(L,y1,X,X₀,ξ0,σ0,Π;tol=tol,maxitr=maxitr)
        
return result

end 

"""

fineQTL_glm(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1)
;L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],tol=1e-4)



Performs SuSiE (Sum of Single Effects model) GLM fine-mapping analysis for a binary trait (logistic regression).



# Arguments

- `G` : a Type of struct, `GenoInfo`. See [`GenoInfo`](@ref).
- `y` : a n x 1 vector of  binary trait
- `X` : a n x p matrix of genetic markers selected from QTL analysis
- `X₀`: a n x 1 vector or n x c matrix of covariates.  The intercept is default if no covariate is added.

## Keyword Arguments

- `L` : an integer. The number of single effects for SuSiE implementation. Default is `10`.
- `Π` : a p x 1 vector of prior inclusion probabilities for SuSiE.  Default is `1/p`, where `p = size(X,2)`. If different probabilities are added to SNPs, the length of Π should be `p`.
- `tol`: tolerance. Default is `1e-4`. 
- `maxitr` : the maximum number of iteration for a case where the algrithm does not coverge.  Default is `1000`.

# Output

 Returns a Type of struct, `ResGLM` per Chromosome (`p1 = # of SNP in a Chromosome`), which includes 

-  `ξ` : a n x 1 vector of variational parameters to fit a logitstic function
-  `β` : a c x 1 vector of fixed effects for covariates
-  `σ0` : a L x 1 vector of hyper-parameters for prior variances for SuSiE
-  `elbo`: a p1 x 1 vector of evidence lower bound (ELBO)
-  `α` : a p1 x L matrix of posterior inclusion probabilities of SuSiE
-  `ν` : a p1 x L matrix of posterior mean of SuSiE
-  `Σ` : a p1 x L matrix of posterior variances of SuSiE


"""
function fineQTL_glm(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},
    X₀::Union{Matrix{Float64},Vector{Float64}}=ones(length(y),1);L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],tol=1e-4,maxitr=1000)

    Chr=sort(unique(G.chr)); n, p=size(X)
    est= @distributed (vcat) for j= eachindex(Chr)
                 midx= findall(G.chr.== Chr[j])
                  #check size of Π
                  if (Π==[1/p]) #default value
                    m=length(midx)
                    Π1 =ones(m)/m #adjusting πⱼ
                   elseif (length(Π)!= p)
                      println("Error. The length of Π should match $(p) SNPs!")
                   else
                    Π1 = Π[midx]
                 end

                 est0= susieGLM(n,L,Π1,y,X[:,midx],X₀;tol=tol,maxitr=maxitr)
                        est0
    end

   return est
end