# """

#     VEM

# A module for posterior and hyper-parameter estimates using a variational expectiation-maximization method.


# """
# module VEM

# using Statistics, LinearAlgebra, Random, StatsBase, Distributions, Distributed




export intOut,intβOut,emNull


function intB!(beta::Vector{Float64},M::Matrix{Float64},
     Vβ̂inv::Matrix{Float64},Xy₀::Vector{Float64},tD::Matrix{Float64},Xt₀::Matrix{Float64}) 
    
    Eq= Vβ̂inv\[Xy₀ tD] # c x n+1
    beta[:]= Eq[:,1];
    M[:,:] =  getXX('N',Xt₀,'N',Eq[:,2:end]) #X₀ΣᵦX₀'λ
    # transpose!(C,Eq[:,2:end]) # C=λX₀Σᵦ
 
end

struct intOut
 Vβ̂inv::Matrix{Float64}
 β̂::Vector{Float64}
 Ŷ::Vector{Float64}
 λ::Vector{Float64}
 tD::Matrix{Float64}
 M::Matrix{Float64}
end

function intβOut(Xy₀::Vector{Float64},yt::Vector{Float64},Xt₀::Matrix{Float64},Σ₀::Matrix{Float64},ξ::Vector{Float64},n::Int64)
    
    c=size(Xt₀,2); beta=zeros(c); tD=copy(Xt₀'); M=zeros(n,n)
    λ= Lambda.(ξ) #2Lambda

    # Xy₀=getXy('T',Xt₀,yt) #X₀'y
    
    # Vβ̂inv= inv(Σ₀)+ BLAS.gemm('N','N',rmul!(tD,λ),Xt₀)  # Σ₀^-₁ +(tD=X₀'*λ)*X₀ :precision Σᵦ
    Λ=Diagonal(sqrt.(λ))
    Vβ̂inv= inv(Σ₀)+ symXX('N',rmul!(tD,Λ)) #forcing to be symmetric
    rmul!(tD,Λ)
    intB!(beta,M,Vβ̂inv,Xy₀,tD,Xt₀)
   
    Ŷ= yt- getXy('T',tD,beta)      # yt - λX₀β̂
    
     return  intOut(Vβ̂inv,beta,Ŷ,λ,tD,M)

end

# function ELBO(Xy₀::Vector{Float64},β̂::Vector{Float64},Vβ̂inv::Matrix{Float64},Σ₀::Matrix{Float64})
#     llbeta= 0.5(Xy₀'*β̂ - logdet(Vβ̂inv)-logdet(Σ₀))# β̂'inv(Σᵦ)β̂ for elbo 
    
#       return llbeta
# end

#E-step

# g for initial values (H0): integration out version
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},Badj::intOut,S::Vector{Float64},τ2::Float64)
    
    Vg[:]= 1.0./(Badj.λ+1.0./(τ2*S))
    # println("pdf of Vg is", isposdef(Vg))
    ghat[:]= Vg.*Badj.Ŷ
    
end


#M-step: H0 for initial values (integration out version)
function mStep!(ξ_new::Vector{Float64},Vg::Vector{Float64},
        ghat::Vector{Float64},Badj::intOut,Xt₀::Matrix{Float64},n::Int64)
  
    V =zeros(n,n);
    V[:,:]=Diagonal(Vg)
    
   
    ξ_new[:]= sqrt.((getXy('N',Xt₀,Badj.β̂)- getXy('N', Badj.M,ghat)+ ghat).^2 
    + Diagonal(symXX('N',Badj.M.*sqrt.(Vg)')+ V+ BLAS.symm('R','U',(Diagonal(1.0./Badj.λ)-2V),Badj.M))*ones(n))
     
    
    # M= symXX('N',rmul!(M,Diagonal(sqrt.(Vg))) + Diagonal(Vg)
    # Badj.M*Diagonal(1.0./Badj.λ-2Vg)
end


    
# For initial values : H0 w/o susie (integration out version)
function ELBO(ξ_new::Vector{Float64},τ2_new::Vector{Float64},Badj::intOut,ghat::Vector{Float64},
        ghat2::Vector{Float64},Vg::Vector{Float64},S::Vector{Float64},
        Xy₀::Vector{Float64},Σ₀::Matrix{Float64},n::Int64)
   
       
    ll= sum(log.(logistic.(ξ_new))- 0.5*ξ_new)  + Badj.Ŷ'*ghat
        # +0.5*Lambda.(ξ_new).*ξ_new.^2 -0.5*tr(getXX('N',Badj.Λ̂,'N',ghat2))
    lbeta= ELBO(Xy₀,Badj.β̂,Badj.Vβ̂inv,Σ₀)   
    # println("τ2_new: $(τ2_new)")
    
    gl = -0.5*(n*log(τ2_new[1])+ sum(log.(S)-log.(Vg))- 1.0 + sum(ghat2./S)/τ2_new[1]) # g
    f=open(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl/test/est_elbo1.txt","a")
    writedlm(f,[τ2_new[1] ll lbeta gl ll+gl+lbeta])
    close(f)

    return ll+gl+lbeta
    
end


# struct Approx0
#     β::Vector{Float64}
#     ξ::Vector{Float64}
#     μ::Vector{Float64}
#     τ2::Float64
#     elbo::Float64
# end


#EM for initial values (H0): integration out version
function emNull(yt,Xt₀,S,τ2,ξ,Σ₀;tol::Float64=1e-4)
    
    
    n,c=size(Xt₀)
    ghat =zeros(n); Vg = zeros(n);# λ = zeros(n)##
    
    ghat2=zeros(n); τ2_new=zeros(1); τ2 =[τ2]; 
    ξ_new = zeros(n); βhat=zeros(c)
    
    Xy₀=getXy('T',Xt₀,yt) #X₀'y
    # Vβ̂inv,Badj = covarAdj(Xy₀,yt,Xt₀,Σ₀,ξ,n)
   
    crit =1.0; el0=0.0;numitr=0
    open(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl/test/est_elbo1.txt","w") 
    #  open("./test/decELBO.txt","w")
    while (crit>=tol)
        ###check again!
         Badj= intβOut(Xy₀,yt,Xt₀,Σ₀,ξ,n) 
         postG!(ghat,Vg,Badj,S,τ2[1])
         emG!(ghat2,τ2_new,Vg,ghat,S)
         
         mStep!(ξ_new,Vg,ghat,Badj,Xt₀,n)

         el1=ELBO(ξ_new,τ2_new,Badj,ghat,ghat2,Vg,S,Xy₀,Σ₀,n)
     
        #  if(el0>el1)
        #   f=open("./test/decELBO.txt","a")
        #     writedlm(f,[numitr τ2_new el1 el1-el0])
        #   close(f)
        #  end
         crit=abs(el1-el0)
        #  crit=norm(ξ_new-ξ)+norm(τ2_new-τ2)+abs(el1-el0)  
        
         ξ=ξ_new; τ2=τ2_new;el0=el1;βhat=Badj.β̂
        
          numitr +=1        
    end
    println(numitr)
    return Approx0(βhat,ξ,ghat,τ2[1],el0)
    
end







function glmmNull(y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},
    S::Vector{Float64};tol=1e-4)

    n=length(y)
# check if covariates are added as input and include the intercept. 
    if(X₀!= ones(n,1)) #&&(size(X₀,2)>1)
        X₀ = hcat(ones(n),X₀)
    end

    
     Xt, Xt₀, yt = rotate(y,X,X₀,T)   

#initialization
    Σ0= 2(cov(Xt₀)+I) # avoid sigularity when only with intercept
    τ0 = 1.99 
# τ0=1.2   
# β0 = glm(X₀,y,Binomial()) |> coef
    sig0=getXX('N',Σ0,'T',Xt₀)
    β̂0=getXy('N',sig0,yt)
    ξ0 =sqrt.(getXy('N',Xt₀,β̂0 ).^2+ Diagonal(getXX('N',Xt₀,'N',sig0).+τ0*S)*ones(n))


    est0= emNull(yt,Xt₀,S,τ0,ξ0,Σ0;tol=tol)
   
 return est0
end
