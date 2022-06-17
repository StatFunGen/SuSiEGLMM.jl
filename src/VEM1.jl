# """

#     VEM

# A module for posterior and hyper-parameter estimates using a variational expectiation-maximization method.


# """
# module VEM

# using Statistics, LinearAlgebra, Random, StatsBase, Distributions, Distributed


## include only integration out of β for glmms (H0 $ H1:QTL)

export intOut,intβOut,emNull,emAlt,glmmNull,glmm1Snp, guessξ0, Approx, scan1SNP


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

#for recall only
# function ELBO(Xy₀::Vector{Float64},β̂::Vector{Float64},Vβ̂inv::Matrix{Float64},Σ₀::Matrix{Float64})
#     llbeta= 0.5(Xy₀'*β̂ - logdet(Vβ̂inv)-logdet(Σ₀))# β̂'inv(Σᵦ)β̂ for elbo 
    
#       return llbeta
# end

#E-step

# g for H0: integration out version
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},Badj::intOut,S::Vector{Float64},τ2::Vector{Float64})
    
    Vg[:]= 1.0./(Badj.λ+1.0./(τ2[1]*S))
    # println("pdf of Vg is", isposdef(Vg))
    ghat[:]=Badj.Ŷ
    lmul!(Diagonal(Vg),ghat)
    
end

# g for QTL 
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},Badj::intOut,S::Vector{Float64},τ2::Vector{Float64},Xt::Vector{Float64},b0::Vector{Float64})

    Vg[:]= 1.0./(Badj.λ+ 1.0./(τ2[1]*S))
    
    ghat[:] = rmul!(Xt[:,:],b0[1])

    ghat[:]= Badj.Ŷ-lmul!(Diagonal(Badj.λ),ghat)
    lmul!(Diagonal(Vg),ghat)

end

#b for QTL
function postB!(b1::Vector{Float64},σ1::Vector{Float64},Badj::intOut,ghat::Vector{Float64},Xt::Vector{Float64},σ0::Vector{Float64})
        
     Λ=convert(Matrix{Float64},Diagonal(Badj.λ));
    σ1[:].=1.0/(Xt'BLAS.symv('U',Λ,Xt)+1.0/σ0[1])
    b1[:].=Xt'*(Badj.Ŷ-BLAS.symv('U',Λ,ghat))
    lmul!(σ1[1],b1)
end

function emB!(b2::Vector{Float64},σ0_new::Vector{Float64},b1::Vector{Float64},σ1::Vector{Float64})
# one of them will be deleted later since they are the same each other.
    b2[:] = b1.^2+ σ1
    σ0_new[:].= mean(b2)
    
end

#M-step: H0  (integration out version)
function mStep!(ξ_new::Vector{Float64},Vg::Vector{Float64},
        ghat::Vector{Float64},Badj::intOut,Xt₀::Matrix{Float64},n::Int64)
  
    V=convert(Matrix{Float64},Diagonal(Vg))
    
   
    ξ_new[:]= sqrt.((getXy('N',Xt₀,Badj.β̂)- getXy('N', Badj.M,ghat)+ ghat).^2 
    + Diagonal(symXX('N',Badj.M.*sqrt.(Vg))+ V+ BLAS.symm('R','U',(Diagonal(1.0./Badj.λ)-2V),Badj.M))*ones(n))
     
    
    # M= symXX('N',rmul!(M,Diagonal(sqrt.(Vg))) + Diagonal(Vg)
    # Badj.M*Diagonal(1.0./Badj.λ-2Vg)
end

# ξ for QTL : expensive!
function mStep!(ξ_new::Vector{Float64},b1::Vector{Float64},σ1::Vector{Float64},Vg::Vector{Float64},ghat::Vector{Float64},
    Badj::intOut,Xt₀::Matrix{Float64},Xt::Vector{Float64},n::Int64)
     
     mu = zeros(n); M1 = zeros(n,n);
     mu[:] = Xt
     rmul!(mu,b1[1])
     mu[:] = mu + ghat
     
     Sig = convert(Matrix{Float64}, Diagonal(Vg))
     Sig = convert(Matrix{Float64},Symmetric(BLAS.syr!('U',σ1[1],Xt,Sig)))
                
    mul!(M1,BLAS.symm('R','U',Sig,Badj.M),Badj.M') #M*Sig*M'      

    ξ_new[:]= sqrt.((getXy('N',Xt₀,Badj.β̂)- getXy('N', Badj.M,mu)+ mu).^2 
                + Diagonal(M1+BLAS.symm('R','U',Diagonal(1.0./Badj.λ)-2Sig,Badj.M)+Sig)*ones(n) )
                

end


    
# For H0 (integration out version)
function ELBO(ξ_new::Vector{Float64},τ2_new::Vector{Float64},Badj::intOut,ghat::Vector{Float64},
        ghat2::Vector{Float64},Vg::Vector{Float64},S::Vector{Float64},
        Xy₀::Vector{Float64},Σ₀::Matrix{Float64},n::Int64)
   
       
    ll= sum(log.(logistic.(ξ_new))- 0.5*ξ_new)  + Badj.Ŷ'*ghat
        # +0.5*Lambda.(ξ_new).*ξ_new.^2 -0.5*tr(getXX('N',Badj.Λ̂,'N',ghat2))
    lbeta= ELBO(Xy₀,Badj.β̂,Badj.Vβ̂inv,Σ₀)   
    # println("τ2_new: $(τ2_new)")
    
    gl = -0.5*(n*log(τ2_new[1])+ sum(log.(S)-log.(Vg))- 1.0 + sum(ghat2./S)/τ2_new[1]) # g
    # f=open(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl/test/est_elbo0.txt","a")
    # writedlm(f,[τ2_new[1] ll lbeta gl ll+gl+lbeta])
    # close(f)

    return ll+gl+lbeta
    
end

# for QTL
function ELBO(ξ_new::Vector{Float64},τ2_new::Vector{Float64},σ0_new::Vector{Float64},Badj::intOut,
    b1::Vector{Float64},b2::Vector{Float64},σ1::Vector{Float64},ghat::Vector{Float64},
    ghat2::Vector{Float64},Vg::Vector{Float64},S::Vector{Float64},Xt::Vector{Float64},
    Xy₀::Vector{Float64},Σ₀::Matrix{Float64},n::Int64)
     
    
      ll0 = ELBO(ξ_new,τ2_new,Badj,ghat,ghat2,Vg,S,Xy₀,Σ₀,n) #loglik for H0
    #   @fastmath @inbounds @views 
      lb = Badj.Ŷ'*(Xt*b1[1]) - 0.5(log(σ0_new[1])-log(σ1[1])-1.0+b2[1]/σ0_new[1]) #loglik for b
    #   f=open(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl/test/est_elbo1.txt","a")
    # writedlm(f,[τ2_new[1] b1[1] ll0 lb ll0+lb])
    # close(f)

      return ll0+lb
   

end

# for recall only
# struct Approx0
#     β::Vector{Float64}
#     ξ::Vector{Float64}
#     μ::Vector{Float64}
#     τ2::Float64
#     elbo::Float64
# end


#EM for H0: integration out version
function emNull(yt,Xt₀,S,τ2,ξ,Σ₀;tol::Float64=1e-4)
    
    
    n,c=size(Xt₀)
    ghat =zeros(n); Vg = zeros(n);# λ = zeros(n)##
    
    ghat2=zeros(n); τ2_new=zeros(1); τ2 =[τ2]; 
    ξ_new = zeros(n); βhat=zeros(c)
    
    Xy₀=getXy('T',Xt₀,yt) #X₀'y
    # Vβ̂inv,Badj = covarAdj(Xy₀,yt,Xt₀,Σ₀,ξ,n)
   
    crit =1.0; el0=0.0;numitr=0
    # open(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl/test/est_elbo0.txt","w") 
    #  open("./test/decELBO.txt","w")
    while (crit>=tol)
        ###check again!
         Badj= intβOut(Xy₀,yt,Xt₀,Σ₀,ξ,n) 
         postG!(ghat,Vg,Badj,S,τ2)
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





"""

     glmmNull(y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},S::Vector{Float64};tol=1e-4) 
     
     Fitting glmm (H0) for score test

"""
function glmmNull(y::Vector{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},
    S::Vector{Float64};tol=1e-4)

    n=length(y)
# check if covariates are added as input and include the intercept. 
    if(X₀!= ones(n,1)) #&&(size(X₀,2)>1)
        X₀ = hcat(ones(n),X₀)
    end

     Xt₀= rotateX(X₀,T)
     yt = rotateY(y,T)

#initialization
     τ0 = 1.99 
     Σ0= 2(cov(Xt₀)+I) # avoid sigularity when only with intercept   
     ξ0=guessξ0(yt,Xt₀,S,Σ0,τ0,n)

    est0= emNull(yt,Xt₀,S,τ0,ξ0,Σ0;tol=tol)
   
 return est0, Xt₀
end

## initial values for H0
function guessξ0(yt::Vector{Float64},Xt₀::Matrix{Float64},S::Vector{Float64},Σ0::Matrix{Float64},τ0::Float64,n::Int64)
    #initialization
  sig0=getXX('N',Σ0,'T',Xt₀)
  β̂0=getXy('N',sig0,yt)
  ξ0 =sqrt.(getXy('N',Xt₀,β̂0 ).^2+ Diagonal(getXX('N',Xt₀,'N',sig0))*ones(n)+τ0*S)

  return ξ0

end

## initial values for H1
function guessξ0(yt::Vector{Float64},Xt₀::Matrix{Float64},Xt::Vector{Float64},S::Vector{Float64},Σ0::Matrix{Float64},τ0::Float64,σ_0::Float64,n::Int64)
    #initialization
  sig0=getXX('N',Σ0,'T',Xt₀)
  β̂0=getXy('N',sig0,yt)
  ξ0 =sqrt.(getXy('N',Xt₀,β̂0 ).^2+ Diagonal(getXX('N',Xt₀,'N',sig0))*ones(n)+σ_0*Xt.^2+τ0*S)

  return ξ0

end

struct Approx
    β::Vector{Float64}
    ν::Vector{Float64}
    σ1::Vector{Float64}
    ξ::Vector{Float64}
    μ::Vector{Float64}
    τ2::Float64
    σ0::Float64
    elbo::Float64
end

## EM for QTL
function emAlt(yt,Xt,Xt₀,S,τ2,σ0::Float64,ξ,Σ₀,n::Int64,c::Int64;tol::Float64=1e-4)
    
    
    # n,c=size(Xt₀)
    ghat =zeros(n); Vg = zeros(n);# λ = zeros(n)##
    
    ghat2=zeros(n); τ2_new=zeros(1); τ2 =[τ2]; 
    b0=zeros(1); b1=zeros(1);b2=zeros(1); σ1=zeros(1)
    σ0_new =zeros(1); σ0=[σ0]
    ξ_new = zeros(n); βhat=zeros(c)
    
    Xy₀=getXy('T',Xt₀,yt) #X₀'y
    # Vβ̂inv,Badj = covarAdj(Xy₀,yt,Xt₀,Σ₀,ξ,n)
   
    crit =1.0; el0=0.0;numitr=0
    # open(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl/test/est_elbo1.txt","w") 
    #  open("./test/decELBO.txt","w")
    while (crit>=tol)
        ###check again!
         Badj= intβOut(Xy₀,yt,Xt₀,Σ₀,ξ,n) 
         postG!(ghat,Vg,Badj,S,τ2,Xt,b0)
         emG!(ghat2,τ2_new,Vg,ghat,S)
         postB!(b1,σ1,Badj,ghat,Xt,σ0)
         emB!(b2,σ0_new,b1,σ1)
         mStep!(ξ_new,b1,σ1,Vg,ghat,Badj,Xt₀,Xt,n)

         el1=ELBO(ξ_new,τ2_new,σ0_new,Badj,b1,b2,σ1,ghat,ghat2,Vg,S,Xt,Xy₀,Σ₀,n)
     
        #  if(el0>el1)
        #   f=open("./test/decELBO.txt","a")
        #     writedlm(f,[numitr τ2_new el1 el1-el0])
        #   close(f)
        #  end
         crit=abs(el1-el0)
        #  crit=norm(ξ_new-ξ)+norm(τ2_new-τ2)+abs(el1-el0)  
        
         ξ=ξ_new; τ2=τ2_new;el0=el1;σ0=σ0_new;βhat=Badj.β̂
        
          numitr +=1        
    end
    println(numitr)
    return Approx(βhat,b1,σ1,ξ,ghat,τ2[1],σ0_new[1],el0)
    
end


# H1 : 1 SNP 
"""

     glmm1Snp(n,τ0,σ_0,Σ0,yt::Vector{Float64},Xt::Vector{Float64},Xt₀::Union{Matrix{Float64},Vector{Float64}},S::Vector{Float64};tol=1e-4)

    fitting H1 model
"""
function glmm1Snp(n,c,τ0,σ_0,Σ0,yt::Vector{Float64},Xt::Vector{Float64},Xt₀::Matrix{Float64},S::Vector{Float64};tol=1e-4)

    #initialization for ξ 
    ξ0=guessξ0(yt,Xt₀,Xt,S,Σ0,τ0,σ_0,n)
    
    est0= emAlt(yt,Xt,Xt₀,S,τ0,σ_0,ξ0,Σ0,n,c;tol=tol)
   
      return est0
end

# 
"""

     scan1SNP(y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},S::Vector{Float64};tol=1e-4)

     QTLscan by SNP level distributed computation

"""
function scan1SNP(y::Vector{Float64},X::Matrix{Float64},X₀::Union{Matrix{Float64},Vector{Float64}},T::Matrix{Float64},
    S::Vector{Float64};tol=1e-4)
 
    nmar =axes(X,2);n,c=size(X₀)
  # check if covariates are added as input and include the intercept. 
     if(X₀!= ones(n,1)) #&&(size(X₀,2)>1)
         X₀ = hcat(ones(n),X₀)
     end
     # initial values of hyper-prameters
     τ0 = 1.99; σ_0 = 1.0
    #  est=zeros(nmar,2)
            Xt, Xt₀, yt = rotate(y,X,X₀,T)   
            Σ0= 2(cov(Xt₀)+I) # initial variance of prior β
         est=@distributed (vcat) for j in nmar
               est0=glmm1Snp(n,c,τ0,σ_0,Σ0,yt,Xt[:,j],Xt₀,S;tol=tol)
               [est0.ν est0]
                                 end
 
           return est[:,1], est[:,2]

end

