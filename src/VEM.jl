# """

#     VEM

# A module for posterior and hyper-parameter estimates using a variational expectiation-maximization method.


# """
# module VEM

# using Statistics, LinearAlgebra, Random, StatsBase, Distributions, Distributed




export postG!, emG, postB!,emB,mStep!,ELBO,emGLMM,emGLM, Result, Null_est, ResGLM

#E-step

#posterior

#g for a full model
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},λ::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},S::Vector{Float64},
        β::Vector{Float64},ξ::Vector{Float64},τ2::Float64,A0::Matrix{Float64},B0::Matrix{Float64})
    
    
    λ[:]= Lambda.(ξ)
   
    #posterior
    Vg[:]= 1.0./(λ+1.0./(τ2*S))
    
    ghat[:]= Diagonal(Vg)*(yt-λ.*(getXy('N',Xt₀,β) + getXy('N',Xt,sum(A0.*B0,dims=2)[:,1])))
    
    
end

# for initial values
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},λ::Vector{Float64},
yt::Vector{Float64},Xt₀::Matrix{Float64},S::Vector{Float64},
        β::Vector{Float64},ξ::Vector{Float64},τ2::Float64)
    
    
    λ[:]= Lambda.(ξ)
    
    #posterior
    Vg[:]= 1.0./(λ+1.0./(τ2*S))
    
    ghat[:]= Diagonal(Vg)*(yt-λ.*(getXy('N',Xt₀,β) ))
    
end


#EM for g
function emG(Vg::Vector{Float64},ghat::Vector{Float64},S::Vector{Float64})
    
    ghat2=zeros(axes(S)); τ2_new=zero(eltype(S)); 
    #e-step : update the second moment 
    ghat2= Vg+ghat.^2 
    #m-step for τ²
    τ2_new= mean(ghat2./S)
 
    return ghat2, τ2_new
    
end


#Xy = getXy(Xt,yt)


# priors : Π,σ0,
#b's
#update adding k<l, B1,A1, k>l, B0,A0
function postB!(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64}, λ::Vector{Float64},ghat::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64},σ0::Vector{Float64},A0::Matrix{Float64},B0::Matrix{Float64},Π::Vector{Float64},L::Int64)
    
    pidx=axes(B0,1)
    ϕ = zeros(pidx); Z=copy(ϕ);
    
    Z0= similar(Z, axes(yt));
    
    ϕ[:]= getXy('T', Xt.^2,λ) # mle of precision
    AB0= A0.*B0;  # #old α_l*b_l
    AB1= zeros(axes(B0))
   

            Sig1[:,:] =  1.0./(1.0./σ0'.+ ϕ)
      #l=1
    Z0= yt - λ.*(getXy('N',Xt₀,β) + getXy('N',Xt,sum(AB0[:,2:end],dims=2)[:,1])+ghat)
            B1[:,1] = Diagonal(Sig1[:,1])*getXy('T',Xt,Z0) 
           # compute α_1
            Z =  0.5*(getXy('T',Xt,Z0)./sqrt.(ϕ)).^2
            # A0[:,1] = Π.*exp.(Z./(1.0.+ϕ/σ0[1]))./sqrt.(ϕ.^(-1).+1)
            A0[:,1] = log.(Π)+ Z./(1.0.+ϕ/σ0[1]) - 0.5*log.(σ0[1]*ϕ.^(-1).+1)
            A0[:,1] = exp.(A0[:,1].-maximum(A0[:,1])) # eliminate max for numerical stability
            A1[:,1]= A0[:,1]/sum(A0[:,1]) # scale to 0< α_1<1
            AB1[:,1]= A1[:,1].*B1[:,1] #update α_1*b_1
    
     for l= 2: L
           
            Z0= yt - λ.*(getXy('N',Xt₀,β) + getXy('N',Xt,sum(hcat(AB1[:,1:l-1],AB0[:,l+1:end]),dims=2)[:,1])+ghat)
            B1[:,l] = Diagonal(Sig1[:,l])*getXy('T',Xt,Z0)
            Z =  0.5*(getXy('T',Xt,Z0)./sqrt.(ϕ)).^2
            # A0[:,l] = Π.*exp.(Z./(1.0.+ϕ/σ0[l]))./sqrt.(ϕ.^(-1).+1)
            A0[:,l] = log.(Π)+ Z./(1.0.+ϕ/σ0[l]) - 0.5*log.(σ0[l]*ϕ.^(-1).+1)
            A0[:,l] = exp.(A0[:,l].-maximum(A0[:,l])) 
            A1[:,l] = A0[:,l]/sum(A0[:,l])
            AB1[:,l]= A1[:,l].*B1[:,l]
      end
    
end

function emB(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64},L::Int64)
       
    σ0_new = zeros(L); AB2=zeros(axes(B1));
       # compute the second moment of b_l & update the hyper-parameter σ0
        for l= 1:L
        AB2[:,l]= A1[:,l].*(B1[:,l].^2+ Sig1[:,l])    
        σ0_new[l]= sum(AB2[:,l]) 
        end
         
    
      return σ0_new, AB2 
        
        
end

#for GLM
#b's
#update adding k<l, B1,A1, k>l, B0,A0
function postB!(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64}, λ::Vector{Float64},
    y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},β::Vector{Float64},σ0::Vector{Float64},A0::Matrix{Float64},B0::Matrix{Float64},Π::Vector{Float64},L::Int64)
        
        pidx=axes(B0,1)
        ϕ = zeros(pidx); Z=zeros(pidx);
        
        Z0= similar(Z, axes(y));
        
        ϕ= getXy('T', X.^2,λ) # mle of precision
        AB0= A0.*B0;  # #old α_l*b_l
            
        Sig1[:,:] =  1.0./(1.0./σ0'.+ ϕ) #posterior Σₗ
        writedlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/sig1_julia.txt",Sig1)
          #l=1
        # Z0= y - λ.*(getXy('N',X₀,β) + getXy('N',X,sum(AB0[:,2:end],dims=2)[:,1]))
        #         Z =  getXy('T',X,Z0)
        #         B1[:,1] = Diagonal(Sig1[:,1])*Z #posterior bₗ
        #        # compute α_1
        #         # Z =  0.5*(getXy('T',X,Z0)./sqrt.(ϕ)).^2
        #         # A0[:,1] = Π.*exp.(Z./(1.0.+ϕ/σ0[1]))./sqrt.(σ0[1]*ϕ.^(-1).+1)
        #         A0[:,1] = log.(Π)+ 0.5*Z.^2 .*Sig1[:,1] + 0.5*log.(Sig1[:,1])
        #         A0[:,1] = exp.(A0[:,1].-maximum(A0[:,1])) # eliminate max for numerical stability
        #         A1[:,1]= A0[:,1]/sum(A0[:,1]) # scale to 0< α_1<1
        #         AB1[:,1]= A1[:,1].*B1[:,1] #update α_1*b_1
        
         for l= 1: L
               
                # Z0= y - λ.*(getXy('N',X₀,β) + getXy('N',X,sum(hcat(AB1[:,1:l-1],AB0[:,l+1:end]),dims=2)[:,1]))
                Z0= (y - λ.*(getXy('N',X₀,β) + getXy('N',X,sum(dropCol(AB0,l),dims=2)[:,1])))
                Z =  getXy('T',X,Z0)
                writedlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/nums-julia.txt",Z)
                B1[:,l] = Diagonal(Sig1[:,l])*Z #posterior bₗ
                  # compute α_1
                # A1[:,l] = exp.(log.(Π)-0.5*(Z.^2)./(1.0.+ϕ./σ0[l]) +0.5*log.(σ0[l]*ϕ.^(-1).+1))
              

                A1[:,l] = log.(Π)+ 0.5*Z.^2 .*Sig1[:,l] + 0.5*log.(Sig1[:,l]) 

                A1[:,l] = exp.(A1[:,l].-maximum(A1[:,l])) # eliminate max for numerical stability
                A1[:,l] = A1[:,l]/sum(A1[:,l]) # scale to 0< α_1<1
                AB0[:,l]= A1[:,l].*B1[:,l] #update α_1*b_1
          end
        
end
    



# M-step: H1
function mStep!(ξ_new::Vector{Float64},β_new::Vector{Float64},A1::Matrix{Float64},B1::Matrix{Float64},AB2::Matrix{Float64},
        ghat::Vector{Float64},ghat2::Vector{Float64},λ::Vector{Float64},
        yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64})
    
    # ξ_new= zeros(axes(yt))
    # β_new= zeros(axes(Xt₀,2))
      ŷ₀ = getXy('N',Xt₀,β)
      AB= getXy('N',Xt,sum(A1.*B1,dims=2)[:,1])
   
      for j in eachindex(yt)
        
         ξ_new[j]  = sum(Xt[j,:].^2.0.*(sum(AB2,dims=2))[:,1]) # E(Xb)^2
      end
    
    ξ_new[:] = sqrt.(ξ_new + ŷ₀.^2 + ghat2 + 2(ŷ₀ +ghat).*AB+ 2(ŷ₀.*ghat))
        
    # β_new[:]= symXX('T',sqrt.(λ).*Xt₀)\getXy('T',Xt₀,(yt- λ.*(AB + ghat)))
     β_new[:]= getXX('T',Xt₀,'N',(λ.*Xt₀))\getXy('T',Xt₀,(yt- λ.*(AB + ghat)))
        
end


#M-step: H0 for initial values
function mStep!(ξ_new::Vector{Float64},β_new::Vector{Float64},
        ghat::Vector{Float64},ghat2::Vector{Float64},λ::Vector{Float64},
        yt::Vector{Float64},Xt₀::Matrix{Float64},β::Vector{Float64})
  
    ŷ₀ = getXy('N',Xt₀,β)

   
    ξ_new[:] = sqrt.(ŷ₀.^2 + ghat2 + 2(ŷ₀.*ghat))  # check for debugging!
    # λ= 2Lambda.(ξ_new) #for check
    β_new[:]=  getXX('T',Xt₀,'N',(λ.*Xt₀))\getXy('T',Xt₀,(yt- λ.*ghat))
            
end

# M-step for GLM
function mStep!(ξ_new::Vector{Float64},β_new::Vector{Float64},A1::Matrix{Float64},B1::Matrix{Float64},AB2::Matrix{Float64},
    λ::Vector{Float64},y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},β::Vector{Float64};nitr=0)

# ξ_new= zeros(axes(yt))
# β_new= zeros(axes(Xt₀,2))
   L= size(A1,2)
   U = zeros(L,L)
   ŷ₀ = getXy('N',X₀,β)
#   AB= getXy('N',X,sum(A1.*B1,dims=2)[:,1])
    AB = getXX('N',X,'N',A1.*B1) # nxL
    B2= sum(AB,dims=2)[:,1]
    ξ_new[:] = getXy('N',X.^2.0,sum(AB2,dims=2)[:,1]) 
 
    for j in eachindex(y)
        U = AB[j,:]*AB[j,:]'
        ξ_new[j] = ξ_new[j]+ sum(U)-tr(U)
  end

    

    
   ξ_new[:] = sqrt.(ξ_new + ŷ₀.^2  + 2(ŷ₀.*B2))
#    a=findall(ξ_new.<0.0) 
#     println("ξ_new at $(nitr) iteration")
#     println(a)
#     display(ξ_new[a])
  
# β_new[:]= symXX('T',sqrt.(λ).*Xt₀)\getXy('T',Xt₀,(yt- λ.*(AB + ghat)))
 β_new[:]= getXX('T',X₀,'N',(λ.*X₀))\getXy('T',X₀,(y- λ.*B2))
    
end


# for a full model
function ELBO(L::Int64,ξ_new::Vector{Float64},β_new::Vector{Float64},σ0_new::Vector{Float64},τ2_new::Float64,
        A1::Matrix{Float64},B1::Matrix{Float64},AB2::Matrix{Float64},Sig1::Matrix{Float64},Π::Vector{Float64},
        ghat2::Vector{Float64},Vg::Vector{Float64},S::Vector{Float64},yt::Vector{Float64},Xt₀::Matrix{Float64})

    n=length(yt); p = size(B1,1); lnb =zeros(L);
    
     elbo0= ELBO(ξ_new,β_new,τ2_new,ghat2,Vg,S,yt,Xt₀) #null part
     # susie part
    for l= 1: L 
        if(sum(A1[:,l].==0.0)>0) #avoid NaN by log(0)
            lnb[l] = 0.0
        else
            lnb[l]  = A1[:,l]'*(log.(A1[:,l])-log.(Π) - 0.5*log.(Sig1[:,l])) 
        end
    end
    
               
      bl= sum(lnb)+0.5*(sum(log.(σ0_new))- L)+ 0.5*sum(sum(AB2,dims=1)'./σ0_new)
     
         
    return elbo0-bl
            
end
    
# For initial values
function ELBO(ξ_new::Vector{Float64},β_new::Vector{Float64},τ2_new::Float64,
        ghat2::Vector{Float64},Vg::Vector{Float64},S::Vector{Float64},yt::Vector{Float64},Xt₀::Matrix{Float64})
   
    n=length(yt);
    ll= sum(log.(logistic.(ξ_new))- 0.5*ξ_new)+ yt'*getXy('N',Xt₀,β_new) #lik
    gl = -0.5*(n*log(τ2_new)+ sum(log.(S)-log.(Vg))- 1.0) - sum(ghat2./S)/τ2_new # g
    
    return ll+gl
    
end

# for GLM
function ELBO(L::Int64,ξ_new::Vector{Float64},β_new::Vector{Float64},σ0_new::Vector{Float64},
    A1::Matrix{Float64},B1::Matrix{Float64},AB2::Matrix{Float64},Sig1::Matrix{Float64},Π::Vector{Float64},
    y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64})

# n=length(y); p = size(B1,1);
 lnb =zeros(L);

 ll= sum(log.(logistic.(ξ_new))- 0.5*ξ_new)+ y'*(getXy('N',X₀,β_new)+ getXy('N',X,sum(A1.*B1,dims=2)[:,1])) #lik
 # susie part
 for l= 1: L 
    if(sum(A1[:,l].==0.0)>0) #avoid NaN by log(0)
        lnb[l] = 0.0
    else
        lnb[l]  = A1[:,l]'*(log.(A1[:,l])-log.(Π) - 0.5*log.(Sig1[:,l]))
    end
end
           
bl= sum(lnb)  +0.5*(sum(log.(σ0_new))- L)+ 0.5*sum(sum(AB2,dims=1)'./σ0_new)
     
return ll-bl
        
end


struct Result
    ξ::Vector{Float64}
    β::Vector{Float64}
    σ0::Vector{Float64}
    τ2::Float64
    elbo::Float64
    α::Matrix{Float64}
    ν::Matrix{Float64}
    # ν2::Matrix{Float64}
    Σ::Matrix{Float64}    
end
    
 

# EM for a full model
function emGLMM(L::Int64,yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},S::Vector{Float64},τ2::Float64,
        β::Vector{Float64},ξ::Vector{Float64},σ0::Vector{Float64},Π::Vector{Float64};tol::Float64=1e-4)
    
    n, p = size(Xt)
    ghat =zeros(n); Vg = zeros(n); λ = zeros(n)
    A0 =repeat(Π,outer=(1,L)) ; 
    B0=zeros(p,L); AB2=zeros(p,L)
   
    A1 =zeros(p,L); B1=zeros(p,L); Sig1=zeros(p,L)
    ghat2=zeros(axes(S)); τ2_new=zero(eltype(S)); 
    σ0_new = zeros(L); ξ_new = zeros(n); β_new=zeros(axes(β))
    
    crit =1.0; el0=0.0;numitr=1
      
    
    while (crit>=tol)
        ###check again!
         postG!(ghat,Vg,λ,yt,Xt,Xt₀,S,β,ξ,τ2,A0,B0)
         ghat2, τ2_new = emG(Vg,ghat,S)
         postB!(A1, B1, Sig1, λ,ghat,yt,Xt,Xt₀,β,σ0,A0,B0,Π,L)
         σ0_new, AB2 = emB(A1, B1, Sig1,L)
         
         mStep!(ξ_new,β_new,A1,B1,AB2,ghat,ghat2,λ,yt,Xt,Xt₀,β)
        
         el1=ELBO(L,ξ_new,β_new,σ0_new,τ2_new,A1,B1,AB2,Sig1,Π,ghat2,Vg,S,yt,Xt₀)
     
         # crit=el1-el0 
         #check later for performance
         crit=norm(ξ_new-ξ)+norm(β_new-β)+abs(τ2_new-τ2)+abs(el1-el0)+norm(A1-A0)+norm(B1-B0)
         
         ξ=ξ_new;β=β_new;σ0=σ0_new; τ2=τ2_new;el0=el1;A0=A1;B0=B1
        
          numitr +=1
        
        
    end
    
    return Result(ξ,β,σ0,τ2,el0,A1, B1, Sig1)
        
end



struct Null_est
    ξ::Vector{Float64}
    β::Vector{Float64}
    μ::Vector{Float64}
    τ2::Float64
    elbo::Float64
end



#EM for initial values
function emGLMM(yt,Xt₀,S,τ2,β,ξ;tol::Float64=1e-4)
    
    
    n = length(yt)
    ghat =zeros(n); Vg = zeros(n); λ = zeros(n)
    
    ghat2=zeros(axes(S)); τ2_new=zero(eltype(S)); 
    ξ_new = zeros(n); β_new=zeros(axes(β))
    
    crit =1.0; el0=0.0;numitr=1
      
    
    while (crit>=tol)
        ###check again!
         postG!(ghat,Vg,λ,yt,Xt₀,S,β,ξ,τ2)
         ghat2, τ2_new = emG(Vg,ghat,S)
         
         mStep!(ξ_new,β_new,ghat,ghat2,λ,yt,Xt₀,β)
        
         el1=ELBO(ξ_new,β_new,τ2_new,ghat2,Vg,S,yt,Xt₀)
     
         # crit=el1-el0 
         crit=norm(ξ_new-ξ)+norm(β_new-β)+abs(τ2_new-τ2)+abs(el1-el0)  
        
         ξ=ξ_new;β=β_new; τ2=τ2_new;el0=el1
        
          numitr +=1        
    end
    
    return Null_est(ξ,β,ghat,τ2,el0)
    
end


struct ResGLM
    ξ::Vector{Float64}
    β::Vector{Float64}
    σ0::Vector{Float64}
    elbo::Float64
    α::Matrix{Float64}
    ν::Matrix{Float64}
    # ν2::Matrix{Float64}
    Σ::Matrix{Float64}    
end
    
 

# EM for GLM
function emGLM(L::Int64,y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64},
        β::Vector{Float64},ξ::Vector{Float64},σ0::Vector{Float64},Π::Vector{Float64};tol::Float64=1e-5)
    
    n, p = size(X)
    λ = zeros(n)
    A0 =repeat(Π,outer=(1,L)) ; 
    B0=zeros(p,L); AB2=zeros(p,L)
   
    A1 =copy(A0); B1=copy(B0); Sig1=zeros(p,L)
    σ0_new = zeros(L); ξ_new = zeros(n); β_new=zeros(axes(β))
    
    crit =1.0; el0=0.0;numitr=1
     
    
    while (crit>=tol)
        ###check again!
        λ= Lambda.(ξ) 
        # println("inter=$(numitr) and λ:")
        # println(λ)
        postB!(A1, B1, Sig1, λ,y,X,X₀,β,σ0,A0,B0,Π,L)
        # println("A1,B1,Sig1")
        # display(A1)
        # display(B1)
        # display(Sig1)
         σ0_new, AB2 = emB(A1, B1, Sig1,L)
        #  println("σ0,AB2")
        #  display(σ0_new)
        #  display(AB2)
         mStep!(ξ_new,β_new,A1,B1,AB2,λ,y,X,X₀,β;nitr=numitr)
        #  println("new ξ, β")
        #  display(ξ_new)
        #  display(β)

        
         el1=ELBO(L,ξ_new,β_new,σ0_new,A1,B1,AB2,Sig1,Π,y,X,X₀)
        #  println("elbo = $(el1)")
     
        #  crit=el1-el0 
         #check later for performance
         crit=norm(ξ_new-ξ)+norm(β_new-β)+abs(el1-el0) +norm(B1-B0)
         
         ξ=ξ_new;β=β_new;σ0=σ0_new;el0=el1;A0=A1;B0=B1
        
          numitr +=1
        
        
    end
    
    return ResGLM(ξ,β,σ0,el0,A1, B1, Sig1)
        
end





