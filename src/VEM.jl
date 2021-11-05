"""

    VEM

A module for posterior and hyper-parameter estimates using a variational expectiation-maximization method.


"""
module VEM

using Statistics, LinearAlgebra, Random, StatsBase, Distributions



#include("Utilities.jl")



#E-step

#posterior

#g
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},Vg_p::Vector{Float64},λ::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},S::Vector{Float64},
        β::Vector{Float64},ξ::Vector{Float64},τ2::Float64,A0::Matrix{Float64},B0::Matrix{Float64})
    
    Vg_p[:]= 1.0./(τ2*S) # prior
    λ[:]= 2Lambda.(ξ)
   
    #posterior
    Vg[:]= 1.0./(λ+Vg_p)
    
    ghat[:]= Diagonal(Vg)*(yt-λ.*(gemv('N',Xt₀,β) + Xt*sum(A0.*B0,dims=2))))
    
    
end

#EM for g
function emG(Vg_p::Vector{Float64},Vg::Vector{Float64},ghat::Vector{Float64},S::Vector{Float64})
    
    ghat2=zero(eltype(S)); τ2_new=copy(ghat2)
    #e-step : update the second moment and sum up for elbo
    numer=(Vg+ghat.^2)
    ghat2=Vg_p'*numer
    #e-step for τ²
    τ2_new= mean(numer./S)
 
    return ghat2, τ2_new
    
end


#Xy = getXy(Xt,yt)


# priors : Π,σ2,
#b's
function postB!(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64}, λ::Vector{Float64},ghat::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64},σ2::Vector{Float64},A0::Matrix{Float64},B0::Matrix{Float64},Π::Vector{Float64},L::Int64)
    
    pidx=axes(B0,1)
    ϕ = zeros(pidx); Z=copy(ϕ);
    
    Z0= similar(Z, axes(yt));
    AB0= similar(B0)
    
    ϕ[:]= gemv('N', Xt.^2,λ) 
    AB0= A0.*B0; #old α_l*b_l
    
    #l=1
          Sig1[:,1] =  1.0./(1/σ2[1].+ ϕ)
            
            Z0= yt - λ.*( gemv('N',Xt₀,β) + Xt*(sum(AB0[:,2:end],dims=2))+ghat)        
            B1[:,l] = Diagonal(Sig1[:,1])*gemv('T',Xt,Z0)
            Z =  0.5*(sqrt.(ϕ).*gemv('T',Xt,Z0)).^2
            A1[:,1] = Π.*sqrt.(ϕ./(σ2[1].+ϕ)).*exp.(Z.* (σ2[1]./(σ2[1].+ϕ)) )
    
    
    
     for l= 2: L
            Sig1[:,l] =  1.0./(1/σ2[l].+ ϕ)
              #need to check if updating b is correct!
            Z0= yt - λ.*( gemv('N',Xt₀,β) + Xt*(sum(hcat(AB0[:,1:l-1],AB0[:,l+1:end]),dims=2))+ghat)        
            B1[:,l] = Diagonal(Sig1[:,l])*gemv('T',Xt,Z0)
            Z =  0.5*(sqrt.(ϕ).*gemv('T',Xt,Z0)).^2
            A1[:,l] = Π.*sqrt.(ϕ./(σ2[l].+ϕ)).*exp.(Z.* (σ2[l]./(σ2[l].+ϕ)) )
    
    end
    
    A1[:,:] = A1./sum(A1,dims=1) #posterior: scale to b/w 0 & 1
    
    
    
end

# #update adding k<l, B1,A1, k>l, B0,A1
# function postB!(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64}, λ::Vector{Float64},ghat::Vector{Float64},
# yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64},σ2::Vector{Float64},A0::Matrix{Float64},B0::Matrix{Float64},Π::Vector{Float64},L::Int64)
    
#     pidx=axes(B0,1)
#     ϕ = zeros(pidx); Z=copy(ϕ);
    
#     Z0= similar(Z, axes(yt));
#     AB0= similar(B0)
    
#     ϕ[:]= gemv('N', Xt.^2,λ) 
#     AB0= A0.*B0; 
#     AB1= zeros(axes(B0))
#     #l=1

#             Sig1[:,1] =  1.0./(1/σ2[1].+ ϕ)
     
#             Z0= yt - λ.*( gemv('N',Xt₀,β) + Xt*(sum(AB0[:,2:end],dims=2)+ghat)
#             B1[:,1] = Diagonal(Sig1[:,1])*gemv('T',Xt,Z0)
#             Z =  0.5*(sqrt.(ϕ).*gemv('T',Xt,Z0)).^2
#             A1[:,1] = Π.*sqrt.(ϕ./(σ2[1].+ϕ)).*exp.(Z.* (σ2[1]./(σ2[1].+ϕ)) )
#             AB1[:,1]= A1[:,1].*B1[:,1]
    
#      for l= 2: L
#             Sig1[:,l] =  1.0./(1/σ0[l].+ ϕ)
#       #need to check if updating b is correct!
#             Z0= yt - λ.*( gemv('N',Xt₀,β) + Xt*(sum(hcat(AB1[:,1:l-1],AB0[:,l:end]),dims=2))+ghat)
#             B1[:,l] = Diagonal(Sig1[:,l])*gemv('T',Xt,Z0)
#             Z =  0.5*(sqrt.(ϕ).*gemv('T',Xt,Z0)).^2
#             A1[:,l] = Π.*sqrt.(ϕ./(σ2[l].+ϕ)).*exp.(Z.* (σ2[l]./(σ2[l].+ϕ)) )
     
#             AB1[:,l]= A1[:,l].*B1[:,l]
    
#     end
    
#         A1[:,:]= A1./sum(A1,dims=1) #posterior prob
    
# end

function emB(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64},σ2::Vector{Float64},L::Int64)
       
    σ0 = zeros(L); AB1=zeros(axes(B1,1)); B2=zeros(axes(B1))
       # compute the second moment of b_l & update the hyper-parameter σ0
        for l= 1:L
        AB1= A1[:,l].*(B1[:,l].^2+ Sig1[:,l])
        B2[:,l]= AB1./σ2[l]
        σ0[l]= sum(AB1) 
     
        end
    
      return σ0, B2
        
        
end
    


#Xy = getXy(Xt,yt)
# M-step

function mStep(A1,B1,ghat,Vg,λ::Vector{Float64},yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64})
    
    ξ_new= zeros(axes(yt))
    β_new= zeros(axes(Xt₀,1))
    
     for i= eachindex(yt)
    ξ_new[i] = (Xt₀*β).^2 + A1
      # dimension check!  (math)
        
        β_new= BLAS.gemm('T','N',Xt₀,(λ.*Xt₀))\BLAS.gemv('T',Xt₀,(yt- λ.*(Xt*sum(A1.*B1,dims=2) + ghat)))
    
    
end





end