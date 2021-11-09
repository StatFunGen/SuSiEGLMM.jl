"""

    VEM

A module for posterior and hyper-parameter estimates using a variational expectiation-maximization method.


"""
module VEM

using Statistics, LinearAlgebra, Random, StatsBase, Distributions



#include("Utilities.jl")




# drawing initial values


#E-step

#posterior

#g
function postG!(ghat::Vector{Float64},Vg::Vector{Float64},λ::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},S::Vector{Float64},
        β::Vector{Float64},ξ::Vector{Float64},τ2::Float64,A0::Matrix{Float64},B0::Matrix{Float64})
    
    
    λ[:]= 2Lambda.(ξ)
   
    #posterior
    Vg[:]= 1.0./(λ+1.0./(τ2*S))
    
    ghat[:]= Diagonal(Vg)*(yt-λ.*(gemv('N',Xt₀,β) + gemv('N',Xt,sum(A0.*B0,dims=2))))
    
    
end

#EM for g
function emG(Vg::Vector{Float64},ghat::Vector{Float64},S::Vector{Float64})
    
    ghat2=zeros(axes(S)); τ2_new=zero(eltype(S)); 
    #e-step : update the second moment 
    ghat2= Vg+ghat.^2 
    #m-step for τ²
    τ2_new= mean(ghat2/.S)
 
    return ghat2, τ2_new
    
end


#Xy = getXy(Xt,yt)


# priors : Π,σ2,
#b's
#update adding k<l, B1,A1, k>l, B0,A0
function postB!(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64}, λ::Vector{Float64},ghat::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64},σ2::Vector{Float64},A0::Matrix{Float64},B0::Matrix{Float64},Π::Vector{Float64},L::Int64)
    
    pidx=axes(B0,1)
    ϕ = zeros(pidx); Z=copy(ϕ);
    
    Z0= similar(Z, axes(yt));
    AB0= similar(B0)
    
    ϕ[:]= gemv('T', Xt.^2,λ) # mle of precision
    AB0= A0.*B0;  # #old α_l*b_l
    # AB1= zeros(axes(B0))
   

            Sig1[:,:] =  1.0./(1.0./σ2'.+ ϕ)
      #l=1
            Z0= yt - λ.*(gemv('N',Xt₀,β) + Xt*(sum(AB0[:,2:end],dims=2))+ghat)
            B1[:,1] = Diagonal(Sig1[:,1])*gemv('T',Xt,Z0) 
           # compute α_1
            Z =  0.5*(gemv('T',Xt,Z0)./sqrt.(ϕ)).^2
            A0[:,1] = Π.*exp.(Z./(1.0.+ϕ/σ2[1]))./sqrt.(σ2[1]*ϕ.+1)
            A1[:,1]= A0[:,1]/sum(A0[:,1]) # scale to 0< α_1<1
            # AB1[:,1]= A1[:,1].*B1[:,1] #update α_1*b_1
    
     for l= 2: L
           
            Z0= yt - λ.*(gemv('N',Xt₀,β) + Xt*(sum(hcat(AB1[:,1:l-1],AB0[:,l+1:end]),dims=2))+ghat)
            B1[:,l] = Diagonal(Sig1[:,l])*gemv('T',Xt,Z0)
            Z =  0.5*(gemv('T',Xt,Z0)./sqrt.(ϕ)).^2
            A0[:,l] = Π.*exp.(Z./(1.0.+ϕ/σ2[l]))./sqrt.(σ2[l]*ϕ.+1)
            A1[:,l] = A0[:,l]/sum(A0[:,l])
            # AB1[:,l]= A1[:,l].*B1[:,l]
      end
    
end

function emB(A1::Matrix{Float64}, B1::Matrix{Float64}, Sig1::Matrix{Float64},σ2::Vector{Float64},L::Int64)
       
    σ0 = zeros(L); AB2=zeros(axes(B1));
       # compute the second moment of b_l & update the hyper-parameter σ0
        for l= 1:L
        AB2[:,l]= A1[:,l].*(B1[:,l].^2+ Sig1[:,l])    
        σ0[l]= sum(AB2[:,l]) 
        end
         
    
      return σ0, AB2 
        
        
end
    


#Xy = getXy(Xt,yt)
# M-step

function mStep!(ξ_new::Vector{Float64},β_new::Vector{Float64},A1::Matrix{Float64},B1::Matrix{Float64},AB2::Matrix{Float64},
        ghat::Vector{Float64},ghat2::Vector{Float64},λ::Vector{Float64},
        yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64})
    
    ξ_new= zeros(axes(yt))
    β_new= zeros(axes(Xt₀,2))
    
     for i= eachindex(yt)
    ξ_new[i] = (Xt₀*β).^2 + Xt*sum(AB2,dims=2) + ghat2 + 2*(Xt₀*β +ghat).*(Xt*sum(A1.*B1,dims=2))
    ξ_new[i]= sqrt.(ξ_new[i])
    end
     
        
        β_new= BLAS.gemm('T','N',Xt₀,(λ.*Xt₀))\BLAS.gemv('T',Xt₀,(yt- λ.*(Xt*sum(A1.*B1,dims=2) + ghat)))
    
    
end



function ELBO(L::Int64,ξ_new::Vector{Float64},β_new::Vector{Float64},σ0::Vector{Float64},τ2_new::Float64,
        A1::Matrix{Float64},B1::Matrix{Float64},AB2::Matrix{Float64},Sig1::Matrix{Float64},Π::Vector{Float64},
        ghat2::Vector{Float64},Vg::Vector{Float64},S::Vector{Float64},yt::Vector{Float64},Xt₀)

    n=length(yt); p = size(B1,1); lnb =zero(1);
# log.(logistic.(ξ))- 0.5*ξ+ yX*β
        # 2nd moment computation
# sum(ghat2/.S)/τ2_new
#0.5*sum(AB2,dims=1)./σ0)  

     ll= sum(log.(logistic.(ξ))- 0.5*ξ+ yt'*gemv('N',Xt₀,β_new)) #lik
     gl = -0.5*(n*log(τ2_new)+ sum(log.(S./Vg))- 1.0) - sum(ghat2/.S)/τ2_new # g
     # susie part
    for l= 1: L 
     lnb  += A1[:,l]'*(log.(A1[:,l]./Π) - 0.5*log.(Sig1[:,l])) 
    end
               
      bl= lnb  +0.5*(sum(log.(σ0)- L)+ 0.5*sum(sum(AB2,dims=1)'./σ0))
         
    return ll+gl-bl
            
end
    


end