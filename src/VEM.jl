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
    
    ghat[:]= Diagonal(Vg)*(yt-λ.*(Xt₀*β + Xt*sum(A0.*B0,dims=2))))
    
    
end

#EM for g
function emG(Vg_p::Vector{Float64},Vg::Vector{Float64},ghat::Vector{Float64},S::Vector{Float64})
    
    ghat2=zero(eltype(S)); τ2=copy(ghat2)
    #e-step : update the second moment and sum up for elbo
    numer=(Vg+ghat.^2)
    ghat2=Vg_p'*numer
    #e-step for τ²
    τ2= mean(numer./S)
 
    return ghat2, τ2
    
end


#b's
function postB(λ::Vector{Float64},
yt::Vector{Float64},Xt::Matrix{Float64},Xt₀::Matrix{Float64},β::Vector{Float64},A0::Matrix{Float64},B0::Matrix{Float64})
    
    
    
end



# M-step

function mStep()
    
    
    
end





end