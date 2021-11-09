


struct GenoData
    snp::Array{String,1}
    chr::Array{Any,1}
    pos::Array{Float64,1} #positon 
    X::Array{Float64,2}  #genotype data: n x p
end
    
 


# X₀: check if it includes intercept
function SuSiEGLMM(L::Int64,Π::Vector{Float64},y,X::GenoData,X₀,T::Union{Matrix{Float64},Array{Float64,3}},
        S::Union{Matrix{Float64},Vector{Float64}};LOCO::Bool=true,tol=1e-4)
    
    n, p = size(X)
    #initialization :
     σ0 = 0.1*ones(L);
     τ2 = mean(σ0); #arbitray
    # need to change
     β = zeros(axes(X₀,2)) 
     ξ = rand(n)
      
     # X₀ = hcat(ones(n),X₀)
    est=[];
    if (LOCO)
        Chr= unique(X.chr); # check if data organized in chr order
        nChr = length(Chr)
     
        
        for j = 1: nChr
          midx = findall(XX.chr.==Chr[j])
       # rotate data
          yt, Xt, Xt₀ = rotate(y,XX.X[:,midx],X₀,T[:,:,j])
         #EM 
          est0 = emLMM(L,yt,Xt,Xt₀,S[:,:,j],τ2,β,ξ,σ0,Π;tol::Float64=1e-4)
          est=[est;est0]       
        end    
    
    else #no LOCO
        
        yt, Xt, Xt₀ = rotate(y,X,X₀,T)
        est = emLMM(L,yt,Xt,Xt₀,S,τ2,β,ξ,σ0,Π;tol::Float64=1e-4)
        
    end #LOCO
    
    
    return est
    
end 