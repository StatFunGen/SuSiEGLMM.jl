


# struct GenoData
#     snp::Array{String,1}
#     chr::Array{Any,1}
#     pos::Array{Float64,1} #positon 
#     X::Array{Float64,2}  #genotype data: n x p
# end



"""

    init()

Returns initial values for parameters τ2, β, ξ to run SuSiEGLMM


"""
function init(y::Vector{Float64},X::GenoData,T::Union{Matrix{Float64},Array{Float64,3}},
        S::Union{Matrix{Float64},Vector{Float64}};X₀::Matrix{Float64}=ones(length(y),1),tol=1e-4)
    
     τ2 = rand(1)*0.5; #arbitray
    # may need to change
     β = zeros(axes(X₀,2)) 
     ξ = rand(n)
     
  
        
        # for j= eachindex(Chr)
      yt, Xt₀ = rotate(y,X₀,T)  
     res= emGLMM(yt,Xt₀,S,τ2,β,ξ;tol=tol)
    
    return res
    
    
end


# X₀: check if it includes intercept
function SuSiEGLMM(L::Int64,Π::Vector{Float64},yt::Vector{Float64},Xt::Matrix{Float64},
        S::Union{Matrix{Float64},Vector{Float64}};Xt₀::Matrix{Float64},tol=1e-4)
    
    n, p = size(X)
    #initialization :
     σ0 = 0.1*ones(L);
     
    
#      X₀ = hcat(ones(n),X₀)
#     est=[];
#     if (LOCO)
#         Chr= unique(X.chr); # check if data organized in chr order
#         nChr = length(Chr)
     
        
#         for j = 1: nChr
#           midx = findall(XX.chr.==Chr[j])
#        # rotate data
#           yt, Xt, Xt₀ = rotate(y,XX.X[:,midx],X₀,T[:,:,j])
#          #EM 
#           est0 = emGLMM(L,yt,Xt,Xt₀,S[:,:,j],τ2,β,ξ,σ0,Π;tol::Float64=1e-4)
#           est=[est;est0]       
#         end    
    
#      else #no LOCO
        
#         yt, Xt, Xt₀ = rotate(y,X,X₀,T)
        est = emGLMM(L,yt,Xt,Xt₀,S,τ2,β,ξ,σ0,Π;tol::Float64=1e-4)
        
#      end #LOCO
    
    
    return est
    
end 

struct GenoInfo
    snp::Array{String,1}
    chr::Array{Any,1}
    pos::Array{Float64,1} #positon 
    # X::Array{Float64,2}  #genotype data: n x p
end


function fineMapping()
    
     #need to work more   
      Chr 
     init()
    
   est= @distributed (vcat) for j= eachindex(Chr)
          est0 = fineMapping1()
            est0
    end
    
    
    
    
end




function fineMapping1(f::Function,args...;kwargs...)                   
        
  #need to work more           
        res = f(args...;kwargs...)
    
    
    return res
        
end