

export susieGLM

function susieGLM(L::Int64,Π::Vector{Float64},y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64};tol=1e-5)
  

  
    if(X₀!= ones(n),1) #&&(size(X₀,2)>1)
        X₀ = hcat(ones(n),X₀)
    end
    
        
    n, c =size(X₀)

#initialization :
 σ0 = 0.1*ones(L);
 β = rand(c)*0.0001
 ξ = rand(n)*0.001
 y[:] =y-0.5*ones(n) # centered y
    result = emGLM(L,y,X,X₀,β,ξ,σ0,Π;tol=tol)
        
return result

end 
