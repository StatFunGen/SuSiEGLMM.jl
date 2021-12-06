

export susieGLM

function susieGLM(L::Int64,Π::Vector{Float64},y::Vector{Float64},X::Matrix{Float64},X₀::Matrix{Float64};tol=1e-5)
  
    n, c =size(X₀)
  
    if(X₀!= ones(n,1)) #&&(size(X₀,2)>1)
        X₀ = hcat(ones(n),X₀)
    end
         

#initialization :
 σ0 = 0.1*ones(L);
 β = rand(c)/sqrt(c)
 ξ = rand(n)/sqrt(n)
 y[:] =y-0.5*ones(n) # centered y
    result = emGLM(L,y,X,X₀,β,ξ,σ0,Π;tol=tol)
        
return result

end 


function fineMapping_GLM(G::GenoInfo,y::Vector{Float64},X::Matrix{Float64},
    X₀::Union{Matrix{Float64},Vector{Float64}};L::Int64=10,Π::Vector{Float64}=[1/size(X,2)],tol=1e-5)

    Chr=sort(unique(G.chr));
    est= @distributed (vcat) for j= eachindex(Chr)
                 midx= findall(G.chr.== Chr[j])
                  #check size of Π
                  if (Π==[1/size(X,2)]) #default value
                    m=length(midx)
                    Π1 =repeat(1/m,m) #adjusting πⱼ
                   elseif (length(Π)!= size(X,2))
                      println("Error. The length of Π should match $(size(X,2)) SNPs!")
                   else
                    Π1 = Π[midx]
                 end

                 est0= susieGLM(L,Π1,y,X,X₀;tol=tol)
                        est0
    end

   return est
end