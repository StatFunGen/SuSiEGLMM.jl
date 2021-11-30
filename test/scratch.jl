#code debugging

using Statistics, Distributions, StatsBase, Random, LinearAlgebra, DelimitedFiles, Distributed

#include("./SuSiEGLMM.jl")
#using .SuSiEGLMM

# include("GIT/SuSiEGLMM.jl/src/GLMM.jl")

info=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/snp_info.bim")
data=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/pop_518ids_4000snps.txt";header=true); #518 x 4000 snps (qtl = 1927th)
#data1=readdlm("../testdata/fam_100fams_4000snps.txt";header=true)

#5th col :sex
Covar = convert(Vector{Float64},data[1][:,5])
Covar[Covar.==1].=0.0
Covar[Covar.==2].=1.0

#last col: trait
y=convert(Vector{Float64},data[1][:,end])

# kinship
K=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/pop_518fams_4000snps.cXX.txt") #518



info1= [info[1:20,:];info[1925:1929,:]]
using Revise
using Pkg
Pkg.activate(homedir()*"/GIT/SuSiEGLMM.jl")
using SuSiEGLMM
G= GenoInfo(info1[:,2],info1[:,1],info1[:,3])



# fill out "NA" 
X = [data[1][:,6:25] data[1][:,1930:1934]]

for j =axes(X,2)
    idx = findall(X[:,j].=="NA")
    X[idx,j].= missing
    X[idx,j] .= mean(skipmissing(X[:,j]))
end

for j =axes(X,2)
    
   println(sum(ismissing.(X[:,j])))
    
end

X = convert(Matrix{Float64},X)
n,p = size(X)
L=3; Π = ones(p)/p

# write small data for debugging in VS-code
# writedlm("./test/smalldataX_covar_y.txt",[X[1:15,11:25] Covar[1:15] y[1:15]])
# writedlm("./test/testinfo.txt",info[11:25,:])
# writedlm("./test/testinfo_loco.txt",[info[11:17,:];info[end-7:end,:]])
# writedlm("./test/testK.txt",K[1:15,1:15])

# K0=zeros(n,n,2);K2=copy(K);K0[:,:,1]=K2; K0[:,:,2]=K
# T1,S1 = eigenK(K0)
# T1,S1 = svdK(K0)


# @time est1= fineMapping_GLMM(G,y,X,Covar,T1[:,:,1],S1[:,1];LOCO=false,tol=1e-4)

#try with cholesky (provide stable eigenvalues)
# ch=cholesky(K)
# F=svd(ch.U)
# T2 = convert(Array{Float64,2}, F.Vt')
# @time est0= fineMapping_GLMM(G,y,X,Covar,T2,F.S.^2;LOCO=false,tol=1e-4)    


F=svd(K;full=true)
# the smallest eigen value close to 0
# # 9.896090246630486e-13 replaced with 0
# put back to Kinship
# K1=F.U*Diagonal(F.S)*F.Vt
# K≈K1 #true

# svd again to the adjusted kinship
# F1= svd(K1;full=true)
# # 2.51072713180801e-17 :worse than K

fineMapping_GLMM(G,y,X,Covar,F.U,F.S;LOCO=false,tol=1e-4)    
@time est2= fineMapping_GLMM(G1,y,X1,Covar,F.U,F.S;LOCO=false, tol=1e-5)

#line by line debugging
Xt, Ct, yt, init_est= initialization(y,X,Covar,F.U,F.S;tol=1e-4, LOCO=false)
    
    # check if covariates are added as input and include the intercept. 
    if(Covar!= ones(length(y),1))
        Covar = hcat(ones(length(y)),Covar)
    end
    println(Covar)
        
#                    Xt₀ = rotateX(X₀,T)
#                    yt = rotateY(y,T)
                   Xt, Ct, yt = rotate(y,X,X₀,F.U)   
                   init_est= SuSiEGLMM.init(yt,Ct,F.S;tol=tol)
       
    return Xt, Xt₀, yt, init_est


    τ2 = rand(1)[1]*0.5; #arbitray
    # may need to change
     β = zeros(axes(Ct,2)) 
     ξ = rand(length(yt))*0.1
      

    emGLMM(yt,Xt₀,S,τ2,β,ξ;tol::Float64=1e-4)
    
    
    n = length(yt)
    ghat =zeros(n); Vg = zeros(n); λ = zeros(n)
    
    ghat2=zeros(axes(F.S)); τ2_new=zero(eltype(F.S)); 
    ξ_new = zeros(n); β_new=zeros(axes(β))
    
    crit =1.0; el0=0.0;numitr=1
      
    
    while (crit>=tol)
        ###check again!
         SuSiEGLMM.postG!(ghat,Vg,λ,yt,Ct,F.S,β,ξ,τ2)
         ghat2, τ2_new = SuSiEGLMM.emG(Vg,ghat,F.S)
         
         SuSiEGLMM.mStep!(ξ_new,β_new,ghat,ghat2,λ,yt,Ct,β)
        
         el1=SuSiEGLMM.ELBO(ξ_new,β_new,τ2_new,ghat2,Vg,F.S,yt,Ct)
     
         # crit=el1-el0 
         crit=norm(ξ_new-ξ)+norm(β_new-β)+abs(τ2_new-τ2)+abs(el1-el0)  
        
         ξ=ξ_new;β=β_new; τ2=τ2_new;el0=el1
        
          numitr +=1        
    end
    




#pip 
 [1.0.-prod(1.0.-est1.α[j,:]) for j=1:p]



#check loco
# @everywhere include("GIT/SuSiEGLMM.jl/src/GLMM.jl")
X1 = data[1][:,6:end-1]
 X1[X1.=="NA"].= missing
for j =axes(X1,2)
    idx =findall(ismissing.(X1[:,j]))
   X1[idx,j] .= mean(skipmissing(X1[:,j]))
end

X1 = convert(Array{Float64,2},X1)

n,p=size(X1)
G1=GenoInfo(info[:,2],info[:,1],info[:,3])




# L=5;Π = rand(p)



# need to fix loco- postB! part.
addprocs(2)
@everywhere using Pkg
@everywhere Pkg.activate("/Users/hyeonjukim/GIT/SuSiEGLMM.jl/")
@everywhere using SuSiEGLMM

@time est2= SuSiEGLMM.fineMapping_GLMM(G1,y,X1,Covar,T1,S1;LOCO=true, tol=1e-5)

@time tstat, pvalue= scoreTest(G1,y,Covar,X1,K0;LOCO=true)

#pip 
[1.0-prod(1.0.-est2[2].α[j,:]) for j=axes(est2[2].α,1)]