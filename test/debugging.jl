using Statistics, Distributions, StatsBase, Random, LinearAlgebra, DelimitedFiles, Distributed

#include("./SuSiEGLMM.jl")
#using .SuSiEGLMM

data=readdlm("./test/smalldataX_covar_y.txt")
info=readdlm("./test/testinfo.txt")
info1=readdlm("./test/testinfo_loco.txt")
K=readdlm("./test/testK.txt")

X=data[:,1:end-2]
y=data[:,end]
C=data[:,end-1]
# addprocs(2)
# @everywhere using Pkg
# @everywhere Pkg.activate("/Users/hyeonjukim/GIT/SuSiEGLMM.jl/")

# @everywhere using Pkg; 
# @everywhere Pkg.activate("/Users/hyeonjukim/GIT/SuSiEGLMM.jl/")
# @everywhere using Revise
# @everywhere using SuSiEGLMM
using Revise
 using Pkg; 
Pkg.activate("/Users/hyeonjukim/GIT/SuSiEGLMM.jl/")

 using SuSiEGLMM
G= GenoInfo(info[:,2],info[:,1],info[:,3])
G1=GenoInfo(info1[:,2],info1[:,1],info1[:,3]) # for loco

n= size(K,1)
K0=zeros(n,n,2);K2=copy(K);K0[:,:,1]=K2; K0[:,:,2]=K


T,S = svdK(K0)



# L=3; Π = ones(p)/p
println("step1")
fineMapping_GLMM(G,y,X,C,T[:,:,1],S[:,1];L=3,LOCO=false,tol=1e-4)

println("step2")
tstat,pval=scoreTest(G,y,C,X,K;LOCO=false)
#loco
println("step3")
est1= SuSiEGLMM.fineMapping_GLMM(G1,y,X,C,T,S;L=3,tol=1e-4)

println("step4")
tstat1,pval1=SuSiEGLMM.scoreTest(G1,y,C,X,K0,tol=1e-4)


T1,S1 = eigenK(K0)

es=fineMapping_GLMM(G,y,X,C,T1[:,:,1],S1[:,1];L=3,LOCO=false,tol=1e-4)


tst,pv=scoreTest(G,y,C,X,K;LOCO=false)
#loco

es1= SuSiEGLMM.fineMapping_GLMM(G1,y,X,C,T1,S1;L=3,tol=1e-4)


tst1,pv1=SuSiEGLMM.scoreTest(G1,y,C,X,K0,tol=1e-4)