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
# K_fam=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/fam_100fams_4000snps.cXX.txt")


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

# @time est1= fineMapping_GLMM(G,y,X,Covar,T1[:,:,1],S1[:,1];LOCO=false,tol=1e-4)


fineMapping_GLMM(G,y,X,Covar,F.U,F.S;LOCO=false,tol=1e-4)    
@time est2= fineMapping_GLMM(G1,y,X1,Covar,F.U,F.S;LOCO=false, tol=1e-5)


for j=axes(K,1)
    K[j,j]=1.0
end




Xt, Ct, yt = rotate(y,X,Covar,T) 
@time res0=susieGLMM(L,Π,yt,Xt,Ct,S;tol=1e-4)
@time res=susieGLMM(L,Π,yt,Xt,Ct,S;tol=1e-5)
@time res2=susieGLMM(L,Π,yt,Xt,Ct,S;tol=1e-6)



# K=I
@time Xt, Ct, yt, init0= initialization(y,X1,ones(n,1),Matrix(1.0I,n,n),ones(n);tol=1e-5)
@time est1= fineMapping_GLMM(G1,y,X1,ones(n,1),Matrix(1.0I,n,n),ones(n);LOCO=false, tol=1e-5)
@time res=susieGLMM(10, ones(p)/p,y,X1,ones(n,1),ones(n);tol=1e-5)

@time glmr=susieGLM(10, ones(p)/p,y,X1,ones(n,1);tol=1e-4)  
#pip 
p=size(X,2)
[[1.0.-prod(1.0.-est1.α[j,:]) for j =1:p] [1.0.-prod(1.0.-res.α[j,:]) for j =1:p]]
[1.0.-prod(1.0.-res.α[j,:]) for j =1:p]
    
#########
X1 = data[1][:,6:end-1]
 X1[X1.=="NA"].= missing
for j =axes(X1,2)
    idx =findall(ismissing.(X1[:,j]))
   X1[idx,j] .= mean(skipmissing(X1[:,j]))
end

X1 = convert(Array{Float64,2},X1)

n,p=size(X1)
G1=GenoInfo(info[:,2],info[:,1],info[:,3])

# K0=zeros(n,n,2);K2=copy(K);K0[:,:,1]=K2; K0[:,:,2]=K
# T1,S1 = svdK(K0)
τ2true= 0.25
btrue=[-sqrt(2);sqrt(2)]./2

idx=findall(G1.chr.==1)

yL= [ones(n) X1[:,180]]*btrue
g=MvNormal(τ2true*Matrix(1.0I,n,n))
yR=rand(g)
y= yL+yR

######## the same simulation in R-version
Random.seed!(113)

n=100; p=10; L=1;
b_true=zeros(p);
B=100;
b_1s=zeros(B); res=[];
for j = 1:B

    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    X=randn(n,p)
    Y= logistic.(X*b_true) .<rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    res0= susieGLM(L, ones(p)/p,Y,X,ones(n,1);tol=1e-4) 
    res=[res;res0]
end

b̂ = [res[j].α[1]*res[j].ν[1] for j=1:B]
α̂ = [res[j].α[1] for j=1:B]
using UnicodePlots
scatterplot(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate")
scatterplot(b_1s,α̂, xlabel="True effects",ylabel="pip")

