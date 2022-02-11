
using Revise

using Statistics, Distributions, StatsBase, Random, LinearAlgebra, DelimitedFiles, Distributed
# @everywhere using Pkg
# @everywher Pkg.activate(homedir()*"/GIT/SuSiEGLMM.jl")
 using SuSiEGLMM

Seed(124)

n =100;p=10;q=5;
 L=1; B=100;
#  τ2=0.4;
#GLM
b_true=zeros(p);
b_1s=zeros(B);

res=[];Tm=zeros(B);

## random covariates

for j = 1:B
    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    # b_true[1]=b_1s[j]
    X=randn(n,p)
    X₀ = randn(n,q)
    δ=randn(q)
    # g=rand(MvNormal(τ2*K)) 
    Y= logistic.(X*b_true+X₀*δ) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    t0=@elapsed res0= susieGLM(L, ones(p)/p,Y,X,X₀;tol=1e-4) 
  # t0=@elapsed  res0= fineQTL_glm(G,Y,X;L=L,tol=1e-4)
    res=[res;res0]; Tm[j]=t0
end

println("min, median, max times for susie-glm including covariates are $(minimum(Tm)), $(median(Tm)),$(maximum(Tm)).")

b̂=[res[j].α[1]*res[j].ν[1] for j=1:B]
α̂ = [res[j].α[1] for j=1:B]
using Plots
ll=@layout[a;b]; 
# l2=@layout[a b;c d]
p1=scatter(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate",label=false,title="SuSiE-GLM+random covariates")
p2=scatter(b_1s,α̂, xlabel="True effects",ylabel="PIP",label=false)
plot(p1,p2,layout=ll)

# correlated covariates with predictors
n =100;p=10;
 L=1; B=100;
#  τ2=0.4;
#GLM
b_true=zeros(p);
b_1s=zeros(B);

res1=[];Tm0=zeros(B);

for j = 1:B
    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    # b_true[1]=b_1s[j]
    X=randn(n,p)
    #correlated covariates
    X₁ = rand(MvNormal([1 0.9;0.9 1]),n)
    X[:,1]=X₁[1,:]
    X₀=convert(Matrix{Float64},X₁[[2],:]')
    δ=randn(1)

    # g=rand(MvNormal(τ2*K)) 
    Y= logistic.(X*b_true+X₀*δ) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    t0=@elapsed res0= susieGLM(L, ones(p)/p,Y,X,X₀;tol=1e-4) 
  # t0=@elapsed  res0= fineQTL_glm(G,Y,X;L=L,tol=1e-4)
    res1=[res1;res0]; Tm0[j]=t0
end

println("min, median, max times for susie-glm including covariates are $(minimum(Tm0)), $(median(Tm0)),$(maximum(Tm0)).")

b̂=[res1[j].α[1]*res1[j].ν[1] for j=1:B]
α̂ = [res1[j].α[1] for j=1:B]


p3=scatter(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate",label=false,title="SuSiE-GLM+correlated covariates")
p4=scatter(b_1s,α̂, xlabel="True effects",ylabel="PIP",label=false)
plot(p3,p4,layout=ll)