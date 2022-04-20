#code debugging


@everywhere using Statistics, Distributions, StatsBase, Random, LinearAlgebra, DelimitedFiles, Distributed

@everywhere using Revise
@everywhere using Pkg
@everywhere Pkg.activate(homedir()*"/GIT/susie-glmm/SuSiEGLMM.jl")
@everywhere using SuSiEGLMM


### causal1:pop
@time info=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/ascertained_pop_12_10.bim");
@time geno=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/ascertained_pop_genotype_12_10.txt";header=true);
@time pheno=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/ascertained_pop_phenotype_12_10.txt";header=true); #503 x 4000 snps
#data1=readdlm("../testdata/fam_100fams_4000snps.txt";header=true)

# covariate: sex
C = pheno[1][:,end-1]
C[C.==1].=0.0
C[C.==2].=1.0

#last col: trait
# y=convert(Vector{Float64},data[1][:,end])
y=pheno[1][:,end]

# kinship
K=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/pop_grm_ped.txt") #518
K0=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/pop_grm.txt");
K0=Symmetric(K0);K0=convert(Matrix{Float64}, K0);
 n=size(K,1)
K1=zeros(n,n,2);
K1[:,:,1]=K; K1[:,:,2]=K0;

G= GenoInfo(info[:,2],info[:,1],info[:,3])

# fill out "NA" 
X = geno[1][:,6:end]

for j =axes(X,2)
    idx = findall(X[:,j].=="NA")
    X[idx,j].= missing
    X[idx,j] .= mean(skipmissing(X[:,j]))
end

# for j =axes(X,2)   
#    println(sum(ismissing.(X[:,j])))
# end

X = convert(Matrix{Float64},X)
n,p = size(X)
# L=3; Π = ones(p)/p
#score test
@time Tstat, pval= scoreTest(K,G,y,X;LOCO=false);

    T, S = svdK(K;LOCO=false)
    Xt, Xt₀, yt,init00= initialization(y,X,ones(n,1),T,S;tol=1e-4)
    T0= computeT(init00,yt,Xt₀,Xt)


@time Tstat1, pval1= scoreTest(K1,G,y,X)

#susie-glm
@time est0= fineQTL_glm(G,y,X;tol=1e-4)
@time est1= fineQTL_glm(G,y,X,C;tol=1e-4)

a1=sum(est0[1].α,dims=2)
a2=sum(est0[2].α,dims=2)

#susie-glmm: verion 1
T, S= svdK(K;LOCO=false)
@time est2 = fineQTL_glmm(G,y,X,ones(n,1),T,S;LOCO=false)
T1, S1= svdK(K1)
@time est3 = fineQTL_glmm(G,y,X,ones(n,1),T1,S1;LOCO=true)
#susie-glmm: version 2

@time est4 =fineQTL_glmm(K,G,y,X;LOCO=false,tol=1e-4)
a3=sum(est4[1].α,dims=2)
a4=sum(est4[2].α,dims=2)
@time est5 =fineQTL_glmm(K1,G,y,X;tol=1e-4)


# K=I
@time Xt, Ct, yt, init0= initialization(y,X1,ones(n,1),Matrix(1.0I,n,n),ones(n);tol=1e-5)
@time est1= fineMapping_GLMM(G1,y,X1,ones(n,1),Matrix(1.0I,n,n),ones(n);LOCO=false, tol=1e-5)
@time res=susieGLMM(10, ones(p)/p,y,X1,ones(n,1),ones(n);tol=1e-5)

@time glmr=susieGLM(10, ones(p)/p,y,X1,ones(n,1);tol=1e-4)  
#pip 
p=size(X,2)
[[1.0.-prod(1.0.-est1.α[j,:]) for j =1:p] [1.0.-prod(1.0.-res.α[j,:]) for j =1:p]]
[1.0.-prod(1.0.-res.α[j,:]) for j =1:p]
    


######## the same simulation in R-version
Seed(124)

X1= (X.-mean(X,dims=2))./std(X,dims=2)
n,p = size(X1)
 L=1; B=100;τ2=0.1;

#GLMM :score test

# n=100; 
# p=10; 

# K2=Symmetric(K0)
b_true=zeros(p);
b_1s=zeros(B); 
init0=[]; init1=[]
Ts=zeros(p,B);
# tt=zeros(B); Ps=zeros(p,B); 
 Ts1=zeros(p,B);#Ps1=zeros(p,B);

res=[]
#add covariates
# c=3

Y0=zeros(n,B);
# K=Matrix(1.0I,n,n)
# T,S = svdK(K;LOCO=false)
T,S = svdK(K0;LOCO=false)
# # H=svd(K2);

for j = 1:B
    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]   
    # b_true[1]=b_1s[j]
    # X=randn(n,p)
    # g=rand(MvNormal(τ2*K)) #theoretical
    g=rand(MvNormal(τ2*K0)) #grm
    # writedlm("./testdata/dataX-julia.csv",X)
    # X₀=randn(n,c)
    # bhat=randn(c)
    # Y= logistic.(X*b_true+g) .>rand(n) #generating binary outcome

    # Y= logistic.(g) .>rand(n)
    Y1=logistic.(X1*b_true+g) .>rand(n)
    # Y= logistic.(X1*b_true+g+X₀*bhat) .>rand(n) # random covariates independent of X:95%
    # Y=convert(Vector{Float64},Y)
    Y1=convert(Vector{Float64},Y1) 
    # writedlm("./testdata/dataY-julia.csv",Y)
    # Y0[:,j]=Y1
    
    t0=@elapsed begin
        # Xt, Xt₀, yt,init00= initialization(Y,X1,X₀,T,S;tol=1e-4)
        # Xt, Xt₀, yt,init00= initialization(Y,X1,ones(n,1),T,S;tol=1e-4)
        # T0= computeT(init00,yt,Xt₀,Xt)
        # Xt, Xt₀, yt,init01= initialization(Y1,X1,ones(n,1),T,S;tol=1e-3)
        # T1= computeT(init01,yt,Xt₀,Xt)
        res0= susieGLMM(L,ones(p)/p,Y1,X1,ones(n,1),T,S;tol=1e-3)
    end
    # init0=[init0;init00]
    # init1=[init1;init01]
    # Ts[:,j]=T0;
    # Ts1[:,j]=T1
    # # Ps[:,j]=ccdf.(Chisq(1),T0); Ps1[:,j]=ccdf.(Chisq(1),T1)
    # tt[j]=t0
    res=[res;res0]
end

# writedlm("./test/y_causal1_1_pop_grm.txt",Y0)
for j=1:B
    Xt, Xt₀, yt,init01= initialization(Y0[:,j],X1,ones(n,1),T,S;tol=1e-4)
    T1= computeT(init01,yt,Xt₀,Xt)
    # Xt, Xt₀, yt,init01= initialization(Y1,X1,ones(n,1),T,S;tol=1e-3)
    # T1= computeT(init01,yt,Xt₀,Xt)
     init1=[init1;init01]
# init1=[init1;init01]
      Ts1[:,j]=T1;
end

[init0[j].τ2 for j=1:B]
tα=percentile(maximum(Tscore,dims=1)[1,:],95)
sum(Tscore1[1,:].>tα)

sum(Ps[1,:].<0.05)
println("min, median, max times for score test are $(minimum(tt)),$(median(tt)), $(maximum(tt)).")
println("min median max times for glm are $(minimum(Tm)),$(median(Tm)), $(maximum(Tm)).")

#test glm
Y= readdlm("./testdata/causal1_pop_Y.csv",',';skipstart=1)[:,1]
res0= susieGLM(L, ones(p)/p,Y,X1,ones(n,1);tol=1e-3) 


#susie-GLMM & glm
#GLM
b_true=zeros(p);
b_1s=zeros(B);

res=[]; #Tm=zeros(B);
init0=[];Tscore=zeros(p,B)
for j = 1:B
    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    # b_true[1]=b_1s[j]
    X=randn(n,p)
    writedlm("./testdata/dataX-julia.csv",X)
    # g=rand(MvNormal(τ2*K)) 
    Y= logistic.(X*b_true) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    writedlm("./testdata/dataY-julia.csv",Y)
    res0= susieGLM(n,L, ones(p)/p,Y,X,ones(n,1);tol=1e-3) 
#   t0=@elapsed  res0= fineQTL_glm(G,Y,X1;L=L,tol=1e-4)
    res=[res;res0]; #Tm[j]=t0
    
    Xt, Xt₀, yt,init00= initialization(Y,X,ones(n,1),Matrix(1.0I,n,n),ones(n);tol=1e-3)
        T0= computeT(init00,yt,Xt₀,Xt)
        init0=[init0;init00]
        Tscore[:,j]=T0;
end




b̂ = [res[2j-1].α[1]*res[2j-1].ν[1] for j=1:B]
α̂ = [res[2j-1].α[1] for j=1:B]
writedlm("./test/glm-cau1.txt",[b̂ α̂ b_1s])
println("min median max times for glm are $(minimum(Tm)),$(median(Tm)), $(maximum(Tm)).")

using UnicodePlots
scatterplot(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate")
scatterplot(b_1s,α̂, xlabel="True effects",ylabel="pip")


#susie-glmm
b_true=zeros(p);
b_1s=zeros(B); 
rglm=[];rglmm=[];
 Tm0=zeros(B);#for K0
 Tmm=zeros(B); #K
#  K=Matrix(1.0I,n,n);

for j = 1:B

    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    # b_true[1]=b_1s[j]
    X=randn(n,p)

    g=rand(MvNormal(τ2*K))
    # writedlm("./testdata/dataX-julia.csv",X)
    Y= logistic.(X*b_true+g) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    # T, S = svdK(K;LOCO=false)
    Xt, Xt₀, yt, init0 = initialization(Y,X,ones(n,1),T,S;tol=1e-4)     
   t0= @elapsed res10 = susieGLMM(L,ones(p)/p,yt,Xt,Xt₀,S,init0;tol=1e-4)
    # @time res10 =susieGLMM(1,ones(p)/p,Y,X1,ones(n,1),T,S) 
#    t0= @elapsed res10=fineQTL_glmm(K,G,Y,X1;L=L,LOCO=false)
    rglmm=[rglmm;res10];# Tm0[j]=t0
    
   t1=@elapsed res0= susieGLM(L,ones(p)/p,Y,X,ones(n,1);tol=1e-4)
    rglm=[rglm;res0]
    Tm0[j]=t0;Tmm[j]=t1
end

b̂1 = [rglm[j].α[1]*rglm[j].ν[1] for j=1:B]
α̂1 = [rglm[j].α[1] for j=1:B]


b̂ = [rglmm[j].α[1]*rglmm[j].ν[1] for j=1:B]
α̂ = [rglmm[j].α[1] for j=1:B]

# b̂ = [res1[2j-1].α[1]*res1[2j-1].ν[1] for j=1:B]
# α̂ = [res1[2j-1].α[1] for j=1:B]

writedlm("./test/glmm-score-susie.txt",[b̂ α̂ b_1s])
println("min, median, max times for susie-glmm are $(minimum(Tmm)), $(median(Tmm)),$(maximum(Tmm)).")
#min, median, max times for susie-glmm are 1.198, 8.719,14.329. for theoretic K
println("min, median, max times for susie-glm are $(minimum(Tm0)), $(median(Tm0)),$(maximum(Tm0)).")
using UnicodePlots
scatterplot(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate")
scatterplot(b_1s,α̂, xlabel="True effects",ylabel="PIP")

ll=@layout[a;b]; l2=@layout[a b;c d]
p1=scatter(b_1s,b̂1,xlabel= "True effects", ylabel="Posterior estimate",label=false,title="SuSiE-GLM")
p2=scatter(b_1s,α̂1, xlabel="True effects",ylabel="PIP",label=false,title="SuSiE-GLM")

p3=scatter(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate",label=false,title="SuSiE-GLMM")
p4=scatter(b_1s,α̂, xlabel="True effects",ylabel="PIP",label=false,title="SuSiEGLMM")
plot(p1,p3,p2,p4,layout=l2)

Xt, Xt₀, yt, init0 = initialization(y[:,1],X,ones(n,1),T,S;tol=1e-4)     
   t0= @elapsed res10 = susieGLMM(10,ones(p)/p,yt,Xt,Xt₀,S,init0;tol=1e-4)
  
   t1=@elapsed res0= susieGLM(10,ones(p)/p,y[:,1],X,ones(n,1);tol=1e-4)

b̂1 = maximum(res0.α.*res0.ν;dims=2)


# # covariates correlated with X (glmm only)
# X2=copy(X1)
# for j = 1:B

#     b_true[1]= randn(1)[1] 
#     b_1s[j] = b_true[1]
#     # b_true[1]=b_1s[j]
#     # X=randn(n,p)
#     # g=rand(MvNormal(τ2*K)) #theoretical 
#     g=rand(MvNormal(τ2*K0)) #grm
#     # writedlm("./testdata/dataX-julia.csv",X)
#     bhat=randn(1)
#     X₀=rand(MvNormal([1 .9;.9 1]),n)
#     X2[:,1] = X₀[1,:]
   
#     Y= logistic.(X1*b_true+g+X₀[2,:].*bhat) .>rand(n) 
#     Y=convert(Vector{Float64},Y)
#     t0=@elapsed begin
#     Xt, Xt₀, yt,init00= initialization(Y,X2,X₀[2,:],T,S;tol=1e-4)
   
#     T0= computeT(init00,yt,Xt₀,Xt)
#     end
#     init0=[init0;init00]
#     Tscore[:,j]=T0
#     Ps[:,j]=ccdf.(Chisq(1),T0)
#     tt[j]=t0
# end
