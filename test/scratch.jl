#code debugging


@everywhere using Statistics, Distributions, StatsBase, Random, LinearAlgebra, DelimitedFiles, Distributed

@everywhere using Revise
@everywhere using Pkg
@everywhere Pkg.activate(homedir()*"/GIT/SuSiEGLMM.jl")
@everywhere using SuSiEGLMM

@time info=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/ascertained_pop_12_10.bim");
@time geno=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/ascertained_pop_genotype_12_10.txt";header=true);
@time pheno=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/ascertained_pop_phenotype_12_10.txt";header=true); #518 x 4000 snps (qtl = 1927th)
#data1=readdlm("../testdata/fam_100fams_4000snps.txt";header=true)

# covariate: sex
C = pheno[1][:,end-1]
C[C.==1].=0.0
C[C.==2].=1.0

#last col: trait
# y=convert(Vector{Float64},data[1][:,end])
y=pheno[1][:,end]

# kinship
K=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal1/pop/pop_grm.txt") #518
# K_fam=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/fam_100fams_4000snps.cXX.txt")
 n=size(K,1)
K1=zeros(n,n,2);
K1[:,:,1]=K; K1[:,:,2]=K;




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
@time Tstat, pval= SuSiEGLMM.scoreTest(K,G,y,X;LOCO=false);

    T, S = svdK(K;LOCO=false)
    Xt, Xt₀, yt,init00= initialization(y,X,ones(n,1),T,S;tol=1e-4)
    T0= computeT(init00,yt,Xt₀,Xt)


@time Tstat1, pval1= SuSiEGLMM.scoreTest(K,G,y,X,C;LOCO=false)

@time Tstat2, pval2= SuSiEGLMM.scoreTest(K1,G,y,X)
@time Tstat3, pval3= SuSiEGLMM.scoreTest(K1,G,y,X,C)

#susie-glm
@time est0= fineQTL_glm(G,y,X;tol=1e-4)
@time est1= fineQTL_glm(G,y,X,C;tol=1e-4)

#susie-glmm: verion 1
@time est2 = fineQTL_glmm(G,y,X,ones(n,1),T,S;LOCO=false)
T1, S1= svdK(K1)
@time est3 = fineQTL_glmm(G,y,X,ones(n,1),T1,S1;LOCO=true)
#susie-glmm: version 2

@time est4 =fineQTL_glmm(K,G,y,X;LOCO=false,tol=1e-4)

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
Random.seed!(124)

#GLM
n=100; p=10; L=1; 
b_true=zeros(p);
B=100;
b_1s=zeros(B); res=[];

for j = 1:B
    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    X=randn(n,p)
    # writedlm("./testdata/dataX-julia.csv",X)
    Y= logistic.(X*b_true) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    res0= susieGLM(L, ones(p)/p,Y,X,ones(n,1);tol=1e-4) 
    res=[res;res0]
end

b̂ = [res[j].α[1]*res[j].ν[1] for j=1:B]
α̂ = [res[j].α[1] for j=1:B]


using UnicodePlots
scatterplot(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate")
scatterplot(b_1s,α̂, xlabel="True effects",ylabel="pip")

#GLMM :scroe test

# n=100; p=10; L=1; 
p=2000;
B=100;τ2=1.2; #K=Matrix(1.0I,n,n);
K2=Symmetric(K)
b_true=zeros(p);b_1s=zeros(B); init0=[]; Ts=zeros(p,B);


# F =cholesky(K2)
# f = svd(F.U)
# T,S = f.Vt, f.S.^2
T,S = svdK(K;LOCO=false)
# H=svd(K2);
for j = 1:B

    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    # X=randn(n,p)
    X1=X[:,1:p]

    g=rand(MvNormal(τ2*K2))
    # writedlm("./testdata/dataX-julia.csv",X)
    Y= logistic.(X1*b_true+g) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    
    Xt, Xt₀, yt,init00= initialization(Y,X1,ones(n,1),T,S;tol=1e-4)
    T0= computeT(init00,yt,Xt₀,Xt)
    init0=[init0;init00]
    Ts[:,j]=T0
end

[init0[j].τ2 for j=1:B]

#susie-GLMM

n=100; p=10; L=1; 
b_true=zeros(p);
B=100;
b_1s=zeros(B); res1=[];
τ2=1.2; K=Matrix(1.0I,n,n);

for j = 1:B

    b_true[1]= randn(1)[1] 
    b_1s[j] = b_true[1]
    # X=randn(n,p)
    X1=X[:,1:p]
    g=rand(MvNormal(τ2*K2))
    # writedlm("./testdata/dataX-julia.csv",X)
    Y= logistic.(X1*b_true+g) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    # T, S = svdK(K;LOCO=false)
    # Xt, Xt₀, yt, init0 = initialization(Y,X,ones(n,1),T,S;tol=1e-4)     
    # res10 = susieGLMM(L,ones(p)/p,yt,Xt,Xt₀,S,init0;tol=1e-4)
    @time res10 =susieGLMM(1,ones(p)/p,Y,X1,ones(n,1),T,S) 
    res1=[res1;res10]
end


b̂ = [res1[j].α[1]*res1[j].ν[1] for j=1:B]
α̂ = [res1[j].α[1] for j=1:B]


using UnicodePlots
scatterplot(b_1s,b̂,xlabel= "True effects", ylabel="Posterior estimate")
scatterplot(b_1s,α̂, xlabel="True effects",ylabel="pip")