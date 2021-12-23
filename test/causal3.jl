@everywhere using Statistics, Distributions, StatsBase, Random, LinearAlgebra, DelimitedFiles, Distributed

@everywhere using Revise
@everywhere using Pkg
@everywhere Pkg.activate(homedir()*"/GIT/SuSiEGLMM.jl")
@everywhere using SuSiEGLMM


### causal3:pop
@time info=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal3/fam_folder/ascertained_fam_12_10.bim");
@time geno=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal3/fam_folder/ascertained_fam_genotype_12_10.txt";header=true);
@time pheno=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal3/fam_folder/ascertained_fam_phenotype_12_10.txt";header=true); #518 x 4000 snps
snps =[13,187,1977]

# covariate: sex
# C = pheno[1][:,end-1]
# C[C.==1].=0.0
# C[C.==2].=1.0

#last col: trait
# y=convert(Vector{Float64},data[1][:,end])
y=pheno[1][:,end]

# kinship
K=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal3/fam_folder/fam_grm_ped.txt") #518
K0=readdlm(homedir()*"/GIT/SuSiEGLMM.jl/testdata/causal3/fam_folder/fam_grm.txt");
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
@time Tstat, pval= SuSiEGLMM.scoreTest(K,G,y,X;LOCO=false);

    T, S = svdK(K;LOCO=false)
    Xt, Xt₀, yt,init00= initialization(y,X,ones(n,1),T,S;tol=1e-4)
    T0= computeT(init00,yt,Xt₀,Xt)


@time Tstat1, pval1= SuSiEGLMM.scoreTest(K1[:,:,2],G,y,X;LOCO=false)

#susie-glm
@time est0= fineQTL_glm(G,y,X;tol=1e-4,L=3)
# @time est1= fineQTL_glm(G,y,X,C;tol=1e-4)

a1=sum(est0[1].α,dims=2)
a2=sum(est0[2].α,dims=2)

#susie-glmm: verion 1
# T, S= svdK(K;LOCO=false)
# @time est2 = fineQTL_glmm(G,y,X,ones(n,1),T,S;LOCO=false,L=3)
# T1, S1= svdK(K1)
# @time est3 = fineQTL_glmm(G,y,X,ones(n,1),T1,S1;LOCO=true,L=3)
#susie-glmm: version 2

@time est4 =fineQTL_glmm(K,G,y,X;LOCO=false,tol=1e-4,L=3)
a3=sum(est4[1].α,dims=2)
a4=sum(est4[2].α,dims=2)
@time est5 =fineQTL_glmm(K1,G,y,X;tol=1e-4,L=3)


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


 L=3; B=100;τ2=0.8;
K0=K0+ (abs(eigmin(K0))+0.001)*I
#GLM
b_true=zeros(p);
b_1s=zeros(B);b_2s=zeros(B); b_3s=zeros(B);

res=[];Tm=zeros(B);

for j = 1:B
    b_true[1]= randn(1)[1] 
    b_true[2]=randn(1)[1]
    b_true[3]=randn(1)[1]
    b_1s[j] = b_true[1]
    b_2s[j] = b_true[2]
    b_3s[j] = b_true[3]
    # X=randn(n,p)
    g=rand(MvNormal(τ2*K))
    Y= logistic.(X*b_true+g) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    # res0= susieGLM(L, ones(p)/p,Y,X,ones(n,1);tol=1e-4) 
  t0=@elapsed  res0= fineQTL_glm(G,Y,X;tol=1e-4)
    res=[res;res0]; Tm[j]=t0
end

# writedlm("./test/glm-cau1.txt",[b̂ α̂ b_1s])
println("min, median, max times for glm are $(minimum(Tm)), $(median(Tm)),$(maximum(Tm)).")
#min, median, max times for glm are 3.19, 4.151,5.13.

b̂1=zeros(B);b̂2=zeros(B);b̂3=zeros(B);
for j=1:B
A = sum(res[2j-1].α.*res[2j-1].ν,dims=2)[:,1]
b̂1[j]=A[1]
b̂2[j]=A[2]
b̂3[j]=A[3]
end
α̂1 = [maximum(res[2j-1].α[1,:]) for j=1:B]
α̂2=[maximum(res[2j-1].α[2,:]) for j=1:B]
α̂3=[maximum(res[2j-1].α[3,:]) for j=1:B]

writedlm("./test/susie-glm-3causal-fam.txt",[b_1s b̂1 α̂1 b_2s b̂2 α̂2 b_3s b̂3 α̂3 ])

using UnicodePlots
scatterplot(b_1s,b̂1,title="susie-glm",xlabel= "True effects", ylabel="Posterior estimate1")
scatterplot(b_1s,α̂1, xlabel="True effects",ylabel="pip1")
scatterplot(b_2s,b̂2,xlabel= "True effects", ylabel="Posterior estimate2")
scatterplot(b_2s,α̂2, xlabel="True effects",ylabel="pip2")
scatterplot(b_3s,b̂3,xlabel= "True effects", ylabel="Posterior estimate3")
scatterplot(b_3s,α̂3, xlabel="True effects",ylabel="pip3")



#GLMM :scroe test

# n=100; p=10; 

# K2=Symmetric(K0)
b_true=zeros(p);
# b_1s=zeros(B); init0=[]; 
Ps=zeros(p,B); Tscore=zeros(B);tt=zeros(p,B);


# F =cholesky(K2)
# f = svd(F.U)
# T,S = f.Vt, f.S.^2
# T,S = svdK(K0;LOCO=false)
# H=svd(K2);
for j = 1:B

    b_true[1]= b_1s[j]
    b_true[2]= b_2s[j]
    b_true[3]= b_3s[j]
    # X=randn(n,p)
    g=rand(MvNormal(τ2*K))
    # writedlm("./testdata/dataX-julia.csv",X)
    Y= logistic.(X*b_true+g) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    
    # Xt, Xt₀, yt,init00= initialization(Y,X1,ones(n,1),T,S;tol=1e-4)
    # T0= computeT(init00,yt,Xt₀,Xt)
    # init0=[init0;init00]
    # Ts[:,j]=T0
   t0= @elapsed Ts,P0= scoreTest(K,G,Y,X;LOCO=false);
   Ps[:,j]=P0; Tscore[j]=t0; tt[:,j]=Ts
end

# writedlm("./test/glmm-scoretest.txt",Ps)
println("min, median, max times for score test are $(minimum(Tscore)), $(median(Tscore)),$(maximum(Tscore)).")
#for thoeretical K
histogram(Ps[1,:],bins=20) #80% at α=0.05
histogram(Ps[2,:],bins=20) # 82%
histogram(Ps[3,:],bins=20) #65%

#susie-GLMM


b_true=zeros(p);
# b_1s=zeros(B); 
res1=[]; Tmm=zeros(B)
#  K=Matrix(1.0I,n,n);

for j = 1:B

    # b_true[1]= randn(1)[1] 
    b_true[1]= b_1s[j]
    b_true[2]= b_2s[j]
    b_true[3]= b_3s[j]
    # X=randn(n,p)

    g=rand(MvNormal(τ2*K))
    # writedlm("./testdata/dataX-julia.csv",X)
    Y= logistic.(X*b_true+g) .>rand(n) #generating binary outcome
    Y=convert(Vector{Float64},Y)
    # writedlm("./testdata/dataY-julia.csv",Y)
    # T, S = svdK(K;LOCO=false)
    # Xt, Xt₀, yt, init0 = initialization(Y,X,ones(n,1),T,S;tol=1e-4)     
    # res10 = susieGLMM(L,ones(p)/p,yt,Xt,Xt₀,S,init0;tol=1e-4)
    # @time res10 =susieGLMM(1,ones(p)/p,Y,X1,ones(n,1),T,S) 
   t0= @elapsed res10=fineQTL_glmm(K,G,Y,X;LOCO=false)
    res1=[res1;res10]; Tmm[j]=t0
end


# writedlm("./test/glmm-score-susie.txt",[b̂ α̂ b_1s])
println("min, median, max times for score test are $(minimum(Tmm)), $(median(Tmm)),$(maximum(Tmm)).")

b̂1=zeros(B);b̂2=zeros(B);b̂3=zeros(B);
for j=1:B
A = sum(res1[2j-1].α.*res1[2j-1].ν,dims=2)[:,1]
b̂1[j]=A[1]
b̂2[j]=A[2]
b̂3[j]=A[3]
end
α̂1 = [maximum(res1[2j-1].α[1,:]) for j=1:B]
α̂2=[maximum(res1[2j-1].α[2,:]) for j=1:B]
α̂3=[maximum(res1[2j-1].α[3,:]) for j=1:B]



using UnicodePlots
scatterplot(b_1s,b̂1,title="susie-glmm",xlabel= "True effects", ylabel="Posterior estimate1")
scatterplot(b_1s,α̂1, xlabel="True effects",ylabel="pip1")
scatterplot(b_2s,b̂2,xlabel= "True effects", ylabel="Posterior estimate2")
scatterplot(b_2s,α̂2, xlabel="True effects",ylabel="pip2")
scatterplot(b_3s,b̂3,xlabel= "True effects", ylabel="Posterior estimate3")
scatterplot(b_3s,α̂3, xlabel="True effects",ylabel="pip3")

