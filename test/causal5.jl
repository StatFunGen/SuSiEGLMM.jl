
#test with arabidopsis data RIL: does not work.

labels=readdlm(homedir()*"/fmulti-lmm/processedData/marlabels_agren.csv",',';skipstart=1);
impgen = readdlm(homedir()*"/GIT/fmulti-lmm/processedData/fullrank_imput.csv",',';skipstart=1);
impgen[impgen.==1.0].=0.0;impgen[impgen.==2.0].=1.0;

pheno =readdlm(homedir()*"/GIT/fmulti-lmm/processedData/pheno2013_imp.csv",',';header=true);
pheno=pheno[1][:,2:end-1];
y=pheno[:,1]
#make binary
y[y.>mean(y)].=1.0
y[y.<=mean(y)].=0.0

using FlxQTL

K=kinshipMan(convert(Matrix{Float64},impgen'))


using SuSiEGLMM
G=GenoInfo(labels[:,1],labels[:,2],labels[:,3])

T,S=svdK(K;LOCO=false)
idx=findall(G.chr.==2)

@time est0=fineMapping_GLMM(G,y,impgen,ones(length(y),1),T,S;L=10,LOCO=false,tol=1e-5)
@time Xt, Ct, yt, init0= initialization(y,impgen,ones(length(y),1),T,S;tol=1e-5)

y1=y[5:end]
X=impgen[5:end,:]
K1=kinshipMan(convert(Matrix{Float64},X'))
T1,S1=svdK(K1;LOCO=false)

@time est0=fineMapping_GLMM(G,y1,X,ones(length(y1),1),T1,S1;L=10,LOCO=false,tol=1e-5)
@time Xt, Ct, yt, init= initialization(y1,X,ones(length(y1),1),T1,S1;tol=1e-5)

Xt, Ct, yt = rotate(y,X,C,T[:,:,1]) 
@time res0=susieGLMM(L,Π,yt,Xt,Ct,S[:,1];tol=1e-4)