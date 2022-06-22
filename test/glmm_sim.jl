
@everywhere using DelimitedFiles, LinearAlgebra, SuSiEGLMM

cidx = collect(1:4)
for j= 1:1000
  # data preparation
  geno= readdlm(string(@__DIR__,"/sim",j,"/fam_sample_",j,".csv"),'\t';skipstart=1)[:,6:end]; # read genotype randomizeData
  info = readdlm(string(@__DIR__,"/sim",j,"/fam_drop_chr4.bim"))  # get chr idx
   
     for l=axes(geno,2) #imputation
        idx = findall(geno[:,l].=="NA")
         geno[idx,l].= missing
         geno[idx,l] .= mean(skipmissing(geno[:,l]))
     end
     trait =geno[:,end]
     geno = geno[:,1:end-1]
 
    #score test
STEST= @distributed (vcat)  for l =1:3
         idx=findall(info[:,1].== cidx[l])
         tstats, pvals, scores, svars = scoreTest(K[:,:,l],trait,geno[:,idx])
         [tstats pvals scores svars]
      end
       idx4=findall(info[:,1].==cidx[end-1])[end]+1
       tstats, pvals, scores, svars = scoreTest(K[:,:,end],trait,geno[:,idx4:end])  
       writedlm(string(@__DIR__,"/sim",j,"/fam_tstat_pval_scr_var.txt"),[STEST;tstats pvals scores svars])

       n,p =axes(geno); K= zeros(n,n,length(cidx)) # read grms
    Threads.@threads for i in eachindex(cidx)
     K[:,:,i] = readdlm(string(@__DIR__,"/sim",j,"/ouput/fam_drop_chr",i,".cXX.txt"))
     K[:,:,i] =Symmetric(K[:,:,i])
     end
     T, S = svdK(K)
     # full model estimation
     B̄=zeros(p);Bvar=zeros(p) 
     for l=1:3
       idx=findall(info[:,1].== cidx[l])
       b̂,est0 = scan1SNP(trait,geno[:,idx],ones(n,1),T[:,:,l],S[:,l])  
       bv= @distributed (vcat)  for t in eachindex(idx)
                      est0[t].σ1
         end
       B̄[idx] = b̄; Bvar[idx]=bv
        end
      
      #for chr4
        b̂,est0 = scan1SNP(trait,geno[:,idx4:end],ones(n,1),T[:,:,end],S[:,end])
        bv= @distributed (vcat) for t in eachindex(idx)
                 est0[t].σ1
            end
      B̄[idx4:end]=b̂; Bvar[idx4:end]=bv
       writedlm(string(@__DIR__,"/sim",j,"/fam_tstat_bhat_postvar.txt"),[B̄ Bvar])
end