
using Distributed 
addprocs(95)
@everywhere using Distributed,DelimitedFiles, LinearAlgebra, Statistics, StatsBase, SuSiEGLMM

Seed(20)

cidx = collect(1:4); Stime=[]; Qtime=[]
for j= 2:1000
  # data preparation
  geno= readdlm(string(@__DIR__,"/sim",j,"/fam_sample_",j,".csv"),'\t';skipstart=1)[:,6:end]; # read genotype 
  info = readdlm(string(@__DIR__,"/sim",j,"/fam_drop_chr4.bim"))  # get chr idx
  
     for l=axes(geno,2) #imputation
        idx1 = findall(geno[:,l].=="NA")
         geno[idx1,l].= missing
         geno[idx1,l] .= mean(skipmissing(geno[:,l]))
     end
     geno = convert(Matrix{Float64},geno)
     trait =geno[:,end]
     geno = geno[:,1:end-1]
     geno = (geno.-mean(geno,dims=1))./std(geno,dims=1)
 
     n,p =axes(geno); K= zeros(n,n,length(cidx)) # read grms
     Threads.@threads for i in eachindex(cidx)
      K[:,:,i] = readdlm(string(@__DIR__,"/sim",j,"/output/fam_drop_chr",i,".cXX.txt"))
      K[:,:,i] =Symmetric(K[:,:,i])
      end
      
    #score test
    ts =@elapsed begin
         STEST= @distributed (vcat)  for l =1:3
         idx=findall(info[:,1].== cidx[l])
         tstats, pvals, scores, svars = scoreTest(K[:,:,l],trait,geno[:,idx])
         [tstats pvals scores svars]
        end
       idx4=findall(info[:,1].==cidx[end-1])[end]+1
       tstats, pvals, scores, svars = scoreTest(K[:,:,end],trait,geno[:,idx4:end])  
      end
       writedlm(string(@__DIR__,"/sim",j,"/fam_tstat_pval_scr_var.txt"),[STEST;tstats pvals scores svars])
       Stime=[Stime;ts]
       T, S = svdK(K)
     # full model estimation
     B̄=zeros(p);Bvar=zeros(p) 
     ts1=@elapsed begin
       for l=1:3
         idx1=findall(info[:,1].== cidx[l])
         b̂,est0 = scan1SNP(trait,geno[:,idx1],ones(n,1),T[:,:,l],S[:,l])  
         bv= @distributed (vcat)  for t in eachindex(idx1)
                      est0[t].σ1
               end
          B̄[idx1] = b̂; Bvar[idx1]=bv
       end

      #for chr4
        b̂,est0 = scan1SNP(trait,geno[:,idx4:end],ones(n,1),T[:,:,end],S[:,end])
        bv= @distributed (vcat) for t =1:length(b̂)
                 est0[t].σ1
            end
        B̄[idx4:end]=b̂; Bvar[idx4:end]=bv
     end
       writedlm(string(@__DIR__,"/sim",j,"/fam_tstat_bhat_postvar.txt"),[B̄ Bvar])
       Qtime=[Qtime;ts1]
end
writedlm(string(@__DIR__,"/fam_min_median_max_time_score_qtl.txt"),[minimum(Stime) median(Stime) maximum(Stime);minimum(Qtime) median(Qtime) maximum(Qtime)])