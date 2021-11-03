#code debugging

using Statistics, Distributions, StatsBase, Random, LinearAlgebr, DelimitedFiles

info=readdlm("./testdata/snp_info.bim")
data=readdlm("./testdata/pop_518ids_4000snps.txt";skipstart=1)[:,5:end]; #518 x 4000 snps
#data1=readdlm("../testdata/fam_100fams_4000snps.txt";header=true)

#filter out NA


#5th col :sex

#last col: trait

# kinship
K=readdlm("./testdata/pop_518fams_4000snps.cXX.txt") #518