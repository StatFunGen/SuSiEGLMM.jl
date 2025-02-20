# SuSiEGLMM.jl
Julia package for sum of single effects regression with generalized linear mixed model

**This package was work in progress as an experiment, and is no longer maintained**


### Julia Installation

Julia can be downloaded from [Julia](https://julialang.org/downloads/) and is automatically recognized by JupyterLab if it has already been installed. Julia can also be called by simlink.

```bash
# Install Julia v1.7.2 (Feb 6, 2022)
wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz
tar -xvf julia-1.7.2-linux-x86_64.tar.gz

# Export Julia into PATH or you can add this line into your ~/.bashrc file
export export PATH="./julia-1.7.2/bin:$PATH"
```



Then by typing `julia` in the command line,  Julia can be invoked

```julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.2 (2022-02-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```



### Installing Julia packages for SuSiE_GLM

You first need to add a package manager `Pkg` and add any necessary packages via `Pkg`

```julia
using Pkg
Pkg.add(url="https://github.com/cumc/SuSiEGLMM.jl.git") # SuSiEGLMM package

# Other necessary packages
Pkg.add(["Statistics", "Distributions", "StatsBase", "Random", "LinearAlgebra", "DelimitedFiles", "Distributed", "GLM"])
Pkg.add("Plots")
```



### Distributed computing

Parallelization (`@distributed`) is performed at the chromosome level; that is, a set of SNPs in each chromosome is assigned to each worker (or process). One can generate workers up to the number of chromosomes. After that packages need to be loaded with `@everyhwere`, so that all workers can access the packages. Note that this distributed computing does not have to send all data to all workers; data are accessible on the main process only.

```julia
using Distributed 
addprocs(2) # for  sall data set case

using Statistics, Distributions, StatsBase, Random, LinearAlgebra
@everywhere using SuSiEGLMM
```



### SuSiE_GLM

Fine mapping for SuSiE-GLM is run by `fineQTL_glm`. For help, type `?fineQTL_glm` about the details.
