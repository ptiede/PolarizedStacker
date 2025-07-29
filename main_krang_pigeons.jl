const filedir = @__DIR__
@show filedir
@info "Using $(Threads.nthreads()) threads"


using Pkg; Pkg.activate(filedir)
# Pkg.add(url="https://github.com/ptiede/EHTModelStacker.jl")
# Pkg.add(url="https://github.com/ptiede/RobustAdaptiveMetropolisSampler.jl")
# Pkg.instantiate()
# Pkg.precompile()

Pkg.precompile()
include(joinpath(filedir, "stacker.jl"))
include(joinpath(filedir, "process_pigeons.jl"))
include(joinpath(filedir, "res_to_chainh5.jl"))





using Comonicon

"""
Runs the stacker on the list of dirs passed as a command line argument

# Arguments

- `1`: The path to the .h5 file with all the chain results
- `2`: The location of the file containing the prior information

# Options
- `-o, --output`: The output directory where the results will be saved. Default is the 
    directory containing the input file, i.e., the first argument.
- `--nrounds`: The number of rounds to run the sampling for. Default is 11.
- `--nchains`: The number of chains to run in parallel. Default is 24.

# Flags

- `-r, --restart`: A flag that says to restart the analysis from the ckpt file

"""
@main function main(d::String, p::String; output::String=dirname(d), restart::Bool=false, 
                    nrounds::Int=11, nchains::Int=24)
    @info "Reading chain $(d)"
    @info "Using prior file $(p)"
    process_pigeons(d, p, output; restart, nchains, nrounds)
    return nothing
end
