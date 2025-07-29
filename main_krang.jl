const filedir = @__DIR__
@show filedir
@info "Using $(Threads.nthreads()) threads"


using Pkg; Pkg.activate(filedir)
# Pkg.add(url="https://github.com/ptiede/EHTModelStacker.jl")
# Pkg.add(url="https://github.com/ptiede/RobustAdaptiveMetropolisSampler.jl")
# Pkg.instantiate()
# Pkg.precompile()


include(joinpath(filedir, "stacker.jl"))
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

# Flags

- `-r, --restart`: A flag that says to restart the analysis from the ckpt file

"""
@main function main(d::String, p::String; output::String=dirname(d), restart::Bool=false)
    @info "Reading chain $(d)"
    @info "Using prior file $(p)"
    process(d, p, output; restart)
    return nothing
end
