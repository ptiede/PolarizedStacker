using Distributed

@everywhere begin
    const filedir = @__DIR__
    @show filedir
end


using Pkg; Pkg.activate(filedir)
Pkg.add(url="https://github.com/ptiede/EHTModelStacker.jl")
Pkg.add(url="https://github.com/ptiede/RobustAdaptiveMetropolisSampler.jl")
Pkg.instantiate()
Pkg.precompile()

@everywhere begin
    using Pkg;Pkg.activate(filedir)
end


include(joinpath(filedir, "stacker.jl"))
include(joinpath(filedir, "res_to_chainh5.jl"))



@everywhere include(joinpath(filedir, "stacker.jl"))
@everywhere include(joinpath(filedir, "res_to_chainh5.jl"))


function make_h5(dir)
    make_hdf5_chain(dirname(dir), joinpath(dir, "stacker_chain.h5"))
end

function loaddir(file)
    string.(reshape(readdlm(file), :))
end


using Comonicon

"""
Runs the stacker on the list of dirs passed as a command line argument

# Arguments

- `d`: The file containing the list of directories
- `p`: The location of the file containing the prior information

# Flags

- `-r, --restart`: A flag that says to restart the analysis from the ckpt file

"""
@main function main(d::String, p::String; restart::Bool=false)
    dd = loaddir(d)
    dirs = mkpath.(joinpath.(dd, "StackedResults"))
    @info dirs
    make_h5.(dirs)
    od = joinpath.(dirs, "stacker_chain.h5")
    pmap(od) do d
        process(d, p, dirname(d); restart)
    end
    return nothing
end
