using Distributed


const filedir = @__DIR__
@show filedir

using Pkg; Pkg.activate(filedir)
Pkg.add(url="https://github.com/ptiede/EHTModelStacker.jl")
Pkg.add(url="https://github.com/ptiede/RobustAdaptiveMetropolisSampler.jl")
Pkg.instantiate()
Pkg.precompile()
