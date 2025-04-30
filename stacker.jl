using EHTModelStacker
using DelimitedFiles
using Distributions
using ParameterHandling
using HypercubeTransform
import HypercubeTransform as HC
using StatsBase
using JLD2
using BlackBoxOptim
using RobustAdaptiveMetropolisSampler
using CSV, DataFrames
using TypedTables
using VLBIImagePriors
using LogDensityProblems

const WT = Union{Truncated{EHTModelStacker.NormalFast{Float64}, Continuous, Float64, Float64, Float64}, DiagonalVonMises{Float64, Float64, Float64}}

struct SnapshotTarget{F,T}
    f::F
    ndim::T
end

(f::SnapshotTarget)(x) = f.f(x)

LogDensityProblems.logdensity(d::SnapshotTarget, x) = d.f(x)
LogDensityProblems.dimension(d::SnapshotTarget) = d.ndim
LogDensityProblems.capabilities(::Type{<:SnapshotTarget}) = LogDensityProblems.LogDensityOrder{0}()

function make_marginal2(μ::T, σ::T, min, max, wrap)::WT where {T}
    μT,σT,minT,maxT = promote(μ, σ, min, max)
    !wrap && return truncated(EHTModelStacker.NormalFast(μT, σT), minT, maxT)::WT
    return DiagonalVonMises(μT, inv(σT^2))::WT
end


function EHTModelStacker.SnapshotWeights(θ, mins, maxs, wrapped, batchsize)
    μ, σ = θ.μ, θ.σ
    dists = Vector{WT}(undef, length(μ))
    dists .= make_marginal2.(μ,σ, mins, maxs, wrapped)
    #dists = EHTModelStacker.NormalFast.(μ, σ)
    transition = EHTModelStacker.MyProduct(dists)#EHTModelStacker.MvNormalFast(μ, Σ.^2)
    prior = EHTModelStacker.MvUniform(mins, maxs)
    return SnapshotWeights(transition, prior, batchsize)
end

struct BatchStackerLklhd{C, T, B}
    chain::C
    min::T
    max::T
    wrapped::B
    batchsize::Int
end




function (l::BatchStackerLklhd)(θ)
    ws = SnapshotWeights(θ, l.min, l.max, l.wrapped, l.batchsize)
    lapprox = lpdf(ws, l.chain)
    return lapprox
end

function create_lklhd(cfile, prior_file; nbatch=1000)
    chain = ChainH5(cfile; nsamples=1024)
    prior = read_prior_table(prior_file,)
    mins, maxs, wrapped = extract_prior(prior, chain.names)
    l = BatchStackerLklhd(chain, mins, maxs, wrapped, nbatch)
    prior = (μ = Product(Uniform.(mins, maxs)), σ = Product(Uniform.(0.0, 10*(maxs-mins))))

    return l, prior, keys(chain)
end

function _mkdf(echain, keys)
    df = DataFrame()
    for (i,k) in enumerate(keys)
        insertcols!(df, Symbol("mean_"*String(k))=>getindex.(echain.μ, i))
    end
    for (i,k) in enumerate(keys)
        insertcols!(df, Symbol("stdd_"*String(k))=>getindex.(echain.σ, i))
    end
    df
end


function readckpt(ckptfile)
    return load(ckptfile)
end

function saveckpt(ckptfile, smplr, tv, logp, keys)
    return save(ckptfile, Dict("smplr"=>smplr,
                               "tv"=>tv,
                               "logp"=>logp,
                               "keys"=>keys))
end


function write_results(file, tv, logp, keys; stride=1)
    df = _mkdf(tv, keys)
    df.logp = logp
    df[1:stride:end,:]|>CSV.write(file)
end

function extract_prior(df, names)
    min  = Float64[]
    max  = Float64[]
    wrap = Bool[]

    for n in names
        i = findall(==(n), df.name)
        @assert length(i) == 1
        push!(min, df.min[first(i)])
        push!(max, df.max[first(i)])
        push!(wrap, df.angular[first(i)])
    end

    return min, max, wrap
end


function read_prior_table(file)
    header = open(file, "r") do io
        String.(split(readline(io), ","))
    end
    header[1] = split(header[1])[2]
    header[2] = split(header[2])[2]
    header[3] = split(header[3])[1]
    header[end] = split(header[end])[1]
    header .= rstrip.(lstrip.(header))
    df = CSV.File(file, header=header, skipto=2, delim=' ')
    # df[!,:name] = Symbol.(df.name)
    df = DataFrame(df)
    df.angular = map(x->ifelse(x=="n", false, true), df.angular)
    df.name = Symbol.(df.name)
    return df
end


function best_image(l, t; ntrials=10, maxiters=70_000)
    ndim = dimension(t)
    fopt(x) = -l(x)
    sols = map(1:ntrials) do i
        sol = bboptimize(fopt; SearchRange = (0.01, 0.99), NumDimensions = ndim, MaxFuncEvals=maxiters, TraceMode=:silent)
        xopt = best_candidate(sol)
        mini = best_fitness(sol)
        @info "Best image $i/$(ntrials) done: minimum: $(mini)"
        return xopt
    end
    lmaps = l.(sols)
    inds = sortperm(lmaps, rev=true)
    return sols[inds], lmaps[inds]
end


function process(
        cfile, prior_file, outdir;
        ntrials = 10, maxiters=70_000,
        restart=false, ckpt_stride=50_000,
        stride=10, nbatch=1_000, nsteps=2_000_000
                )

    mkpath(outdir)
    ckptfile = joinpath(outdir, replace(basename(cfile), ".h5"=>"_ckpt.jld2"))
    chainfile = joinpath(outdir, replace(basename(cfile), ".h5"=>"_ha_trunc.csv"))
    nbatches = nsteps÷ckpt_stride

    l, prior, k = create_lklhd(cfile, prior_file; nbatch=nbatch)



    t = ascube(prior)
    ff = let t=t, l=l
        x-> begin
            any(x->(x<(0)||x>1), x) && return -1e300
            l(HypercubeTransform.transform(t, x))
        end
    end
    f = SnapshotTarget(ff, dimension(t))

    if !restart || !isfile(ckptfile)

        xopts, l0s = best_image(f, t; ntrials, maxiters)
        println("After 2 runs the estimated logdensity of the MAP estimate are $(first.(l0s))")
        p0 = first(xopts)

        if ckpt_stride > nsteps
            println("Checkpoint stide > nsteps resetting")
            ckpt_stride = nsteps
        end
        Minit = 0.001#[0.01*(maxs .- mins)..., 0.001*(maxs .- mins)...]
        smplr = RAM(p0, Minit)
        chain = @timed sample(f, smplr, ckpt_stride; show_progress=false, output_log_probability_x=true)
        c = [HC.transform(t, chain.value.chain[i,:]) for i in axes(chain.value.chain, 1)]
        tv    = Table(c)
        state = chain.value.state
        logp  = chain.value.log_probabilities_x
        # Starting checkpoint
        saveckpt(ckptfile, state, tv, logp, k)
        println("Done batch 1 this took $(chain.time) seconds")
        println("I am guessing this entire run will take $(chain.time*nbatches/3600.0) hours to finish")
        write_results(chainfile, tv, logp, k)
        nstart=2
    else
        ckpt = readckpt(ckptfile)
        state = ckpt["smplr"]
        tv    = ckpt["tv"]
        logp  = ckpt["logp"]
        @show typeof(state)
        nstart = length(tv)÷ckpt_stride + 1
        @info "Reading in checkpoint file $(ckptfile)"
        @info "According to checkpoint I am on batch $(nstart)"
    end

    for i in nstart:nbatches
        println("On batch $i/$nbatches")
        # We are early so lets speed up the learning
        if i < nbatches÷2
            state = RAM(state.x, state.M.mat; step=1)
        end
        tv_b,_,state_b,logp_b = sample(f, state, ckpt_stride; show_progress=false, output_log_probability_x=true)

        #extend results and set up next run
        c_b = [HC.transform(t, tv_b[i,:]) for i in axes(tv_b, 1)]
        tv    = vcat(tv, Table(c_b))
        logp  = vcat(logp, logp_b)
        state = state_b
        println("Writing checkpoint")
        saveckpt(ckptfile, state, tv, logp, k)
        write_results(chainfile, tv, logp, k, stride=stride)
    end

end
