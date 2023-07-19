using EHTModelStacker
using DelimitedFiles
using Distributions
using ParameterHandling
using HypercubeTransform
import HypercubeTransform as HC
using StatsBase
using JLD2
using BlackBoxOptim
using Pigeons
using CSV, DataFrames
using SplittableRandoms: SplittableRandom
using TypedTables
using Random

function make_marginal(μ::T, σ::T, min, max, wrap) where {T}
    μT,σT,minT,maxT = promote(μ, σ, min, max)
    !wrap && return truncated(EHTModelStacker.NormalFast(μT, σT), minT, maxT)
end


function EHTModelStacker.SnapshotWeights(θ, mins, maxs, wrapped, batchsize)
    μ, σ = θ.μ, θ.σ
    dists = make_marginal.(μ,σ, mins, maxs, wrapped)
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

struct StackerPotential{F}
    f::F
    dim::Int
end

struct FlatPrior{D}
    dims::D
end
(::FlatPrior)(x) = zero(eltype(x))
(p::StackerPotential)(x) = p.f(x)
Pigeons.create_reference_log_potential(p::StackerPotential, ::Inputs) = FlatPrior(p.dim)
Pigeons.sample_iid!(::FlatPrior, replica) = rand!(replica.rng, replica.state)

Pigeons.create_state_initializer(my_potential::StackerPotential, ::Inputs) = my_potential
Pigeons.initialization(p::StackerPotential, rng::SplittableRandom, ::Int) = rand(rng, p.dim)

function process(
        cfile, prior_file, outdir;
        nrounds = 11, nchains=24,
        restart=false,
        nbatch=1_000,
                )

    mkpath(outdir)
    chainfile = joinpath(outdir, replace(basename(cfile), ".h5"=>"_ha_trunc.csv"))
    @info "Reading in $(cfile)"
    @info "Reading in $(prior_file)"
    @info "Output to $(chainfile)"

    if !restart
        l, prior, k = create_lklhd(cfile, prior_file; nbatch=nbatch)
        t = ascube(prior)

        f = let t=t, pr=prior
            x-> begin
                any(x->(x<(0)||x>1), x) && return -1e300
                lp = l(HypercubeTransform.transform(t, x))
                return lp
            end
        end

        pot = StackerPotential(f, dimension(t))

        @info "Starting Sampling"
        pt = pigeons(target = pot,
                     n_rounds = nrounds,
                     recorder_builders=[Pigeons.traces, Pigeons.log_sum_ratio, Pigeons.round_trip],
                     multithreaded=true, checkpoint=true,
                     n_chains=nchains,
                     explorer = Pigeons.SliceSampler(; w = 0.05),
                     seed = 2
                    )

        trace = get_sample(pt, nchains)
        tv = HypercubeTransform.transform.(Ref(t), trace) |> Table
        logp  = f.(trace)
        write_results(chainfile, tv, logp, k)
    else
        @info "Resuming from checkpoint"
        pt_from_checkpoint = PT("results/latest")
        pt = pigeons(pt_from_checkpoint)
        trace = get_sample(pt, nchains)
        tv = transform.(Ref(t), trace) |> Table
        logp  = l.(eachrow(tv))
        write_results(chainfile, tv, logp, k)
    end
end
