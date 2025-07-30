using EHTModelStacker
using DelimitedFiles
using Distributions
import HypercubeTransform as HC
using HypercubeTransform: ascube, dimension
using StatsBase
using JLD2
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

struct _SnapshotTransform{T, L}
    t::T
    l::L
end

function (st::_SnapshotTransform)(x)
    any(x->(x<(0)||x>1), x) && return -1e300
    return st.l(HC.transform(st.t, x))
end



function EHTModelStacker.SnapshotWeights(θ, mins, maxs, wrapped, batchsize; scheduler=:serial)
    μ, σ = θ.μ, θ.σ
    dists = Vector{WT}(undef, length(μ))
    dists .= make_marginal2.(μ,σ, mins, maxs, wrapped)
    #dists = EHTModelStacker.NormalFast.(μ, σ)
    transition = EHTModelStacker.MyProduct(dists)#EHTModelStacker.MvNormalFast(μ, Σ.^2)
    prior = EHTModelStacker.MvUniform(mins, maxs)
    return SnapshotWeights(transition, prior, batchsize, scheduler)
end

struct BatchStackerLklhd{C, T, B}
    chain::C
    min::T
    max::T
    wrapped::B
    batchsize::Int
    scheduler::Symbol
end




function (l::BatchStackerLklhd)(θ)
    ws = SnapshotWeights(θ, l.min, l.max, l.wrapped, l.batchsize; scheduler = l.scheduler)
    lapprox = lpdf(ws, l.chain)
    return lapprox
end

function create_lklhd(cfile, prior_file; nbatch=1000, scheduler=:serial)
    chain = ChainH5(cfile; nsamples=5000)
    prior = read_prior_table(prior_file,)
    mins, maxs, wrapped, restrict = extract_prior(prior, chain.names)
    l = BatchStackerLklhd(chain, mins, maxs, wrapped, nbatch, scheduler)
    σl = map(mins, maxs, restrict) do ml, mu, r
        if r
            return (mu - ml)/20
        else
            return (mu - ml)/2
        end
    end
    prior = (μ = Product(Uniform.(mins, maxs)), σ = Product(Exponential.(σl)))

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
    restrict = Bool[]

    for n in names
        i = findall(==(n), df.name)
        @assert length(i) == 1
        push!(min, df.min[first(i)])
        push!(max, df.max[first(i)])
        push!(wrap, df.angular[first(i)])
        push!(restrict, df.restrict[first(i)])
    end

    return min, max, wrap, restrict
end



function read_prior_table(file)
    header = open(file, "r") do io
        String.(split(readline(io), ","))
    end
    header[1] = split(header[1])[2]
    header[2] = split(header[2])[2]
    header[3] = split(header[3])[1]
    if length(header) > 6
        header[end-1] = split(header[end-1])[1]
    end
    header[end] = split(header[end])[1]
    header .= rstrip.(lstrip.(header))
    df = CSV.read(file, DataFrame, header=header, skipto=2, delim=' ')
    df = DataFrame(df)
    if !hasproperty(df, :restrict)
        df.restrict = fill("n", nrow(df))
    end
    df.angular = map(x->ifelse(x=="n", false, true), df.angular)
    df.restrict = map(x->ifelse(x=="n", false, true), df.restrict)
    df.name = Symbol.(df.name)
    return df
end
