using Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

using PythonCall
using Printf
using Tables
using CSV
using DataFrames
using Comonicon
using EHTModelStacker
using JLD2
using NamedTupleTools
ehtim = pyimport("ehtim")
include("quick_analysis.jl")

include("stacker.jl")

function construct_meanstd(df, labels::Vector{String})
    means = DataFrame([Pair(l,df[!, "mean_"*l]) for l in labels])
    std = DataFrame([Pair(l, df[!, "stdd_"*l]) for l in labels])
    return means, std
end

function construct_meanstd(df)
    na = names(df)
    single = split.(na[findall(startswith("mean"), na)], "_") .|> last .|> String
    return construct_meanstd(df, single)
end


function extract_cp_params(x::DataFrameRow)
    d = Tables.getcolumn(x, :d)
    x0 = 0.0
    y0 = 0.0
    ilistre = findall(x->occursin(r"beta\d-re",x), names(x))
    ilistim = findall(x->occursin(r"beta\d-im",x), names(x))
    beta_list_re = Tables.getcolumn.(Ref(x), ilistre)
    beta_list_im = Tables.getcolumn.(Ref(x), ilistim)
    beta_list = complex.(beta_list_re, beta_list_im)

    ilistcp_re = findall(x->occursin(r"betacpol\d-re",x), names(x))
    ilistcp_im = findall(x->occursin(r"betacpol\d-im",x), names(x))
    beta_list_pol_re = Tables.getcolumn.(Ref(x), ilistcp_re)
    beta_list_pol_im = Tables.getcolumn.(Ref(x), ilistcp_im)
    beta_list_pol = complex.(beta_list_pol_re, beta_list_pol_im)
    # There is a constant offset
    prepend!(beta_list_pol, [-0.0114])

    alpha = Tables.getcolumn(x, :alpha)
    return (; d, x0, y0, beta_list, beta_list_pol, alpha)
end


function extract_lp_params(x::DataFrameRow)
    d = Tables.getcolumn(x, :d)
    x0 = 0.0
    y0 = 0.0

    # Sort first to ensure that the names are ordered properly
    ind = sortperm(names(x))
    ilistre = findall(x->occursin(r"beta\d-re",x), names(x)[ind])
    ilistim = findall(x->occursin(r"beta\d-im",x), names(x)[ind])
    beta_list_re = Tables.getcolumn.(Ref(x[ind]), ilistre)
    beta_list_im = Tables.getcolumn.(Ref(x[ind]), ilistim)
    beta_list = complex.(beta_list_re, beta_list_im)


    ilistre_lp = findall(x->occursin(r"betapol-*\d-re",x), names(x)[ind])
    ilistim_lp = findall(x->occursin(r"betapol-*\d-im",x), names(x)[ind])
    betapol_list_re = Tables.getcolumn.(Ref(x[ind]), ilistre_lp)
    betapol_list_im = Tables.getcolumn.(Ref(x[ind]), ilistim_lp)
    betapol_list_usrt = complex.(betapol_list_re, betapol_list_im)
    nmax = (length(betapol_list_im)-1)÷2
    beta_list_pol = zero(betapol_list_usrt)
    beta_list_pol[begin:nmax] .= @view betapol_list_usrt[nmax:-1:begin]
    beta_list_pol[nmax+1] = betapol_list_usrt[nmax+1]
    beta_list_pol[nmax+2:end] = betapol_list_usrt[nmax+2:end]

    alpha = Tables.getcolumn(x, :alpha)
    return (; d, x0, y0, beta_list, beta_list_pol, alpha)
end

function make_image(r, cp)
    m = ehtim.model.Model()
    p  = if cp
            extract_cp_params(r)
        else
            extract_lp_params(r)
        end
    m = m.add_thick_mring(1.0, p.d*ehtim.RADPERUAS, p.alpha*ehtim.RADPERUAS, 0.0*ehtim.RADPERUAS, 0.0*ehtim.RADPERUAS, beta_list=p.beta_list, beta_list_pol=p.beta_list_pol)
    img = m.make_image(100.0*ehtim.RADPERUAS, 64)
    return img
end

function make_samples(m, s, mins, maxs, wrapped; nsamples=100)
    name  = Symbol.(names(m))
    dists = make_marginal.(Tuple(m), Tuple(s), mins, maxs, wrapped)
    return DataFrame([NamedTuple{Tuple(name)}(Tuple(rand.(dists))) for _ in 1:nsamples])
end


function image_stats_table(outdir, ms, ss, prior_file, cp; nsamples=200)
    mkpath(outdir)
    prior = read_prior_table(prior_file)
    mins, maxs, wrapped = extract_prior(prior, Symbol.(names(ms)))
    i = 0
    stats = []
    for i in 1:nrow(ms)
        @info "$(i)/$(nrow(ms))"
        samples = make_samples(ms[i,:], ss[i,:], mins, maxs, wrapped; nsamples)
        st = map(eachrow(samples)) do s
            img = make_image(s, cp)
            s = quick_analysis(img)
            d = pyconvert(Dict, s)
            bt = pyconvert(Vector, d["betas"])
            bn = pyconvert(Vector, d["beta_ms"])
            b_nt = NamedTuple{Tuple([Symbol("beta_$i") for i in bn])}(Tuple(bt))
            return merge((m_net = d["m_net"], m_avg = d["m_avg"], v_net = d["v_net"], v_avg = d["v_avg"]), b_nt)
        end
        dd = DataFrame(st)
        push!(stats, NamedTuple{Tuple(Symbol.(names(dd)))}(Tuple(map(mean, eachcol(dd)))))
    end
    CSV.write(joinpath(outdir, "image_stats.csv"), DataFrame(stats))
    return DataFrame(stats)
end

function postprocess(fstack, outdir, prior, cirpol=true; burnfrac=0.1, nsamples=100)
    df = CSV.File(fstack) |> DataFrame
    nstart = floor(Int, nrow(df)*burnfrac)
    step = floor(Int, (nrow(df) - nstart)/(nsamples))
    nrange = range(nstart, length=nsamples, step=step)
    @info length(nrange)
    dfsub = df[nrange, :]
    ms, ss = construct_meanstd(dfsub)
    @info "Is this a circular polarized stacking: $(cirpol)"
    return image_stats_table(outdir, ms, ss, prior, cirpol)
end

"""
    postprocess the stacking results producing a set of image files

# Args

-  `s`: The path to the csv file of the stacking results
-  `o`: The directory where you want to save the images
-  `p`: The prior file used in the stacking procedure

# Flags

- `-c, --circ`: A flag that signals the stacking was using circular pol. Otherwise we assume linear


# Options

- `-n, --nsamples=<int>`: The number of images sampled. Default is 500


"""
@main function main(s::String, o::String, p::String; circ::Bool=false, nsamples::Int=500)
    postprocess(s, o, p, circ; nsamples=nsamples)
end
