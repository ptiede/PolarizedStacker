using Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

using PythonCall
using Printf
using Tables
using CSV
using DataFrames
using Comonicon
ehtim = pyimport("ehtim")

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
    beta_list_cpol_re = Tables.getcolumn.(Ref(x), ilistcp_re)
    beta_list_cpol_im = Tables.getcolumn.(Ref(x), ilistcp_im)
    beta_list_cpol = complex.(beta_list_cpol_re, beta_list_cpol_im)

    alpha = Tables.getcolumn(x, :alpha)
    return (; d, x0, y0, beta_list, beta_list_cpol, alpha)
end

function sample_images_cp(outdir, chain)
    mkpath(outdir)
    i = 0
    for r in eachrow(chain)
        m = ehtim.model.Model()
        p  = extract_cp_params(r)
        m = m.add_thick_mring(1.0, p.d*ehtim.RADPERUAS, p.alpha*ehtim.RADPERUAS, 0.0*ehtim.RADPERUAS, 0.0*ehtim.RADPERUAS, beta_list=p.beta_list, beta_list_cpol=p.beta_list_cpol)
        img = m.make_image(100.0*ehtim.RADPERUAS, 128)
        fname = @sprintf "image_%05d.fits" i
        img.save_fits(joinpath(outdir, fname))
        i += 1
    end
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

function sample_images_lp(outdir, chain)
    mkpath(outdir)
    i = 0
    for r in eachrow(chain)
        m = ehtim.model.Model()
        p  = extract_lp_params(r)
        m = m.add_thick_mring(1.0, p.d*ehtim.RADPERUAS, p.alpha*ehtim.RADPERUAS, 0.0*ehtim.RADPERUAS, 0.0*ehtim.RADPERUAS, beta_list=p.beta_list, beta_list_pol=p.beta_list_pol)
        img = m.make_image(100.0*ehtim.RADPERUAS, 64)
        fname = @sprintf "image_%05d.fits" i
        img.save_fits(joinpath(outdir, fname))
        i += 1
    end
end

function postprocess(fstack, outdir, linpol=true; burnfrac=0.75, nsamples=100)
    df = CSV.File(fstack) |> DataFrame
    nstart = ceil(Int, nrow(df)*burnfrac)
    step = ceil(Int, (nrow(df) - nstart)/(nsamples))
    nrange = range(nstart, length=nsamples, step=step)
    @info length(nrange)
    dfsub = df[nrange, :]
    ms, ss = construct_meanstd(dfsub)
    linpol  &&(@info "Using Linpol"; return sample_images_lp(outdir, ms))
    !linpol &&(@info "Using Cirpol"; return sample_images_cp(outdir, ms))
end

"""
    postprocess the stacking results producing a set of image files

# Args

-  `s`: The path to the csv file of the stacking results
-  `o`: The directory where you want to save the images

# Flags

- `-c, --circ`: A flag that signals the stacking was using circular pol. Otherwise we assume linear


# Options

- `-n, --nsamples=<int>`: The number of images sampled. Default is 500


"""
@main function main(s::String, o::String; circ::Bool=false, nsamples::Int=500)
    postprocess(s, o, !(circ); nsamples=nsamples)
end
