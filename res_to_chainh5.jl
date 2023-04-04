#=
This is a conversion script. The stacking algorithms requires the use of a ChainH5
object which is loaded from a specific HDF5 format. This script will convert from
a eht-imaging output to the required format. To use it just run the

`make_hdf5_chain(dir, outname)`

where `dir` is the directory where all the files are stored and `outname`
is the named of the output HDF5 file.
=#


using EHTModelStacker
using NPZ
using DataFrames
using CSV


function make_hdf5_chain(dir, outname)
    dfchain, dfsum = load_chains(dir)
    EHTModelStacker.write2h5(dfchain, dfsum, outname)
end


function readparamnames(pfile)
    open(pfile, "r") do io
        lines = readlines(io)[2:end]
        names = split.(lines) .|> first

        return names
    end
end

function readlogz(sfile)
    logz = open(sfile, "r") do io
        lines = readlines(io)
        logz = parse(Float64, lines[end])
        return logz
    end
    return logz
end



function load_chains(dir)
    files = filter(endswith(".npy"), readdir(dir, join=true))
    zfiles = replace.(files, Ref("_samples.npy"=>"_logz.txt"))
    ind = parse.(Int, first.(split.(last.(split.(files, Ref("scan="))), Ref("_"))))

    sind = sortperm(ind)
    times = 1:length(ind)
    logz = fill(-1e300, length(zfiles))

    dfsum = DataFrame(scan=ind, time=times, logz=logz)
    pfile = replace(first(files), "_samples.npy"=>"_params.txt")
    pnames = readparamnames(pfile)

    dfs = map(files) do f
        read_chain(f, pnames)
    end

    return dfs[sind], dfsum

end

function read_chain(file, pnames)
    params = npzread(file)
    df = DataFrame([params[:,i] for i in eachindex(pnames)], pnames; makeunique=true)
    return df
end
