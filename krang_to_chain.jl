#=
This is a conversion script. The stacking algorithms requires the use of a ChainH5
object which is loaded from a specific HDF5 format. This script will convert from
a eht-imaging output to the required format. To use it just run the

`make_hdf5_chain(dir, outname)`

where `dir` is the directory where all the files are stored and `outname`
is the named of the output HDF5 file.
=#


using EHTModelStacker
using DataFrames
using CSV
using Glob

function load_chains(cfiles, summaries)
    dfc = CSV.read.(cfiles, DataFrame)
    dfs = reduce(vcat, CSV.read.(summaries, DataFrame))
    dfs.scan = map(summaries) do f
        parse(Int, string(first(split(basename(f), "-"))[5:7]))
    end
    sind = sortperm(dfs[!, :scan])
    return dfc[sind], dfs[sind, :]

end

function load_and_write_h5(cfiles, summaries, outname)
    dfchain, dfsum = load_chains(cfiles, summaries)
    EHTModelStacker.write2h5(dfchain, dfsum, outname)
end

cfiles = readdir(glob"Data/KrangTest/chains/*skychain.csv")
sfiles = readdir(glob"Data/KrangTest/chains/*dfsum.csv")

load_and_write_h5(cfiles, sfiles, "Data/KrangTest/chain.h5")