using Pigeons 
using SplittableRandoms: SplittableRandom
using Random


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
Pigeons.sample_iid!(::FlatPrior, replica, shared) = Random.rand!(replica.rng, replica.state)
Pigeons.default_reference(p::StackerPotential) = FlatPrior(p.dim)

Pigeons.initialization(p::StackerPotential, rng::SplittableRandom, ::Int) = rand(rng, p.dim)

function process_pigeons(
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
        l, prior, k = create_lklhd(cfile, prior_file; nbatch=nbatch, scheduler=:serial)
        t = ascube(prior)

        f = _SnapshotTransform(t, l)

        pot = StackerPotential(f, dimension(t))

        @info "Starting Sampling"
        pt = pigeons(target = pot,
                     n_rounds = nrounds,
                     record=[Pigeons.traces, Pigeons.log_sum_ratio, Pigeons.round_trip],
                     multithreaded=true, checkpoint=true,
                     n_chains=nchains,
                     explorer = Pigeons.SliceSampler(; w = 0.05),
                     seed = 2
                    )

        trace = dropdims(sample_array(pt); dims=3)[:, 1:end-1]
        tv = HC.transform.(Ref(t), eachrow(trace)) |> Table
        logp = l.(eachrow(tv))
        write_results(chainfile, tv, logp, k)
    else
        @info "Resuming from checkpoint"
        pt_from_checkpoint = PT("results/latest")
        pt = pigeons(pt_from_checkpoint)
        trace = dropdims(sample_array(pt); dims=3)[:, 1:end-1]
        tv = HC.transform.(Ref(t), eachrow(trace)) |> Table
        logp = l.(eachrow(tv))
        write_results(chainfile, tv, logp, k)
    end
end
