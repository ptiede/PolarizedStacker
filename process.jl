using BlackBoxOptim
using RobustAdaptiveMetropolisSampler


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

    if Threads.nthreads() > 1
        scheduler = :dynamic
    else
        scheduler = :serial
    end 
    l, prior, k = create_lklhd(cfile, prior_file; nbatch=nbatch, scheduler=scheduler)



    t = ascube(prior)
    ff = _SnapshotTransform(t, l)
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
