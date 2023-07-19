
function snapshot_plot(times, quants, means, xlabel, ylabel; dt=1)
    fig = Figure(;resolution=(300, 600))
    ax = Axis(fig[1,1]; xlabel, ylabel)
    t0 = times[begin]
    cs = Makie.wong_colors()
    for i in eachindex(quants)
        density!(ax, quants[i], offset=(times[i]-t0)*dt, color=(cs[1], 0.2), strokecolor=(:blue, 0.3), strokewidth=2)
    end
    vlines!(means, color=(:black, 0.8))
    return fig, ax
end


ns = l.chain.names
for n in ns
    fig, ax = snapshot_plot(l.chain.times, getparam(l.chain, n).chain, df[1:50:end, Symbol("mean_", n)], String(n), "time - t₀")
    display(fig)
    save("orig_snapshot_plots_"*String(n)*".png", fig)
end
