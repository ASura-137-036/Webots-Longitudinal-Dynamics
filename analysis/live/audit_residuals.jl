using JLD2, DataFrames, Printf, Statistics

# The payload to audit is the first argument, so the same audit can be run on any variant:
#   julia audit_residuals.jl params-opt-v2.2
const LABEL = get(ARGS, 1, "params-opt-v2.1")
payload = JLD2.load(joinpath(normpath(joinpath(@__DIR__, "..", "results")), LABEL * ".jld2"))["saved"]
println("auditing ", LABEL, " on ", payload.log_file)
t        = Float64.(payload.time_s)
throttle = Float64.(payload.throttle)
brake    = Float64.(payload.brake)

residual = [
    ("heave [mm]", 1000 .* (Float64.(payload.sim_com_height) .- Float64.(payload.measured_com_z))),
    ("pitch [deg]", rad2deg.(Float64.(payload.sim_pitch_rad) .- Float64.(payload.measured_pitch_rad))),
    ("speed [km/h]", 3.6 .* (Float64.(payload.sim_velocity_ms) .- Float64.(payload.measured_speed_ms)))]
speed = 3.6 .* Float64.(payload.measured_speed_ms)

phase_of(thr, brk, v) =
    v < 1.0            ? :standstill :
    brk > 0.1          ? :brake      :
    thr > 0.1          ? (abs(v - 63.3) < 2.0 ? :limiter : :launch) : :coast
phases = [phase_of(throttle[i], brake[i], speed[i]) for i in eachindex(t)]

println("samples: ", length(t), "  phases: ", sort(unique(phases)))
for (name, r) in residual
    finite = isfinite.(r)
    total  = sum(abs2, r[finite])
    @printf("\n%s   overall RMSE %.4f, total squared %.4e\n", name,
        sqrt(total / count(finite)), total)
    @printf("  %-11s %6s %10s %9s %9s\n", "phase", "n", "RMSE", "share", "mean")
    for phase in (:launch, :brake, :limiter, :coast, :standstill)
        mask = finite .& (phases .== phase)
        count(mask) == 0 && continue
        @printf("  %-11s %6d %10.4f %8.1f%% %9.4f\n", phase, count(mask),
            sqrt(mean(abs2, r[mask])), 100 * sum(abs2, r[mask]) / total, mean(r[mask]))
    end
end

println("\nworst 1 s bins by squared residual")
for (name, r) in residual
    bins = [(t0, sum(abs2, r[(t .>= t0) .& (t .< t0 + 1) .& isfinite.(r)]))
            for t0 in 0.0:1.0:maximum(t) - 1]
    sort!(bins; by=last, rev=true)
    print("\n", name, ": ")
    for (t0, value) in bins[1:min(8, length(bins))]
        @printf("t=%.0f(%.1e)  ", t0, value)
    end
end
println()
