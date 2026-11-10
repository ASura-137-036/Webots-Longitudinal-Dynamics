# Does the Webots vehicle library's own force law explain the measurement as well as the fitted model?
#
# This is deliberately NOT a fit. Every candidate is evaluated algebraically, sample by sample, from the
# recorded pedals, rpm and speed - no ODE, no integration - so no candidate can accumulate drift and no
# parameter is adjusted. It is the check that keeps the shipped model honest: if the library's own laws
# describe the recording better than the fit does, the fit is carrying something it should not.
#
#   julia --project=analysis analysis/live/verify_webots_laws.jl
#
# Three candidates:
#   shipped     the fitted model's longitudinal force, parameters loaded from the fitted payload
#   webots_cap  the library's laws plus the wheel-spin guard at driver.c:52 (50 rad/s2 -> 17.9 m/s2)
#   webots_raw  the same laws without the guard, i.e. what the torque curve alone would ask for
#
# The measured acceleration is a central difference of the GPS ground speed, the same channel the fit uses.

using CSV, DataFrames, Printf, Statistics, JLD2

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CSV_FILE = joinpath(REPO, "analysis", "longdyn_train.csv")
# The fitted payload. `export_params.jl` writes the viewer's JSON straight from these same numbers, so
# loading the payload here evaluates exactly the model the viewer is running, without a JSON dependency.
const PAYLOAD = joinpath(REPO, "analysis", "results", "params-opt-v2.34.jld2")

# Webots vehicle-library constants, read from the project's own proto and the shipped library source.
const WHEEL_RADIUS = 0.358       # Car.proto wheelFrontLeft tireRadius, also the fitted value to 1 %
const GEAR_RATIO = 9.6           # our EV proto gearRatio[1]
const MAX_TORQUE = 9000.0        # our EV proto engineMaxTorque [N m] at the motor
const MAX_POWER = 900000.0       # our EV proto engineMaxPower [W]
const ENGINE_MAX_RPM = 4500.0    # Car.proto default, our proto leaves it commented out
const BRAKE_COEFFICIENT = 8000.0 # our EV proto brakeCoefficient [N m per unit intensity]
const BRAKE_SCALE = 0.806        # fitted in v2.34; the recording's achieved/predicted ratio is 0.66-0.85
const SPIN_GUARD = 50.0 * WHEEL_RADIUS   # driver.c ACCELERATION_THRESHOLD [rad/s2] * r [m] = 17.9 m/s2
const GRAVITY = 9.81

const FRAME = CSV.File(CSV_FILE; comment="#") |> DataFrame
@printf("read %d samples, %d columns from %s\n", nrow(FRAME), ncol(FRAME),
    basename(CSV_FILE))

# The fit compares against the GPS ground speed, so the same channel is differentiated here.
const T = Float64.(FRAME.t_s)
const V = Float64.(FRAME.speed_long_ms)
const THROTTLE = Float64.(FRAME.throttle_applied)
const BRAKE = Float64.(FRAME.brake_applied)
const RPM = Float64.(FRAME.engine_rpm)

# Central difference, one-sided at the ends. dt is a constant 5 ms, so a two-sample span is 10 ms and
# the GPS channel's quantisation stays below 0.1 m/s2.
const ACCELERATION = let
    a = zeros(length(T))
    for i in 2:(length(T) - 1)
        a[i] = (V[i + 1] - V[i - 1]) / (T[i + 1] - T[i - 1])
    end
    a[1] = a[2]
    a[end] = a[end - 1]
    a
end

# ---------------------------------------------------------------------------------------------
# The shipped model: the fitted parameter set, so this is the model the viewer is actually running
# ---------------------------------------------------------------------------------------------

const SAVED = JLD2.load(PAYLOAD)["saved"]
const VALUES = Dict(name => Float64(value)
                    for (name, value) in zip(SAVED.param_names, SAVED.param_fitted))
const P = (; (Symbol(k) => v for (k, v) in VALUES)...)

@printf("loaded %s: %d fields, objective %.4f\n", basename(PAYLOAD), length(VALUES),
    Float64(SAVED.objective))

"""
Longitudinal force of the shipped model, exactly the terms of `halfcar_rhs!`:

    demand = throttle * throttle_force * (T_avail / max_torque) * redline
    drive  = smooth_min(demand, traction_acceleration * total_mass)
    resistance = resistance * |v| * tanh(v / resistance_smoothing)
    brake  = 4 C brake_scale intensity / r * tanh(v / brake_smoothing)

`rpm` is computed from road speed the way the model does, not taken from the log, because that is the
feedback the fit experiences.
"""
function shipped_force(velocity, throttle, brake)
    rpm = abs(velocity) / P.wheel_radius * P.gear_ratio * 60.0 / (2pi)
    torque = rpm <= 0.0 ? P.max_torque : min(P.max_torque, P.max_power * 60.0 / (2pi * rpm))
    redline = 0.5 * (1.0 - tanh((rpm - P.engine_max_rpm) / P.redline_span))
    demand = throttle * P.throttle_force * (torque / P.max_torque) * redline
    cap = P.traction_acceleration * P.total_mass
    k = 200.0
    drive = 0.5 * (demand + cap - sqrt((demand - cap)^2 + k^2))
    resistance = P.resistance * abs(velocity) * tanh(velocity / P.resistance_smoothing)
    brake_force = 4.0 * P.brake_coefficient * P.brake_scale * brake / P.wheel_radius *
        tanh(velocity / P.brake_smoothing)
    return drive - brake_force - resistance
end

"""
Longitudinal force from the vehicle library's own laws, using the `engine_rpm` the log recorded.

Reproduces `compute_output_torque()` in driver.c:169-196 and `update_brake()` at 379-390. The `rpm ==
engine_max_rpm` branch is the library's hard zero, not a taper.

`guard` switches the wheel-spin limit at driver.c:191-194 on or off. It is applied to the *drive* torque
only: the library scales `output_torque` before distributing it to the wheels, and the brake is a
separate device that the guard never touches.
"""
function webots_force(velocity, throttle, brake, rpm_measured; guard=true)
    engine_torque = rpm_measured <= 0.0 ? MAX_TORQUE :
        min(MAX_TORQUE, MAX_POWER * 60.0 / (2pi * rpm_measured))
    output_torque = engine_torque * throttle * GEAR_RATIO
    # The library clamps rpm at engine_max_rpm and then zeroes the output there, which is the redline cut.
    rpm_measured >= ENGINE_MAX_RPM && (output_torque = 0.0)
    if guard
        output_torque = min(output_torque, SPIN_GUARD * P.total_mass * WHEEL_RADIUS)
    end
    drive = output_torque / WHEEL_RADIUS
    brake_force = 4.0 * BRAKE_COEFFICIENT * BRAKE_SCALE * brake / WHEEL_RADIUS *
        tanh(velocity / 0.05)
    # No aero term and no constant friction: the library has neither, and the coast fit agrees.
    resistance = 141.0 * abs(velocity)
    return drive - brake_force - resistance
end

# ---------------------------------------------------------------------------------------------
# Phases, matching `audit_residuals.jl` so the numbers line up with the earlier audit
# ---------------------------------------------------------------------------------------------

phase_of(throttle, brake, speed_kmh) =
    speed_kmh < 1.0 ? :standstill :
    brake > 0.1 ? :brake :
    throttle > 0.1 ? (abs(speed_kmh - 63.3) < 2.0 ? :limiter : :launch) : :coast

const PHASES = [phase_of(THROTTLE[i], BRAKE[i], 3.6V[i]) for i in eachindex(T)]

const CANDIDATES = [
    ("shipped", (v, thr, brk, rpm) -> shipped_force(v, thr, brk)),
    ("webots_cap", (v, thr, brk, rpm) -> webots_force(v, thr, brk, rpm; guard=true)),
    ("webots_raw", (v, thr, brk, rpm) -> webots_force(v, thr, brk, rpm; guard=false)),
]

println("\nlongitudinal acceleration: measured (GPS central difference) against three force laws")
println("mass used for a = F/m: ", round(P.total_mass; digits=1), " kg\n")

results = Dict{String, Vector{Float64}}()
for (name, force_law) in CANDIDATES
    predicted = [force_law(V[i], THROTTLE[i], BRAKE[i], RPM[i]) / P.total_mass
                 for i in eachindex(T)]
    results[name] = predicted
end

@printf("%-12s %10s %10s %10s %10s %10s\n", "candidate", "all RMSE", "launch", "brake",
    "coast", "limiter")
for (name, _) in CANDIDATES
    predicted = results[name]
    error = predicted .- ACCELERATION
    finite = isfinite.(error)
    cells = Float64[]
    for phase in (:launch, :brake, :coast, :limiter)
        mask = finite .& (PHASES .== phase)
        push!(cells, count(mask) == 0 ? NaN : sqrt(mean(abs2, error[mask])))
    end
    @printf("%-12s %10.3f %10.3f %10.3f %10.3f %10.3f\n", name, sqrt(mean(abs2, error[finite])),
        cells...)
end

println("\nmean bias, a_predicted - a_measured [m/s2], and the equivalent force error at ",
    round(P.total_mass; digits=0), " kg")
@printf("%-12s %10s %10s %10s %10s\n", "candidate", "launch", "brake", "coast", "limiter")
for (name, _) in CANDIDATES
    predicted = results[name]
    error = predicted .- ACCELERATION
    finite = isfinite.(error)
    cells = String[]
    for phase in (:launch, :brake, :coast, :limiter)
        mask = finite .& (PHASES .== phase)
        count(mask) == 0 ? push!(cells, "     n/a") :
            push!(cells, @sprintf("%10.3f", mean(error[mask])))
    end
    @printf("%-12s %s\n", name, join(cells, " "))
end
@printf("%-12s %10.2f %10.2f %10.2f %10.2f  kN\n", "1 m/s2 =",
    P.total_mass / 1000, P.total_mass / 1000, P.total_mass / 1000, P.total_mass / 1000)

# The launch phase is the one the plan turns on, so break it down by speed as well: a cap that is right
# should be right across the whole range, not just on the average.
println("\nlaunch acceleration by speed bin [g]: measured, shipped, cap, raw")
@printf("%8s %6s %10s %10s %10s %10s\n", "v[km/h]", "n", "measured", "shipped", "cap", "raw")
for low in 0.0:5.0:60.0
    mask = (PHASES .== :launch) .& (3.6 .* V .>= low) .& (3.6 .* V .< low + 5)
    count(mask) < 20 && continue
    @printf("%8.0f %6d %10.3f %10.3f %10.3f %10.3f\n", low + 2.5, count(mask),
        mean(ACCELERATION[mask]) / GRAVITY,
        mean(results["shipped"][mask]) / GRAVITY,
        mean(results["webots_cap"][mask]) / GRAVITY,
        mean(results["webots_raw"][mask]) / GRAVITY)
end

println("\ncoast acceleration by speed bin [g]")
@printf("%8s %6s %10s %10s %10s %10s\n", "v[km/h]", "n", "measured", "shipped", "cap", "raw")
for low in 0.0:5.0:65.0
    mask = (PHASES .== :coast) .& (3.6 .* V .>= low) .& (3.6 .* V .< low + 5)
    count(mask) < 20 && continue
    @printf("%8.0f %6d %10.3f %10.3f %10.3f %10.3f\n", low + 2.5, count(mask),
        mean(ACCELERATION[mask]) / GRAVITY,
        mean(results["shipped"][mask]) / GRAVITY,
        mean(results["webots_cap"][mask]) / GRAVITY,
        mean(results["webots_raw"][mask]) / GRAVITY)
end

# The `:coast` phase is defined as throttle <= 0.1, which also catches the tail of every ramp. With a
# 9.6:1 reduction a throttle of 0.05 already commands 12 kN, so those samples are not coasting at all and
# they dominate the comparison above. Repeat it on samples that really have both pedals at zero.
const TRUE_COAST = (THROTTLE .< 1.0e-6) .& (BRAKE .< 1.0e-6) .& (V .> 1.0)
println("\ntrue coast (throttle = brake = 0, v > 1 m/s): ", count(TRUE_COAST), " samples of ",
    count(PHASES .== :coast), " in the :coast phase")
@printf("%-12s %10s %10s\n", "candidate", "coast RMSE", "coast bias")
for (name, _) in CANDIDATES
    error = results[name][TRUE_COAST] .- ACCELERATION[TRUE_COAST]
    @printf("%-12s %10.3f %10.3f\n", name, sqrt(mean(abs2, error)), mean(error))
end
@printf("%8s %6s %10s %10s %10s %10s\n", "v[km/h]", "n", "measured", "shipped", "cap", "raw")
for low in 0.0:5.0:65.0
    mask = TRUE_COAST .& (3.6 .* V .>= low) .& (3.6 .* V .< low + 5)
    count(mask) < 20 && continue
    @printf("%8.0f %6d %10.3f %10.3f %10.3f %10.3f\n", low + 2.5, count(mask),
        mean(ACCELERATION[mask]) / GRAVITY,
        mean(results["shipped"][mask]) / GRAVITY,
        mean(results["webots_cap"][mask]) / GRAVITY,
        mean(results["webots_raw"][mask]) / GRAVITY)
end

println("\nlaunch detail: mean and median predicted acceleration")
const LAUNCH = isfinite.(ACCELERATION) .& (PHASES .== :launch)
@printf("  launch samples %d, measured a: mean %.3f, median %.3f, p05 %.3f, p95 %.3f m/s2\n",
    count(LAUNCH), mean(ACCELERATION[LAUNCH]), median(ACCELERATION[LAUNCH]),
    quantile(ACCELERATION[LAUNCH], 0.05), quantile(ACCELERATION[LAUNCH], 0.95))
for (name, _) in CANDIDATES
    @printf("  %-12s mean %.3f, median %.3f\n", name, mean(results[name][LAUNCH]),
        median(results[name][LAUNCH]))
end
# Samples where the throttle is steady and the car is genuinely pulling are the ones the cap is meant to
# explain; ramp tails and limiter chatter are not comparable with a steady-state force law.
const STEADY_LAUNCH = LAUNCH .& (THROTTLE .> 0.25) .& (ACCELERATION .> 0.0)
@printf("\n  steady accelerating launch (throttle > 0.25, a > 0): %d samples\n",
    count(STEADY_LAUNCH))
@printf("  %-12s %10s %10s\n", "candidate", "rmse", "bias")
for (name, _) in CANDIDATES
    error = results[name][STEADY_LAUNCH] .- ACCELERATION[STEADY_LAUNCH]
    @printf("  %-12s %10.3f %10.3f\n", name, sqrt(mean(abs2, error)), mean(error))
end

# Decision support: which candidate wins where, on the phase that owns the residual share.
println("\nverdict")
for (phase, label) in ((:launch, "launch"), (:coast, "coast"), (:brake, "brake"))
    mask = isfinite.(ACCELERATION) .& (PHASES .== phase)
    count(mask) == 0 && continue
    scores = [(name, sqrt(mean(abs2, results[name][mask] .- ACCELERATION[mask])))
              for (name, _) in CANDIDATES]
    best = argmin(last.(scores))
    @printf("  %-8s best: %-11s  rmse %.3f m/s2   (cap gain over shipped: %+.3f)\n",
        label, scores[best][1], scores[best][2],
        scores[2][2] - scores[1][2])
end
for (mask, label) in ((TRUE_COAST, "true coast"),)
    scores = [(name, sqrt(mean(abs2, results[name][mask] .- ACCELERATION[mask])))
              for (name, _) in CANDIDATES]
    best = argmin(last.(scores))
    @printf("  %-8s best: %-11s  rmse %.3f m/s2   (cap gain over shipped: %+.3f)\n",
        label, scores[best][1], scores[best][2], scores[2][2] - scores[1][2])
end
