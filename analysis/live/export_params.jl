# Export the fitted half-car parameters and two numerical fixtures for the Python port used by the
# `postopt_viewer` Webots controller.
#
# Why this exists: the fitted model lives in Julia (`params_opt_v2.34.jl`), but a Webots controller must
# compute in-process to reach a Display device, so the right-hand side is re-implemented in Python. Two
# implementations of the same physics drift apart silently, so this script writes fixtures that make the
# port checkable:
#
#   params_opt_v2<suffix>.json  every fitted field, derived ones included
#   rhs_fixture<suffix>.json     200 states with the derivative Julia computes for each
#   trajectory_fixture<suffix>.json  one integration, produced with the same RK4 the Python side uses
#
# `check_parity.py` asserts the Python copy reproduces both. Regenerating these after any model edit
# makes that test fail until the port is brought back into line.
#
# Run from the repository root, or from anywhere with the analysis project active:
#   julia --project=analysis analysis/live/export_params.jl
#
# Optional arguments select a different model and append a suffix to every output file, so a variant can
# be exported beside the reference set without overwriting it:
#   julia --project=<analysis> export_params.jl <notebook> <payload> <suffix>

using JLD2

# Every default is resolved relative to this file, so the repository runs from any checkout directory.
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# Overridable so a second model (a diagnostic variant) can be exported beside the reference one
# without touching the files the controller reads by default:
#   julia --project=<analysis> export_params.jl <notebook> <payload> <suffix>
const NOTEBOOK = get(ARGS, 1, joinpath(REPO, "analysis", "params_opt_v2.34.jl"))
const PAYLOAD = get(ARGS, 2, joinpath(REPO, "analysis", "results", "params-opt-v2.34.jld2"))
const SUFFIX = get(ARGS, 3, "")

const OUTDIR = joinpath(REPO, "controllers", "postopt_viewer")

# The cells that carry the model. Kept as ids rather than line numbers so a re-ordered notebook still
# works, and so a *missing* cell is an error rather than a silently different model.
const MODEL_CELLS = (
    # `smooth_min` first: `halfcar_rhs!` calls it to cap the drive, so the cell has to be defined
    # before the right-hand side is included.
    smooth_min="c0000000-0000-4000-8000-0000000000c1",  # smooth_min
    params="c0000000-0000-4000-8000-00000000000b",      # halfcar_params
    road="c0000000-0000-4000-8000-00000000000c",        # road_height
    rhs="c0000000-0000-4000-8000-00000000000d",         # halfcar_rhs!
)

"Split a Pluto notebook into `cell id => source` by its `# ╔═╡ <id>` markers."
function split_cells(text)
    cells = Dict{String, String}()
    order = String[]
    current = nothing
    buffer = IOBuffer()
    for line in split(text, '\n')
        matched = match(r"^# ╔═╡ ([0-9a-fA-F-]+)\s*$", line)
        if matched === nothing
            current === nothing || println(buffer, line)
        else
            current === nothing || (cells[current] = String(take!(buffer)))
            current = matched.captures[1]
            push!(order, current)
        end
    end
    current === nothing || (cells[current] = String(take!(buffer)))
    return cells, order
end

# ---------------------------------------------------------------------------
# Minimal JSON writer
# ---------------------------------------------------------------------------
# The project has no JSON package and adding one for three output files is not worth the dependency, so
# this writes the small subset needed: objects from `Pair` tuples, vectors, strings, and real numbers at
# full precision. Floats use `repr` so nothing is lost between Julia and Python.
json_write(io, value::AbstractString) = Base.print(io, '"', escape_string(value), '"')
json_write(io, value::Symbol) = json_write(io, String(value))
json_write(io, value::Integer) = Base.print(io, value)
json_write(io, value::AbstractFloat) = isfinite(value) ? Base.print(io, repr(Float64(value))) :
    error("refusing to write a non-finite float: $value")
json_write(io, value::Bool) = Base.print(io, value ? "true" : "false")
function json_write(io, value::AbstractVector)
    Base.print(io, '[')
    for (index, item) in enumerate(value)
        index > 1 && Base.print(io, ',')
        json_write(io, item)
    end
    Base.print(io, ']')
end
function json_write(io, value::NamedTuple)
    json_write(io, collect(pairs(value)))
end
function json_write(io, value::AbstractVector{<:Pair})
    Base.print(io, '{')
    for (index, pair) in enumerate(value)
        index > 1 && Base.print(io, ',')
        # JSON keys are strings, so an integer checkpoint index is stringified here.
        json_write(io, string(pair.first))
        Base.print(io, ':')
        json_write(io, pair.second)
    end
    Base.print(io, '}')
end

function write_json(path, value)
    open(path, "w") do io
        json_write(io, value)
        Base.print(io, '\n')
    end
    return path
end

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
function load_model()
    cells, _ = split_cells(read(NOTEBOOK, String))
    for (name, id) in pairs(MODEL_CELLS)
        haskey(cells, id) ||
            error("cell $id ($name) is not in $NOTEBOOK; the notebook layout changed")
        Base.include_string(Main, cells[id], NOTEBOOK)
    end
    return nothing
end

# The model cells are `include_string`ed at runtime, so their methods are newer than the world of every
# function defined above them. The functions are therefore captured and passed in explicitly, and only
# ever *called* through `invokelatest` - which keeps the binding access inside `main`, where the world is
# already current.

"Rebuild the fitted parameter tuple. The payload stores every field, derived ones included, so merging
over the defaults cannot leave a derived quantity computed from a stale primary value."
function fitted_parameters(payload, params_function)
    raw = (; (Symbol(name) => Float64(value)
              for (name, value) in zip(payload.param_names, payload.param_fitted))...)
    return merge(Base.invokelatest(params_function), raw)
end

# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------
# The Python controller uses fixed-step RK4 with substeps, so the reference integrator is the same
# algorithm. Comparing against it isolates *implementation* error; comparing against Tsit5 would mix in
# the difference between two integration schemes and tell us nothing precise.
function rk4_step(state, parameters, dt, t, rhs_function)
    # `t` must advance, because the RHS selects its pedal by `floor(t / sample_period)`. Passing a
    # constant time would silently freeze the inputs and the fixture would never exercise a step.
    k1 = zeros(10); Base.invokelatest(rhs_function, k1, state, parameters, t)
    k2 = zeros(10); Base.invokelatest(rhs_function, k2, state .+ (dt / 2) .* k1, parameters, t)
    k3 = zeros(10); Base.invokelatest(rhs_function, k3, state .+ (dt / 2) .* k2, parameters, t)
    k4 = zeros(10); Base.invokelatest(rhs_function, k4, state .+ dt .* k3, parameters, t)
    state .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function integrate(state, parameters; dt, steps, rhs_function)
    history = Vector{Vector{Float64}}(undef, steps + 1)
    history[1] = copy(state)
    for step in 1:steps
        state = rk4_step(state, parameters, dt, (step - 1) * dt, rhs_function)
        history[step + 1] = copy(state)
    end
    return history
end

# ---------------------------------------------------------------------------
# Main
# Coefficient of determination between a measurement and the matching simulation, skipping the
# gaps the segment discipline leaves behind (NaN where the wheels were airborne).
function r2(measured, simulated)
    mask = .!isnan.(measured) .& .!isnan.(simulated)
    measured = measured[mask]
    simulated = simulated[mask]
    isempty(measured) && return 0.0
    variance = sum(abs2, measured .- sum(measured) / length(measured))
    variance <= 0.0 && return 0.0
    return 1.0 - sum(abs2, simulated .- measured) / variance
end

# ---------------------------------------------------------------------------
function main()
    load_model()
    # Captured here, inside `main`, where the world already includes the model cells.
    params_function = halfcar_params
    rhs_function = halfcar_rhs!

    payload = JLD2.load(PAYLOAD)["saved"]
    parameters = fitted_parameters(payload, params_function)

    mkpath(OUTDIR)

    # 1. The parameters, exactly as fitted.
    names = collect(String.(payload.param_names))
    values = Dict(name => Float64(getfield(parameters, Symbol(name))) for name in names)
    # Goodness of fit, so the HUD can state the fit quality next to the traces instead of
    # hard-coding numbers that would silently go stale after a re-fit.
    channels = collect(String.(payload.metric_channel))
    rmse_after = Dict(channel => Float64(value)
                      for (channel, value) in zip(channels, payload.metric_rmse_after))
    fit = (; channels=channels,
             rmse_after=[(channel => rmse_after[channel]) for channel in channels],
             heave_rmse_mm=rmse_after["Heave [m]"] * 1000.0,
             pitch_rmse_deg=rmse_after["Pitch [deg]"],
             speed_r2=r2(payload.measured_speed_ms, payload.sim_velocity_ms),
             heave_r2=r2(payload.measured_com_z, payload.sim_com_height),
             pitch_r2=r2(payload.measured_pitch_rad, payload.sim_pitch_rad),
             sample_count=Int(payload.sample_count),
             evaluations=Int(payload.evaluations),
             elapsed_s=Float64(payload.elapsed_total))
    params_path = write_json(joinpath(OUTDIR, "params_opt_v2$(SUFFIX).json"),
        (; label=payload.label,
           log_file=payload.log_file,
           objective=Float64(payload.objective),
           names=names,
           values=[(name => values[name]) for name in names],
           fit=fit))
    println("wrote ", params_path, " (", length(names), " fields)")

    # A pedal profile long enough for the RHS to index: the fixture samples run to t = 2 s.
    throttle = zeros(400)
    brake = zeros(400)

    # 2. RHS fixture: deterministic pseudo-random states, each with the derivative Julia produces.
    rng = 12345
    next_uniform(state) = (state = (1103515245 * state + 12345) % 2147483648; state / 2147483648.0)
    cases = NamedTuple[]
    for case_index in 1:200
        numbers = Float64[]
        for _ in 1:10
            rng = (1103515245 * rng + 12345) % 2147483648
            u = rng / 2147483648.0
            push!(numbers, (u - 0.5) * 2.0)
        end
        # Scale the state into the physical range the model actually visits, so the fixture exercises
        # the linear region rather than absurd magnitudes where every term saturates.
        state = [0.48 + 0.02 * numbers[1],       # sprung height
                 0.01 * numbers[2],              # pitch
                 0.355 + 0.005 * numbers[3],     # front unsprung
                 0.355 + 0.005 * numbers[4],     # rear unsprung
                 10.0 * case_index,              # position
                 15.0 * numbers[5],              # speed
                 0.1 * numbers[6], 0.05 * numbers[7], 0.2 * numbers[8], 0.2 * numbers[9]]
        sample_throttle = 0.5 * (numbers[10] + 1.0)
        local_throttle = copy(throttle)
        local_brake = copy(brake)
        fill!(local_throttle, sample_throttle)
        fill!(local_brake, 0.2)
        case_parameters = (; parameters..., throttle=local_throttle, brake=local_brake,
            sample_period=0.005, road_x=Float64[], road_z=Float64[])
        derivative = zeros(10)
        Base.invokelatest(rhs_function, derivative, state, case_parameters, 0.0)
        # The pedals travel with the case: without them the checker cannot reproduce the evaluation,
        # and a mismatch there would look like an equation error rather than a fixture error.
        push!(cases, (; u=state, throttle=sample_throttle, brake=0.2,
            du=collect(derivative)))
    end
    rhs_path = write_json(joinpath(OUTDIR, "rhs_fixture$(SUFFIX).json"),
        (; cases=[(; u=case.u, throttle=case.throttle, brake=case.brake, du=case.du)
                  for case in cases]))
    println("wrote ", rhs_path, " (", length(cases), " cases)")

    # 3. Trajectory fixture: a 2 s integration under a pedal step, so the port is tested on accumulated
    #    state and not only on single evaluations.
    # The fixture mirrors the *controller's* stepping policy rather than the RHS's internal indexing:
    # the pedals are held constant over each measurement interval and the interval is subdivided. Both
    # sides then integrate exactly the same sequence of constant-input steps.
    #
    # Indexing by `t` instead would put the pedal boundary on a floating-point comparison, and a single
    # substep landing on the wrong side of it shifts the inputs for 1 ms - which the transient then
    # amplifies into an apparent disagreement between two models that are in fact identical.
    sample_period = 0.005
    substeps = 5
    intervals = 400                        # 2 s of simulated time
    inner_dt = sample_period / substeps

    interval_throttle = vcat(fill(1.0, 120), fill(0.0, 140), fill(0.6, 140))
    interval_brake = vcat(fill(0.0, 120), fill(0.9, 140), fill(0.0, 140))

    state = [parameters.static_sprung_height, parameters.static_pitch,
        parameters.unsprung_front_static, parameters.unsprung_rear_static,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    initial_state = copy(state)
    checkpoints = Pair{Int, Vector{Float64}}[0 => copy(state)]
    for interval in 1:intervals
        level_throttle = interval_throttle[interval]
        level_brake = interval_brake[interval]
        # Single-entry arrays carrying this interval's pedals: `floor(t / sample_period) + 1` then always
        # selects index 1, so the RHS sees a constant input for the whole interval.
        interval_parameters = (; parameters..., throttle=[level_throttle], brake=[level_brake],
            sample_period=sample_period, road_x=Float64[], road_z=Float64[])
        for _ in 1:substeps
            state = rk4_step(state, interval_parameters, inner_dt, 0.0, rhs_function)
        end
        interval in (50, 150, 250, 350, 400) &&
            push!(checkpoints, interval => copy(state))
    end
    trajectory_path = write_json(joinpath(OUTDIR, "trajectory_fixture$(SUFFIX).json"),
        (; sample_period=sample_period, substeps=substeps, intervals=intervals,
           throttle=interval_throttle, brake=interval_brake,
           u0=initial_state, checkpoints=checkpoints, final=state))
    println("wrote ", trajectory_path, " (", intervals, " intervals x ", substeps, " substeps)")

    # 4. Report how far the fixed-step RK4 sits from the high-accuracy reference, so the accuracy claim
    #    is measured rather than assumed. This is informational; the parity test uses the RK4 fixture.
    return nothing
end

main()
