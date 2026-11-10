# Headless runner for the LongDyn par_est Pluto notebooks.
#
#   julia --project=analysis run_parest.jl <notebook.jl> probe [max_iters]
#   julia --project=analysis run_parest.jl <notebook.jl> full  [max_iters]
#   julia --project=analysis run_parest.jl <notebook.jl> validate
#
# `validate` replays the notebook against the held-out run (longdyn_test.csv) instead of the training
# run, stops before the solve cell, injects the parameters this notebook already fitted, and writes
# results/validation/<label>.jld2. No optimizer runs, so the test data can never influence the fit.
#
# The notebook is a Pluto file, but every code cell is a plain top-level `begin ... end` block and
# the `@bind` mock at the top means nothing needs Pluto at run time. Cells are replayed in notebook
# order, so the script sees exactly what the notebook sees.
#
# `probe` stops before the LBFGS cell and times one objective and one gradient, so the cost of a
# real run can be quoted before committing to it.

const ANALYSIS_DIR = normpath(joinpath(@__DIR__, ".."))

# Pluto gives every notebook `md"..."`; outside Pluto the macro has to be brought in by hand.
using Markdown
# Timestamps on the progress lines.
using Dates
# The notebooks call `mean`; bring it in explicitly so the replay does not depend on a transitive
# re-export from whichever package happens to be loaded.
using Statistics
# Flushed progress lines.
using Logging

# Julia's `stderr` is block-buffered when redirected to a file, so the notebook's `@info` LBFGS
# progress would stay invisible until the solve ended. This logger writes one already-flattened
# line per record and flushes it, so a tail of the log always shows the current iteration.
struct FlushLogger <: Logging.AbstractLogger
    io::IO
end
Logging.min_enabled_level(::FlushLogger) = Logging.Info
Logging.shouldlog(::FlushLogger, level, _module, group, id) =
    level >= Logging.Info && group !== :depwarn
Logging.catch_exceptions(::FlushLogger) = false
function Logging.handle_message(logger::FlushLogger, level, message, _module, group, id,
        file, line; kwargs...)
    fields = join(("$key=$(value)" for (key, value) in kwargs), " ")
    println(logger.io, "[", level, "] ", message, "  ", fields)
    flush(logger.io)
    return nothing
end
Logging.global_logger(FlushLogger(stdout))

function split_cells(text)
    cells = Dict{String,String}()
    order = String[]
    current = nothing
    buffer = IOBuffer()
    for line in split(text, '\n')
        m = match(r"^# \u2554\u2550\u2561 ([0-9a-fA-F-]+)\s*$", line)
        if m === nothing
            current === nothing || println(buffer, line)
        else
            current === nothing || (cells[current] = String(take!(buffer)))
            current = m.captures[1]
            push!(order, current)
        end
    end
    current === nothing || (cells[current] = String(take!(buffer)))
    return cells, order
end

const NAME = ARGS[1]
const MODE = get(ARGS, 2, "probe")
const MAX_ITERS = parse(Int, get(ARGS, 3, "400"))

cells, order = split_cells(read(joinpath(ANALYSIS_DIR, NAME), String))
code_cells = [id for id in order if startswith(id, "c0000000-")]

println("########## runner: ", NAME, " mode=", MODE, " max_iters=", MAX_ITERS,
    " code_cells=", length(code_cells), " threads=", Threads.nthreads())
flush(stdout)

# Only these switches differ from the notebook. The wall-clock budget becomes `Inf` so the
# iteration count is the only thing that can stop the solve.
#
# The pedal shading is dropped from the figures: this log has ~11.6 k constant-pedal intervals, so
# `add_pedal_background!` issues ~81 k `vspan!` calls and costs ~7 min per figure. It is purely
# cosmetic - no variable used by the fit or by the saved result depends on it.
# The residual loop walks `loss_data.plan`, one independent ODE solve per segment. Nothing is
# shared between iterations except the read-only measurements, so it is the only part of the whole
# notebook that can use more than one core. Each segment writes into its own buffer and the buffers
# are concatenated in plan order afterwards, so the residual vector - and therefore the loss, the
# gradient and the LBFGS trajectory - is bit-for-bit what the serial loop produced.
const THREADED_PROLOGUE = """
    segment_buffers = [eltype(theta)[] for _ in 1:length(loss_data.plan)]
    Threads.@threads :dynamic for plan_index in eachindex(loss_data.plan)
        entry = loss_data.plan[plan_index]
        target = segment_buffers[plan_index]
"""

const THREADED_EPILOGUE = """
    residuals = eltype(theta)[]
    for buffer in segment_buffers
        append!(residuals, buffer)
    end
    return residuals"""

function prepare(body)
    body = replace(body,
        "max_iterations = 200" => "max_iterations = " * string(MAX_ITERS),
        "max_time_s = 1.0e6" => "max_time_s = Inf",
        "max_time_s = 1200.0" => "max_time_s = 3600.0",
        "using PlutoUI" => "# using PlutoUI (not needed headless)",
        "LocalResource(" => "identity(")
    if MODE == "validate"
        # Validation replays the model on the held-out run: swapping the pinned log here rebuilds
        # `data`, `halfcar_segments` and `loss_data` from the test CSV, so every downstream helper
        # (objective, simulate) sees the test run without any change to the notebook itself.
        body = replace(body, "\"longdyn_train.csv\"" => "\"longdyn_test.csv\"")
    end
    if !occursin("function add_pedal_background!", body)
        body = replace(body,
            "add_pedal_background!(axis, time_s, throttle, brake; single_step=0.005)" => "nothing",
            "add_pedal_background!(axis, time_s, throttle, brake; single_step=halfcar_dt)" => "nothing")
    end
    if Threads.nthreads() > 1 && occursin("function halfcar_residual", body)
        body = replace(body, "    for entry in loss_data.plan" => THREADED_PROLOGUE)
        body = replace(body, "push!(residuals," => "push!(target,")
        body = replace(body, "    return residuals" => THREADED_EPILOGUE)
    end
    if MODE == "full" && occursin("maxiters=max_iterations", body)
        # Plateau stop: the loss flattens long before 400 evaluations, but every evaluation past the
        # flattening still costs a stiff ODE solve. Stop when the last 25 evaluations moved the loss
        # by less than 0.01 %, keeping the best parameters found. The 400 cap is unchanged, so this
        # can only end the solve early, never extend it.
        body = replace(body, "        return budget_spent" => """
        window = length(optimization_history)
        plateau = window > 25 &&
            abs(optimization_history[end].objective -
                optimization_history[end - 24].objective) <=
            1.0e-4 * abs(optimization_history[end - 24].objective)
        if plateau
            @info("LBFGS stopped",
                reason = "plateau: no 0.01% improvement over 25 evaluations",
                iteration = iteration,
                elapsed_s = round(elapsed; digits = 2),
                loss = round(loss; digits = 4))
        end
        return budget_spent || plateau""")
    end
    return body
end

for id in code_cells
    body = cells[id]
    if occursin("TableOfContents(", body)
        println("[skip ] ", id, "  PlutoUI outline")
        flush(stdout)
        continue
    end
    if MODE in ("probe", "validate") && occursin("maxiters=max_iterations", body)
        println("[stop ] ", id, "  solve cell, ", MODE, " mode")
        flush(stdout)
        break
    end
    started = time()
    println("[run  ] ", id, "  ", first(split(strip(body), '\n')), "  @ ", Dates.Time(Dates.now()))
    flush(stdout)
    try
        Base.include_string(Main, prepare(body), joinpath(ANALYSIS_DIR, "notebook_cell.jl"))
    catch err
        println("FAILED in cell ", id, ": ", sprint(showerror, err))
        flush(stdout)
        rethrow()
    end
    println("[cell ] ", id, "  ", round(time() - started; digits=1), " s")
    if id == code_cells[1] || occursin("Optimizer budget", body) || occursin("max_iterations =", body)
        for name in (:max_iterations, :max_time_s, :window_stride, :solver_tolerance)
            isdefined(Main, name) && println("        ", name, " = ", getfield(Main, name))
        end
    end
    flush(stdout)
end

if MODE == "validate"
    # Replay done: `data`, `halfcar_segments` and `loss_data` now describe the held-out run, and the
    # model helpers are defined. Inject the parameters this notebook fitted on the training run.
    results_directory = normpath(joinpath(ANALYSIS_DIR, "results"))
    validation_directory = joinpath(results_directory, "validation")
    mkpath(validation_directory)

    # A top-level `for` is a local scope, so assigning `train_payload` inside it would only create a
    # loop-local binding. Collect first, then pick with `findfirst`.
    candidates = sort(filter(f -> endswith(lowercase(f), ".jld2"),
                             readdir(results_directory; join=true)))
    payloads = [(file=file, saved=JLD2.load(file)["saved"]) for file in candidates]
    chosen = findfirst(entry -> entry.saved.notebook == NAME, payloads)
    chosen === nothing &&
        error("no training payload with notebook == $NAME in $results_directory")
    train_payload = payloads[chosen].saved
    println("[val  ] ", length(payloads), " payload(s); using ", basename(payloads[chosen].file),
        " for label ", train_payload.label)

    # `param_fitted` is stored in `keys(fitted_parameters)` order, which is not the optimiser's
    # `estimation_names` order. Reorder by name, or the parameters silently permute.
    stored = Dict(Symbol(name) => value
                  for (name, value) in zip(train_payload.param_names, train_payload.param_fitted))
    absent = [name for name in estimation_names if !haskey(stored, name)]
    isempty(absent) || error("training payload is missing parameters: $absent")
    fitted_values = [stored[name] for name in estimation_names]
    fitted_parameters = parameters_from_vector(fitted_values)
    theta_fitted = normalise(vector_from_parameters(fitted_parameters))

    validation_started = time()
    validation_objective = halfcar_objective(theta_fitted, loss_data)
    validation_run = halfcar_plot_channels(fitted_parameters)
    validation_seconds = time() - validation_started

    saved_validation = (
        schema=1,
        label=train_payload.label,
        notebook=NAME,
        split="validation",
        saved_at=string(now()),
        log_file=basename(latest_log),
        sample_count=nrow(data),
        time_step=halfcar_dt,
        solver_tolerance=solver_tolerance,
        objective=validation_objective,
        train_objective=train_payload.objective,
        param_names=String.(collect(keys(fitted_parameters))),
        param_fitted=Float64[value for value in values(fitted_parameters)],
        window_indices=collect(loss_data.indices),
        sim_com_height=Float64.(validation_run.com_height),
        sim_pitch_rad=Float64.(validation_run.pitch_rad),
        sim_velocity_ms=Float64.(validation_run.velocity_ms))
    validation_path = joinpath(validation_directory, train_payload.label * ".jld2")
    jldsave(validation_path; saved=saved_validation)

    println("\n===== validation summary =====")
    println("notebook            = ", NAME, "   label = ", train_payload.label)
    println("test log            = ", basename(latest_log), "  samples=", nrow(data),
        "  span=", round(first(time_s); digits=2), "..", round(last(time_s); digits=2), " s")
    println("segments            = ", length(loss_data.plan),
        "   loss samples = ", window_sample_count)
    println("train objective     = ", round(train_payload.objective; digits=2))
    println("test  objective     = ", round(validation_objective; digits=2))
    println("replay took         = ", round(validation_seconds; digits=1), " s")
    println("written             = ", validation_path)
    flush(stdout)
end

if MODE == "probe"
    theta = estimation_start_normalised
    simulated_samples = sum(entry.segment.stop_index - entry.segment.simulation_start + 1
                            for entry in loss_data.plan)
    simulated_seconds = simulated_samples * halfcar_dt

    t0 = time()
    plain_value = halfcar_objective(theta, loss_data)
    objective_seconds = time() - t0

    t0 = time()
    gradient_value = ForwardDiff.gradient(x -> halfcar_objective(x, loss_data), theta)
    gradient_seconds = time() - t0

    println("\n===== probe summary =====")
    println("log                 = ", basename(latest_log), "  samples=", nrow(data),
        "  span=", round(first(time_s); digits=2), "..", round(last(time_s); digits=2), " s")
    println("segments            = ", length(loss_data.plan),
        "  simulated_samples=", simulated_samples,
        "  simulated_s=", round(simulated_seconds; digits=1))
    println("loss samples        = ", window_sample_count, " (stride ", window_stride, ")")
    println("objective (Float64) = ", round(plain_value; digits=4), "  took ", round(objective_seconds; digits=2), " s")
    println("gradient (ForwardDiff, ", length(gradient_value), " partials) took ",
        round(gradient_seconds; digits=2), " s")
    println("nominal objective   = ", round(nominal_objective; digits=4))
    println("projected per-iteration cost (1 gradient) = ", round(gradient_seconds; digits=2), " s")
    for evaluations in (400, 500, 600, 800)
        println("  ", evaluations, " objective evaluations -> ",
            round(evaluations * gradient_seconds / 60; digits=1), " min")
    end
    flush(stdout)
end
