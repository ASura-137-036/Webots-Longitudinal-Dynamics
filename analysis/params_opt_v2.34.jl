### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ c0000000-0000-4000-8000-000000000001
begin
# Render notebook narrative and section headings as Markdown.
using PlutoUI
# Read the measurement files into tabular data.
using CSV
# Provide DataFrame storage, row iteration, and table utilities.
using DataFrames
# Provide `mean` for the scalar objective.
using Statistics
# Use WGLMakie for interactive figures rendered in the browser.
using WGLMakie
# Integrate the half-car model with the DifferentialEquations solvers.
using DifferentialEquations
# Fit the model parameters to the measured response.
using Optimization
using OptimizationOptimJL
# Supply gradients to the optimizer.
using ForwardDiff
# Probe for a free loopback port for the WGLMakie render server.
using Sockets
# Save the fitted parameters, loss history and traces so other notebooks can compare them.
using JLD2
# Timestamp each saved result so a stale file is obvious.
using Dates
end

# Cell 4: Packages used by the data reader, the model, the figures and the optimizer.

# ╔═╡ c0000000-0000-4000-8000-000000000002
md"""
# Retuning a Simplified Half-Car Model from Measured Response Data

!!! note "The shipped model"
    This notebook is the reference fit for the project. It estimates the uncertain coefficients of the
    half-car model from one measured run and writes the parameter set that the Webots viewer and the HUD
    movie consume.

    **Mass anchored, not estimated.** The previous three fits disagreed on mass by 630 kg while agreeing on
    specific force, which is the signature of a direction the recording does not constrain:

    | | `sprung_mass` | `total_mass` | `throttle_force / total_mass` |
    |---|---|---|---|
    | v2.31 | 2561 kg | 2685 kg | 77.35 |
    | v2.32 | 2103 kg | 2190 kg | 78.20 |
    | v2.33 (pinned 2700) | 2700 kg | 2822 kg | 78.18 |

    Only the ratio is measured. v2.32 therefore bought its 0.7 % fit advantage by sliding mass to 2190 kg -
    below the real car's kerb weight - so the fit quality is real but the mass it reports is not.

    This variant removes that direction instead of fitting it. Two anchors:

    - **One unsprung mass per axle, shared front and rear** (`unsprung_mass`). The four wheels are the same
      `VehicleWheel` PROTO with the same `tireRadius`, so nothing distinguishes them. v2.31 left both free
      and split them 1.6:1, which is not physical; v2.32 and v2.33 both wanted symmetry, and v2.31 and
      v2.33 independently agree on ~123 kg of total unsprung mass.
    - **Total anchored to 2700 kg**, the proto's calibrated figure. `sprung_mass` is then dependent:
      `sprung = 2700 - 2 * unsprung`. This is the reading v2.31 corroborates, having fitted 2685 kg from
      dynamics alone - 0.6 % away.

    Estimated set: **21 parameters**, down from 23. Everything else - the box widening on `wheel_radius`,
    `spring_front` and `tyre_spring`, the drive cap, the brake law and the single linear resistance - is
    inherited from v2.32 unchanged.

    The question this answers: with the mass direction closed, does the model still fit as well, and what
    does `throttle_force` settle at when it can no longer trade against mass?

    **The widened boxes inherited from v2.32.** v2.31 finished with three parameters pressed against a
    ceiling, so their reported values were "as far as this box allows" rather than measurements:

    | parameter | v2.31 box | v2.32 / v2.34 box | v2.31 result | where it sat |
    |---|---|---|---|
    | `wheel_radius` | 0.3508 … 0.3652 (±2 %) | 0.3401 … 0.3759 (±5 %) | 0.36519 | 99.9 % |
    | `spring_front` | 70 … 140 kN/m | 60 … 220 kN/m | 139 667 | 99.5 % |
    | `tyre_spring` | 120 … 900 kN/m | 120 … 1800 kN/m | 896 350 | 99.5 % |

    v2.32 showed what those were worth: `spring_front` fell to 120 kN/m once free, so its v2.31 ceiling was
    compensating for the others rather than reflecting the data, while `tyre_spring` genuinely wanted more
    room and rose 33 % past its old wall.

    **Longitudinal force.** Three terms, no more:

        F = drive - brake_force - resistance

    - **Drive.** The vehicle proto's electric motor, `T = min(9000 N m, 900 kW * 60 / (2 pi rpm))` through
      the `9.6:1` reduction, scaled by throttle and normalised by `max_torque` so that a single identified
      parameter, `throttle_force`, carries the whole traction magnitude. `rpm` comes from road speed the
      way the vehicle library computes it, and the drive ceiling is a `tanh` at `4500 rpm` rather than a
      `clamp`: a flat region would leave its switching point invisible to the gradient.
    - **Wheel-spin cap.** The vehicle library does not let the wheels spin up faster than
      `ACCELERATION_THRESHOLD 50 rad/s2` (`driver.c:52`), scaling the output torque down when they try.
      Translated to the body that is a ceiling on acceleration, which is what actually sets the launch
      response: the recording saturates near 1.9 g while the torque curve alone asks for 4.4 g. The cap is
      applied as `smooth_min(demand, traction_acceleration * total_mass)` with `traction_acceleration`
      fitted, anchored at `50 * 0.358 m = 17.9 m/s2`; a smooth minimum rather than `min` keeps the
      switching point differentiable.
    - **Brake.** A constant pad torque, because that is what the vehicle applies. The library sets
      `damping_constant = |brake * brake_coefficient / omega|` precisely to cancel the speed dependence of
      its damping device, so the torque is `brake * brake_coefficient` per wheel at any road speed;
      four wheels divided by the rolling radius give the contact force. A `tanh` edge fades it to zero at
      standstill so it can never push the car backwards.
    - **Resistance.** One term, linear in speed:

          resistance = resistance * |v| * tanh(v / resistance_smoothing)

      Measured rather than assumed. With both pedals at zero, a least-squares fit of the recorded
      deceleration gives `F = -141 v` over 11 591 samples: proportional to speed, with no meaningful
      constant part. The same test is repeated by `analysis/live/verify_webots_laws.jl`, which evaluates
      the force laws algebraically against the measurement without integrating anything. The `tanh` again
      carries the sign and keeps the force smooth through standstill.

    **Result.** Twenty-three estimated parameters, none of them pinned to a bound.

    | | training | held out |
    |---|---|---|
    | objective | 8.84 | 7.72 |
    | heave RMSE | 1.607 mm | 1.453 mm |
    | pitch RMSE | 0.176° | 0.172° |
    | speed RMSE | 1.93 km/h | 2.17 km/h |
    | speed R² | 0.990 | 0.987 |

    Fitted values that matter downstream: `resistance` 179.2 N per m/s, `traction_acceleration`
    14.32 m/s², `throttle_force` 207.7 kN, sprung mass 2561 kg on a 2685 kg total.

    **Where the parameters are soft.** `throttle_force` and `total_mass` only ever appear as a ratio below
    the speed where the cap stops binding, so the two are not separately identified: starting the fit from
    `resistance = 0` instead of 100 reaches the same objective within 0.5 % with `resistance` 141.9 and a
    2160 kg total. The fit quality is start-independent; the individual values are not. Three parameters
    also sit within 0.5 % of their ceilings (`wheel_radius`, `spring_front`, `tyre_spring`).

This notebook estimates the uncertain coefficients of a small vehicle model from a measured run. The
model is deliberately simple - six degrees of freedom and a handful of force laws - so that every
parameter keeps a physical meaning and can be defended after the fit. Where a quantity cannot be
measured directly, it is estimated from the data rather than assumed.

## The idea

The model is integrated with an ODE solver, and that solve is differentiated with respect to the
model parameters by automatic differentiation. That turns "which coefficients explain this
measurement?" into a smooth optimisation problem: the mismatch between the simulated and the
measured response is a differentiable function of the parameter vector, and a quasi-Newton method
(BFGS) minimises it using exact gradients. No finite differences, no surrogate model, and no
separate sensitivity equations.

BFGS was chosen over LBFGS by measurement on this problem rather than by default: at twenty-four
parameters the two store the same amount of curvature, and BFGS reached a lower objective in less
wall time here because its line search copes better with the occasional ill-conditioned candidate.
Section 6.1 shows the resulting convergence history.

Two consequences matter in practice:

- The same machinery estimates quantities that are awkward to instrument directly - suspension
  rates, damper coefficients, the mass split between axles - from the response they produce.
- A coefficient the data cannot constrain is not invented by the optimiser. It drifts to whichever
  bound is nearest and the residual stops improving. That is a diagnostic rather than a failure, and
  the bounds table reports it explicitly.

## What the measurement contains

The input is one straight-line pass under a prescribed pedal schedule: ramps, steps, sinusoids and
stops, spanning standstill to the speed limiter. Because the inputs are open-loop, the response is
informative about the vehicle rather than about a driver.

| feature | what it constrains |
|---|---|
| speed plateau at the 4500 rpm drive limit | the drive limit: torque falls to zero there, so top speed is set by gearing rather than by a force balance |
| constant `F/v` over 6.3 … 9.7 m/s while coasting | passive resistance is dominated by a term proportional to speed |
| pitch response to pedal steps | suspension rates and damping, and the pitch inertia |
| wheel lift-off events | unsprung mass and tyre rate |

The resistance row is the one that decides the shape of the passive term, and it is unambiguous: the
coast deceleration tracks speed rather than its square, so a single speed-proportional loss describes it
and nothing else is needed.

## Which parameters are known and which are estimated

Every parameter enters with a prior and an uncertainty band, and the band is what the optimiser is
allowed to move it within:

- **Known values** carry a small tolerance. Wheelbase, wheel radius, overall gear ratio, drive torque
  and drive power are nameplate figures, so they move by a couple of percent at most - enough to
  absorb build tolerance, not enough to re-identify the hardware.
- **Estimated values** carry the uncertainty of the estimate they came from. Suspension rates, damper
  coefficients, the sprung mass split, pitch inertia and the CG height are either derived from a
  static test or carried over from a comparable machine, so they get a wide band.
- **Unknown terms** get the widest band of all. The drive-force scale, the tyre friction and the
  tyre rates are the coefficients this notebook is really solving for; their bounds exist only to keep
  the optimiser inside a physically sensible region.

## Normalisation

Each physical parameter `p` is mapped onto a dimensionless `u` by an affine transform:

```
u = (p - centre) / half_width      # lower bound ↦ -1, upper bound ↦ +1
p = centre + half_width * u        # inverse
```

where `centre = (lower + upper) / 2` and `half_width = (upper - lower) / 2`. The prior interval for
each parameter therefore becomes the identical box `u ∈ [-1, 1]` for every parameter, and a step of
`0.1` always means "move a tenth of the way along the uncertainty interval" — the same relative move
for a 20 mm wheel radius and a 300 kW drive limit. That conditioning is what BFGS benefits from.

## How this notebook is organised

1. **Setup** — the packages, the shared colour palette, and the render server.
2. **Measured data** — read the measurement file, classify the wheel lift-off samples, and draw the
   measurement on its own in the stacked layout.
3. **Half-car model** — parameters, road profile, right-hand side, segment integration and the
   segmentation of the measured run.
4. **Pre-optimization parameter estimate** — the starting estimate and its baseline trajectory, kept
   completely separate from the optimizer.
5. **Optimization** — the loss window, the parameter bounds mapped onto `-1 … 1`, the loss data, the
   residual and the BFGS solve with ForwardDiff gradients.
6. **Results** — the fitted parameter table, the measurement vs before vs after comparison, and the
   error metrics over the window.
"""

# Cell 1: Notebook title, purpose and how the sections are organised.

# ╔═╡ c0000000-0000-4000-8000-000000000031
begin
    # PlutoUI builds this outline from the markdown headings above and below, so the section
    # structure is exactly what appears in the widget. `aside=true` docks it to the side of the
    # page, where it stays visible while scrolling.
    isdefined(PlutoUI, :TableOfContents) ||
        error("This PlutoUI version has no TableOfContents widget; update PlutoUI to 0.7.54 or newer")
    TableOfContents(title="Contents", depth=3, indent=true, aside=true)
end

# Cell 2: Table of contents docked to the side of the page.

# ╔═╡ c0000000-0000-4000-8000-000000000060
md"""
## 1. Setup

The packages, the shared figure palette, and a free port for the render server.

The palette is defined once here so that every figure in the notebook draws the measurement in the
same colour, the pre-optimization simulation in a second colour, and the post-optimization simulation
in a third. Because each panel plots only one measured channel, the channels themselves need no
individual colours — the y-axis label identifies them.
"""

# Cell 3: Section 1: setup - packages, shared palette and render server.

# ╔═╡ c0000000-0000-4000-8000-000000000030
begin
    # One colour per role, shared by every figure, so the same line always means the same thing.
    measured_color = :black       # the measured response
    before_color = :darkorange    # simulation with the pre-optimization guess
    after_color = :royalblue      # simulation with the optimized parameters

    # The three lift classes are disjoint, so each gets its own shape as well as its own colour. The
    # hues stay clear of the three line colours, so a marker never reads as a data series.
    lift_styles = (
        front=(color=:crimson, marker=:utriangle, label="Front lift"),
        rear=(color=:purple, marker=:dtriangle, label="Rear lift"),
        all=(color=:saddlebrown, marker=:star5, label="All-wheel lift"),
    )
end

# Cell 5: One colour per role, shared by every figure.

# ╔═╡ c0000000-0000-4000-8000-00000000001e
begin
    """
        port_is_open(port)

    True when something is already listening on the loopback `port`.
    """
    function port_is_open(port)
        probe = try
            Sockets.connect(Sockets.localhost, port)
        catch
            return false
        end
        close(probe)
        return true
    end

    # Bonito serves the figures from a local HTTP server. Unless it is told otherwise it binds its
    # **default port 9384**, and it does not move when that port is taken: binding fails, the figure
    # never loads, and Pluto leaves the cell spinning on "loading" forever. 9384 is also what every
    # other WGLMakie session on this machine wants, this notebook's twins included.
    #
    # Three rules matter, and all three were learned the hard way:
    #
    #   * `configure_server!` must always be called, even when the chosen port looks free. Skipping
    #     it silently falls back to 9384, which is the exact failure above.
    #   * the port must be chosen `deterministically`, never by asking the OS for any free port. A
    #     random pick changes on every re-run, and figures already on the page keep pointing at the
    #     old port, so they spin forever. A fixed preference scans upward, so a re-run lands on the
    #     same number and already-rendered figures survive.
    #   * it must be configured at most `once` per session. Reconfiguring after the first figure has
    #     been served silently breaks the live page.
    #
    # Each notebook owns a distinct preference so they never contend with each other:
    # `_normz_par_est` 9381, `_physics_study` 9386, `_original` 9396.
    render_port = let
        preferred = 9386
        if !port_is_open(preferred)
            preferred
        else
            first(candidate for candidate in (preferred + 1):(preferred + 15)
                if !port_is_open(candidate))
        end
    end

    # A workspace global, not task-local storage: Pluto runs each cell in its own task, so anything
    # task-scoped would be forgotten and the server would be reconfigured on every re-run.
    if !@isdefined(wglmakie_configured)
        WGLMakie.Bonito.configure_server!(listen_port=render_port)
        wglmakie_configured = true
    end
end

# Cell 6: Claim a free port for the WGLMakie render server.

# ╔═╡ c0000000-0000-4000-8000-000000000061
md"""
## 2. Measured data

2.1 reads the measurement file, validates the channels the model needs, and derives the signals
plotted below. It also classifies every sample by which axle is airborne.

The lift classes are mutually exclusive:

| class | meaning |
|---|---|
| **front lift** | a front wheel is off the ground and both rear wheels stay down |
| **rear lift** | a rear wheel is off the ground and both front wheels stay down |
| **all-wheel lift** | a front wheel and a rear wheel are airborne together |

Samples before the vehicle settles are reported as *settling* and never counted as lift, because
every contact flag still reads zero during the start-up transient.

2.2 prints a one-line summary of the run. 2.3 shades the pedal bands, 2.4 overlays the lift classes,
and 2.5 draws the measurement on its own so the raw response can be judged before any model is
involved.
"""

# Cell 7: Section 2: measured data - load, lift classification and figures.

# ╔═╡ c0000000-0000-4000-8000-000000000003
begin
    """
        read_measurement(path)

    Load one recording into a `NamedTuple` of plain `Float64` vectors, one field per channel the model
    and the figures use. The recording is far wider than that - bounding-box corners, frame vectors
    and other diagnostics - and multi-channel time series are the dominant memory cost of this
    notebook, so the unread channels are never allocated.

    The selection is checked against the file's own header rather than the narrowed table, so a
    channel that exists but was not selected is never reported as missing. Used for both the training
    and the test recording, so the two are guaranteed to be read the same way.
    """
    function read_measurement(path)
        isfile(path) || error("Measurement file $(path) is missing; copy a recorded run here")
        recorded = names(CSV.read(path, DataFrame; comment="#", limit=0))
        unavailable = [name for name in used_columns if !(name in recorded)]
        isempty(unavailable) ||
            error("$(basename(path)) is missing required channels: $(unavailable)")
        table = CSV.read(path, DataFrame; comment="#", select=used_columns)
        measurement = NamedTuple{Tuple(Symbol.(used_columns))}(
            Tuple(Float64.(table[!, name]) for name in used_columns))
        # Release the tabular container: its column lookup and string columns are dead weight once
        # the `Float64` vectors exist, and they would otherwise sit in memory for the whole run.
        table = nothing
        GC.gc()
        return measurement
    end

    # Only the channels the model and the figures actually use.
    wheel_names = ("fl", "fr", "rl", "rr")
    used_columns = ["t_s", "s_travelled_m", "com_z", "gps_z", "pitch_rad", "speed_kmh",
        "speed_long_ms", "throttle_applied", "brake_applied", "engine_rpm",
        ["wheel_$(wheel)_contact" for wheel in wheel_names]...]

    # The analysis is pinned to one frozen measurement file that lives beside this notebook, not to
    # "the newest file in a directory": a further experiment writes a new timestamped CSV there, and
    # with a directory scan that silently changed the data under every fit and every figure.
    # Re-freeze by copying a new run over this file.
    latest_log = normpath(joinpath(@__DIR__, "longdyn_train.csv"))
    data = read_measurement(latest_log)

    # Number of samples, kept as a name because the table it was derived from no longer exists.
    sample_count = length(data.t_s)
    # Plot every n-th sample. A 200 Hz recording carries far more points than a 700 px panel can
    # resolve, so drawing all of them costs render time and browser memory for no visible detail.
    # This affects the figures only: the loss window, the segment plan and every reported statistic
    # stay at full resolution, so no fitted value can depend on it.
    plot_stride = 5
    plot_index = 1:plot_stride:sample_count

    # Convert the simulation-time column to numeric seconds for Makie axes.
    time_s = Float64.(data.t_s)
    # Time between plotted samples, used only to size a degenerate single-point axis.
    plot_step = plot_stride * (time_s[2] - time_s[1])
    # Prepare a short summary identifying the selected run and its sample/time range.
    summary = md"""
Selected **$(basename(latest_log))** with $(sample_count) samples
from $(round(first(time_s); digits=3)) s to $(round(last(time_s); digits=3)) s.
"""

    # Express CG vertical motion relative to the first recorded CG height.
    heave_cg_m = Float64.(data.com_z) .- Float64(first(data.com_z))
    # Express GPS vertical motion relative to the first recorded GPS height.
    heave_gps_m = Float64.(data.gps_z) .- Float64(first(data.gps_z))
    # Convert the recorded pitch angle from radians to degrees for plotting.
    pitch_deg = rad2deg.(Float64.(data.pitch_rad))
    # Read the recorded vehicle speed for the time-series panel.
    speed_kmh = Float64.(data.speed_kmh)
    # Convert the drive-speed channel to a numeric vector.
    rpm = Float64.(data.engine_rpm)
    # Torque ceiling of the drive model.
    torque_limit_nm = 9000.0
    # Power ceiling of the drive model.
    power_limit_w = 900000.0
    # Evaluate the torque curve at each RPM, using the torque ceiling at zero RPM.
    available_torque_nm = [rpm <= 0.0 ? torque_limit_nm :
        min(torque_limit_nm, power_limit_w * 60.0 / (2pi * rpm))
        for rpm in rpm]
    # Convert applied pedal channels to numeric vectors for shading and plotting.
    throttle = Float64.(data.throttle_applied)
    brake = Float64.(data.brake_applied)

    # A wheel is airborne when its contact flag is present and zero.
    airborne = Dict(wheel => [ismissing(flag) ? false : flag == 0
            for flag in getproperty(data, Symbol("wheel_", wheel, "_contact"))]
        for wheel in wheel_names)

    # The recording begins before the vehicle has settled: every flag is zero for the first samples,
    # which is a start-up transient rather than a lift-off. Anything before the first fully grounded
    # sample is therefore reported as "settling" and never counted as lift.
    all_grounded = [!(airborne["fl"][i] || airborne["fr"][i] || airborne["rl"][i] ||
        airborne["rr"][i]) for i in 1:sample_count]
    settled_index = findfirst(identity, all_grounded)
    settled_index === nothing && error("No sample has all four wheels grounded")

    # Three mutually exclusive lift classes, decided per axle and reported only after settling:
    #   all   - a front wheel and a rear wheel are both airborne
    #   front - a front wheel is airborne and no rear wheel is
    #   rear  - a rear wheel is airborne and no front wheel is
    #   none  - all four wheels are grounded
    lift_class = Vector{String}(undef, sample_count)
    for index in 1:sample_count
        if index < settled_index
            lift_class[index] = "settling"
            continue
        end
        front_airborne = airborne["fl"][index] || airborne["fr"][index]
        rear_airborne = airborne["rl"][index] || airborne["rr"][index]
        lift_class[index] = front_airborne ? (rear_airborne ? "all" : "front") :
            (rear_airborne ? "rear" : "none")
    end

    # The figures only need the sample indices of each airborne class.
    lift_indices = Dict(name => findall(==(name), lift_class) for name in ("front", "rear", "all"))
    lift_summary = let
        names = ["settling", "none", "front", "rear", "all"]
        counts = [count(==(name), lift_class) for name in names]
        DataFrame(lift_class=names, samples=counts,
            share_percent=round.(100.0 .* counts ./ sample_count; digits=2))
    end
end

# Cell 8: Load, validate and classify the measured run.

# ╔═╡ c0000000-0000-4000-8000-000000000004
summary

# Cell 9: Summary of the selected run.

# ╔═╡ c0000000-0000-4000-8000-000000000005
"""
    add_pedal_background!(axis, x_values, throttle_values, brake_values; kwargs...)

Shade contiguous pedal intervals behind an axis. The x and pedal vectors must be aligned and
ordered; keyword arguments control the colours and opacity.

Intervals are grouped by pedal **category** - idle, throttle, brake, or both - rather than by exact
level. A recorded schedule ramps its pedals continuously, so every sample has a slightly different
level; grouping by level would emit one span per ramp step, and on a long run that is tens of
thousands of plot objects across the figure, each one a real cost to build and to draw. Grouping by
category collapses that to roughly one span per schedule phase and looks the same at figure scale.
"""
function add_pedal_background!(axis, x_values, throttle_values, brake_values;
    single_step=0.005, activity_threshold=0.005,
    throttle_alpha=0.14, brake_alpha=0.14, overlap_alpha=0.20,
    throttle_color=:seagreen, brake_color=:orangered, overlap_color=:gold)
    sample_count = length(x_values)  # Use the x-axis vector length as the expected sample count.
    sample_count == 0 && return nothing  # Nothing can be shaded when there are no samples.
    # Reject mismatched vectors before indexing or drawing spans.
    length(throttle_values) == sample_count && length(brake_values) == sample_count ||
        throw(ArgumentError("x, throttle, and brake vectors must have equal lengths"))

    # Extend the first sample halfway backward, or by a configured step for a single point.
    edge_left(index) = index == 1 ? x_values[1] -
        (sample_count > 1 ? (x_values[2] - x_values[1]) / 2 : single_step / 2) :
        (x_values[index - 1] + x_values[index]) / 2
    # Extend the final sample halfway forward and otherwise use midpoints between samples.
    edge_right(index) = index == sample_count ? x_values[end] +
        (sample_count > 1 ? (x_values[end] - x_values[end - 1]) / 2 : single_step / 2) :
        (x_values[index] + x_values[index + 1]) / 2

    category_of(throttle_level, brake_level) =
        throttle_level > activity_threshold ?
            (brake_level > activity_threshold ? :both : :throttle) :
            (brake_level > activity_threshold ? :brake : :idle)

    function shade_segment!(first_index, last_index, category)
        category === :idle && return nothing  # Leave inactive pedal intervals unshaded.
        color, alpha = category === :both ? (overlap_color, overlap_alpha) :
            category === :throttle ? (throttle_color, throttle_alpha) :
            (brake_color, brake_alpha)
        # Draw one span covering all samples in the contiguous segment.
        vspan!(axis, edge_left(first_index), edge_right(last_index); color=(color, alpha))
        return nothing  # Return no plot object from this mutating helper.
    end

    segment_start = 1  # Start grouping at the first sample.
    segment_category = category_of(throttle_values[1], brake_values[1])
    for sample_index in 2:sample_count
        category = category_of(throttle_values[sample_index], brake_values[sample_index])
        if category != segment_category
            shade_segment!(segment_start, sample_index - 1, segment_category)
            segment_start = sample_index  # Begin a new segment at the changed sample.
            segment_category = category
        end
    end
    shade_segment!(segment_start, sample_count, segment_category)  # Final segment.
    nothing  # Complete the helper without returning a separate display value.
end

# Cell 10: Shade throttle and brake intervals behind the axes.

# ╔═╡ c0000000-0000-4000-8000-000000000006
"""
    add_lift_markers!(axis, x_values, y_values; lift_indices=lift_indices, kwargs...)

Overlay the lift-off samples on a time-aligned series, one marker style per airborne class. The
classes are mutually exclusive: a front lift is a front-axle wheel off the ground while both rear
wheels stay down, a rear lift is the mirror image, and an all-wheel lift is both axles airborne. The
start-up transient before the vehicle settles is not a lift and is never plotted.

`lift_indices` defaults to the training classification; pass the test recording's own classification
to annotate a figure drawn from it, so markers always match the trace they sit on.
"""
function add_lift_markers!(axis, x_values, y_values; lift_indices=lift_indices,
        marker_size=8, stroke_color=:white, stroke_width=0.8, show_label=false)
    length(x_values) == length(y_values) ||
        throw(ArgumentError("series x and y vectors must have equal lengths"))
    for (name, style) in pairs(lift_styles)
        indices = lift_indices[String(name)]
        isempty(indices) && continue
        scatter!(axis, x_values[indices], y_values[indices];
            color=style.color, marker=style.marker, markersize=marker_size,
            strokecolor=stroke_color, strokewidth=stroke_width,
            label=show_label ? style.label : "")
    end
    return nothing
end

# Cell 11: Overlay the three lift-off classes on a time series.

# ╔═╡ c0000000-0000-4000-8000-000000000008
begin
    timeseries_figure = let
        fig = Figure(size=(700, 1620), backgroundcolor=:white)
        ax_cg = Axis(fig[1, 1], ylabel="Heave at CG (m)", title="Measured response")
        ax_gps = Axis(fig[2, 1], ylabel="Heave at GPS (m)")
        ax_pitch = Axis(fig[3, 1], ylabel="Pitch (deg)")
        ax_speed = Axis(fig[4, 1], ylabel="Speed (km/h)")
        ax_rpm = Axis(fig[5, 1], ylabel="Motor speed (RPM)")
        ax_torque = Axis(fig[6, 1], ylabel="Available torque (Nm)")
        ax_inputs = Axis(fig[7, 1], ylabel="Pedal level", xlabel="Simulation time (s)",
            yticks=([0.0, 0.5, 1.0], ["0", "0.5", "1"]), limits=(nothing, (0.0, 1.05)))

        axes_all = (ax_cg, ax_gps, ax_pitch, ax_speed, ax_rpm, ax_torque, ax_inputs)
        for axis in axes_all
            # Repeat the same pedal intervals behind every response channel.
            add_pedal_background!(axis, time_s[plot_index], throttle[plot_index],
                brake[plot_index]; single_step=plot_step)
        end

        # Each response panel plots exactly one measured channel, so they all share one colour and
        # the y-axis label says which channel it is. Only every `plot_stride`-th sample is drawn.
        channels = ((ax_cg, heave_cg_m), (ax_gps, heave_gps_m), (ax_pitch, pitch_deg),
            (ax_speed, speed_kmh), (ax_rpm, rpm), (ax_torque, available_torque_nm))
        for (axis, channel) in channels
            lines!(axis, time_s[plot_index], channel[plot_index];
                color=measured_color, linewidth=1.5)
            add_lift_markers!(axis, time_s, channel)
        end

        # The pedal panel is the only panel with two series, so it keeps two colours.
        for (channel, color) in ((throttle, :seagreen), (brake, :orangered))
            lines!(ax_inputs, time_s[plot_index], channel[plot_index];
                color=color, linewidth=1.2)
            add_lift_markers!(ax_inputs, time_s, channel)
        end

        # Link the time axes so a zoom or pan on any panel applies to all of them.
        linkxaxes!(axes_all...)
        for axis in (ax_cg, ax_gps, ax_pitch, ax_speed, ax_rpm, ax_torque)
            hidexdecorations!(axis; grid=false)
        end
        rowsize!(fig.layout, 7, Fixed(150))

        legend_elements = [
            LineElement(color=measured_color, linewidth=1.5),
            LineElement(color=:seagreen, linewidth=1.2),
            LineElement(color=:orangered, linewidth=1.2),
            [MarkerElement(color=style.color, marker=style.marker, markersize=8,
                strokecolor=:white, strokewidth=0.8) for style in values(lift_styles)]...
        ]
        legend_labels = ["Measured", "Throttle", "Brake",
            [style.label for style in values(lift_styles)]...]
        Legend(fig[8, 1], legend_elements, legend_labels; orientation=:horizontal,
            nbanks=3, tellwidth=false, framevisible=false, labelsize=10,
            patchsize=(18, 10), rowgap=3, colgap=8)
        rowsize!(fig.layout, 8, Fixed(115))
        fig
    end
end

# Cell 12: Stacked measured-time-series figure.

# ╔═╡ c0000000-0000-4000-8000-000000000009
md"""
## 3. Half-car model

Six physical degrees of freedom: sprung heave `z_s`, pitch `θ` (positive = nose down), front and
rear unsprung heave `z_uf`/`z_ur`, plus the longitudinal position `x` and velocity `v`. The
equations are a standard two-axle quarter-car pair coupled through the sprung body.

The section is ordered bottom-up: configuration constants, then the parameters, the road profile,
the right-hand side, a single-segment integration, the segmented driver used for plotting, and
finally the segmentation of the measured run itself.

`halfcar_segment_history` solves a single segment and returns the raw state matrix; its element type
follows the parameters, which is what lets ForwardDiff push `Dual` numbers through the solver.
`halfcar_simulate` wraps it into the full-length `NaN`-padded channels used for plotting.
"""

# Cell 13: Section 3: half-car model - parameters, road, RHS and segmentation.

# ╔═╡ c0000000-0000-4000-8000-00000000000a
begin
    # Road table: empty means a flat surface, which is the assumption for this dataset.
    road_stations_x = Float64[]
    road_stations_z = Float64[]
    # Sampling grid and the segment rules (0.1 s minimum length and a 2-sample pad at 200 Hz).
    halfcar_dt = 0.005
    halfcar_wheel_names = ("fl", "fr", "rl", "rr")
    halfcar_min_segment_samples = 20
    halfcar_pad_indices = 2
    # Integration tolerance shared by every simulation — the baseline, the loss and the final runs —
    # so that a comparison never mixes two different solver accuracies.
    solver_tolerance = 1.0e-5
end

# Cell 14: Model configuration - road table, sample period and segment rules.

# ╔═╡ c0000000-0000-4000-8000-00000000000b
"""
    halfcar_params(; kwargs...)

Complete parameter set of the half-car model, as a named tuple. Parameters are either known
figures with a small tolerance or estimates with a wide one - the bounds table in section 5 records
which is which - and `throttle_force` is the unknown that the fit identifies from the measured
speed. Everything derived (masses, axle loads, CG split, static heights, preloads) is computed here
once, so the model function only ever sees numbers. Pass any field by name to override it, e.g.
`halfcar_params(throttle_force=34000.0)`.
"""
function halfcar_params(; wheelbase=2.71, wheel_radius=0.358,
        spring_front=110000.0, spring_rear=110000.0,
        damper_front=5700.0, damper_rear=5700.0,
        tyre_spring=250000.0, tyre_damper=300.0,
        unsprung_mass=31.0,
        pitch_inertia=3500.0, cg_height=0.75,
        max_torque=9000.0, max_power=900000.0, gear_ratio=9.6, traction_acceleration=17.9,
        engine_max_rpm=4500.0, redline_span=80.0,
        throttle_force=16000.0, brake_coefficient=8000.0, brake_scale=1.0,
        brake_smoothing=0.05,
        resistance=100.0, resistance_smoothing=0.01, gravity=9.81,
        static_sprung_height=0.4789, static_pitch=0.008185,
        front_mass_share=0.071 / (0.071 + 0.049))

    # Mass is anchored, not estimated. The vehicle proto states `physics { mass 2700 }` on the Car body
    # node with the four `VehicleWheel` nodes on the default `Physics {}`, and the reading adopted here is
    # that 2700 is the vehicle's *total*: v2.31 fitted 2685 kg from dynamics alone, 0.6 % away, and both
    # fits that resolved the vertical dynamics independently agree on ~123 kg of unsprung mass.
    #
    # One unsprung mass per axle, shared front and rear, because the four wheels are the same
    # `VehicleWheel` PROTO with the same `tireRadius` - nothing distinguishes them. v2.31 left the two free
    # and split them 1.6:1, which is not physical; v2.32 and v2.33 both wanted symmetric values near
    # 22-33 kg per wheel. Tying them removes that free direction rather than fitting it.
    #
    # The sprung mass then follows: `total = sprung + 2 * unsprung`, so anchoring the total to 2700 makes
    # sprung a dependent quantity. The estimated set shrinks to 21 parameters and the
    # `throttle_force`/mass degeneracy that made v2.31 and v2.32 disagree by 460 kg is gone by
    # construction - there is no mass direction left for the optimiser to slide along.
    unsprung_front = unsprung_mass
    unsprung_rear = unsprung_mass
    sprung_mass = 2700.0 - unsprung_front - unsprung_rear

    total_mass = sprung_mass + unsprung_front + unsprung_rear
    front_axle_load = total_mass * gravity * front_mass_share
    rear_axle_load = total_mass * gravity * (1.0 - front_mass_share)
    # The suspension carries the sprung mass only; the tyre carries the whole axle.
    suspension_force_front = front_axle_load - unsprung_front * gravity
    suspension_force_rear = sprung_mass * gravity - suspension_force_front
    # CG split from the sprung load share, so the static pitch moment is exactly zero.
    cg_to_front = wheelbase * suspension_force_rear /
        (suspension_force_front + suspension_force_rear)
    cg_to_rear = wheelbase - cg_to_front
    # One unsprung mass per axle, so the rates are twice the per-wheel values.
    suspension_stiffness_front = 2.0 * spring_front
    suspension_stiffness_rear = 2.0 * spring_rear
    suspension_damping_front = 2.0 * damper_front
    suspension_damping_rear = 2.0 * damper_rear
    tyre_stiffness_front = 2.0 * tyre_spring
    tyre_stiffness_rear = 2.0 * tyre_spring
    tyre_damping_front = 2.0 * tyre_damper
    tyre_damping_rear = 2.0 * tyre_damper
    unsprung_front_static = wheel_radius - front_axle_load / tyre_stiffness_front
    unsprung_rear_static = wheel_radius - rear_axle_load / tyre_stiffness_rear
    body_front_static = static_sprung_height - cg_to_front * static_pitch
    body_rear_static = static_sprung_height + cg_to_rear * static_pitch

    return (; wheelbase, wheel_radius, sprung_mass, unsprung_front, unsprung_rear,
        unsprung_mass,
        pitch_inertia, cg_height, max_torque, max_power, gear_ratio, traction_acceleration,
        engine_max_rpm, redline_span, throttle_force,
        brake_coefficient, brake_scale, brake_smoothing,
        resistance, resistance_smoothing, gravity,
        spring_front, spring_rear, damper_front, damper_rear, tyre_spring, tyre_damper,
        front_mass_share,
        static_sprung_height, static_pitch, total_mass, front_axle_load, rear_axle_load,
        cg_to_front, cg_to_rear, suspension_stiffness_front, suspension_stiffness_rear,
        suspension_damping_front, suspension_damping_rear, tyre_stiffness_front,
        tyre_stiffness_rear, tyre_damping_front, tyre_damping_rear,
        unsprung_front_static, unsprung_rear_static, body_front_static, body_rear_static,
        suspension_force_front, suspension_force_rear,
        tyre_force_front=front_axle_load, tyre_force_rear=rear_axle_load)
end

# Cell 15: Half-car parameters as a named tuple.

# ╔═╡ c0000000-0000-4000-8000-00000000000c
"""
    road_height(stations_x, stations_z, position)

Road height under a contact patch, linearly interpolated between the supplied stations. An empty
table is a perfectly flat surface and positions outside the table read zero, so a flat run needs no
road data while an uneven surface only needs its stations added.
"""
function road_height(stations_x, stations_z, position)
    isempty(stations_x) && return 0.0
    (position <= first(stations_x) || position >= last(stations_x)) && return 0.0
    upper_index = searchsortedfirst(stations_x, position)
    lower_index = upper_index - 1
    span = stations_x[upper_index] - stations_x[lower_index]
    span <= 0.0 && return stations_z[lower_index]
    return stations_z[lower_index] + (position - stations_x[lower_index]) / span *
        (stations_z[upper_index] - stations_z[lower_index])
end

# Cell 16: Road height under a contact patch.

# ╔═╡ c0000000-0000-4000-8000-0000000000c1
"""
    smooth_min(a, b; k)

Differentiable minimum, `0.5(a + b - sqrt((a - b)^2 + k^2))`.

A plain `min` is continuous but not continuously differentiable: at the switching point its slope jumps
from 1 to 0, so ForwardDiff returns one side's slope and the optimiser is handed a kink it cannot
differentiate through. This rounds the corner over a band of width `k` instead. The band is deliberately
narrow - a few tens of newtons against forces of tens of kilonewtons - so the model stays within a
fraction of a percent of the hard minimum wherever it matters while remaining smooth everywhere.
"""
smooth_min(a, b; k=200.0) = 0.5 * (a + b - sqrt((a - b)^2 + k^2))

# Cell 16b: Differentiable minimum, for the drive and brake limits.

# ╔═╡ c0000000-0000-4000-8000-00000000000d
"""
    halfcar_rhs!(du, u, p, t)

The half-car model, in the form DifferentialEquations.jl expects. `u` holds six coordinates and
their rates: `z_s, θ, z_uf, z_ur, x, v, dz_s, dθ, dz_uf, dz_ur`. `p` holds the parameters from
`halfcar_params` plus the held pedal vectors, the sample period and the road table.

`θ` is positive nose down, so the front of the body sits at
`z_s - a·θ` and the rear at `z_s + b·θ`.
"""
function halfcar_rhs!(du, u, p, t)
    z_s, theta, z_uf, z_ur, position, velocity = u[1], u[2], u[3], u[4], u[5], u[6]
    dz_s, dtheta, dz_uf, dz_ur = u[7], u[8], u[9], u[10]

    # Pedals are held constant between measurement samples.
    sample = clamp(floor(Int, t / p.sample_period) + 1, 1, length(p.throttle))
    throttle = p.throttle[sample]
    brake = p.brake[sample]

    # Longitudinal force: a drive limited by the motor curve and by wheel spin, a constant-torque brake,
    # and one speed-proportional resistance. The drive is normalised by `max_torque` so that a single
    # scale, `throttle_force`, carries the whole traction magnitude and is therefore identifiable from the
    # measured speed.
    rpm = abs(velocity) / p.wheel_radius * p.gear_ratio * 60.0 / (2pi)
    # Constant torque to the base speed, constant power after it, normalised by `max_torque` so
    # `throttle_force` stays the single traction scale.
    torque = rpm <= 0.0 ? p.max_torque : min(p.max_torque, p.max_power * 60.0 / (2pi * rpm))
    # The drive limit is a speed ceiling, not a torque ceiling: the recorded signal saturates at a
    # fixed rpm, above which no further torque is delivered. Below it the power-limited torque still
    # exceeds what the tyres can take, so the ceiling - not the power limit - sets the top speed.
    # A `clamp` would leave a flat region whose switching time the autodiff cannot see, so the edge
    # is a tanh: one far below `engine_max_rpm`, zero far above, smooth in between.
    redline = 0.5 * (1.0 - tanh((rpm - p.engine_max_rpm) / p.redline_span))
    demand = throttle * p.throttle_force * (torque / p.max_torque) * redline
    # What actually limits the launch is not the torque curve but the wheel-spin guard in the vehicle
    # library. driver.c:52 defines ACCELERATION_THRESHOLD 50 rad/s2 and driver.c:191-194 scales the
    # output torque down by 50/|a_wheel| whenever the wheels spin up faster than that, so the car can
    # never pull harder than about 50 rad/s2 * r = 17.9 m/s2 no matter what the motor delivers. The
    # recording confirms it: launch saturates near 1.9 g while the torque curve asks for 4.4 g.
    #
    # The cap is a body acceleration, `traction_acceleration`, rather than the library's wheel
    # acceleration, because the guard acts on a slipping tyre and the body follows through the slip
    # ratio, not one for one; 17.9 m/s2 is the anchor the fit starts from, not an assumed identity.
    # `smooth_min` rather than `min`: a hard minimum is C0 but not C1, so its switching point would be
    # invisible to the gradient and the optimiser would see a kink it cannot differentiate through.
    drive = smooth_min(demand, p.traction_acceleration * p.total_mass)
    # One resistance term, proportional to speed. Measured, not assumed: with both pedals at zero the
    # recorded deceleration is a least-squares fit of `F = -141 v` over 11 591 samples, i.e. it scales
    # with speed and has no meaningful constant part. `analysis/live/verify_webots_laws.jl` repeats that
    # comparison algebraically, sample by sample, without integrating anything.
    #
    # The tanh keeps the sign and fades the force to zero at standstill, so it can never drive the car
    # backwards; a hard `sign(velocity)` would jump between +R and -R at zero and make the right-hand
    # side discontinuous.
    resistance = p.resistance * abs(velocity) * tanh(velocity / p.resistance_smoothing)
    # The brake is a constant torque, because that is what the vehicle applies. The controller that
    # produced the recording drives a Webots `Brake` device, and the vehicle library cancels the speed
    # dependence of that damping device explicitly:
    #   damping_constant = |brake * brake_coefficient / omega|   =>   torque = brake * brake_coefficient
    # so the pad torque is set by the pedal alone, at any road speed. Four wheels, divided by the
    # rolling radius to give the contact force.
    #
    # No grip cap, deliberately. Clipping the demand at `tyre_friction * m * g` is what limits a real
    # tyre, but it turns the fit into a one-parameter search for the friction coefficient and lets it
    # press against its own ceiling while other terms compensate. Without it the brake force is fully
    # determined by the pedal and the known `brake_coefficient`, and the fit can only answer whether a
    # constant torque explains the recording by itself.
    #
    # The torque must still vanish at standstill or it would drive the car backwards. Webots gets that
    # from the same division by speed (the damping constant diverges, holding the wheel); here it is a
    # `tanh` edge, which is smooth and can be differentiated.
    brake_force = 4.0 * p.brake_coefficient * p.brake_scale * brake / p.wheel_radius *
        tanh(velocity / p.brake_smoothing)
    longitudinal = drive - brake_force - resistance

    # Vertical forces as perturbations about the measured static ride height.
    body_front = z_s - p.cg_to_front * theta
    body_rear = z_s + p.cg_to_rear * theta
    road_front = road_height(p.road_x, p.road_z, position + p.cg_to_front)
    road_rear = road_height(p.road_x, p.road_z, position - p.cg_to_rear)
    suspension_front = p.suspension_force_front +
        p.suspension_stiffness_front * ((z_uf - p.unsprung_front_static) -
            (body_front - p.body_front_static)) +
        p.suspension_damping_front * (dz_uf - (dz_s - p.cg_to_front * dtheta))
    suspension_rear = p.suspension_force_rear +
        p.suspension_stiffness_rear * ((z_ur - p.unsprung_rear_static) -
            (body_rear - p.body_rear_static)) +
        p.suspension_damping_rear * (dz_ur - (dz_s + p.cg_to_rear * dtheta))
    # The tyre acts between wheel and road, so raising the wheel reduces its compression.
    tyre_front = p.tyre_force_front +
        p.tyre_stiffness_front * (road_front - (z_uf - p.unsprung_front_static)) -
        p.tyre_damping_front * dz_uf
    tyre_rear = p.tyre_force_rear +
        p.tyre_stiffness_rear * (road_rear - (z_ur - p.unsprung_rear_static)) -
        p.tyre_damping_rear * dz_ur

    # A positive pitch is nose down, so a forward force at ground level pitches the body up.
    pitch_moment = -p.cg_to_front * suspension_front + p.cg_to_rear * suspension_rear -
        p.cg_height * longitudinal

    du[1] = dz_s
    du[2] = dtheta
    du[3] = dz_uf
    du[4] = dz_ur
    du[5] = velocity
    du[6] = longitudinal / p.total_mass
    du[7] = (suspension_front + suspension_rear - p.sprung_mass * p.gravity) / p.sprung_mass
    du[8] = pitch_moment / p.pitch_inertia
    du[9] = (tyre_front - suspension_front - p.unsprung_front * p.gravity) / p.unsprung_front
    du[10] = (tyre_rear - suspension_rear - p.unsprung_rear * p.gravity) / p.unsprung_rear
    return nothing
end

# Cell 17: The half-car right-hand side for DifferentialEquations.jl.

# ╔═╡ c0000000-0000-4000-8000-00000000000e
"""
    halfcar_segment_history(parameters, segment, data; kwargs...)

Integrate one segment with DifferentialEquations.jl and return the `10 × n` state history, one
column per measurement sample.

The element type follows the parameters: passing a `Float64` set gives a `Float64` history, passing
ForwardDiff `Dual` numbers gives a `Dual` history. That is what makes the same function usable both
for plotting and inside the optimizer's gradient.
"""
function halfcar_segment_history(parameters, segment, data;
        road_x=Float64[], road_z=Float64[], dt=0.005, abstol=1.0e-6, reltol=1.0e-6)
    first_index = segment.simulation_start
    last_index = segment.stop_index
    indices = first_index:last_index
    segment_samples = length(indices)

    # Measured state at the first simulated sample.
    body = Float64(data.com_z[first_index])
    pitch = Float64(data.pitch_rad[first_index])
    body_front = body - parameters.cg_to_front * pitch
    body_rear = body + parameters.cg_to_rear * pitch
    # Unsprung masses are not measured: start them at their quasi-static equilibrium.
    front_offset = (body_front - parameters.body_front_static) *
        parameters.suspension_stiffness_front /
        (parameters.suspension_stiffness_front + parameters.tyre_stiffness_front)
    rear_offset = (body_rear - parameters.body_rear_static) *
        parameters.suspension_stiffness_rear /
        (parameters.suspension_stiffness_rear + parameters.tyre_stiffness_rear)
    initial_state = [body, pitch,
        parameters.unsprung_front_static + front_offset,
        parameters.unsprung_rear_static + rear_offset,
        Float64(data.s_travelled_m[first_index]),
        Float64(data.speed_long_ms[first_index]), 0.0, 0.0, 0.0, 0.0]

    segment_parameters = (; parameters...,
        throttle=Float64.(data.throttle_applied[indices]),
        brake=Float64.(data.brake_applied[indices]),
        sample_period=dt, road_x=road_x, road_z=road_z)
    save_times = collect(0:(segment_samples - 1)) .* dt
    problem = ODEProblem(halfcar_rhs!, initial_state, (0.0, save_times[end]), segment_parameters)
    solution = solve(problem, Tsit5(); saveat=save_times, abstol=abstol, reltol=reltol)
    # A solver that gives up early returns fewer saved points. Report that as a NaN history so the
    # loss can treat the candidate as invalid instead of indexing past the end of the matrix.
    length(solution.u) == segment_samples || return fill(NaN, 10, segment_samples)
    return reduce(hcat, solution.u)
end

# Cell 18: Single-segment integration, type generic for ForwardDiff.

# ╔═╡ c0000000-0000-4000-8000-00000000000f
"""
    halfcar_simulate(parameters, segments, data; kwargs...)

Integrate every segment and return the channels as full-length vectors, `NaN` outside the segments
so lift-off intervals appear as gaps. This is the plotting path: it allocates `Float64` buffers and
so must be called with ordinary numbers, not ForwardDiff duals.

Channels: `com_height`, `sprung_height`, `pitch_rad`, `velocity_ms`, `position_m`,
`wheel_load_front_n`, `wheel_load_rear_n` and `available_torque_nm`.
"""
function halfcar_simulate(parameters, segments, data;
        road_x=Float64[], road_z=Float64[], dt=0.005, abstol=1.0e-6, reltol=1.0e-6)
    sample_count = length(data.t_s)
    com_height = fill(NaN, sample_count)
    sprung_height = fill(NaN, sample_count)
    pitch_rad = fill(NaN, sample_count)
    velocity_ms = fill(NaN, sample_count)
    position_m = fill(NaN, sample_count)
    wheel_load_front_n = fill(NaN, sample_count)
    wheel_load_rear_n = fill(NaN, sample_count)
    available_torque_nm = fill(NaN, sample_count)

    for segment in eachrow(segments)
        first_index = segment.simulation_start
        last_index = segment.stop_index
        last_index < first_index && continue
        indices = first_index:last_index
        segment_samples = length(indices)

        history = halfcar_segment_history(parameters,
            (simulation_start=first_index, stop_index=last_index), data;
            road_x=road_x, road_z=road_z, dt=dt, abstol=abstol, reltol=reltol)

        com_height[indices] = (parameters.sprung_mass .* history[1, :] .+
            parameters.unsprung_front .* history[3, :] .+
            parameters.unsprung_rear .* history[4, :]) ./ parameters.total_mass
        sprung_height[indices] = history[1, :]
        pitch_rad[indices] = history[2, :]
        velocity_ms[indices] = history[6, :]
        position_m[indices] = history[5, :]
        front_station = history[5, :] .+ parameters.cg_to_front
        rear_station = history[5, :] .- parameters.cg_to_rear
        wheel_load_front_n[indices] = parameters.tyre_force_front .+
            parameters.tyre_stiffness_front .* (road_height.(Ref(road_x), Ref(road_z),
                front_station) .- (history[3, :] .- parameters.unsprung_front_static))
        wheel_load_rear_n[indices] = parameters.tyre_force_rear .+
            parameters.tyre_stiffness_rear .* (road_height.(Ref(road_x), Ref(road_z),
                rear_station) .- (history[4, :] .- parameters.unsprung_rear_static))

        # Available drive torque: the motor curve at the modelled motor speed.
        segment_rpm = abs.(history[6, :]) ./ parameters.wheel_radius .*
            parameters.gear_ratio .* 60.0 ./ (2pi)
        available_torque_nm[indices] = [rpm <= 0.0 ? parameters.max_torque :
            min(parameters.max_torque, parameters.max_power * 60.0 / (2pi * rpm))
            for rpm in segment_rpm]
    end

    return (; com_height, sprung_height, pitch_rad, velocity_ms, position_m,
        wheel_load_front_n, wheel_load_rear_n, available_torque_nm)
end

# Cell 19: Segmented simulation driver used for plotting.

# ╔═╡ c0000000-0000-4000-8000-000000000010
"""
    extract_segments(data; kwargs...)

Split a measurement record into the intervals in which **all four wheels** report ground contact. The
boundaries come straight from the `wheel_*_contact` flags, so a new run is segmented automatically.

Runs shorter than `min_segment_samples` are discarded: they are landing or take-off transients with
no usable dynamic content (the current run contains a single-sample one). `simulation_start` is the
segment start plus `pad_indices`, which skips the tyre settling transient that follows a wheel
touching down before the model starts.

Returns the raw contiguous `runs` and the `segments` table.
"""
function extract_segments(data; wheel_names=("fl", "fr", "rl", "rr"),
        min_segment_samples=20, pad_indices=2)
    sample_count = length(data.t_s)
    # One lookup per wheel, outside the per-sample loop.
    contact = Dict(wheel => getproperty(data, Symbol("wheel_", wheel, "_contact"))
        for wheel in wheel_names)
    # A wheel counts as grounded only when the flag is present and equal to 1.
    grounded = [all(!ismissing(contact[wheel][i]) && contact[wheel][i] == 1
        for wheel in wheel_names) for i in 1:sample_count]

    runs = Tuple{Int, Int}[]
    run_start = 0
    for i in 1:sample_count
        if grounded[i]
            run_start == 0 && (run_start = i)
        elseif run_start != 0
            push!(runs, (run_start, i - 1))
            run_start = 0
        end
    end
    run_start != 0 && push!(runs, (run_start, sample_count))

    # Keep runs long enough to survive the pad and still hold samples to compare.
    kept = [run for run in runs if (run[2] - run[1] + 1) >= min_segment_samples]
    segments = DataFrame(
        segment=collect(1:length(kept)),
        start_index=[run[1] for run in kept],
        stop_index=[run[2] for run in kept],
        simulation_start=[run[1] + pad_indices for run in kept],
        sample_count=[run[2] - run[1] + 1 for run in kept],
        simulated_count=[run[2] - (run[1] + pad_indices) + 1 for run in kept],
        start_time_s=[Float64(data.t_s[run[1]]) for run in kept],
        stop_time_s=[Float64(data.t_s[run[2]]) for run in kept],
        start_speed_kmh=[Float64(data.speed_kmh[run[1]]) for run in kept])
    return (runs=runs, segments=segments)
end

# Cell 21: Split the measured run into all-wheels-grounded segments.

# ╔═╡ c0000000-0000-4000-8000-000000000011
begin
    halfcar_extraction = extract_segments(data;
        wheel_names=halfcar_wheel_names,
        min_segment_samples=halfcar_min_segment_samples,
        pad_indices=halfcar_pad_indices)
    halfcar_all_runs = halfcar_extraction.runs
    halfcar_segments = halfcar_extraction.segments

    segment_summary = md"""
Found **$(length(halfcar_all_runs))** contiguous all-wheels-grounded runs;
**$(nrow(halfcar_segments))** of them are simulated with a $(halfcar_pad_indices)-sample pad.
Only the segments overlapping the loss window take part in the optimization.
"""
end

# Cell 22: Segmentation of the currently loaded run.

# ╔═╡ c0000000-0000-4000-8000-000000000032
"""
    halfcar_plot_channels(parameters)

Run the segmented simulation with the plotting tolerances and return its channels. A thin wrapper so
the figure code never repeats solver options.
"""
halfcar_plot_channels(parameters) = halfcar_simulate(parameters, halfcar_segments, data;
    road_x=road_stations_x, road_z=road_stations_z, dt=halfcar_dt,
    abstol=solver_tolerance, reltol=solver_tolerance)

# Cell 20: Simulation adapter used by the figures.

# ╔═╡ c0000000-0000-4000-8000-000000000050
md"""
## 4. Pre-optimization parameter guess

This section is the **starting point** of the fit, kept in its own place so it is never confused with
the fitted result.

`guess_parameters` holds the full half-car parameter set at its pre-optimization estimate. Nothing
here is fitted:

| group | source |
|---|---|
| wheelbase, wheel radius, gear ratio, drive torque and drive power | nameplate values with a build tolerance |
| static ride height and static pitch | measured from the settled part of the record |
| spring, damper, sprung mass and pitch inertia | estimates from a comparable machine |
| tyre rates, CG height, drive force, wheel-spin cap, linear resistance | unknowns, bounded only by physical plausibility |

The optimizer below may move only the values listed in the bounds table, and only within the width
justified by their uncertainty. Every other field stays at the estimate.
"""

# Cell 23: Section 4: pre-optimization parameter guess, kept separate from the fit.

# ╔═╡ c0000000-0000-4000-8000-000000000051
begin
    # The pre-optimization estimate. `halfcar_params()` returns the documented defaults, which makes
    # this the single source of truth for every unfitted value in the notebook.
    guess_parameters = halfcar_params()
    guess_table = DataFrame(
        parameter=String.(keys(guess_parameters)),
        guess=[Float64(value) for value in values(guess_parameters)])
end

# Cell 24: The pre-optimization parameter guess.

# ╔═╡ c0000000-0000-4000-8000-000000000052
begin
    # Baseline trajectory of the guess. Computed here, before the optimizer runs, so the comparison
    # figure only ever consumes it.
    before_run = halfcar_plot_channels(guess_parameters)
    nothing
end

# Cell 25: Baseline trajectory produced by the guess alone.

# ╔═╡ c0000000-0000-4000-8000-000000000012
md"""
## 5. Optimization

This section reads top to bottom in the order the fit is built:

1. **Window and budget** — which samples the loss sees, and how long the solve may run.
2. **Bounds** — how far each parameter may move, taken from the guess in section 4.
3. **Coordinates** — the normalised vector the optimizer actually varies.
4. **Window selection and scaling** — how the strided window is chosen and each channel normalised.
5. **Loss data** — the measured samples and channel scales, precomputed once.
6. **Residual and objective** — the function that turns a parameter vector into a scalar.
7. **Problem** — the Optimization.jl wrapper with bounds and autodiff.
8. **Contract check** — proof that the loss really is a function of the parameters.
9. **Solve** — BFGS with a callback that streams progress.

Every continuous parameter gets a lower and an upper bound, and the width follows our uncertainty:
nameplate values are trusted to a couple of percent, while estimates such as the tyre rates or the CG
height are given a wide range and the unknown force scales a wider one still.

Those bounds are then mapped onto a normalised coordinate, so the optimizer never sees metres,
kilograms, newtons or watts. It only ever sees a vector in `[-1, 1]`, where `-1` is the lower bound,
`0` the middle of the trusted range and `+1` the upper bound. Because the map is affine, the gradients
that ForwardDiff produces are simply rescaled by `half_width`, and the result converts back to
physical units with the one-line inverse.

The loss compares heave, pitch and velocity over the selected window, each divided by its own spread
so a decimetre of heave and ten km/h of speed contribute comparably.
"""

# Cell 26: Section 5: optimization - window, bounds, loss and solve.

# ╔═╡ c0000000-0000-4000-8000-000000000013
begin
    # Trajectory the loss sees: 1-based sample indices into the full log, inclusive.
    # The window opens 2.5 s in, so the initial settling transient - the car dropping onto its
    # springs, with every contact flag still zero - never enters the residual. It is a time
    # lookup rather than a sample count so it survives a log with a different time step.
    loss_window_start_s = 2.5
    window_start_index = something(findfirst(index -> time_s[index] >= loss_window_start_s,
        eachindex(time_s)), 1)
    window_stride = 5
    window_end_index = sample_count     # inclusive; the last sample of the log

    # Optimizer budget. The callback stops BFGS at the time limit and keeps the best parameters
    # found so far, so a long run can never lock up the notebook.
    max_iterations = 40
    # Effectively unconstrained: the callback stops only when `elapsed > max_time_s`, so a very
    # large finite value lets all 200 iterations finish. Finite rather than `Inf` so the budget
    # text below still renders a number, and the per-iteration timing plot is meaningful.
    max_time_s = 1200.0
end

# Cell 27: Loss window and optimizer budget.

# ╔═╡ c0000000-0000-4000-8000-000000000014
begin
    """
    Build the optimization bounds as a table: parameter name, nominal value, lower and upper bound,
    and the uncertainty that justifies the width.
    """
    estimation_parameter_bounds = let
        # The nominal column is the pre-optimization estimate, so the bounds and the starting point
        # can never drift away from section 4.
        nominal = guess_parameters
        rows = [
            (:wheelbase,            2.656,     2.764,     "known: nameplate, ±2 %"),
            (:wheel_radius,         0.3401,    0.3759,    "weakly known: nameplate ±5 %, widened - it finished on the ±2 % ceiling"),
            (:gear_ratio,           9.504,     9.696,     "known: nameplate, ±1 %"),
            (:max_torque,           8820.0,    9180.0,    "known: nameplate, ±2 %"),
            (:max_power,            882000.0,  918000.0,  "known: nameplate, ±2 %"),
            (:static_sprung_height, 0.4589,    0.4989,    "measured: settled ride height, ±20 mm"),
            (:static_pitch,         0.004185,  0.012185,  "measured: settled pitch, ±0.004 rad"),
            (:spring_front,         60000.0,   220000.0,  "weakly known: comparable machine, widened - it finished on the ceiling"),
            (:spring_rear,          70000.0,   140000.0,  "estimated: comparable machine"),
            (:damper_front,         3000.0,    12000.0,   "estimated: comparable machine"),
            (:damper_rear,          3000.0,    12000.0,   "estimated: comparable machine"),
            (:front_mass_share,     0.54,      0.64,      "estimated: axle-load split"),
            (:unsprung_mass,        20.0,      100.0,     "unknown: one per axle, shared front and rear; ~31 kg per wheel from v2.31/v2.33 agreement"),
            (:pitch_inertia,        1500.0,    8000.0,    "unknown: broad physical range"),
            (:cg_height,            0.30,      1.10,      "unknown: broad physical range"),
            (:tyre_spring,          120000.0,  1800000.0, "unknown: identified from the data, widened - it finished on the ceiling"),
            (:tyre_damper,          0.0,       3000.0,    "unknown: identified from the data"),
            (:throttle_force,       5000.0,    250000.0,  "unknown: identified from the data, bounded by the drivetrain"),
            (:traction_acceleration, 8.0,      30.0,      "known: driver.c ACCELERATION_THRESHOLD 50 rad/s2 x r = 17.9 m/s2, one-sided band"),
            (:brake_scale,          0.20,      2.00,      "unknown: gain on the known pad torque"),
            (:resistance,           0.0,       600.0,     "unknown: linear resistance, ~141 N per m/s measured from the coast"),
        ]
        DataFrame(
            parameter=[string(row[1]) for row in rows],
            nominal=[Float64(getfield(nominal, row[1])) for row in rows],
            lower=[Float64(row[2]) for row in rows],
            upper=[Float64(row[3]) for row in rows],
            confidence=[row[4] for row in rows])
    end

    estimation_names = Symbol.(estimation_parameter_bounds.parameter)
    estimation_lower = estimation_parameter_bounds.lower
    estimation_upper = estimation_parameter_bounds.upper
    estimation_nominal = estimation_parameter_bounds.nominal

    # Normalised coordinate: the lower bound maps to -1 and the upper bound to +1, so every parameter
    # gets the same search box regardless of its physical units or magnitude.
    estimation_center = (estimation_lower .+ estimation_upper) ./ 2
    estimation_half_width = (estimation_upper .- estimation_lower) ./ 2
    estimation_start_normalised =
        (estimation_nominal .- estimation_center) ./ estimation_half_width

    estimation_parameter_bounds.normalised_start = estimation_start_normalised
    estimation_parameter_bounds
end

# Cell 28: Bounds for every continuous parameter, scaled by confidence.

# ╔═╡ c0000000-0000-4000-8000-000000000015
begin
"""
    parameters_from_vector(values)

Rebuild the full parameter named tuple from a physical vector, one entry per optimized parameter.
Unlisted parameters keep the values from `halfcar_params()`.
"""
function parameters_from_vector(values)
    overrides = Dict{Symbol, Any}(estimation_names[i] => values[i]
        for i in eachindex(estimation_names))
    return halfcar_params(; (key => value for (key, value) in overrides)...)
end

"""
    vector_from_parameters(parameters)

Inverse of `parameters_from_vector`: pull the physical parameters back out of the named tuple.
"""
function vector_from_parameters(parameters)
    return [Float64(getfield(parameters, name)) for name in estimation_names]
end

"""
    denormalise(normalised)

Map normalised parameters in `-1 … 1` onto their physical range: `-1` gives the lower bound, `0` the
middle of the trusted range and `+1` the upper bound. The map is affine, so ForwardDiff duals pass
through unchanged and only pick up a `half_width` factor.
"""
function denormalise(normalised)
    return estimation_center .+ estimation_half_width .* normalised
end

"""
    normalise(values)

Inverse of `denormalise`: express physical parameters in the normalised `-1 … 1` coordinate.
"""
function normalise(values)
    return (values .- estimation_center) ./ estimation_half_width
end

"""
    parameters_from_normalised(normalised)

Rebuild the parameter named tuple from the normalised `-1 … 1` vector the optimizer works on.
"""
function parameters_from_normalised(normalised)
    return parameters_from_vector(denormalise(normalised))
end
end

# Cell 29: Map between the optimizer vector and the parameter named tuple.

# ╔═╡ c0000000-0000-4000-8000-000000000016
begin
"""
    window_selection(data, segments, start_index, stride, end_index)

Return the sample indices the loss sees, and the segments that have to be simulated to produce
those samples. `end_index <= 0` means the end of the log. Samples outside a simulated segment are
dropped, because the model cannot produce a value there.
"""
function window_selection(data, segments, start_index, stride, end_index)
    # Derived from the dataset itself, not from a global: this must work on any recording, and a
    # global row count would build a mask of the wrong length on a differently-sized one.
    total_samples = length(data.t_s)
    last_index = end_index <= 0 ? total_samples : min(end_index, total_samples)
    first_index = clamp(start_index, 1, last_index)
    requested = collect(first_index:max(1, stride):last_index)

    simulated = falses(total_samples)
    for row in eachrow(segments)
        simulated[row.simulation_start:row.stop_index] .= true
    end
    indices = [index for index in requested if simulated[index]]

    overlapping = segments[(segments.stop_index .>= first_index) .&
        (segments.simulation_start .<= last_index), :]
    return (indices=indices, segments=overlapping)
end

"""
    channel_scale(values)

Standard deviation of a measured channel, used to put every loss channel on the same footing.
Non-finite samples are ignored, so a single missing measurement cannot collapse the scale to the
fallback. Falls back to 1 for a degenerate window.
"""
function channel_scale(values)
    finite = [Float64(value) for value in values if isfinite(value)]
    length(finite) < 2 && return 1.0
    average = sum(finite) / length(finite)
    spread = sqrt(sum((value - average)^2 for value in finite) / (length(finite) - 1))
    return isfinite(spread) && spread > 0.0 ? spread : 1.0
end
end

# Cell 30: Window selection and loss scaling.

# ╔═╡ c0000000-0000-4000-8000-000000000017
begin
    """
        build_loss_data(data, segments, start_index, stride, end_index)

    Freeze everything the residual needs for one dataset: the windowed sample indices, a per-segment
    plan that precomputes each sample's column in the state history and its position in the flattened
    measured arrays, the measured channels, and the per-channel scales.

    Factored out of the training path so the test recording is scored by exactly the same loss, with
    no second copy of this logic to drift.
    """
    function build_loss_data(data, segments, start_index, stride, end_index)
        selection = window_selection(data, segments, start_index, stride, end_index)

        # Where each windowed sample lives inside the flattened measured arrays below.
        position_of = Dict(index => position for (position, index) in enumerate(selection.indices))

        # Per simulated segment: the samples inside it, their columns in the state history and their
        # positions in the measured arrays. Precomputing the positions keeps the residual loop free
        # of a linear search per sample.
        plan = [(segment=row,
                indices=[index for index in selection.indices
                    if row.simulation_start <= index <= row.stop_index]) for row in eachrow(selection.segments)]
        filter!(entry -> !isempty(entry.indices), plan)
        plan = [(segment=entry.segment,
                indices=entry.indices,
                columns=[index - entry.segment.simulation_start + 1 for index in entry.indices],
                positions=[position_of[index] for index in entry.indices]) for entry in plan]

        heave_measured = Float64.(coalesce.(data.com_z[selection.indices], NaN))
        pitch_measured = Float64.(coalesce.(data.pitch_rad[selection.indices], NaN))
        speed_measured = Float64.(coalesce.(data.speed_long_ms[selection.indices], NaN))

        # Channel weighting. Every channel is divided by its own spread, so the three contribute
        # equally; raising this sharpens the velocity match at the expense of heave and pitch.
        # Measured on this run: weight 2.0 moves speed RMSE 6.89 -> 6.48 km/h and heave
        # 1.91 -> 2.04 mm, so the trade is real but small and 1.0 stays the principled default.
        speed_weight = 1.0
        scales = (heave=channel_scale(heave_measured), pitch=channel_scale(pitch_measured),
            speed=channel_scale(speed_measured) / speed_weight)
        # A degenerate scale would silently un-normalise a channel, so fail here instead.
        for (name, value) in pairs(scales)
            (isfinite(value) && value > 0.0) ||
                error("channel_scale produced $name = $value; the loss window has no usable spread")
        end

        return (indices=selection.indices,
            plan=plan,
            # The measurement this loss belongs to. `halfcar_residual` simulates against it, so
            # carrying it here is what lets the same loss score a second recording instead of always
            # reading the training data.
            measurement=data,
            heave_scale=scales.heave,
            pitch_scale=scales.pitch,
            speed_scale=scales.speed,
            heave_measured=heave_measured,
            pitch_measured=pitch_measured,
            speed_measured=speed_measured)
    end

    loss_data = build_loss_data(data, halfcar_segments,
        window_start_index, window_stride, window_end_index)

    heave_scale_text = round(loss_data.heave_scale; digits=4)
    pitch_scale_text = round(loss_data.pitch_scale; digits=5)
    speed_scale_text = round(loss_data.speed_scale; digits=3)
    window_first_index = first(loss_data.indices)
    window_last_index = last(loss_data.indices)
    window_sample_count = length(loss_data.indices)
    window_segment_count = length(loss_data.plan)
    optimization_window_summary = Markdown.parse("""
The loss sees **$(window_sample_count)** samples from index $(window_first_index) to
$(window_last_index) with stride $(window_stride), spread over **$(window_segment_count)**
simulated segments. Channel scales: heave $(heave_scale_text) m,
pitch $(pitch_scale_text) rad, speed $(speed_scale_text) m/s.
""")
end

# Cell 31: Build the loss data for the selected window.

# ╔═╡ c0000000-0000-4000-8000-000000000018
begin
"""
    halfcar_residual(theta, loss_data)

Residual vector of the loss. `theta` is the parameter vector in the optimizer's coordinates — the
`x` of `OptimizationProblem(f, x0; p=…)`, *not* the `p` of `ODEProblem(f, u0, tspan, p)`, which
holds the physical model parameters. Here `theta` lives in the normalised `-1 … 1` box and is mapped
onto its physical range before the model runs. `loss_data` carries the frozen measurements.

For every windowed sample the heave, pitch and velocity differences between the measurement and the
model are appended, each divided by its channel scale. Exactly three entries are pushed per sample:
when the model produced a non-finite value a large sentinel penalty is used instead, so the residual
length is the same for every candidate and the objective stays continuous.

The residual is built in the element type of `theta`, so ForwardDiff duals flow through the affine
map untouched.
"""
function halfcar_residual(theta, loss_data)
    parameters = parameters_from_normalised(theta)
    residuals = eltype(theta)[]
    @assert eltype(residuals) === eltype(theta) "residual element type must follow the parameter type"

    # A non-finite model output becomes a penalty large enough to reject the candidate outright,
    # without changing the residual length. A plain float converts to the element type on push.
    sentinel_penalty = 1.0e6

    for entry in loss_data.plan
        # The measurement carried by `loss_data`, not a global: the same residual then scores the
        # training run and the test run without a second copy of this function.
        history = halfcar_segment_history(parameters, entry.segment, loss_data.measurement;
            road_x=road_stations_x, road_z=road_stations_z, dt=halfcar_dt,
            abstol=solver_tolerance, reltol=solver_tolerance)

        for (column, position) in zip(entry.columns, entry.positions)
            model_speed = history[6, column]
            model_heave = (parameters.sprung_mass * history[1, column] +
                parameters.unsprung_front * history[3, column] +
                parameters.unsprung_rear * history[4, column]) / parameters.total_mass
            model_pitch = history[2, column]

            if isfinite(model_speed) && isfinite(model_heave) && isfinite(model_pitch)
                push!(residuals, (loss_data.heave_measured[position] - model_heave) /
                    loss_data.heave_scale)
                push!(residuals, (loss_data.pitch_measured[position] - model_pitch) /
                    loss_data.pitch_scale)
                push!(residuals, (loss_data.speed_measured[position] - model_speed) /
                    loss_data.speed_scale)
            else
                push!(residuals, sentinel_penalty, sentinel_penalty, sentinel_penalty)
            end
        end
    end
    return residuals
end

"""
    halfcar_objective(theta, loss_data)

Scalar loss handed to Optimization.jl: the sum of squared normalised residuals, as a function of the
normalised parameter vector `theta` only.
"""
halfcar_objective(theta, loss_data) = mean(abs2, 10 .* halfcar_residual(theta, loss_data))
end

# Cell 32: Residual and objective for the optimizer.

# ╔═╡ c0000000-0000-4000-8000-000000000019
begin
    # Bounds force Fminbox(BFGS), which requires gradients, so the AD backend is attached here.
    # The box wrapper is unchanged from the LBFGS formulation: only the inner solver differs, so the
    # convergence history can be compared between the two without a formulation change.
    # They sit at -1 and +1 because the search space is the normalised coordinate, not the physical
    # parameters.
    optimization_function = OptimizationFunction(halfcar_objective,
        Optimization.AutoForwardDiff())
    optimization_problem = OptimizationProblem(optimization_function,
        estimation_start_normalised;
        lb=fill(-1.0, length(estimation_names)), ub=fill(1.0, length(estimation_names)),
        p=loss_data)

    # Sanity check before the solver starts: the nominal parameters must produce a finite loss.
    nominal_objective = halfcar_objective(estimation_start_normalised, loss_data)
    (problem=optimization_problem, nominal_objective=nominal_objective)
end

# Cell 33: Build the Optimization problem with bounds.

# ╔═╡ c0000000-0000-4000-8000-000000000020
begin
    """
    Verify that the loss really is a function of the normalised parameter vector alone: the same
    input must give the same output, the residual length must not depend on the candidate, and the
    ForwardDiff gradient must agree with central finite differences. Returns a table of the checks.
    """
    loss_contract_checks = let
        start = estimation_start_normalised
        base = halfcar_objective(start, loss_data)
        repeat_value = halfcar_objective(start, loss_data)
        expected_length = 3 * sum(length(entry.indices) for entry in loss_data.plan)

        # Nudge every parameter to make sure the length and value respond to the input rather than
        # staying constant.
        perturbed = start .+ fill(1.0e-3, length(start))
        perturbed_length = length(halfcar_residual(perturbed, loss_data))

        gradient = ForwardDiff.gradient(theta -> halfcar_objective(theta, loss_data), start)
        zero_gradient = [string(estimation_names[i]) for i in eachindex(gradient)
            if gradient[i] == 0.0]

        # Central differences on the whole gradient along one fixed direction. A direction aggregates
        # every component, so the comparison stays well conditioned where individual components are
        # near zero. The step has to stay small: the estimate is a chord across the loss surface, and
        # a step of 0.1 or 0.3 in a 24-dimensional normalised box lands far enough away that the
        # model has genuinely changed regime, which reads as a 6 % discrepancy that is not a
        # gradient error. Small steps are only usable if the objective is smooth at their scale, and
        # the adaptive ODE controller is what breaks that: it accepts or rejects steps on the primal
        # value, so a tiny parameter change can shift the accepted sequence and jump the loss by a
        # few percent. Evaluating the differences at a tighter solver tolerance removes that jitter
        # - at 3e-2/1e-2/3e-3 the estimates then converge onto the autodiff value to 0.05 % - while
        # the fit keeps the cheaper working tolerance. A single step is still a lottery, so the
        # estimate is the median of three.
        steps = (3.0e-2, 1.0e-2, 3.0e-3)
        difference_tolerance = 1.0e-8
        tolerance = 5.0e-2
        finite_difference_error = let
            function difference_objective(theta)
                saved = solver_tolerance
                global solver_tolerance = difference_tolerance
                value = halfcar_objective(theta, loss_data)
                global solver_tolerance = saved
                return value
            end
            direction = [sin(1.0 * index) + 1.5 for index in eachindex(start)]
            direction ./= sqrt(sum(abs2, direction))
            ad_value = sum(gradient .* direction)
            estimates = [begin
                             forward = difference_objective(start .+ step .* direction)
                             backward = difference_objective(start .- step .* direction)
                             (forward - backward) / (2 * step)
                         end for step in steps]
            numeric = sort(estimates)[2]     # median of the three step sizes
            scale = max(abs(numeric), abs(ad_value), 1.0e-12)
            abs(numeric - ad_value) / scale
        end

        DataFrame(check=["deterministic", "residual length", "gradient non-zero",
                "gradient matches finite differences", "objective finite"],
            value=[base == repeat_value,
                perturbed_length == expected_length,
                isempty(zero_gradient),
                finite_difference_error < tolerance,
                isfinite(base)],
            detail=["objective $(round(base; digits=6)) twice",
                "$(perturbed_length) of $(expected_length) entries",
                isempty(zero_gradient) ? "all $(length(gradient)) components non-zero" :
                    "zero for " * join(zero_gradient, ", "),
                "directional derivative, relative error $(round(finite_difference_error; sigdigits=3))",
                "start objective = $(round(base; digits=4))"])
    end

    # A failed check means the loss is not a pure function of the parameters, so stop before solving.
    failed_checks = loss_contract_checks[loss_contract_checks.value .== false, :]
    isempty(failed_checks) || error("loss contract failed: " *
        join(failed_checks.check, ", "))
    loss_contract_checks
end

# Cell 34: Prove the loss is a pure function of the parameter vector.

# ╔═╡ c0000000-0000-4000-8000-00000000001a
begin
    optimization_history = NamedTuple{(:iteration, :objective, :elapsed),
        Tuple{Int, Float64, Float64}}[]
    optimization_start_time = time()
    # Mutable holder keeps the callback's state without assigning to a global, which Pluto forbids.
    optimization_best = Ref(Inf)

    """
    Record every objective evaluation and stop BFGS once the wall-clock budget is spent.
    The optimizer calls this as `callback(state, loss)`; returning `true` terminates the solve.
    Every call reports the iteration number, the elapsed time and the loss value with `@info`,
    so the notebook streams progress while the solve is running. Under `Fminbox` the loss is
    the solver's internal objective, so compare it between iterations rather than against
    `nominal_objective`.
    """
    function optimization_callback(state, loss)
        elapsed = time() - optimization_start_time
        iteration = state.iter
        push!(optimization_history, (iteration=iteration, objective=loss, elapsed=elapsed))
        improved = loss < optimization_best[]
        improved && (optimization_best[] = loss)

        @info("BFGS progress",
            iteration = iteration,
            elapsed_s = round(elapsed; digits=2),
            loss = round(loss; digits=4),
            best_loss = round(optimization_best[]; digits=4))

        budget_spent = elapsed > max_time_s
        if budget_spent
            @info("BFGS stopped",
                reason = "wall-clock budget of $(max_time_s) s reached",
                iteration = iteration,
                elapsed_s = round(elapsed; digits=2),
                loss = round(loss; digits=4))
        end
        return budget_spent
    end

    optimization_solution = solve(optimization_problem, BFGS();
        maxiters=max_iterations, callback=optimization_callback)
    optimization_elapsed = time() - optimization_start_time

    # `solution.objective` is the solver's own (Fminbox) objective, so it is not comparable to the
    # plain least-squares baseline. Re-evaluate the same function the baseline used for reporting.
    fitted_objective = halfcar_objective(optimization_solution.u, loss_data)
    solver_objective = optimization_solution.objective
    final_reduction = nominal_objective > 0.0 ?
        100.0 * (1.0 - fitted_objective / nominal_objective) : 0.0

    @info("Optimization finished",
        retcode = optimization_solution.retcode,
        evaluations = length(optimization_history),
        objective = round(fitted_objective; digits=4),
        solver_objective = round(solver_objective; digits=4),
        objective_delta = round(solver_objective - fitted_objective; digits=4),
        reduction_percent = round(final_reduction; digits=2),
        elapsed_s = round(optimization_elapsed; digits=1))

    (retcode=optimization_solution.retcode,
        evaluations=length(optimization_history),
        objective=fitted_objective,
        solver_objective=solver_objective,
        elapsed_s=round(optimization_elapsed; digits=1))
end

# Cell 35: Run BFGS with ForwardDiff gradients.

# ╔═╡ c0000000-0000-4000-8000-000000000062
md"""
## 6. Results

6.1 shows how the solve progressed. 6.2 tabulates the estimate, the fitted value and the uncertainty
bounds side by side. 6.3 works out which samples the model never simulated, so the comparison figure
can shade them. 6.4 draws the measurement, the pre-optimization simulation and the post-optimization
simulation on the same axes, with the loss window marked. 6.5 reports the error over that window.
6.6 replays the fitted parameters on a second recording that the optimizer never saw, and reports the
same errors there.
"""

# Cell 36: Section 6: results - convergence, parameters, comparison figure and metrics.

# ╔═╡ c0000000-0000-4000-8000-00000000001b
begin
    # The fitted parameters. The table below compares them with `guess_parameters` from section 4.
    fitted_normalised = Float64.(optimization_solution.u)
    fitted_values = denormalise(fitted_normalised)
    fitted_parameters = parameters_from_vector(fitted_values)

    estimation_results = DataFrame(
        parameter=estimation_parameter_bounds.parameter,
        confidence=estimation_parameter_bounds.confidence,
        nominal=estimation_parameter_bounds.nominal,
        normalised_start=estimation_start_normalised,
        normalised=fitted_normalised,
        fitted=fitted_values,
        lower=estimation_parameter_bounds.lower,
        upper=estimation_parameter_bounds.upper)
    estimation_results.relative_change = (estimation_results.fitted .-
        estimation_results.nominal) ./ abs.(estimation_results.nominal)
    # A normalised value near ±1 means the fit is pressed against a confidence limit.
    estimation_results.at_bound = abs.(estimation_results.normalised) .>= 0.99

    objective_before_text = round(nominal_objective; digits=4)
    objective_after_text = round(fitted_objective; digits=4)
    solver_objective_text = round(solver_objective; digits=4)
    reduction_text = round(100.0 * (1.0 - fitted_objective / nominal_objective); digits=2)
    elapsed_text = round(optimization_elapsed; digits=1)
    budget_text = round(max_time_s; digits=0)
    evaluation_count = length(optimization_history)
    bound_count = count(estimation_results.at_bound)
    estimation_summary = Markdown.parse("""
Least-squares objective **$(objective_before_text)** to **$(objective_after_text)** in
$(elapsed_text) s over $(evaluation_count) evaluations, a $(reduction_text) % reduction
(limit: $(max_iterations) iterations, $(budget_text) s).
$(bound_count) of $(nrow(estimation_results)) parameters ended within 1 % of a bound in normalised
coordinates.

The solver's own Fminbox objective was $(solver_objective_text). It is augmented by the box barrier,
so it is not comparable to the plain objective above and is reported only for reference.
""")
    estimation_results
end

# Cell 37: Parameter table before and after optimization.

# ╔═╡ c0000000-0000-4000-8000-00000000001f
begin
"""
    unseen_sample_spans(plan, total_samples)

Return the sample-index intervals inside `1:total_samples` that the model never simulated. `plan` is
the loss plan: one entry of sample indices per simulated segment. Every sample between the first and
last index of an entry was covered by a simulation, so the unseen intervals are the stretches
outside those ranges — the out-of-window parts of the log and the wheel lift-off gaps between
segments.

The intended stride inside a segment is *not* reported as unseen, because those samples lie inside a
simulated segment even though the loss only evaluated every n-th one.
"""
function unseen_sample_spans(plan, total_samples)
    total_samples < 1 && return Tuple{Int, Int}[]

    used = falses(total_samples)
    for entry in plan
        isempty(entry) && continue
        first_sample = clamp(first(entry), 1, total_samples)
        last_sample = clamp(last(entry), 1, total_samples)
        used[first_sample:last_sample] .= true
    end

    unseen = Tuple{Int, Int}[]
    run_start = 0
    for sample in 1:total_samples
        if used[sample]
            run_start > 0 && push!(unseen, (run_start, sample - 1))
            run_start = 0
        elseif run_start == 0
            run_start = sample
        end
    end
    run_start > 0 && push!(unseen, (run_start, total_samples))
    return unseen
end
end

# Cell 38: Intervals of the recording the model never simulated.

# ╔═╡ c0000000-0000-4000-8000-00000000001c
begin
    after_run = halfcar_plot_channels(fitted_parameters)

    halfcar_figure = let
        fig = Figure(size=(700, 1160), backgroundcolor=:white)
        ax_heave = Axis(fig[1, 1], ylabel="Heave (m)", title="Measurement vs model")
        ax_pitch = Axis(fig[2, 1], ylabel="Pitch (deg)")
        ax_speed = Axis(fig[3, 1], ylabel="Speed (km/h)")
        ax_position = Axis(fig[4, 1], ylabel="Position (m)")
        ax_torque = Axis(fig[5, 1], ylabel="Available torque (Nm)")
        ax_pedals = Axis(fig[6, 1], ylabel="Pedal level", xlabel="Simulation time (s)",
            yticks=([0.0, 0.5, 1.0], ["0", "0.5", "1"]), limits=(nothing, (0.0, 1.05)))

        axes_all = (ax_heave, ax_pitch, ax_speed, ax_position, ax_torque, ax_pedals)

        # Shade the stretches the model never simulated. Drawn before the pedal bands so the pedal
        # colours stay on top, and painted in every panel so the gaps line up across the stack. The
        # spans are the exact complement of the simulated segment ranges, so the intended stride
        # inside a segment is not shaded.
        unseen_spans = unseen_sample_spans([entry.indices for entry in loss_data.plan],
            sample_count)
        for axis in axes_all
            for (first_unseen, last_unseen) in unseen_spans
                vspan!(axis, time_s[first_unseen], time_s[last_unseen]; color=(:gray60, 0.18))
            end
        end

        for axis in axes_all
            add_pedal_background!(axis, time_s[plot_index], throttle[plot_index],
                brake[plot_index]; single_step=plot_step)
        end

        measured_heave = Float64.(data.com_z)
        measured_speed = Float64.(data.speed_long_ms) .* 3.6
        measured_position = Float64.(data.s_travelled_m)

        # Each panel holds one measured channel plus the two simulations, so one colour per role is
        # enough: the measurement is black, and the pre- and post-optimization lines are both solid
        # and differ only by colour.
        panels = (
            (ax_heave, measured_heave, before_run.com_height, after_run.com_height),
            (ax_pitch, pitch_deg, rad2deg.(before_run.pitch_rad),
                rad2deg.(after_run.pitch_rad)),
            (ax_speed, measured_speed, before_run.velocity_ms .* 3.6,
                after_run.velocity_ms .* 3.6),
            (ax_position, measured_position, before_run.position_m, after_run.position_m),
            (ax_torque, available_torque_nm, before_run.available_torque_nm,
                after_run.available_torque_nm),
        )
        for (axis, measured, before, after) in panels
            lines!(axis, time_s[plot_index], measured[plot_index];
                color=measured_color, linewidth=1.5)
            add_lift_markers!(axis, time_s, measured)
            lines!(axis, time_s[plot_index], before[plot_index];
                color=before_color, linewidth=1.4)
            lines!(axis, time_s[plot_index], after[plot_index];
                color=after_color, linewidth=1.8)
        end

        # The pedal panel is the only panel with two series, so it keeps two colours.
        lines!(ax_pedals, time_s[plot_index], throttle[plot_index];
            color=:seagreen, linewidth=1.2)
        lines!(ax_pedals, time_s[plot_index], brake[plot_index];
            color=:orangered, linewidth=1.2)
        add_lift_markers!(ax_pedals, time_s, throttle)

        # Mark the window the loss actually saw; the curves still show the whole run.
        window_first = first(loss_data.indices)
        window_last = last(loss_data.indices)
        for axis in axes_all
            vlines!(axis, time_s[window_first]; color=(:black, 0.65), linewidth=1.3,
                linestyle=:dashdot)
            vlines!(axis, time_s[window_last]; color=(:black, 0.65), linewidth=1.3,
                linestyle=:dashdot)
        end

        linkxaxes!(axes_all...)
        for axis in (ax_heave, ax_pitch, ax_speed, ax_position, ax_torque)
            hidexdecorations!(axis; grid=false)
        end
        rowsize!(fig.layout, 6, Fixed(150))

        legend_elements = [
            LineElement(color=measured_color, linewidth=1.5),
            LineElement(color=before_color, linewidth=1.4),
            LineElement(color=after_color, linewidth=1.8),
            LineElement(color=(:black, 0.65), linewidth=1.3, linestyle=:dashdot),
            PolyElement(color=(:gray60, 0.45)),
            [MarkerElement(color=style.color, marker=style.marker, markersize=8,
                strokecolor=:white, strokewidth=0.8) for style in values(lift_styles)]...
        ]
        legend_labels = ["Measured", "Before optimization", "After optimization", "Loss window",
            "Not used by the loss", [style.label for style in values(lift_styles)]...]
        Legend(fig[7, 1], legend_elements, legend_labels; orientation=:horizontal,
            nbanks=3, tellwidth=false, framevisible=false, labelsize=10,
            patchsize=(18, 10), rowgap=3, colgap=8)
        rowsize!(fig.layout, 7, Fixed(115))
        fig
    end
end

# Cell 39: Measured, before-optimization and after-optimization comparison.

# ╔═╡ c0000000-0000-4000-8000-00000000001d
begin
    """
    RMSE, MAE and R² of one channel over the windowed samples only, comparing a simulation result
    with the measurement.

    `SS_tot` is taken about the measured mean *over the same samples*, so the two sums share their
    sample set and R² cannot exceed 1 by construction. R² is reported alongside RMSE because they
    answer different questions: a channel with a small spread can show a tiny RMSE while being almost
    uncorrelated with the measurement, and R² exposes that.
    """
    function window_error(measured, simulated, indices)
        differences = Float64[]
        measurements = Float64[]
        for index in indices
            value = measured[index]
            (ismissing(value) || isnan(simulated[index])) && continue
            push!(differences, Float64(value) - simulated[index])
            push!(measurements, Float64(value))
        end
        isempty(differences) && return (rmse=NaN, mae=NaN, r2=NaN, count=0)
        ss_residual = sum(abs2, differences)
        centre = sum(measurements) / length(measurements)
        ss_total = sum(abs2, measurements .- centre)
        return (rmse=sqrt(ss_residual / length(differences)),
            mae=sum(abs, differences) / length(differences),
            r2=ss_total > 0.0 ? 1.0 - ss_residual / ss_total : NaN,
            count=length(differences))
    end

    estimation_metrics = let
        indices = loss_data.indices
        channels = ["Heave [m]", "Pitch [deg]", "Speed [km/h]"]
        measured = [data.com_z, pitch_deg, Float64.(data.speed_long_ms) .* 3.6]
        before = [before_run.com_height, rad2deg.(before_run.pitch_rad),
            before_run.velocity_ms .* 3.6]
        after = [after_run.com_height, rad2deg.(after_run.pitch_rad),
            after_run.velocity_ms .* 3.6]

        before_errors = [window_error(measured[i], before[i], indices) for i in 1:3]
        after_errors = [window_error(measured[i], after[i], indices) for i in 1:3]
        DataFrame(channel=channels,
            rmse_before=[error.rmse for error in before_errors],
            rmse_after=[error.rmse for error in after_errors],
            mae_before=[error.mae for error in before_errors],
            mae_after=[error.mae for error in after_errors],
            r2_after=[error.r2 for error in after_errors],
            samples=[error.count for error in after_errors])
    end
end

# Cell 40: Measured-versus-model error over the loss window.

# ╔═╡ c0000000-0000-4000-8000-0000000000e0
md"""
### 6.7 Saving the result

Each notebook writes one file into `results/`, named by its own label, holding everything the
comparison notebook needs in a single flat record: the provenance, the objective, the whole
optimization history, the parameters at the guess and at the fit, the measured channels, the
simulated traces, the loss window and the per-channel metrics. The table below prints the exact path
this run produced.

The traces are stored, not just the parameters, because a study may compare several models that do
not share their force laws. A reader that loaded only parameters would have to re-simulate, and it
could not do so for a model whose physics no longer exists in the tree. The measured channels are
stored for the same reason: a saved result must stay plottable even if the recording is deleted.

Everything is a plain array or a `Vector{String}` - no `DataFrame` - so the file can be read back
without the reader having to guess at types.
"""

# Cell 41: Save the parameters, loss history and traces for later comparison.

# ╔═╡ c0000000-0000-4000-8000-0000000000f0
md"""
## Reproducing this

| requirement | value |
|---|---|
| Julia | 1.13 (`Project.toml` and `Manifest.toml` sit beside this notebook) |
| dataset | `longdyn_train.csv`, 83 MB, beside this notebook |
| runtime | ~40 min for a full fit on one machine; the numerical core is a single-threaded ODE solve |

Activate the environment, then either open the notebook in Pluto or replay it headlessly:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

The dataset is a multi-channel recording: one CSV with a `#`-prefixed metadata block followed by a
header of named channels and one row per sample. Only fifteen of its channels are used; the loader
names them explicitly, so an unrelated recording can be substituted as long as it carries the same
column names.

The dataset is tracked with Git LFS because it is larger than a comfortable plain-git blob. If the
file is missing after cloning, run `git lfs pull`.
"""

# ╔═╡ c0000000-0000-4000-8000-0000000000e2
md"""
### 6.1 Convergence history

The objective against evaluation index and against wall time. The faint line is the value at each
evaluation and the solid line the running best: a quasi-Newton line search does not decrease
monotonically, so the running best is the honest summary of how quickly the solve converged and where
it stalled.

The two panels answer different questions. The left one says how much work was needed; the right one
says what that work cost, which is what matters when comparing solvers or budgets. The dashed line is
the pre-optimization objective, so the vertical distance from it is the improvement the fit bought.
"""

# Cell 37: Objective against evaluation and against elapsed time.

# ╔═╡ c0000000-0000-4000-8000-0000000000e3
begin
    convergence_figure = let
        iterations = [point.iteration for point in optimization_history]
        objectives = [point.objective for point in optimization_history]
        elapsed = [point.elapsed for point in optimization_history]
        running_best = accumulate(min, objectives)

        fig = Figure(size=(760, 330), backgroundcolor=:white)
        ax_iteration = Axis(fig[1, 1], ylabel="Objective",
            xlabel="Evaluation", title="Convergence of the fit")
        ax_time = Axis(fig[1, 2], xlabel="Elapsed time (s)")

        for axis in (ax_iteration, ax_time)
            hlines!(axis, [nominal_objective]; color=(:gray40, 0.9), linewidth=1.2,
                linestyle=:dash)
        end
        # Raw value first so the running best is drawn over it.
        lines!(ax_iteration, iterations, objectives; color=(before_color, 0.35), linewidth=1.0)
        lines!(ax_iteration, iterations, running_best; color=after_color, linewidth=2.0)
        lines!(ax_time, elapsed, objectives; color=(before_color, 0.35), linewidth=1.0)
        lines!(ax_time, elapsed, running_best; color=after_color, linewidth=2.0)
        scatter!(ax_iteration, [iterations[argmin(objectives)]], [minimum(objectives)];
            color=measured_color, markersize=9, strokecolor=:white, strokewidth=1.0)

        linkyaxes!(ax_iteration, ax_time)
        hideydecorations!(ax_time; grid=false)

        legend_elements = [
            LineElement(color=(before_color, 0.35), linewidth=1.0),
            LineElement(color=after_color, linewidth=2.0),
            LineElement(color=(:gray40, 0.9), linewidth=1.2, linestyle=:dash),
            MarkerElement(color=measured_color, marker=:circle, markersize=9,
                strokecolor=:white, strokewidth=1.0),
        ]
        Legend(fig[2, 1:2], legend_elements,
            ["Objective at each evaluation", "Running best", "Pre-optimization objective",
                "Best value found"];
            orientation=:horizontal, nbanks=4, tellwidth=false, framevisible=false, labelsize=10,
            patchsize=(18, 10), colgap=10)
        rowsize!(fig.layout, 2, Fixed(50))
        fig
    end
end

# Cell 38: The loss curve, against evaluations and against time.

# ╔═╡ c0000000-0000-4000-8000-0000000000e4
md"""
### 6.2 Parameters before and after

The estimate, the fitted value and the uncertainty bounds side by side. A fitted value pressed against
its bound is a warning rather than a result: it means the model was reaching for something the bound
denied.
"""

# ╔═╡ c0000000-0000-4000-8000-0000000000ea
md"""
### 6.3 Samples the model never simulated

The model only integrates the intervals in which all four wheels report ground contact, so the
lift-off gaps and anything outside the loss window have no prediction. This subsection derives those
stretches exactly, as the complement of the simulated segment ranges, so the figure below can shade
them rather than leaving the reader to guess whether a gap is missing data or a deliberate refusal to
predict.
"""

# ╔═╡ c0000000-0000-4000-8000-0000000000eb
md"""
### 6.4 Measured response against the model

The measurement in black, the pre-optimization simulation in orange and the fitted one in blue. The
vertical dash-dot lines bound the samples the loss saw: the curves run over the whole recording, so
the parts outside that window show how the model behaves where it was not fitted.
"""

# ╔═╡ c0000000-0000-4000-8000-0000000000ec
md"""
### 6.5 Error on the training data

Every statistic in this subsection is computed over the samples the loss scored, so it measures how
well the fit reproduces the data it was given. That is the number to compare against section 6.6: the
distance between the two is what the fit lost by being asked to extrapolate.
"""

# ╔═╡ c0000000-0000-4000-8000-0000000000e5
md"""
### 6.6 Validation on test data

Section 6.5 scores the model on the same samples the fit was given, so those numbers are *training*
errors: the parameters were chosen to make them small. This section scores the same parameters on a
second recording that the optimizer never saw. It is validation data, not training data - the fit
cannot have memorised it, because its inputs never entered the objective.

The two recordings come from the same source and cover the same manoeuvre families with different
levels, durations and ordering. Nothing about the model is re-estimated here: the fitted parameters
are replayed unchanged, so a small error on this data cannot come from having seen it.

Two caveats in reading the table below:

- **The two loss values are not on one scale.** Each normalises every channel by the spread of the
  dataset it is given, which is right for fitting but means the test loss is not a ratio of the
  training one. Read the per-channel errors in physical units (mm, deg, km/h) for the like-for-like
  comparison, and the two loss values only as a general indication.
- **Only all-wheels-grounded intervals are simulated**, so the model declines to predict across
  lift-off gaps rather than guessing through them. Those stretches appear as grey spans in the figure
  and are excluded from every statistic.
"""

# Cell 41: Load the test recording and replay the fitted parameters on it.

# ╔═╡ c0000000-0000-4000-8000-0000000000e6
begin
    test_log = normpath(joinpath(@__DIR__, "longdyn_test.csv"))
    test_data = read_measurement(test_log)

    # The model integrates on a fixed grid, so a recording sampled at a different period would shift
    # every simulated sample against the measurement and quietly inflate the errors.
    test_period = median(diff(Float64.(test_data.t_s)))
    abs(test_period - halfcar_dt) <= 0.05 * halfcar_dt ||
        error("Test recording has a sample period of $(test_period) s, expected $(halfcar_dt) s")

    test_extraction = extract_segments(test_data;
        wheel_names=halfcar_wheel_names,
        min_segment_samples=halfcar_min_segment_samples,
        pad_indices=halfcar_pad_indices)
    test_segments = test_extraction.segments

    # The same window rule as training, but at full resolution: nothing is being fitted here, so
    # there is no reason to stride.
    test_loss_data = build_loss_data(test_data, test_segments,
        window_start_index, 1, 0)

    test_run = halfcar_simulate(fitted_parameters, test_segments, test_data;
        road_x=road_stations_x, road_z=road_stations_z, dt=halfcar_dt,
        abstol=solver_tolerance, reltol=solver_tolerance)

    # `halfcar_objective` takes the optimiser's coordinate, so the fitted vector is passed straight
    # through: the parameters are identical to the ones the fit ended on.
    test_objective = halfcar_objective(fitted_normalised, test_loss_data)

    # The test recording's own lift classification, so the markers match the trace they annotate.
    test_airborne = Dict(wheel => [ismissing(flag) ? false : flag == 0
            for flag in getproperty(test_data, Symbol("wheel_", wheel, "_contact"))]
        for wheel in halfcar_wheel_names)
    test_grounded = [!(test_airborne["fl"][i] || test_airborne["fr"][i] ||
        test_airborne["rl"][i] || test_airborne["rr"][i]) for i in 1:length(test_data.t_s)]
    test_settled = something(findfirst(identity, test_grounded), 1)
    test_lift_class = Vector{String}(undef, length(test_data.t_s))
    for index in 1:length(test_data.t_s)
        if index < test_settled
            test_lift_class[index] = "settling"
            continue
        end
        front_airborne = test_airborne["fl"][index] || test_airborne["fr"][index]
        rear_airborne = test_airborne["rl"][index] || test_airborne["rr"][index]
        test_lift_class[index] = front_airborne ? (rear_airborne ? "all" : "front") :
            (rear_airborne ? "rear" : "none")
    end
    test_lift_indices = Dict(name => findall(==(name), test_lift_class)
        for name in ("front", "rear", "all"))

    Markdown.parse("""
Test recording **$(basename(test_log))**: $(length(test_data.t_s)) samples over
$(round(Float64(test_data.t_s[end]); digits=1)) s, of which **$(length(test_loss_data.indices))**
are scored across **$(length(test_loss_data.plan))** simulated segments. Fitted parameters replayed
unchanged; no re-estimation.
""")
end

# Cell 42: The test recording, segmented and simulated at the fitted parameters.

# ╔═╡ c0000000-0000-4000-8000-0000000000e7
md"""
The measurement is black and the model's prediction on these unseen inputs is blue. Grey spans are
samples the model never simulated - the all-wheels-grounded segmentation - so a gap means the model
declined to predict, not that a measurement is missing.
"""

# Cell 43: Measured test data against the model.

# ╔═╡ c0000000-0000-4000-8000-0000000000e8
begin
    test_time = Float64.(test_data.t_s)
    test_stride = 5
    test_index = 1:test_stride:length(test_time)

    test_figure = let
        fig = Figure(size=(760, 820), backgroundcolor=:white)
        ax_heave = Axis(fig[1, 1], ylabel="Heave (m)",
            title="Validation on test data - not used for fitting")
        ax_pitch = Axis(fig[2, 1], ylabel="Pitch (deg)")
        ax_speed = Axis(fig[3, 1], ylabel="Speed (km/h)")
        ax_pedals = Axis(fig[4, 1], ylabel="Pedal level", xlabel="Time (s)",
            yticks=([0.0, 0.5, 1.0], ["0", "0.5", "1"]), limits=(nothing, (0.0, 1.05)))
        axes_all = (ax_heave, ax_pitch, ax_speed, ax_pedals)

        # NaN in a simulated channel marks a sample outside every simulated segment.
        unsimulated = isnan.(Float64.(test_run.com_height))
        span_start = 0
        for sample in eachindex(unsimulated)
            if unsimulated[sample]
                span_start == 0 && (span_start = sample)
            elseif span_start != 0
                span_start < sample - 1 && for axis in axes_all
                    vspan!(axis, test_time[span_start], test_time[sample - 1];
                        color=(:gray60, 0.18))
                end
                span_start = 0
            end
        end

        for axis in axes_all
            add_pedal_background!(axis, test_time[test_index],
                Float64.(test_data.throttle_applied)[test_index],
                Float64.(test_data.brake_applied)[test_index]; single_step=test_stride * halfcar_dt)
        end

        panels = (
            (ax_heave, Float64.(test_data.com_z), test_run.com_height, 1.0),
            (ax_pitch, rad2deg.(Float64.(test_data.pitch_rad)),
                rad2deg.(test_run.pitch_rad), 1.0),
            (ax_speed, Float64.(test_data.speed_long_ms) .* 3.6,
                test_run.velocity_ms .* 3.6, 1.0),
        )
        for (axis, measured, simulated, _) in panels
            lines!(axis, test_time[test_index], measured[test_index];
                color=measured_color, linewidth=1.5)
            add_lift_markers!(axis, test_time, measured; lift_indices=test_lift_indices)
            lines!(axis, test_time[test_index], simulated[test_index];
                color=after_color, linewidth=1.7)
        end

        lines!(ax_pedals, test_time[test_index],
            Float64.(test_data.throttle_applied)[test_index]; color=:seagreen, linewidth=1.2)
        lines!(ax_pedals, test_time[test_index],
            Float64.(test_data.brake_applied)[test_index]; color=:orangered, linewidth=1.2)

        linkxaxes!(axes_all...)
        for axis in (ax_heave, ax_pitch, ax_speed)
            hidexdecorations!(axis; grid=false)
        end
        rowsize!(fig.layout, 4, Fixed(150))

        legend_elements = [
            LineElement(color=measured_color, linewidth=1.5),
            LineElement(color=after_color, linewidth=1.7),
            PolyElement(color=(:gray60, 0.45)),
            [MarkerElement(color=style.color, marker=style.marker, markersize=8,
                strokecolor=:white, strokewidth=0.8) for style in values(lift_styles)]...
        ]
        legend_labels = ["Measured (test)", "Model (fitted)", "Not simulated",
            [style.label for style in values(lift_styles)]...]
        Legend(fig[5, 1], legend_elements, legend_labels; orientation=:horizontal, nbanks=3,
            tellwidth=false, framevisible=false, labelsize=10, patchsize=(18, 10),
            rowgap=3, colgap=8)
        rowsize!(fig.layout, 5, Fixed(100))
        fig
    end
end

# Cell 44: The test-data figure.

# ╔═╡ c0000000-0000-4000-8000-0000000000e9
begin
    """
    Compute every error statistic on the test recording, over the scored samples only.
    """
    test_metrics = let
        indices = test_loss_data.indices
        channels = ["Heave [m]", "Pitch [deg]", "Speed [km/h]"]
        measured = [Float64.(test_data.com_z), rad2deg.(Float64.(test_data.pitch_rad)),
            Float64.(test_data.speed_long_ms) .* 3.6]
        simulated = [test_run.com_height, rad2deg.(test_run.pitch_rad),
            test_run.velocity_ms .* 3.6]
        errors = [window_error(measured[i], simulated[i], indices) for i in 1:3]
        DataFrame(channel=channels,
            rmse=[error.rmse for error in errors],
            mae=[error.mae for error in errors],
            r2=[error.r2 for error in errors],
            samples=[error.count for error in errors])
    end

    """
    One table for both datasets: the same statistics side by side, so the generalisation gap is read
    off adjacent rows rather than inferred. The loss block sits at the bottom because the two loss
    values are normalised by different spreads and are therefore not directly comparable.
    """
    validation_summary = let
        rows = NamedTuple[]
        channels = ["Heave [m]", "Pitch [deg]", "Speed [km/h]"]
        for (dataset, training) in (("Training", true), ("Test", false))
            for (index, channel) in enumerate(channels)
                if training
                    push!(rows, (dataset=dataset, channel=channel, statistic="RMSE",
                        value=estimation_metrics.rmse_after[index]))
                    push!(rows, (dataset=dataset, channel=channel, statistic="MAE",
                        value=estimation_metrics.mae_after[index]))
                    push!(rows, (dataset=dataset, channel=channel, statistic="R²",
                        value=estimation_metrics.r2_after[index]))
                else
                    push!(rows, (dataset=dataset, channel=channel, statistic="RMSE",
                        value=test_metrics.rmse[index]))
                    push!(rows, (dataset=dataset, channel=channel, statistic="MAE",
                        value=test_metrics.mae[index]))
                    push!(rows, (dataset=dataset, channel=channel, statistic="R²",
                        value=test_metrics.r2[index]))
                end
            end
        end
        for (dataset, value) in (("Training", fitted_objective), ("Test", test_objective))
            push!(rows, (dataset=dataset, channel="-", statistic="Loss", value=value))
        end
        table = DataFrame(dataset=[row.dataset for row in rows],
            channel=[row.channel for row in rows],
            statistic=[row.statistic for row in rows],
            value=round.([row.value for row in rows]; sigdigits=6))
        table
    end
end

# Cell 45: Error statistics on both datasets, side by side.

# ╔═╡ c0000000-0000-4000-8000-0000000000e1
begin
    # One label per notebook. The comparison notebook discovers files by name, so this is the only
    # place that decides what this model is called in the figures.
    result_label = "params-opt-v2.34"
    results_directory = normpath(joinpath(@__DIR__, "results"))
    mkpath(results_directory)
    result_path = joinpath(results_directory, result_label * ".jld2")

    saved_result = (
        schema=1,
        label=result_label,
        notebook="params_opt_v2.34.jl",
        saved_at=string(now()),
        log_file=basename(latest_log),
        log_path=latest_log,
        sample_count=sample_count,
        time_step=halfcar_dt,
        solver_tolerance=solver_tolerance,
        retcode=string(optimization_solution.retcode),
        evaluations=length(optimization_history),
        elapsed_total=optimization_elapsed,
        objective=fitted_objective,
        solver_objective=solver_objective,
        reduction_percent=final_reduction,
        loss_iteration=[point.iteration for point in optimization_history],
        loss_objective=[point.objective for point in optimization_history],
        loss_elapsed=[point.elapsed for point in optimization_history],
        param_names=String.(collect(keys(fitted_parameters))),
        param_fitted=Float64[value for value in values(fitted_parameters)],
        param_guess=Float64[getfield(guess_parameters, name)
                           for name in keys(fitted_parameters)],
        time_s=collect(time_s),
        measured_com_z=Float64.(data.com_z),
        measured_pitch_rad=Float64.(data.pitch_rad),
        measured_speed_ms=Float64.(data.speed_long_ms),
        measured_position_m=Float64.(data.s_travelled_m),
        throttle=Float64.(data.throttle_applied),
        brake=Float64.(data.brake_applied),
        lift_class=String.(lift_class),
        window_indices=collect(loss_data.indices),
        sim_com_height=Float64.(after_run.com_height),
        sim_pitch_rad=Float64.(after_run.pitch_rad),
        sim_velocity_ms=Float64.(after_run.velocity_ms),
        sim_position_m=Float64.(after_run.position_m),
        sim_available_torque_nm=Float64.(after_run.available_torque_nm),
        metric_channel=String.(estimation_metrics.channel),
        metric_rmse_before=Float64.(estimation_metrics.rmse_before),
        metric_rmse_after=Float64.(estimation_metrics.rmse_after),
        metric_mae_before=Float64.(estimation_metrics.mae_before),
        metric_mae_after=Float64.(estimation_metrics.mae_after),
        metric_r2_after=Float64.(estimation_metrics.r2_after),
        # Validation on the test recording. Additive fields: the comparison notebook checks a
        # required-field list and ignores anything extra, so the schema stays at 1.
        test_log_file=basename(test_log),
        test_sample_count=length(test_data.t_s),
        test_objective=test_objective,
        test_metric_channel=String.(test_metrics.channel),
        test_metric_rmse=Float64.(test_metrics.rmse),
        test_metric_mae=Float64.(test_metrics.mae),
        test_metric_r2=Float64.(test_metrics.r2))

    jldsave(result_path; saved=saved_result)

    DataFrame(item=["file", "size (kB)", "log", "loss points", "objective"],
        value=[result_path,
            string(round(filesize(result_path) / 1024; digits=1)),
            saved_result.log_file,
            string(length(saved_result.loss_objective)),
            string(round(saved_result.objective; digits=2))])
end

# Cell 42: Write results/params-opt-v2.34.jld2.

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Sockets = "6462fe0b-24de-5631-8697-dd941f90decc"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
CSV = "~1.3.0"
DataFrames = "~1.8.2"
DifferentialEquations = "~8.1.1"
ForwardDiff = "~1.4.7"
JLD2 = "~0.6.7"
Optimization = "~5.9.1"
OptimizationOptimJL = "~0.4.23"
PlutoUI = "~0.7.84"
WGLMakie = "~0.13.17"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "8d772395a91c9109cf824922d1a2a052dbae637f"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "344f3d221a6bee75925b5c97a30cfd870f12a138"
registries = "General"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.4"

[[deps.ANSIColoredPrinters]]
git-tree-sha1 = "574baf8110975760d391c710b6341da1afa48d8c"
registries = "General"
uuid = "a4c015fc-c6ff-483c-b24f-f7ea428134e9"
version = "0.0.1"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "dee8388a9bec89c8713ad176f96960ae2f98598a"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.4"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptiveArrayPools]]
deps = ["Preferences", "Printf", "Random"]
git-tree-sha1 = "86d0ae34bc46373170064ef33c12a303994ab736"
registries = "General"
uuid = "4f381ef7-9af0-4cbe-99d4-cf36d7b0f233"
version = "0.3.12"

    [deps.AdaptiveArrayPools.extensions]
    AdaptiveArrayPoolsCUDAExt = "CUDA"
    AdaptiveArrayPoolsMetalExt = "Metal"

    [deps.AdaptiveArrayPools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
registries = "General"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
registries = "General"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
registries = "General"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
registries = "General"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
registries = "General"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.BinaryHeaps]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0adb6d8d0a4bae93a07d79c8d59eaf617f4c59da"
registries = "General"
uuid = "b2a6c25c-c996-4615-94ab-6e519ffb8690"
version = "1.1.0"

[[deps.Bonito]]
deps = ["ANSIColoredPrinters", "Base64", "CodecZlib", "Colors", "CommonMark", "Dates", "Deno_jll", "HTTP", "Hyperscript", "JSON", "LinearAlgebra", "Markdown", "MbedTLS", "MsgPack", "Observables", "OrderedCollections", "PrecompileTools", "Random", "RelocatableFolders", "SHA", "Scratch", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "f947a8b5967714bf5b7d82070b647631de1f10d9"
registries = "General"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "5.2.0"

    [deps.Bonito.extensions]
    BonitoDocumenterExt = ["Documenter", "MarkdownAST"]

    [deps.Bonito.weakdeps]
    Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
    MarkdownAST = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "b1b04c236ff7ed92e69d64b620448b3da00933d4"
registries = "General"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.12.8"
weakdeps = ["ChainRulesCore", "ForwardDiff"]

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveChainRulesCoreExt = ["ChainRulesCore", "ForwardDiff"]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "0db159493e83ecb2d122f9c40138d89b70d43963"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.1.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
registries = "General"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "DataStrings", "Dates", "Downloads", "Durations", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "Printf", "Tables", "Unicode"]
git-tree-sha1 = "feb15aa037b0004ed9a988a05bb6f5833a670ac7"
registries = "General"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "1.3.0"

    [deps.CSV.extensions]
    CSVDataDecimalsExt = "DataDecimals"
    CSVFilePathsBaseExt = "FilePathsBase"
    CSVInlineStringsExt = "InlineStrings"

    [deps.CSV.weakdeps]
    DataDecimals = "3e2245cb-6932-498d-a7cf-39d39de8bde3"
    FilePathsBase = "48062228-2e41-5def-b9a4-89aafe57970f"
    InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ChainRulesCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "295fa9b898db109bc56df62c0fc20bd105cec00e"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.8"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "3496e2b359c27793d12bec54ec94be7983a6ceb2"
registries = "General"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.2"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "d4101e848e8d3f585d61d244c2fe0c80a70e6b3b"
registries = "General"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.1.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
registries = "General"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
registries = "General"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
registries = "General"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "af523e59de94de05b9b367a2ca4d63233aacd222"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.5"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "96fce5c80e0834e96ab6d24fb384fd945c9cb2d4"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.2"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "0737a3e59f9d94957f5b177fa2ca40901636b85b"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.4"

[[deps.CommonMark]]
deps = ["PrecompileTools"]
git-tree-sha1 = "7c8fe02c7eb6fe22e89d3990123a2a931ca8fcfb"
registries = "General"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "1.0.4"

    [deps.CommonMark.extensions]
    CommonMarkMarkdownASTExt = "MarkdownAST"
    CommonMarkMarkdownExt = "Markdown"

    [deps.CommonMark.weakdeps]
    Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
    MarkdownAST = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "b56df0c58a9d37b4a926f84f3f4f283a5987680e"
registries = "General"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.9"

[[deps.ConcreteStructs]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a72c3b5ce6d2a477f55b5c9b8756e91e695c67f6"
registries = "General"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.8"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
registries = "General"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
registries = "General"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
registries = "General"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStrings]]
git-tree-sha1 = "b8032263d364177a0b192ea15f277df4e6ab07e2"
registries = "General"
uuid = "48204bd6-5611-42ea-b167-fc1d713f9be2"
version = "1.1.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1e28f37cbdca4b594ad86ecfe2ee8f25796ebff"
registries = "General"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.7.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
registries = "General"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "BracketingNonlinearSolve", "DocStringExtensions", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "RespecializeParams", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "47230845a4944d292c07079838bdfee25bffebf8"
registries = "General"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "7.21.6"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDynamicQuantitiesExt = "DynamicQuantities"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseFlexUnitsExt = "FlexUnits"
    DiffEqBaseForwardDiffExt = ["ForwardDiff"]
    DiffEqBaseGTPSAExt = "GTPSA"
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseMooncakeExt = "Mooncake"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FlexUnits = "76e01b6b-c995-4ce6-8559-91e72a3d4e95"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentialEquations]]
deps = ["OrdinaryDiffEq", "PrecompileTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "d167917674ca92bbf51b56a3e00d019792695110"
registries = "General"
uuid = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
version = "8.1.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "0693d8b0a4608ff289d228ab4c598df5894845cd"
registries = "General"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.21"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "2ddbc341251ed713d39e503b504593a5d6789551"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.132"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Durations]]
deps = ["Dates"]
git-tree-sha1 = "407b209d76168dcc4355f9dcc4d3b3aaa60dfb3e"
registries = "General"
uuid = "41ac09f5-0a95-4810-aa18-c88bf48ee130"
version = "1.4.1"

    [deps.Durations.extensions]
    DurationsArrowExt = "Arrow"
    DurationsTimeZonesExt = "TimeZones"

    [deps.Durations.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
    TimeZones = "f269a46b-ccf7-5d73-abea-4c690281aa53"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
registries = "General"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "d88c803e26799193db42e47f6e1e28eeb0322e4d"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.27"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
registries = "General"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0f666deaf8099a7d7ac229a74402ada88898be1"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.9.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "fdcc9d4a253e0041d37c95487f9ae027c3ec002d"
registries = "General"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "093bf5e9ea63b36032240033ec2f149088fc0926"
registries = "General"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "1.5.1"

    [deps.FastBroadcast.extensions]
    FastBroadcastPolyesterExt = "Polyester"
    FastBroadcastStaticExt = "Static"
    FixedSizeArraysExt = "FixedSizeArrays"

    [deps.FastBroadcast.weakdeps]
    FixedSizeArrays = "3821ddf9-e5b5-40d5-8e25-6813ab96b5e2"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
registries = "General"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastInterpolations]]
deps = ["AdaptiveArrayPools", "HelpPlots", "LinearAlgebra", "Preferences", "Printf"]
git-tree-sha1 = "f7bcd496f0a2e4e1174e7e1f667f27fb671fd7df"
registries = "General"
uuid = "9ea80cae-fc13-4c00-8066-6eaedb12f34b"
version = "0.4.20"

    [deps.FastInterpolations.extensions]
    FastInterpolationsChainRulesCoreExt = "ChainRulesCore"
    FastInterpolationsColorVectorSpaceExt = ["ColorTypes", "ColorVectorSpace"]
    FastInterpolationsEnzymeExt = "Enzyme"
    FastInterpolationsForwardDiffExt = "ForwardDiff"
    FastInterpolationsRecipesBaseExt = "RecipesBase"
    FastInterpolationsSymbolicsExt = ["Symbolics", "SymbolicUtils"]

    [deps.FastInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
    ColorVectorSpace = "c3611d14-8923-5661-9e6a-0046d554d3a4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.FastPower]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6bcbd090fc6b6f022fc0ad96f77ba31b9577282"
registries = "General"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.5.1"

    [deps.FastPower.extensions]
    FastPowerEnzymeExt = "Enzyme"
    FastPowerForwardDiffExt = "ForwardDiff"
    FastPowerMeasurementsExt = "Measurements"
    FastPowerMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    FastPowerMooncakeExt = "Mooncake"
    FastPowerReverseDiffExt = "ReverseDiff"
    FastPowerTrackerExt = "Tracker"

    [deps.FastPower.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
registries = "General"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
registries = "General"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
registries = "General"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "93eb9afa3a53b29d8a49714b83878225402b3281"
registries = "General"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "3.4.1"

    [deps.FindFirstFunctions.extensions]
    FindFirstFunctionsEnzymeCoreExt = "EnzymeCore"
    FindFirstFunctionsMooncakeExt = "Mooncake"

    [deps.FindFirstFunctions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "5031f23e040bf17082e5b52422d77b5e844eefb1"
registries = "General"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.33.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "e1e9ba3a8f5d71e6ae698f138a1362aeabb42e9f"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.18.3+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a55669aa77ae8bbdc7310c4027112500ff5ec874"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.7"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
registries = "General"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d95a7a3a7e0f2fffc6291bd33035830aab01310b"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.15.0+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
registries = "General"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e894be022a2eba58463939f1bf1f7f7006f27f52"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.19+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
registries = "General"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2bcce3ad6f6977d617928d7707fdc86ac83cce03"
registries = "General"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.13.0"

    [deps.FunctionWrappersWrappers.extensions]
    FunctionWrappersWrappersEnzymeExt = ["Enzyme", "EnzymeCore"]
    FunctionWrappersWrappersMooncakeExt = "Mooncake"

    [deps.FunctionWrappersWrappers.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "f60436514e6d73f8ee069ae4ee0ff038fe96b69d"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.2"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "ec46c5825710fa1a15d468acb2d93cc939a7a5fe"
registries = "General"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.13"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3efbbc027441271444dcd0c0a46f2d119dc4329"
registries = "General"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "6.1.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "ef70da5e123a06a29e2d6ddff0f09985bc226491"
registries = "General"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.3"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "EnumX", "Logging", "PrecompileTools", "Random", "Reseau", "SHA", "TranscodingStreams", "URIs", "UUIDs", "Zlib_jll"]
git-tree-sha1 = "f756cc1b518604435938ebca5dddf08c4921ea50"
registries = "General"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "2.14.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f700c32c1e9e11b576592043f270e6b33868a56f"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15000.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HelpPlots]]
git-tree-sha1 = "07d389d81ca31e42d68bd88cfa1e900a63b7d378"
registries = "General"
uuid = "5880b2e8-ad0a-48c9-bacd-30dc9c1357e0"
version = "1.1.0"

    [deps.HelpPlots.extensions]
    HelpPlotsExt = "Plots"

    [deps.HelpPlots.weakdeps]
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
registries = "General"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
registries = "General"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
registries = "General"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "f0f005f997dfb8c5fe23920d99458a9619873893"
registries = "General"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.10"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
registries = "General"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
registries = "General"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
registries = "General"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
registries = "General"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
registries = "General"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "1c531bf0f8a5c60a340926e058fd3f209b5eef5d"
registries = "General"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.12"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticMakieExt = "Makie"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
registries = "General"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"
weakdeps = ["Makie", "Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
registries = "General"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "123311cf9834c6ee6a8347bba9af68f255615391"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.11.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecCore", "ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "9ce2e7c49ae4a7035b7d60db3a553b1f6c16875d"
registries = "General"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.7"

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "06d90f699285d055e34f0b014636a75a17ff2252"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.13.4"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]
    JSONSerializationExt = ["Serialization"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
registries = "General"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e06b451acfb1c6fa5ea753b5042207226271f455"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "b13076e4fff6ea0d0a5ddd9b939a11ca2027755a"
registries = "General"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d7c65411dda4bfa0beb7490476122b4f805cf31"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.3.0+0"

[[deps.LHLFactorization]]
deps = ["LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "51d99d9892d0f95b4c9d4d7310785971f2d48298"
registries = "General"
uuid = "2faa5264-e118-4071-8864-e10559f68c7c"
version = "2.2.2"

    [deps.LHLFactorization.extensions]
    LHLFactorizationPolyesterExt = "Polyester"
    LHLFactorizationSparseExt = ["PureKLU", "SparseArrays"]

    [deps.LHLFactorization.weakdeps]
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PureKLU = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
registries = "General"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "d4816abce26971e3237b46a47b99991f306e4832"
registries = "General"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LineSearch]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "FastClosures", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "ReactantCore", "SciMLBase", "SciMLJacobianOperators", "StaticArraysCore"]
git-tree-sha1 = "53cf132c453dcfcf10f1cb9a701a07c99b7e842e"
registries = "General"
uuid = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
version = "0.1.20"
weakdeps = ["LineSearches"]

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "a09a85e94e681def2446752cddaf9401a4d82b6d"
registries = "General"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.8.2"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LinearSolve]]
deps = ["AMD", "ArrayInterface", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "LHLFactorization", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "OpenBLAS_jll", "PrecompileTools", "Preferences", "PureKLU", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "SparseArrays", "SparseColumnPivotedQR", "StaticArraysCore"]
git-tree-sha1 = "e3e84c5c330ea65568a85aef00031e2b870b0e99"
registries = "General"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "5.18.3"

    [deps.LinearSolve.extensions]
    LinearSolveAMDGPUExt = "AMDGPU"
    LinearSolveAMGXExt = ["AMGX", "CUDA"]
    LinearSolveAlgebraicMultigridExt = "AlgebraicMultigrid"
    LinearSolveArnoldiMethodExt = "ArnoldiMethod"
    LinearSolveArpackExt = "Arpack"
    LinearSolveBLISExt = ["blis_jll", "LAPACK_jll"]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCKTSOExt = "CKTSO"
    LinearSolveCUDAExt = ["cuSOLVER"]
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveCUSOLVERRFExt = "CUSOLVERRF"
    LinearSolveChainRulesCoreExt = "ChainRulesCore"
    LinearSolveCliqueTreesExt = "CliqueTrees"
    LinearSolveConjugateGradientsExt = "ConjugateGradients"
    LinearSolveElementalExt = "Elemental"
    LinearSolveEnzymeExt = "EnzymeCore"
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveFastLapackInterfaceExt = "FastLapackInterface"
    LinearSolveForwardDiffExt = "ForwardDiff"
    LinearSolveGinkgoExt = "Ginkgo"
    LinearSolveHSLExt = "HSL"
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveJacobiDavidsonExt = "JacobiDavidson"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMUMPSExt = "MUMPS"
    LinearSolveMetalExt = "Metal"
    LinearSolveMooncakeExt = "Mooncake"
    LinearSolvePETScExt = ["PETSc", "SparseMatricesCSR"]
    LinearSolvePETScMPIExt = ["PETSc", "PartitionedArrays", "SparseMatricesCSR"]
    LinearSolveParUExt = "ParU_jll"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolvePartitionedSolversExt = ["PartitionedArrays", "PartitionedSolvers"]
    LinearSolvePureUMFPACKExt = "PureUMFPACK"
    LinearSolveReactantExt = "Reactant"
    LinearSolveRecursiveFactorizationExt = ["RecursiveFactorization", "TriangularSolve"]
    LinearSolveSLATEExt = "SLATE_jll"
    LinearSolveSTRUMPACKExt = "STRUMPACK_jll"
    LinearSolveSparspakExt = "Sparspak"
    LinearSolveSpecializingFactorizationsExt = "SpecializingFactorizations"
    LinearSolveSuperLUDISTExt = "SuperLUDIST"

    [deps.LinearSolve.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AMGX = "c963dde9-0319-47f5-bf0c-b07d3c80ffa6"
    AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
    ArnoldiMethod = "ec485272-7323-5ecc-a04f-4719b315124d"
    Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CKTSO = "8e543fc2-2ef1-4e2d-a99e-39beaa046845"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    CUSOLVERRF = "a8cc9031-bad2-4722-94f5-40deabb4245c"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    ConjugateGradients = "f59de78d-195d-4e7b-a078-2e47da4c3ad6"
    Elemental = "902c3f28-d1ec-5e7e-8399-a24c3845ee38"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    FastLapackInterface = "29a986be-02c6-4525-aec4-84b980013641"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Ginkgo = "4c8bd3c9-ead9-4b5e-a625-08f1338ba0ec"
    HSL = "34c5aeac-e683-54a6-a0e9-6e0fdc586c50"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    JacobiDavidson = "11c68b98-9c9b-11e8-267b-bbb95576cead"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    LAPACK_jll = "51474c39-65e3-53ba-86ba-03b1b862ec14"
    MUMPS = "55d2b088-9f4e-11e9-26c0-150b02ea6a46"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    ParU_jll = "9e0b026c-e8ce-559c-a2c4-6a3d5c955bc9"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    PartitionedArrays = "5a9dfac6-5c52-46f7-8278-5e2210713be9"
    PartitionedSolvers = "11b65f7f-80ac-401b-9ef2-3db765482d62"
    PureUMFPACK = "b7e1f0a2-3c4d-4e5f-9a0b-1c2d3e4f5a6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
    SLATE_jll = "cb01ff19-587c-516c-bcf9-532d49da298d"
    STRUMPACK_jll = "86fbd0b9-476f-557c-b766-62c724b42d8c"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    Sparspak = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
    SpecializingFactorizations = "fa08b7a1-13d3-4faf-875d-5cbc1520e3f3"
    SuperLUDIST = "4cd002a6-0da4-410d-a012-232df062f478"
    TriangularSolve = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
    blis_jll = "6136c539-28a5-5bf0-87cc-b183200dce32"
    cuSOLVER = "887afef0-6a32-4de5-add4-7827692ba8fc"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
registries = "General"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
registries = "General"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FastInterpolations", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "8ff1cd11162828e9f517e32d83b49d2af63391c6"
registries = "General"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.17"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
registries = "General"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "PrecompileTools"]
git-tree-sha1 = "f2cde0ae772162f20287b803e623966d0d94dea9"
registries = "General"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.8"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "0e4b139a1f0f0b8c054fcdfd3be5f610cfb8c787"
registries = "General"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.2.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
registries = "General"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
registries = "General"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.10.20"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
registries = "General"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "f96d38936d92d610318dec4d3b5ef37b0373c20f"
registries = "General"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
registries = "General"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "NonlinearSolveBase", "NonlinearSolveFirstOrder", "NonlinearSolveQuasiNewton", "NonlinearSolveSpectralMethods", "PrecompileTools", "Preferences", "Reexport", "SciMLBase", "Setfield", "SimpleNonlinearSolve", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "90afae5b2b33dcd1098e4c1a7b6066f18741bf6f"
registries = "General"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "4.32.0"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = ["NLsolve", "LineSearches"]
    NonlinearSolvePETScExt = ["PETSc", "MPI", "SparseArrays"]
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSundialsExt = "Sundials"

    [deps.NonlinearSolve.weakdeps]
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnumX", "EnzymeCore", "FastClosures", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "RespecializeParams", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "f83b1fb97c02b99390346e13a71a027a59dc2459"
registries = "General"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.55.1"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseChainRulesCoreExt = "ChainRulesCore"
    NonlinearSolveBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseMooncakeExt = ["ChainRulesCore", "Mooncake"]
    NonlinearSolveBaseReverseDiffExt = "ReverseDiff"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"
    NonlinearSolveBaseTrackerExt = "Tracker"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.NonlinearSolveFirstOrder]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConcreteStructs", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "Setfield", "Sobol", "StaticArraysCore"]
git-tree-sha1 = "95a105b647b656dfbd7985ede8eb0eaf4b836505"
registries = "General"
uuid = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
version = "2.10.0"

[[deps.NonlinearSolveQuasiNewton]]
deps = ["ArrayInterface", "CommonSolve", "ConcreteStructs", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "StaticArraysCore"]
git-tree-sha1 = "db0bca65e9c849333023a0d5827813947d003f09"
registries = "General"
uuid = "9a2c21bd-3a47-402d-9113-8faf9a0ee114"
version = "1.15.3"
weakdeps = ["ForwardDiff"]

    [deps.NonlinearSolveQuasiNewton.extensions]
    NonlinearSolveQuasiNewtonForwardDiffExt = "ForwardDiff"

[[deps.NonlinearSolveSpectralMethods]]
deps = ["CommonSolve", "ConcreteStructs", "LineSearch", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "c3e5a641a867b7f37155a78086cf3f22134c088f"
registries = "General"
uuid = "26075421-4e9a-44e1-8bd1-420ed7ad02b2"
version = "1.8.3"
weakdeps = ["ForwardDiff"]

    [deps.NonlinearSolveSpectralMethods.extensions]
    NonlinearSolveSpectralMethodsForwardDiffExt = "ForwardDiff"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
registries = "General"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
registries = "General"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "OpenJPH_jll", "Zlib_jll", "Zstd_jll", "libdeflate_jll"]
git-tree-sha1 = "afa66dc1246b28b68600d053715e00a379268980"
registries = "General"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.5.3+0"

[[deps.OpenJPH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "1b1f3cabb1130670812b848f4623d69d578ea805"
registries = "General"
uuid = "95196869-6d61-5868-8eee-c38c33804e0c"
version = "0.32.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "81f338ad984b25bc82471db4d235e7b8bf21b8d6"
registries = "General"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.3.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Printf", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "b4d47691ca17e5f8e26cc7e453f6ab92d480285f"
registries = "General"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "5.9.1"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DifferentiationInterface", "DocStringExtensions", "FastClosures", "LinearAlgebra", "PrecompileTools", "SciMLBase", "SciMLLogging", "SparseArrays", "SparseConnectivityTracer", "SparseMatrixColorings", "SymbolicIndexingInterface"]
git-tree-sha1 = "d58d389e754d6641aab95950779a214b5e43c93a"
registries = "General"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "5.10.1"

    [deps.OptimizationBase.extensions]
    OptimizationChainRulesCoreExt = "ChainRulesCore"
    OptimizationEnzymeExt = ["ChainRulesCore", "Enzyme"]
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMLDataDevicesExt = "MLDataDevices"
    OptimizationMLUtilsExt = "MLUtils"
    OptimizationMooncakeExt = "Mooncake"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSymbolicAnalysisExt = "SymbolicAnalysis"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SymbolicAnalysis = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "OptimizationBase", "SciMLBase", "SparseArrays"]
git-tree-sha1 = "214a40c3e5bf26fbf240ce15dceff0f3d8794b56"
registries = "General"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.4.23"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.OrdinaryDiffEq]]
deps = ["ADTypes", "CommonSolve", "DiffEqBase", "DocStringExtensions", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore", "OrdinaryDiffEqDefault", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "bae9b8e5f1d00e51c554648e5094e215164a19f8"
registries = "General"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "7.8.1"

[[deps.OrdinaryDiffEqBDF]]
deps = ["ADTypes", "ArrayInterface", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "OrdinaryDiffEqSDIRK", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "06dddbb32c023b4fc42f620bc7ff98f72d3d1f5b"
registries = "General"
uuid = "6ad6398a-0878-4a85-9266-38940aa047c8"
version = "2.4.14"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "BinaryHeaps", "CommonSolve", "ConstructionBase", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FindFirstFunctions", "FunctionWrappers", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "PrecompileTools", "Preferences", "Printf", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "5765dd5605c88f1db2b3630df6f291751a96cd16"
registries = "General"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "4.19.1"

    [deps.OrdinaryDiffEqCore.extensions]
    OrdinaryDiffEqCoreMooncakeExt = "Mooncake"
    OrdinaryDiffEqCorePolyesterExt = "Polyester"
    OrdinaryDiffEqCoreSparseArraysExt = "SparseArrays"

    [deps.OrdinaryDiffEqCore.weakdeps]
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.OrdinaryDiffEqDefault]]
deps = ["ADTypes", "DiffEqBase", "EnumX", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "PrecompileTools", "Preferences", "Reexport", "SciMLBase"]
git-tree-sha1 = "1ce6ba1f52c94059089f835f173fddda90233959"
registries = "General"
uuid = "50262376-6c5a-4cf5-baba-aaf4f84d72d7"
version = "2.6.2"

[[deps.OrdinaryDiffEqDifferentiation]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DiffEqBase", "DifferentiationInterface", "FastBroadcast", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEqCore", "SciMLBase", "SciMLOperators", "SparseMatrixColorings", "StaticArraysCore"]
git-tree-sha1 = "7bab95310a175e7b24b419cdc49d234c35a13a22"
registries = "General"
uuid = "4302a76b-040a-498a-8c04-15b101fed76b"
version = "3.12.5"
weakdeps = ["SparseArrays"]

    [deps.OrdinaryDiffEqDifferentiation.extensions]
    OrdinaryDiffEqDifferentiationSparseArraysExt = "SparseArrays"

[[deps.OrdinaryDiffEqNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "DiffEqBase", "FastBroadcast", "FastClosures", "ForwardDiff", "LinearAlgebra", "LinearSolve", "MuladdMacro", "NonlinearSolve", "NonlinearSolveBase", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "PreallocationTools", "RecursiveArrayTools", "SciMLBase", "SciMLOperators", "SciMLPublic", "SimpleNonlinearSolve", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "19dcb7f30de00a97d4c4e39094f398eeca31080a"
registries = "General"
uuid = "127b3ac7-2247-4354-8eb6-78cf4e7c58e8"
version = "2.9.10"

[[deps.OrdinaryDiffEqRosenbrock]]
deps = ["ADTypes", "ArrayInterface", "DiffEqBase", "DifferentiationInterface", "FastBroadcast", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "LinearSolve", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqRosenbrockTableaus", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "4d6354525db068b9d1578389274b7e6cabc663b4"
registries = "General"
uuid = "43230ef6-c299-4910-a778-202eb28ce4ce"
version = "2.7.5"

[[deps.OrdinaryDiffEqRosenbrockTableaus]]
git-tree-sha1 = "bab0d4ccee6fcb89a786b89207774314523e2e5d"
registries = "General"
uuid = "b4bd8bb3-f80f-41d2-9b21-73a655b304b9"
version = "2.4.2"

[[deps.OrdinaryDiffEqSDIRK]]
deps = ["ADTypes", "CommonSolve", "ConstructionBase", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "9e2bd5473cf2fb7c813f7c3f93a1f5a61b5a699a"
registries = "General"
uuid = "2d112036-d095-4a1e-ab9a-08536f3ecdbf"
version = "2.9.8"

[[deps.OrdinaryDiffEqTsit5]]
deps = ["CommonSolve", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "371de3eb5985febd9d7babb0937520653f2b15ca"
registries = "General"
uuid = "b1df2697-797e-41e3-8120-5422d3b24e4a"
version = "2.1.5"

[[deps.OrdinaryDiffEqVerner]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "5a66a18325e678f260e849a62840cb4903ece511"
registries = "General"
uuid = "79d7bb75-1356-48c1-b8c0-6832512096c2"
version = "2.4.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
registries = "General"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
registries = "General"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
registries = "General"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "91503bbab255aec9374145accd1b09c7e108cd54"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.5"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
registries = "General"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "9869376035f47756b8b8e711fd0d06a06d51da6f"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.84"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
registries = "General"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
registries = "General"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cd00e28071a75c98664663f87c2ae447d25c0503"
registries = "General"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.7.1"

    [deps.PreallocationTools.extensions]
    PreallocationToolsEnzymeCoreExt = "EnzymeCore"
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
registries = "General"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Printf"]
git-tree-sha1 = "ce5ba3edfdeba107a721748d53b5f5887f415853"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.13.0"

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"
    ProgressMeterIJuliaExt = "IJulia"

    [deps.ProgressMeter.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.PureKLU]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "ad0e84647fe7b3c2ae02aaeace1522139aaf6a27"
registries = "General"
uuid = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
version = "1.6.0"
weakdeps = ["ForwardDiff"]

    [deps.PureKLU.extensions]
    PureKLUForwardDiffExt = "ForwardDiff"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
registries = "General"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
registries = "General"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.ReactantCore]]
deps = ["ExpressionExplorer", "MacroTools"]
git-tree-sha1 = "1fe4e616752a0c4818ae88a504fc59b0fb016659"
registries = "General"
uuid = "a3311ec8-5e00-46d5-b541-4f83e724a433"
version = "0.1.26"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "47fe0f2a25b5e98412c65227f8a649d26bb7f1b9"
registries = "General"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "4.5.6"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsCUDAExt = "CUDA"
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsFastBroadcastPolyesterExt = ["FastBroadcast", "Polyester"]
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsMooncakeExt = "Mooncake"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Reseau]]
deps = ["NetworkOptions", "OpenSSL_jll", "PrecompileTools", "Random", "SHA"]
git-tree-sha1 = "13d0465bbd89b45183c0363b635dc82d5843593f"
registries = "General"
uuid = "802f3686-a58f-41ce-bb0c-3c43c75bba36"
version = "1.6.6"

[[deps.RespecializeParams]]
deps = ["FunctionWrappersWrappers", "PrecompileTools"]
git-tree-sha1 = "140c074bb63518a0e1ec2c8aba2a8a5719bcf21c"
registries = "General"
uuid = "9fe22ead-9e00-4db2-8b46-706a60d40f5e"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "c5e8a0a52c0d5d043214de04f11f4d9c2c6ccdb8"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.13"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
registries = "General"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "PrecompileTools", "SHA", "Serialization"]
git-tree-sha1 = "9bb075002b40849582905394bb3031fb77aa102d"
registries = "General"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.27"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "37ac518b3c924f110d82d50173eaf4ebeaf27ad7"
registries = "General"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.8.1"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FindFirstFunctions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "Markdown", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "85f85c3c5b949e7d691047e758e3fdb77c21310c"
registries = "General"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "3.58.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseFunctionPropertiesExt = "FunctionProperties"
    SciMLBaseMakieExt = ["Makie", "GeometryBasics"]
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReactantCoreExt = "ReactantCore"
    SciMLBaseReactantExt = "Reactant"
    SciMLBaseReverseDiffExt = ["ReverseDiff", "ForwardDiff"]
    SciMLBaseStaticArraysExt = "StaticArrays"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore", "ZygoteRules", "FillArrays"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    FunctionProperties = "f62d2435-5019-4c03-9749-2d4c77af0cbc"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "781d5413c2991e581da86979b0d9f86f9a034a1c"
registries = "General"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.20"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "PrecompileTools", "Preferences"]
git-tree-sha1 = "3aafee7edc2ec7e71f50dcc6816cb031fac390a0"
registries = "General"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "2.1.0"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "Adapt", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "3feb8a7bce88bb3ef051a4eda8106004ad040b6f"
registries = "General"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.30.4"

    [deps.SciMLOperators.extensions]
    SciMLOperatorsLoopVectorizationExt = "LoopVectorization"
    SciMLOperatorsSparseArraysExt = "SparseArrays"

    [deps.SciMLOperators.weakdeps]
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "4b8fc0ffe22b4ecba7a02f1fec0b3757db014d0c"
registries = "General"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.6"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "4910c2c2f334d74735b7a8fd02f837ead352576a"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.11"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "89880a2cca5cabc68c9023f58e544beda1d2fc1f"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.2.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "57aa595158717ef165e6f5ab639fe2e3178c0a2b"
registries = "General"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
registries = "General"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "8d9ec24ec7bd126abf97b44ba5bdd6d280572095"
registries = "General"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "2.14.6"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2c79185e5261b159474903598c571ddd9f12ff7b"
registries = "General"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.6"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
registries = "General"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SparseColumnPivotedQR]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "8cf1a59c78dd6f900feb366e5a574f99ed278ef1"
registries = "General"
uuid = "a57abbd0-fea5-4d57-96be-5e525945e8e4"
version = "2.1.8"
weakdeps = ["AMD"]

    [deps.SparseColumnPivotedQR.extensions]
    SparseColumnPivotedQRAMDExt = "AMD"

[[deps.SparseConnectivityTracer]]
deps = ["ADTypes", "DocStringExtensions", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "277d8929a56912e7342458d4db22ee45e4c0e75c"
registries = "General"
uuid = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
version = "1.2.4"

    [deps.SparseConnectivityTracer.extensions]
    SparseConnectivityTracerChainRulesCoreExt = "ChainRulesCore"
    SparseConnectivityTracerLogExpFunctionsExt = "LogExpFunctions"
    SparseConnectivityTracerNNlibExt = "NNlib"
    SparseConnectivityTracerNaNMathExt = "NaNMath"
    SparseConnectivityTracerSpecialFunctionsExt = "SpecialFunctions"

    [deps.SparseConnectivityTracer.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Random", "SparseArrays"]
git-tree-sha1 = "0730f14b07c68fd3b3a6fae89a829a5124a7ae7e"
registries = "General"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.29"

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsAMDGPUExt = "AMDGPU"
    SparseMatrixColoringsCUDAExt = ["CUDA", "cuSPARSE"]
    SparseMatrixColoringsCliqueTreesExt = "CliqueTrees"
    SparseMatrixColoringsColorsExt = "Colors"
    SparseMatrixColoringsGPUArraysExt = "GPUArrays"
    SparseMatrixColoringsJuMPExt = ["JuMP", "MathOptInterface"]

    [deps.SparseMatrixColorings.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
    cuSPARSE = "b26da814-b3bc-49ef-b0ee-c816305aa060"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "88ef0d98204515bc53dd4aaea5f7788882d1b66c"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
registries = "General"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "8a8a52f887f38d44014e24ee46b8a7311c625f6d"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.24"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "4daaddbe37aa80b4551bf7c56adf53e7cfcdf35a"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.4"

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "b04bfab37484defe300b2f9ed64469abab2431ff"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.12.0"

    [deps.StructUtils.extensions]
    StructUtilsDBInterfaceExt = ["DBInterface", "Tables"]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DBInterface = "a10d1c49-ce27-4219-8d33-6db1a4562965"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrecompileTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "42dd4c193a1bacea5b6adf78b6224de907b06351"
registries = "General"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.56"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "81c9b4137edfe56a56efcdcb35d721b2ce3e2416"
registries = "General"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.8"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
registries = "General"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
registries = "General"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TimerOutputs]]
deps = ["ExprTools", "PrettyTables", "Printf", "Tables"]
git-tree-sha1 = "2244393f04554e24b54896b430b2ab3850a2232d"
registries = "General"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "1.2.2"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
registries = "General"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
registries = "General"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "eb3b85ca904a1d1647b15acc091b113b8874b648"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "1f0f9f401753701a7e4113b5056ca38d33875b55"
registries = "General"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.29.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "798486379d2b1ef502c5f68e18885b8274183b66"
registries = "General"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.13.17"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
registries = "General"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
registries = "General"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
registries = "General"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "21e05cef1411bfe6b22a1ca8cbdc64a297d4b447"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdeflate_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "11c07205bc3b3b7833b8fb36beb4e4764d0126a7"
registries = "General"
uuid = "46979653-d7f6-5232-b59e-dd310c4598de"
version = "1.26.0+1"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "e067c8bae65bb40866552296a2d5b4dc65b8928f"
registries = "General"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.80.702+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "21cbc8fc2ac0bb182ed461e556d9c062ae6a3164"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.24.1+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "52d3b9475133c3bc8c0a7f90f18bfc8cdb5a443a"
registries = "General"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ed2d8d4ad96c03d3abe3382cdbbdea9489b63636"
registries = "General"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2023.1.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╠═c0000000-0000-4000-8000-000000000002
# ╠═c0000000-0000-4000-8000-000000000031
# ╠═c0000000-0000-4000-8000-000000000060
# ╠═c0000000-0000-4000-8000-000000000001
# ╠═c0000000-0000-4000-8000-000000000030
# ╠═c0000000-0000-4000-8000-00000000001e
# ╠═c0000000-0000-4000-8000-000000000061
# ╠═c0000000-0000-4000-8000-000000000003
# ╠═c0000000-0000-4000-8000-000000000004
# ╠═c0000000-0000-4000-8000-000000000005
# ╠═c0000000-0000-4000-8000-000000000006
# ╠═c0000000-0000-4000-8000-000000000008
# ╠═c0000000-0000-4000-8000-000000000009
# ╠═c0000000-0000-4000-8000-00000000000a
# ╠═c0000000-0000-4000-8000-00000000000b
# ╠═c0000000-0000-4000-8000-00000000000c
# ╠═c0000000-0000-4000-8000-00000000000d
# ╠═c0000000-0000-4000-8000-00000000000e
# ╠═c0000000-0000-4000-8000-00000000000f
# ╠═c0000000-0000-4000-8000-000000000032
# ╠═c0000000-0000-4000-8000-000000000010
# ╠═c0000000-0000-4000-8000-000000000011
# ╠═c0000000-0000-4000-8000-000000000050
# ╠═c0000000-0000-4000-8000-000000000051
# ╠═c0000000-0000-4000-8000-000000000052
# ╠═c0000000-0000-4000-8000-000000000012
# ╠═c0000000-0000-4000-8000-000000000013
# ╠═c0000000-0000-4000-8000-000000000014
# ╠═c0000000-0000-4000-8000-000000000015
# ╠═c0000000-0000-4000-8000-000000000016
# ╠═c0000000-0000-4000-8000-000000000017
# ╠═c0000000-0000-4000-8000-000000000018
# ╠═c0000000-0000-4000-8000-000000000019
# ╠═c0000000-0000-4000-8000-000000000020
# ╠═c0000000-0000-4000-8000-00000000001a
# ╠═c0000000-0000-4000-8000-000000000062
# ╠═c0000000-0000-4000-8000-00000000001b
# ╠═c0000000-0000-4000-8000-00000000001f
# ╠═c0000000-0000-4000-8000-00000000001c
# ╠═c0000000-0000-4000-8000-00000000001d
# ╠═c0000000-0000-4000-8000-0000000000e0
# ╠═c0000000-0000-4000-8000-0000000000e1
# ╠═c0000000-0000-4000-8000-0000000000f0
# ╠═c0000000-0000-4000-8000-0000000000e2
# ╠═c0000000-0000-4000-8000-0000000000e3
# ╠═c0000000-0000-4000-8000-0000000000e4
# ╠═c0000000-0000-4000-8000-0000000000ea
# ╠═c0000000-0000-4000-8000-0000000000eb
# ╠═c0000000-0000-4000-8000-0000000000ec
# ╠═c0000000-0000-4000-8000-0000000000e5
# ╠═c0000000-0000-4000-8000-0000000000e6
# ╠═c0000000-0000-4000-8000-0000000000e7
# ╠═c0000000-0000-4000-8000-0000000000e8
# ╠═c0000000-0000-4000-8000-0000000000e9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
