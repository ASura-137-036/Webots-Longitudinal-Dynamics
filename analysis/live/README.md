# Live half-car HUD (post-optimisation)

Makes a short clip where the fitted half-car model is drawn over the Webots Range Rover it is
predicting: a side-on tracking shot of the car, with heave and pitch plots in the upper corners, the
throttle and brake trace underneath them, and the current numbers in the lower left. Everything on
screen comes out of the recording itself — the 3D view and the panel are composited by Webots, not
afterwards.

The model is the **v2.34** fit from `analysis/params_opt_v2.34.jl`: a drive limited both by the motor
curve and by the vehicle library's wheel-spin guard, a constant pad torque with a smooth standstill edge,
and a single passive resistance proportional to speed. The vehicle total mass is anchored to the proto's
`2700 kg` and one unsprung mass is shared by both axles, which leaves twenty-one estimated parameters —
none of them near a bound.

The deliverable is the world and the controller:

* `worlds/highway_overtake_postopt.wbt`
* `controllers/postopt_viewer/`

## Pipeline

```powershell
# 1. (only if the schedule or the fitted parameters changed)
python analysis/live/make_schedule.py

# 2. drive, draw and record; writes media/postopt_raw.mp4 and media/hud_samples.csv
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --mode=realtime `
    --stdout --stderr "worlds\highway_overtake_postopt.wbt"

# 3. re-encode for delivery
ffmpeg -i media/postopt_raw.mp4 -c:v libx264 -preset slow -crf 19 -pix_fmt yuv420p `
    -vf "scale=in_range=full:out_range=tv" -colorspace bt709 -color_range tv `
    -movflags +faststart media/postopt_hud.mp4
```

`--mode=realtime` is required: `--mode=fast` and `--no-rendering` turn off 3D rendering entirely, so
the movie comes out as a still frame with a live panel on top. That is worth stating because it is
easy to conclude from it that the scene cannot be captured at all — it can, in real time. The run
takes about a minute of wall clock for 23 s of simulation. The controller writes its output to
`media/`, which is where the delivered artefacts live.

`export_params.jl` regenerates the parameter file and the parity fixtures from the Julia notebook;
`check_parity.py` asserts the Python port matches the Julia right-hand side to 1e-10 and a full
trajectory to 1e-9. Both default to the reference model, so no arguments are needed:

```powershell
julia --project=analysis analysis/live/export_params.jl
python analysis/live/check_parity.py
```

That pair rewrites `controllers/postopt_viewer/params_opt_v2.json` and the two fixtures from
`analysis/params_opt_v2.34.jl` and `analysis/results/params-opt-v2.34.jld2`, and then proves the Python
port still reproduces the model. Run it after **any** edit to the physics — a stale fixture after a model
change is exactly the failure the check exists to surface.

Both scripts also take an optional suffix so a second fitted model can sit beside the reference one
instead of replacing it, which is how the variants during development were compared:

```powershell
julia --project=analysis analysis/live/export_params.jl `
    analysis/params_opt_v2.34.jl analysis/results/params-opt-v2.34.jld2 _variant
python analysis/live/check_parity.py _variant
```

and the world is then replayed against it with `POSTOPT_PARAMS=params_opt_v2_variant.json`, which leaves
the reference file untouched. That is how the development variants were compared: the suffix keeps each
fitted model beside the reference instead of overwriting it.

## Which parameters the panel uses

`PARAMS_FILE` defaults to **`params_opt_v2.json`**, the v2.34 fit: held-out objective `7.65`, with
speed `2.16 km/h` and heave `1.6 mm`. It is the only variant whose parameters all sit away from their
bounds *and* whose masses are physical.

The fitted values that shape the movie: `resistance` `180.0 N per m/s`, `traction_acceleration`
`14.28 m/s²`, `throttle_force` `209.8 kN`, sprung mass `2601.8 kg` on a `2700 kg` total.

**Masses are anchored, not fitted.** The recording does not constrain mass: three earlier fits landed at
`2685`, `2190` and `2822 kg` while all agreeing on `throttle_force / total_mass` to within 1 %, because
below the speed where the drive cap binds only the ratio is observable. Rather than let the optimiser
slide along that unidentifiable direction, the total is fixed at the vehicle proto's calibrated `2700 kg`
and the sprung mass follows from it. The axle split is likewise unidentified — three free-split fits gave
front:rear unsprung ratios of `1.60`, `0.97` and `3.82`, the last with a 10 kg rear wheel — so one
unsprung mass is shared by both axles.

That leaves one assumption worth stating: the shared unsprung mass settles at `49.1 kg per axle`, i.e.
**24.5 kg per wheel**, which is light for a 20-inch SUV wheel and tyre (typically 35–45 kg). It sits at
36 % of its box, so it is not a boundary artefact, and it has no meaningful effect on longitudinal
prediction — but it is assumed rather than measured.

The passive resistance in `halfcar_model.py` is one speed-proportional term, faded to zero at
standstill so it can never push the car backwards:

    resistance = resistance * |v| * tanh(v / resistance_smoothing)

It is measured rather than assumed: with both pedals at zero the recorded deceleration is a least-squares
fit of `F = -141 v` over 11 591 samples. `analysis/live/verify_webots_laws.jl` repeats that comparison
algebraically, sample by sample, without integrating anything.

The drive is capped as well as rpm-limited, because the vehicle library does not let the wheels spin up
faster than `ACCELERATION_THRESHOLD 50 rad/s²` (`driver.c:52`):

    drive = smooth_min(throttle * throttle_force * torque_ratio * redline,
                       traction_acceleration * total_mass)

`smooth_min` rather than `min` so the switching point stays differentiable.

The brake law is the constant torque the Webots vehicle library actually applies, faded to zero at
standstill:

    brake_force = 4 * brake_coefficient * brake_scale * brake / wheel_radius * tanh(v / brake_smoothing)

`brake_smoothing` and `resistance_smoothing` are both read from the parameter file, so no smoothing
constant is duplicated in Python.

## How the panel is drawn

**Two speeds, deliberately.** `car.getCurrentSpeed()` is the mean of the four wheel rotation rates
times the radius (`Webots driver.c`), so it is what the *schedule* is advanced on — the phases then stay
identical to the recorded run. The panel, the log and the model all use the **body** speed instead:
`node.getVelocity()` projected on the car's forward axis. The two differ sharply whenever the wheels
lock; during a 50 % brake pulse at 63 km/h the wheel-based number reports the equivalent of 9 g, which
is what makes the plotted trace collapse and recover while the pedal is steady. `hud_samples.csv`
therefore carries `speed_body_kmh`, matching the GPS ground speed the model is fitted against.

The HUD is a `Display` device textured onto a plane, which the controller parks one metre in front of
the camera on every step. At that distance a 0.8284 m x 0.4660 m plane spans the 45 degree frustum
exactly, so the 1280x720 panel texture covers the frame with no margins.

The panel is cleared to **fully transparent** every frame (`setAlpha(0)` then a fill of the whole
canvas, which sets alpha without painting colour), so only the cards are opaque and the simulation
shows through everywhere else. The cards use alpha 0.9, so the sky and the grass bank tint them
slightly and they do not look pasted on.

That clear is load-bearing rather than cosmetic: Webots creates the Display's texture as an opaque
one, and only recreates it as a blended one once a sub-opaque alpha is used.

The pedal strip is drawn on the same six-second window and the same horizontal scale as the two
plots, so a throttle step lines up vertically with the heave and pitch it causes. Throttle grows up
from the centre line, brake down.

## Things that are easy to get wrong

* **A `Display` with a `Shape` child is shown twice by Webots.** It is textured onto the shape *and*
  also drawn as a viewport overlay in the top-left of the 3D view, with a cyan border, at its own
  scale. Both end up in the movie, so the panel appears as a ghost under the scene copy. Hide it per
  world in the perspective file — `renderingDevicePerspectives: rrs:hud;0;…` in
  `worlds/.highway_overtake_postopt.wbproj`, the `0` being the visible flag — or set
  `HKCU\Software\Cyberbotics\Webots-R2025a\View3d\hideAllDisplayOverlays` to `true`. The perspective
  file is per user and is easy to lose, so check a recorded frame before believing the panel is
  clean.

  That file **is committed here with the flag already at `0`**, which is why the delivered movie has a
  single clean panel. Webots rewrites it on every run and will reset the flag to `1` if the file is
  missing, so if a recording suddenly shows a duplicated panel, check that line first.
* **The board hangs off the car but must be still in the world.** A `Display` is a robot device, so it
  can only live inside the vehicle's node tree, and it inherits the body's pitch and roll. The
  controller converts the pose it wants into the body's frame each step. Two consequences that are
  easy to miss: the body's frame origin is not where `getPosition()` reports the vehicle, which the
  placement absorbs by correcting from the previous frame's actual position; and the body moves one
  step ahead of the world-anchored camera before the frame is drawn, so the placement aims a step
  back along the motion. Without that second term the panel slides sideways in proportion to speed —
  about 120 px at 40 km/h.
* **The camera frame is not what the documentation implies.** A Webots camera looks along its local
  **+X** with local **+Z** up, so `orientation 0 0 0 1` looks down +X with a level horizon, and the
  side view here is `0 0 -1 1.5707963267948966`. `docs/reference/viewpoint.md` describes +x as "to
  the right", which reads as a different convention.
* **`drawText`'s y is the top of the text, not its baseline**, again despite `display.md`. Measured
  on a rendered frame: the ink lands 2–3 px below the requested y regardless of size. Anything that
  assumes a baseline offset pushes every label down by about 0.7 em.
* **The texture orientation is a half turn away from the obvious one.** The plane mapping that reads
  correctly for a camera looking along -Y is x→-X, y→+Z, z→+Y. With the natural x→+X, y→+Z, z→+Y the
  text is mirrored, and if the plane's normal faces away the whole board disappears.
* **Window aspect and recording aspect differ.** The preview window is not 16:9, so the board that
  spans a 1920x1080 recording is slightly cropped or slightly small on screen. Judge the framing from
  a recorded frame, not from `exportImage` on a window.
* **The schedule amplitudes are chosen to keep all four tyres planted.** The fitted model was
  identified on data where the wheels never leave the road; full throttle from rest and full braking
  at speed lift an axle in Webots (`coulombFriction 8`, soft springs), and no half-car model of this
  form can represent that. `make_schedule.py` documents the numbers.
* **One font family, four sizes.** `WbDisplayFont::loadFace` on Windows appends the whole TrueType
  file to a list that is never freed, once per `setFont` call, so text draws are queued and flushed
  sorted by size and the size changes at most four times per frame.

## Output

| file | what |
| --- | --- |
| `media/postopt_hud.mp4` | 1280x720, 25 fps, 23.08 s, H.264 High / yuv420p bt709, ~11 MB |
| `media/postopt_hud.gif` | 800x450, 10 fps, 9 s loop of the pulse train, ~16 MB |
| `media/poster_t*.png` | 720p stills, plus `_350px.png` legibility previews |
| `media/hud_samples.csv` | every sample the panel drew from, with the pedal inputs, one row per step |

`postopt_raw.mp4` is not committed: it is the uncompressed capture Webots writes during a run, and the
delivered files are derived from it by re-encoding.

Poster times land on the interesting transients: `06.00` the first die-down, `14.40` a throttle
pulse, `21.00` the final die-down. The gif covers `12–21 s`, which spans the pulse train.

`ffmpeg` is not on `PATH` on this machine, but Webots ships one. It needs the MinGW runtime alongside
it, so both directories have to be added before it will start:

```powershell
$bin = "C:\Program Files\Webots\msys64\mingw64\bin"
$env:PATH = "$bin\cpp;$bin;" + $env:PATH   # cpp\ holds libwinpthread-1.dll
```

**`hud_samples.csv` is the only record of which model was drawn.** The panel's `source_note` and the
three `fit_*` values are passed to `Hud` and stored, but nothing in `hud.py` actually draws them, so the
frame itself carries no model identifier. Line 1 of the CSV (`# label,...`) plus the controller's
`postopt_viewer: parameters ... (label ...)` console line are what prove the movie is the model you
think it is.

## Two Webots defects this pipeline works around

Both were found by measuring the process, and both are commented at the constant they set.

* **The display image is re-uploaded, and a copy leaked, on every panel refresh** — about 70 MB per
  refresh, and it does *not* scale down with the canvas, so a 25 Hz refresh at 1080p exhausts memory
  part-way through a take. Two things follow. The canvas is 1280x720 rather than 1920x1080: the
  layout is authored in a 1920x1080 design space and scaled by `SCALE`, which keeps the panel at the
  full frame size for a quarter of the pixels. And the refresh rate is the full 25 Hz, because
  anything less makes the traces tick — a 8 Hz panel is frozen for three frames and then jumps
  16 px, which is what "jumpy" looks like. Roughly 40 GB peaks during a take; a smaller panel would
  not help, so budget for it.
* **The movie encoder aborts with an access violation on a full-length 1920x1080 take.** Short takes
  are fine and 1280x720 is fine, so `MOVIE_WIDTH`/`MOVIE_HEIGHT` are 720p.

A third, unrelated trap: `Node.getVelocity()` has no matching `enableVelocityTracking()` in the
Python API, so it is a supervisor round trip per call; calling it at 200 Hz was what pushed the
run over the edge. The controller takes the body speed from a finite difference of the pose it
already tracks.
