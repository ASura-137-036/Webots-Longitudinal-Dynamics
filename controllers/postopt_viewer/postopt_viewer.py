#!/usr/bin/env python3
"""Drive the fitted half-car model live beside Webots and record the result.

The vehicle is driven by the open-loop throttle/brake schedule in
``schedule_postopt.json``.  On every physics step the half-car model from
``params_opt_v2.34.jl`` is integrated forward from the measured state, and both the
measurement and the prediction are drawn onto a Webots `Display` that is textured onto a
plane one metre in front of the camera.  Because the plane is a real `Shape` in the scene it is
captured by `movieStartRecording`, so the HUD ends up *inside* the movie instead of being
composited afterwards.

This controller only reads the simulation; it never writes to the log produced by
``longdyn_logger``, and it does not touch that file.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

# Sibling modules live next to this file, but Webots starts the controller with an
# arbitrary working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vehicle import Car

import halfcar_model as hc
from hud import Hud

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAMS_FILE = os.path.join(HERE, "params_opt_v2.json")
# The reference fit is v2.34 (the `label` field in the JSON and `analysis/params_opt_v2.34.jl`): a drive
# limited by the motor curve and by the library's wheel-spin guard, a constant-torque brake, and a single
# speed-proportional resistance, with the vehicle total mass anchored to the proto's 2700 kg and one
# unsprung mass shared by both axles. Objective 8.80 on the training run and 7.65 held out, with speed
# 1.91 km/h and heave 1.6 mm; twenty-one estimated parameters, none near a bound. Set POSTOPT_PARAMS to a
# file name in this directory (or an absolute path) to overlay a different fitted parameter set without
# moving the reference JSON.
SCHEDULE_FILE = os.path.join(HERE, "schedule_postopt.json")
OUTPUT_DIRECTORY = os.path.abspath(os.path.join(HERE, "..", "..", "media"))
MOVIE_FILE = os.path.join(OUTPUT_DIRECTORY, "postopt_raw.mp4")
HUD_SAMPLES_FILE = os.path.join(OUTPUT_DIRECTORY, "hud_samples.csv")

CONTROL_PERIOD_MS = 5
TARGET_GEAR = 1

# Samples of measurement to collect after the wheels regain contact before the model is
# restarted, matching `halfcar_pad_indices` in the notebooks.
PAD_INDICES = 2
# The vehicle is spawned above its ride height and drops onto the springs during the first
# second, so the plot's zero is latched here rather than at the spawn height.
ARM_TIME = 2.0  # [s] from the start of the run
MAX_RUN_TIME = float(os.environ.get("POSTOPT_MAX_TIME", "240"))  # [s] hard stop
# Webots' encoder aborts with an access violation on a full-length 1920x1080 take in this build
# (short takes are fine, and 1280x720 is fine), so the movie is 720p. The panel itself is still
# rendered at 1920x1080 and downsampled by the projection, so its text stays crisp.
MOVIE_WIDTH = 1280
MOVIE_HEIGHT = 720
MOVIE_QUALITY = 90
MOVIE_WAIT_S = 180.0  # [s] wall clock allowed for the encoder to flush the movie

# --- Camera framing -------------------------------------------------------
# A side-on tracking shot at 12 m, the height of the car's roof, with the default 45 degree
# field of view: the car covers half the frame width and sits on the horizon line, and the
# sky and the grass bank above it are left clear for the two plot cards.
CAMERA_DISTANCE = 12.0     # [m] lateral offset, on the +Y side of the car
CAMERA_FIELD_OF_VIEW = 0.7853981634  # [rad] horizontal, Webots' default 45 degrees
CAMERA_HEIGHT = 1.85       # [m] above the road: the roofline lands on the horizon
CAMERA_SIDE = +1           # +1: camera on +Y, so the car's nose reads to the right

# The panel is a Display textured onto a plane one metre in front of the camera, where the
# plane spans the frustum exactly: 2 * tan(22.5 deg) = 0.8284271 m wide, and 0.46599 m tall
# for a 16:9 image. Everything the panel does not paint stays transparent, so the scene shows
# through between the cards. The board is a child of the car - that is the only way the
# controller can own the Display device - so its pose is written in the car's frame each step
# to hold it still in the world while the body pitches over it.
BOARD_DISTANCE = 1.0
BOARD_SIZE = (0.8284271, 0.46599)
# The board's world axes as a row-major 3x3, mapping its local x, y and z onto world
# (-X, +Z, +Y): the image's x runs to screen right, its y up the screen, and the texture
# normal back towards the camera.
BOARD_BASIS = (-1.0, 0.0, 0.0,
               0.0, 0.0, 1.0,
               0.0, 1.0, 0.0)

# --- Contact classification ----------------------------------------------
CONTACT_HEIGHT_MAX = 0.20   # [m] a contact point this low is on the ground plane
CONTACT_SIDE_MIN = 0.45     # [m] |lateral| window that excludes chassis contact points
CONTACT_SIDE_MAX = 1.30
CONTACT_AXLE_TOLERANCE = 0.55  # [m] |longitudinal| window around an axle
CALIBRATION_SAMPLES = 40       # contact points collected before the axles are known
DEBUG_CONTACT = os.environ.get("POSTOPT_DEBUG_CONTACT", "0") != "0"


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _matmul(a, b):
    """Row-major 3x3 product."""
    return tuple(sum(a[row * 3 + k] * b[k * 3 + col] for k in range(3))
                 for row in range(3) for col in range(3))


def _transpose(m):
    return tuple(m[col * 3 + row] for row in range(3) for col in range(3))


def _apply(m, v):
    return tuple(m[row * 3] * v[0] + m[row * 3 + 1] * v[1] + m[row * 3 + 2] * v[2]
                 for row in range(3))


def matrix_to_axis_angle(m):
    """Convert a row-major 3x3 rotation to the `x y z angle` form Webots stores."""
    cosine = max(-1.0, min(1.0, (m[0] + m[4] + m[8] - 1.0) * 0.5))
    angle = math.acos(cosine)
    if angle < 1e-6:
        return [0.0, 0.0, 1.0, 0.0]
    if math.pi - angle < 1e-4:
        # Half turns are the common case here, and the antisymmetric part vanishes exactly
        # at half a turn, so the axis comes out of the symmetric part instead.
        axis = [math.sqrt(max(0.0, (m[0] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (m[4] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (m[8] + 1.0) * 0.5))]
        primary = max(range(3), key=lambda index: axis[index])
        if axis[primary] > 1e-9:
            others = [index for index in range(3) if index != primary]
            for index in others:
                # R_ij = 2 a_i a_j, so each remaining component follows from the pair
                axis[index] = (m[primary * 3 + index] + m[index * 3 + primary]) / \
                              (4.0 * axis[primary])
        return axis + [angle]
    scale = 1.0 / (2.0 * math.sin(angle))
    return [(m[7] - m[5]) * scale, (m[2] - m[6]) * scale, (m[3] - m[1]) * scale, angle]


def place_board(board, car_node, world_position, world_basis, motion=(0.0, 0.0, 0.0)):
    """Put the board at a world pose, expressed in the frame of the body it hangs from.

    The board is a child of the vehicle, so it can only be moved in the body's frame, and the
    body's frame origin is not the position the API reports for the vehicle. Rather than
    assume a frame, the placement is corrected from where the board actually is: the residual
    is measured in world metres and converted back through the body rotation, so it converges
    in a single step and stays exact as the body pitches.

    `motion` is how far the body will travel before the next frame is drawn. The camera is set
    in world coordinates and does not travel with it, so without this the panel slides across
    the frame in proportion to speed.
    """
    inverse = _transpose(car_node.getOrientation())  # row-major, world from body
    current = board.getPosition()
    delta = _apply(inverse, (world_position[0] - current[0] - motion[0],
                             world_position[1] - current[1] - motion[1],
                             world_position[2] - current[2] - motion[2]))
    translation = board.getField("translation").getSFVec3f()
    board.getField("translation").setSFVec3f(
        [translation[index] + delta[index] for index in range(3)])
    board.getField("rotation").setSFRotation(
        matrix_to_axis_angle(_matmul(inverse, world_basis)))


# ---------------------------------------------------------------------------
# Open-loop schedule
# ---------------------------------------------------------------------------


class Schedule:
    """Sequential throttle/brake phases, with an optional speed-based early exit."""

    def __init__(self, phases, start_time):
        self.phases = phases
        self.index = 0
        self.start = start_time
        self.commands = (0.0, 0.0)
        self.finished = False

    @property
    def name(self):
        return "done" if self.finished else self.phases[self.index]["name"]

    def update(self, t, speed_kmh):
        while not self.finished:
            phase = self.phases[self.index]
            elapsed = t - self.start
            done = elapsed >= float(phase["duration"])
            limit = phase.get("end_speed_below")
            # The guard on `elapsed` keeps a phase that starts already below the limit from
            # being skipped in the same step it was entered.
            if limit is not None and elapsed > 0.05 and speed_kmh <= float(limit):
                done = True
            if not done:
                self.commands = (float(phase.get("throttle", 0.0)),
                                 float(phase.get("brake", 0.0)))
                return self.commands
            self.start = t
            self.index += 1
            if self.index >= len(self.phases):
                self.finished = True
                self.commands = (0.0, 0.0)
                return self.commands
        return self.commands


# ---------------------------------------------------------------------------
# The model, restarted on every stretch of continuous ground contact
# ---------------------------------------------------------------------------


class SegmentModel:
    """Integrates the half-car model across one contact segment.

    The state is re-seeded from the measurement whenever ground contact is regained, which
    is exactly the discipline the fit was trained with: the model is only ever asked to
    predict across a stretch in which all four wheels stay on the road.
    """

    def __init__(self, params, dt, start_time, pad_indices=PAD_INDICES):
        self.params = params
        self.dt = dt
        self.start_time = start_time
        self.pad_indices = pad_indices
        self.state = None
        self.pending = 0
        self.throttle = 0.0
        self.brake = 0.0
        # Latched once the body has settled so both traces start from zero together.
        self.measured_heave_zero = None
        self.model_heave_zero = None
        self.pitch_zero = None

    @property
    def active(self):
        return self.state is not None

    def _arm(self, t, measured):
        """Latch the display zero, but not before the body has settled onto its springs.

        The vehicle is spawned above its ride height and drops onto the suspension during the
        first second, so zeroing on the spawn position would leave the whole trace sitting on
        a ~140 mm offset instead of showing the transient.
        """
        if self.measured_heave_zero is not None or self.state is None:
            return
        if t - self.start_time < ARM_TIME:
            return
        self.measured_heave_zero = measured["com_z"]
        self.model_heave_zero = hc.com_height(self.state, self.params)
        self.pitch_zero = measured["pitch"]

    def observe(self, contact, measured, commands, t):
        """Advance one control step and return the new state, or None while inactive."""
        if contact == "all":
            self.state = None
            self.pending = 0
            self._remember(commands)
            return None

        if self.state is None:
            self.pending += 1
            if self.pending < self.pad_indices:
                self._remember(commands)
                return None
            self.state = hc.initial_state(self.params, measured["com_z"], measured["pitch"],
                                          measured["position"], measured["speed"])
            self._remember(commands)
            self._arm(t, measured)
            return None

        # The pedals of the interval that just elapsed drive the state to the current time.
        self.state = hc.advance(self.state, self.params, self.throttle, self.brake, self.dt)
        self._remember(commands)
        self._arm(t, measured)
        return self.state

    def _remember(self, commands):
        self.throttle, self.brake = commands

    def heave_mm(self, state):
        if state is None or self.model_heave_zero is None:
            return None
        return (hc.com_height(state, self.params) - self.model_heave_zero) * 1000.0

    def pitch_deg(self, state):
        if state is None or self.pitch_zero is None:
            return None
        return math.degrees(state[1] - self.pitch_zero)

    def measured_heave_mm(self, com_z):
        # None before the display zero is armed: a trace that has not started yet must not be
        # drawn as a flat line at zero, which would read as data.
        if self.measured_heave_zero is None:
            return None
        return (com_z - self.measured_heave_zero) * 1000.0

    def measured_pitch_deg(self, pitch):
        if self.pitch_zero is None:
            return None
        return math.degrees(pitch - self.pitch_zero)


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def enable_device(robot, name, period):
    device = robot.getDevice(name)
    if device is None:
        raise RuntimeError("the vehicle has no device named '%s'" % name)
    device.enable(period)
    return device


class ContactReader:
    """Classify the axles as on or off the ground: 'none', 'front', 'rear' or 'all'.

    Contact points are classified geometrically rather than by the id of the Solid they
    belong to: a point sitting on the ground plane, off the centre line and near an axle can
    only be a tyre contact patch, which also keeps chassis and road contacts out.

    The two axle positions are *learnt* from the first moments of the run instead of being
    assumed. The half-car model measures them from the centre of gravity, whereas contact
    points are reported in the vehicle node's frame, and the two frames differ by an offset
    that depends on the PROTO.
    """

    def __init__(self, params):
        self.params = params
        self.front_x = None
        self.rear_x = None
        self._samples = []

    @property
    def calibrated(self):
        return self.front_x is not None

    def classify(self, node):
        position = node.getPosition()
        # `getOrientation` returns the rows of the local-to-world rotation matrix.
        r = node.getOrientation()
        front = rear = False
        grounded = []
        for contact_point in node.getContactPoints(True):
            point = contact_point.getPoint()
            if point[2] > CONTACT_HEIGHT_MAX:
                continue
            dx = point[0] - position[0]
            dy = point[1] - position[1]
            dz = point[2] - position[2]
            local_y = r[1] * dx + r[4] * dy + r[7] * dz
            if not (CONTACT_SIDE_MIN <= abs(local_y) <= CONTACT_SIDE_MAX):
                continue
            local_x = r[0] * dx + r[3] * dy + r[6] * dz
            grounded.append(local_x)
            if self.front_x is None:
                continue
            if abs(local_x - self.front_x) < CONTACT_AXLE_TOLERANCE:
                front = True
            elif abs(local_x - self.rear_x) < CONTACT_AXLE_TOLERANCE:
                rear = True

        if self.front_x is None:
            self._samples.extend(grounded)
            if len(self._samples) >= CALIBRATION_SAMPLES:
                self.rear_x = min(self._samples)
                self.front_x = max(self._samples)
                wheelbase = self.front_x - self.rear_x
                expected = float(self.params["wheelbase"])
                print("postopt_viewer: axles calibrated at x=%.3f and x=%.3f m "
                      "(wheelbase %.3f m, model %.3f m)"
                      % (self.rear_x, self.front_x, wheelbase, expected), flush=True)
                if abs(wheelbase - expected) > CONTACT_AXLE_TOLERANCE:
                    sys.stderr.write("postopt_viewer: learnt wheelbase %.3f m disagrees with "
                                     "the model's %.3f m\n" % (wheelbase, expected))
                self._samples = []
            return "none"

        if DEBUG_CONTACT:
            print("postopt_viewer: contacts front=%s rear=%s" % (front, rear), flush=True)
        if front and rear:
            return "none"
        if front:
            return "rear"
        if rear:
            return "front"
        return "all"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def schedule_path():
    """The schedule, as the first controller argument or the POSTOPT_SCHEDULE override."""
    for candidate in ((sys.argv[1] if len(sys.argv) > 1 else None),
                      os.environ.get("POSTOPT_SCHEDULE")):
        if not candidate:
            continue
        if not os.path.isabs(candidate) and not os.path.isfile(candidate):
            candidate = os.path.join(HERE, candidate)
        return candidate
    return SCHEDULE_FILE


def write_hud_samples(rows, payload, fit):
    """Write the panel's input stream, header first, so the clip can be reproduced or audited."""
    os.makedirs(os.path.dirname(HUD_SAMPLES_FILE), exist_ok=True)
    with open(HUD_SAMPLES_FILE, "w", newline="") as handle:
        handle.write("# label,%s\n" % payload.get("label", ""))
        handle.write("# log_file,%s\n" % payload.get("log_file", ""))
        handle.write("# heave_rmse_mm,%r\n" % float(fit.get("heave_rmse_mm", 0.0)))
        handle.write("# pitch_rmse_deg,%r\n" % float(fit.get("pitch_rmse_deg", 0.0)))
        handle.write("# speed_r2,%r\n" % float(fit.get("speed_r2", 0.0)))
        handle.write("# sample_count,%s\n" % fit.get("sample_count", 0))
        handle.write("# evaluations,%s\n" % fit.get("evaluations", 0))
        writer = csv.writer(handle)
        writer.writerow(["t_s", "speed_body_kmh", "heave_measured_mm", "pitch_measured_deg",
                         "heave_model_mm", "pitch_model_deg", "contact", "throttle",
                         "brake", "phase"])
        for row in rows:
            t, speed, heave, pitch, heave_model, pitch_model, contact, throttle, brake, \
                phase = row
            writer.writerow([repr(t), repr(speed),
                             "" if heave is None else repr(heave),
                             "" if pitch is None else repr(pitch),
                             "" if heave_model is None else repr(heave_model),
                             "" if pitch_model is None else repr(pitch_model),
                             contact, repr(throttle), repr(brake), phase])
    print("postopt_viewer: wrote %s (%d samples)" % (HUD_SAMPLES_FILE, len(rows)),
          flush=True)


def params_path():
    """The parameter file, as the POSTOPT_PARAMS override or the reference fit."""
    candidate = os.environ.get("POSTOPT_PARAMS")
    if not candidate:
        return PARAMS_FILE
    if not os.path.isabs(candidate) and not os.path.isfile(candidate):
        candidate = os.path.join(HERE, candidate)
    return candidate


def main():
    parameter_file = params_path()
    with open(parameter_file) as handle:
        payload = json.load(handle)
    params = hc.load_parameters(parameter_file)
    print("postopt_viewer: parameters %s (label %s)"
          % (os.path.basename(parameter_file), params["_label"]), flush=True)
    fit = payload.get("fit", {})
    with open(schedule_path()) as handle:
        phases = json.load(handle)

    car = Car()
    timestep = int(car.getBasicTimeStep())
    dt = CONTROL_PERIOD_MS / 1000.0
    steps_per_control = max(1, int(round(CONTROL_PERIOD_MS / float(timestep))))

    node = car.getSelf()
    if node is None:
        raise RuntimeError("the vehicle must be a supervisor: set 'supervisor TRUE'")
    node.enablePoseTracking(CONTROL_PERIOD_MS, None)
    node.enableContactPointsTracking(CONTROL_PERIOD_MS, True)

    devices = {"gps": enable_device(car, "gps", CONTROL_PERIOD_MS),
               "imu": enable_device(car, "imu", CONTROL_PERIOD_MS)}

    display = car.getDevice("hud") if os.environ.get("POSTOPT_HUD", "1") != "0" else None
    if display is None and os.environ.get("POSTOPT_HUD", "1") != "0":
        raise RuntimeError("the world has no Display device named 'hud'")
    board = car.getFromDef("POSTOPT_HUD")
    camera = car.getFromDef("POSTOPT_CAMERA")
    if board is None or camera is None:
        raise RuntimeError("the world needs DEF POSTOPT_HUD and DEF POSTOPT_CAMERA")

    # The camera is a side-on tracking shot. Webots' camera frame looks along its local +X with
    # local +Z up, so a -90 degree turn about Z looks along -Y with world +Z up: the horizon is
    # level and the car, which travels along -X, has its nose on the right of frame.
    camera.getField("orientation").setSFRotation([0.0, 0.0, -1.0, 1.5707963267948966])
    camera.getField("fieldOfView").setSFFloat(CAMERA_FIELD_OF_VIEW)

    hud = None if display is None else Hud(
        display,
        fit_heave_mm=float(fit.get("heave_rmse_mm", 0.0)),
        fit_pitch_deg=float(fit.get("pitch_rmse_deg", 0.0)),
        fit_speed_r2=float(fit.get("speed_r2", 0.0)),
        source_note="%s - fitted on %s (%s samples, %s BFGS evaluations)" % (
            payload.get("label", "half-car"),
            payload.get("log_file", "?"),
            format(int(fit.get("sample_count", 0)), ","),
            fit.get("evaluations", "?")))

    car.setThrottle(0.0)
    car.setBrakeIntensity(0.0)
    car.setSteeringAngle(0.0)
    car.setGear(TARGET_GEAR)

    if car.step() == -1:
        return 0

    # The camera tracks the car along the road at a fixed height, so the body's heave and pitch
    # stay visible against the road instead of being cancelled by the camera.
    camera.getField("position").setSFVec3f(
        [node.getPosition()[0], node.getPosition()[1] + CAMERA_SIDE * CAMERA_DISTANCE,
         CAMERA_HEIGHT])

    record = os.environ.get("POSTOPT_RECORD", "1") != "0"
    if record:
        os.makedirs(os.path.dirname(MOVIE_FILE), exist_ok=True)
        # The codec argument is ignored by Webots (only MPEG-4/AVC is available) and the
        # frame rate is derived from the basic time step so playback is real time.
        car.movieStartRecording(MOVIE_FILE, MOVIE_WIDTH, MOVIE_HEIGHT, 0, MOVIE_QUALITY,
                                1, False)
        # `movieIsReady` is only true while no recording is running, so there is nothing to
        # wait for here: recording starts on the next step.
        if car.movieFailed():
            sys.stderr.write("postopt_viewer: movie recording failed to start\n")
            record = False
        else:
            print("postopt_viewer: recording to %s" % MOVIE_FILE, flush=True)

    schedule = Schedule(phases, car.getTime())
    print("postopt_viewer: %d phases, nominal %.1f s"
          % (len(phases), sum(float(phase["duration"]) for phase in phases)), flush=True)

    # Debug aid: POSTOPT_SNAPSHOT_AT="3,8.5" writes a JPEG of the viewport at those times.
    snapshot_times = sorted(float(value) for value in
                            os.environ.get("POSTOPT_SNAPSHOT_AT", "").split(",")
                            if value.strip())
    snapshot_directory = os.path.dirname(MOVIE_FILE)
    sweep_distances = []
    orientations = [value for value in
                    os.environ.get("POSTOPT_SWEEP_ORIENT", "").split(",") if value.strip()]
    if orientations:
        distances = [float(value) for value in
                     os.environ.get("POSTOPT_SWEEP_DIST", "17.55").split(",") if value.strip()]
        for text in orientations:
            parts = [float(part) for part in text.replace(":", " ").split()]
            for distance in distances:
                sweep_distances.append((parts, distance))

    contacts = ContactReader(params)
    # Every sample that drives the panel is also written to disk, so the same video can be
    # rendered offline without depending on the 3D view being capturable.
    hud_rows = []

    model = SegmentModel(params, dt, car.getTime())
    reference = (node.getPosition()[0], node.getPosition()[1])
    previous_position = node.getPosition()
    travelled = 0.0
    previous_phase = None
    step_index = 1
    start_time = car.getTime()
    t = start_time

    while True:
        # Step first, then read: Webots publishes the contact points of a physics step to the
        # supervisor during that step, so they are only populated after `step()` returns.
        if car.step() == -1:
            break
        t = car.getTime()

        if steps_per_control == 1 or step_index % steps_per_control == 0:
            position = node.getPosition()
            delta = (position[0] - reference[0], position[1] - reference[1])
            travelled = delta[0] * -1.0  # the car drives along -X
            # Two speeds, because they answer different questions. `getCurrentSpeed()` is the mean of
            # the four wheel rotation rates times the radius (Webots `driver.c`), so a locked wheel
            # reads a collapsed speed - during a 50 % brake pulse at 63 km/h it reports the equivalent
            # of 9 g. It is still what the schedule is advanced on, so the phases stay identical to
            # the recorded training run. The panel and the model, however, need the body speed: the
            # half-car state is a velocity, and the recorded channel it is fitted against is the GPS
            # ground speed.
            speed_kmh = car.getCurrentSpeed()
            # Body ground speed from the tracked pose, not `getVelocity()`. The supervisor's
            # velocity query is a round trip to Webots on every control step, and at 200 Hz that
            # allocates faster than the process can reclaim - it is enough to exhaust memory over a
            # 23 s clip. The pose is already read here for the board placement, so a finite
            # difference of it costs nothing and is the same quantity once projected on the body's
            # forward axis.
            body_velocity = ((position[0] - previous_position[0]) / dt,
                             (position[1] - previous_position[1]) / dt,
                             0.0)
            forward = node.getOrientation()
            forward_axis = (forward[0], forward[3], forward[6])
            speed_body_kmh = 3.6 * (forward_axis[0] * body_velocity[0] +
                                    forward_axis[1] * body_velocity[1] +
                                    forward_axis[2] * body_velocity[2])
            pitch = devices["imu"].getRollPitchYaw()[1]
            com_z = node.getCenterOfMass()[2]
            contact = contacts.classify(node)

            throttle, brake = schedule.update(t, speed_kmh)
            if schedule.name != previous_phase:
                print("postopt_viewer: t=%6.2f s  phase %s  throttle %.2f brake %.2f"
                      % (t, schedule.name, throttle, brake), flush=True)
                previous_phase = schedule.name
            if schedule.finished:
                break
            car.setThrottle(throttle)
            car.setBrakeIntensity(brake)

            measured = {"com_z": com_z, "pitch": pitch, "position": travelled,
                        "speed": speed_body_kmh / 3.6}
            state = model.observe(contact, measured, (throttle, brake), t)

            heave_measured = model.measured_heave_mm(com_z)
            pitch_measured = model.measured_pitch_deg(pitch)
            heave_model = model.heave_mm(state)
            pitch_model = model.pitch_deg(state)
            hud_rows.append((t, speed_body_kmh, heave_measured, pitch_measured, heave_model,
                             pitch_model, contact, throttle, brake, schedule.name))

            if hud is not None:
                hud.push(t, speed_body_kmh, heave_measured, pitch_measured, heave_model,
                         pitch_model, contact, throttle, brake)
                hud.maybe_draw(t, schedule.name)

            car_position = node.getPosition()
            motion = tuple(car_position[index] - previous_position[index] for index in range(3))
            previous_position = car_position
            camera_position = (car_position[0],
                               car_position[1] + CAMERA_SIDE * CAMERA_DISTANCE,
                               CAMERA_HEIGHT)
            camera.getField("position").setSFVec3f(list(camera_position))
            # The board sits one metre down the camera's line of sight, so it spans the frame.
            place_board(board, node,
                        (camera_position[0], camera_position[1] - BOARD_DISTANCE,
                         camera_position[2]),
                        BOARD_BASIS, motion)

            while snapshot_times and t - start_time >= snapshot_times[0]:
                target = snapshot_times.pop(0)
                path = os.path.join(snapshot_directory, "snapshot_t%05.2f.jpg" % target)
                car.exportImage(path, 92)
                print("postopt_viewer: snapshot %s" % path, flush=True)

            if sweep_distances and t - start_time >= 1.0:
                for index, (orientation, distance) in enumerate(sweep_distances):
                    camera.getField("orientation").setSFRotation(list(orientation))
                    camera.getField("position").setSFVec3f(
                        [board_position[0],
                         board_position[1] - CAMERA_SIDE * distance,
                         camera_z])
                    for _ in range(3):
                        car.step()
                    path = os.path.join(snapshot_directory, "sweep_%02d.jpg" % index)
                    car.exportImage(path, 92)
                    print("postopt_viewer: sweep %s %s dist %.2f -> %s"
                          % (index, orientation, distance, path), flush=True)
                sweep_distances = []

            if t - start_time > MAX_RUN_TIME:
                print("postopt_viewer: maximum run time reached", flush=True)
                break
            step_index += 1

    if hud is not None:
        hud.draw(car.getTime(), "done")
    write_hud_samples(hud_rows, payload, fit)
    car.setThrottle(0.0)
    car.setBrakeIntensity(1.0)

    if record:
        car.movieStopRecording()
        deadline = time.time() + MOVIE_WAIT_S
        while not car.movieIsReady() and time.time() < deadline:
            if car.movieFailed():
                sys.stderr.write("postopt_viewer: movie encoding failed\n")
                break
            if car.step() == -1:
                break
        print("postopt_viewer: movie %s"
              % ("saved" if not car.movieFailed() else "FAILED"), flush=True)

    # Quit from inside the controller: the run must not linger as a background process.
    car.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
