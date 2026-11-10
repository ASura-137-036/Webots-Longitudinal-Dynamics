"""Longitudinal dynamics data logger for the Range Rover Sport SVR.

World
-----
``worlds/highway_overtake_longdyn.wbt``: a straight 5 km highway with a single
physics-enabled ``RangeRoverSportSVR``.  The vehicle *must* have ``supervisor TRUE``
because the logger reads the pose, the centre of mass and the tyre contact points
from the scene tree.

What it does
------------
1. Applies an open-loop throttle/brake schedule (``SCHEDULE`` below, or a JSON file
   given as the first controller argument: ``controllerArgs [ "my_run.json" ]``).
2. Applies a minimal lateral controller so the car holds the straight line it started
   on.  It only exists to cancel the small lateral disturbances (tyre asymmetry,
   steering offsets) that would otherwise add lateral acceleration to the data.
3. Streams the complete vehicle state to ``logs/run_<timestamp>.csv`` (one row every
   ``CONTROL_PERIOD_MS`` = 5 ms of simulation time, i.e. 200 Hz; the log grid is
   independent of the world time step, so the rows stay 5 ms apart if the step is
   refined further).
4. Prints the whole schedule to the Webots console before it starts, then one line each
   time a phase is entered (with its real start time and the number of samples logged
   so far), so the running schedule is visible live in the console.

Schedule format
---------------
A schedule is a list of phases.  Keys (all optional except ``name``)::

    name             label written in the log
    duration         maximum phase duration [s]        (default: infinite)
    throttle         throttle target at the end of the phase, 0..1   (default 0)
    brake            brake intensity target, 0..1                    (default 0)
    ramp             time to go linearly from the previous value to the target [s]
                     (default 0 = step input)
    end_speed_below  end the phase early when the speed drops below this value [km/h]
    end_speed_above  end the phase early when the speed rises above this value [km/h]
    block            group label written in the CSV 'block' column
    type             "step" (default) ramps the throttle/brake to the targets above,
                     the other types drive ONE channel themselves; the other pedal is
                     kept at zero, so throttle and brake are never applied together:
                     "sine"      channel sine = mean + amp*sin(2*pi*f*t_phase),
                                 clamped at zero (used for both throttle and brake)
                     "pulse"     channel = amp*sin(pi*t_phase/duration), 0 -> amp -> 0
                     "square"    channel square wave 0 <-> amp at f Hz
                     "regulate"  closed loop speed hold, used only to place the car at
                                 the right speed before a measured event; labelled with
                                 block "reg" so it can be filtered out of the analysis

Schedule macros (expanded by expand_phases() before the run):
    {"type": "sweep", ...}             one pulse per level, no reset in between
        channel throttle|brake, prefix, levels [], hold, gap, block
    {"type": "regulated_sweep", ...}   the same, but every level starts from the same
                                       speed: reset = {"mode": "hold", "speed_kmh":,
                                       "band":, "duration":}
    {"type": "pulse_train", ...}       count pulses of one level separated by gap
        channel, level, pulse, gap, count, prefix, block
    {"type": "alternate", ...}         alternating throttle and brake pulses
        throttle_level, brake_level, pulse, gap, count, prefix, block
    {"type": "ramp_event", ...}        ramp up, hold, die down on one channel
        channel, amplitude, attack, hold, release, end_speed_above, name, block
    {"type": "square_alternate", ...}  throttle for t, then brake for t, NO gap:
        t, count, throttle_level, brake_level, prefix, block

Inputs of the non-"step" types:
    channel          throttle | brake (sine, pulse, square)
    duration         required, the waveform period is derived from it
    amplitude        waveform height 0..1            (pulse, square)
    mean, amplitude, frequency                       (sine)
    t                seconds per half cycle           (square_alternate)

Overlap invariant: if a phase ever asks for throttle and brake at the same time (a
`ramp`/`step` phase can, since it interpolates both), Schedule.update() drops the weaker
command and counts the event; the count is printed in the run summary and must be 0.
validate_phases() additionally reports such phases before the run starts.

The pitch of the car is the response of the suspension, not of the input: an event
shorter than about half a second barely moves the accelerometer (the body has not
started to rotate yet), and the peak pitch lags the input step by the quarter period
of the pitch mode (~0.5 s with this model).  Events of 0.4-1.0 s are the useful range.

Known limitation of the vehicle model
-------------------------------------
The tyre contacts behave like a strong static friction when the vehicle stands still:
a partial throttle applied from rest only makes it creep, whatever the contact
parameters.  Roughly 2500 N of drive force are needed to break away, which is more than
the engine delivers below ~75 % throttle.  The default ``SCHEDULE`` therefore starts
with a short full throttle ``launch`` phase; afterwards every throttle level produces
the acceleration predicted by the engine model.  Keep a launch phase in your own
schedules, or start them while the car is already rolling.

Log channels
------------
The ``#`` comment header of the produced CSV documents the model constants and the
sensor locations.  In short:

* ``t_s``, ``phase``, ``t_phase_s``            - time base and applied input phase
* ``throttle_cmd``/``brake_cmd``               - inputs sent to the vehicle
* ``engine_torque_nm``/``wheel_torque_total_nm`` - drive torque of the engine model
* ``speed_kmh``/``speed_ms``                   - ``Driver.getCurrentSpeed()`` (wheel based)
* ``speed_ground_ms``/``speed_long_ms``        - GPS ground speed (total / along the car)
* ``s_travelled_m``                            - distance along the initial heading
* ``engine_rpm``, ``gear``, ``control_mode``   - drivetrain state
* ``accel_*``, ``gyro_*``                      - IMU at the chassis centre of mass
* ``roll_rad``, ``pitch_rad``, ``yaw_rad``     - InertialUnit
* ``gps_*``, ``pos_*``, ``rot**``, ``fwd_*``   - world pose of the vehicle
* ``com_*``                                    - world centre of mass of the whole car
* ``lat_error``, ``head_error``                - deviation from the initial straight line
* ``body_min_*``/``body_max_*``                - world axis-aligned bounding box
* ``obb_lower_*``/``obb_central_*``            - world corners of the two collision boxes
* ``wheel_*``                                  - per wheel: encoder, speed, slip ratio,
                                                 contact flag, contact count and contact
                                                 height (a flag of 0 means the wheel is
                                                 off the ground)

Webots conventions used here
----------------------------
* ``setSteeringAngle()`` is in radians, **positive steers right**.
* ``setThrottle()`` selects torque control; ``setBrakeIntensity()`` adds a damping
  proportional to the wheel speed (it is not a constant braking torque).
* ``getCurrentSpeed()`` is in km/h.
* ``Accelerometer`` values include gravity (a car at rest reads +9.81 m/s^2 on z).
"""

import csv
import json
import math
import os
import re
import sys
import time as wall_clock

import numpy as np

from vehicle import Car # type: ignore

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

LOG_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
EXIT_ON_FINISH = True      # quit the simulation once the manoeuvre is over

# Sampling period of the log [ms]; it is independent of the world time step, so the CSV
# row spacing stays the same if the physics step changes.  One row is written every
# CONTROL_PERIOD_MS of simulation time (0.005 s = 200 Hz).
CONTROL_PERIOD_MS = 5
FLUSH_INTERVAL = 50       # number of steps between two writes to the disk

MAX_RUN_TIME = 900.0      # [s] hard limit for the whole manoeuvre
MAX_LATERAL_ERROR = 2.5   # [m] abort the run if the car leaves its lane
ROAD_END_X = -9800.0      # [m] abort the run before the end of the road (two 5 km segments)

LATERAL_CONTROLLER = {
    "heading_gain": 0.80,          # [rad/rad]
    "cross_track_gain": 1.50,      # [1/s]  (Stanley style, divided by the speed)
    "yaw_rate_gain": 0.06,         # [s]
    "max_steering_angle": 0.35,    # [rad]
    "deadband_lateral": 0.01,      # [m]    do not steer inside the deadband
    "deadband_heading": 0.0015,    # [rad]
    "filter_time_constant": 0.06,  # [s]    first order smoothing of the command
    "min_speed": 2.0,              # [m/s]  floor used to scale the cross-track term
}

# The library does not shift gears by itself: gear 0 is neutral and disengaged, gear 1
# and above are the forward gears of the Car PROTO 'gearRatio' array, -1 is reverse.
# A fixed gear keeps the run repeatable (no shift transient in the data).
TARGET_GEAR = 1

# Throttle / brake steps separated by coast phases (drag identification), followed by
# brake steps and a full stop.  Edit freely or pass a JSON file as controller argument.
#
# The first "launch" phase at full throttle is required: in this simulation the tyre
# contact holds the wheels like a static friction when the car stands still, so a
# partial throttle applied from rest only makes the vehicle creep (it needs roughly
# 2500 N of drive force to break away).  Once the car is rolling every throttle level
# gives the acceleration expected from the engine model, hence all identification
# phases come after the launch.
SCHEDULE = [
    {"name": "settle",     "duration": 2.0, "throttle": 0.00, "brake": 0.00},
    # Breakaway only. A partial throttle from rest just creeps, because the tyres hold the car like
    # static friction, so one short full-throttle phase gets it rolling; it is kept short (0.6 s)
    # so every ramp below still starts from near standstill instead of from the redline.
    {"name": "launch",     "duration": 0.6, "throttle": 1.00, "brake": 0.00},

    # Ramp-up / die-down pairs.  Each accelerates from near rest at a fixed throttle, brakes to a
    # standstill, and holds the brake so it stays there.  Full throttle gets 3 s and the lesser
    # levels progressively longer, because a smaller force needs more time to reach the plateau.
    # `end_speed_below` ends a die-down at rest; the following hold keeps the brake applied, because
    # brake torque is proportional to wheel speed and alone it only decays asymptotically.
    {"name": "ramp_100_a_up",   "duration": 3.0, "throttle": 1.00, "brake": 0.00},
    {"name": "ramp_100_a_down", "duration": 5.0, "throttle": 0.00, "brake": 1.00,
     "end_speed_below": 0.4},
    {"name": "ramp_100_a_hold", "duration": 0.8, "throttle": 0.00, "brake": 1.00},

    {"name": "ramp_075_up",   "duration": 3.5, "throttle": 0.75, "brake": 0.00},
    {"name": "ramp_075_down", "duration": 5.0, "throttle": 0.00, "brake": 1.00,
     "end_speed_below": 0.4},
    {"name": "ramp_075_hold", "duration": 0.8, "throttle": 0.00, "brake": 1.00},

    {"name": "ramp_050_up",   "duration": 4.5, "throttle": 0.50, "brake": 0.00},
    {"name": "ramp_050_down", "duration": 5.0, "throttle": 0.00, "brake": 1.00,
     "end_speed_below": 0.4},
    {"name": "ramp_050_hold", "duration": 0.8, "throttle": 0.00, "brake": 1.00},

    {"name": "ramp_030_up",   "duration": 6.0, "throttle": 0.30, "brake": 0.00},
    {"name": "ramp_030_down", "duration": 5.0, "throttle": 0.00, "brake": 1.00,
     "end_speed_below": 0.4},
    {"name": "ramp_030_hold", "duration": 0.8, "throttle": 0.00, "brake": 1.00},

    # A repeat of the full-throttle pair: the same input twice, so the two responses can be
    # compared directly and a residual is never fitted to a single unrepeated event.
    {"name": "ramp_100_b_up",   "duration": 3.0, "throttle": 1.00, "brake": 0.00},
    {"name": "ramp_100_b_down", "duration": 5.0, "throttle": 0.00, "brake": 1.00,
     "end_speed_below": 0.4},
    {"name": "ramp_100_b_hold", "duration": 0.8, "throttle": 0.00, "brake": 1.00},

    # One clean coast from speed: the resistance sample.
    {"name": "coast_id",   "duration": 2.5, "throttle": 0.00, "brake": 0.00},

    # Pitch events.  The car is at rest here: the throttle events cannot accelerate it
    # (the tyres hold it like a static friction, see the note above), but they load the
    # rear suspension, and the brake events keep the nose down, so the pitch response
    # to an input step is still recorded.
    {"name": "accel_snap",  "type": "pulse", "channel": "throttle",
     "amplitude": 1.00, "duration": 0.4},
    {"name": "brake_snap",  "type": "pulse", "channel": "brake",
     "amplitude": 1.00, "duration": 0.5},
    {"name": "pitch_rest",  "duration": 1.5, "throttle": 0.00, "brake": 0.00},

    # Resonance: the pitch mode is excited by one channel at a time, never both.
    {"name": "res_sweep", "type": "square", "channel": "throttle",
     "amplitude": 1.00, "frequency": 0.50, "duration": 3.0},
    {"name": "res_settle", "duration": 1.0, "throttle": 0.00, "brake": 0.00},
    {"name": "res_sweep_bk", "type": "square", "channel": "brake",
     "amplitude": 0.70, "frequency": 0.50, "duration": 3.0},
    {"name": "final_stop",  "duration": 2.0, "throttle": 0.00, "brake": 1.00},
]

# ---------------------------------------------------------------------------
# Vehicle description
# ---------------------------------------------------------------------------
# Nothing below is hard-coded to a vehicle model: the powertrain, gearbox and brake
# numbers are read from the 'customData' string the Car PROTO writes (the exact string
# the Webots vehicle library parses), the collision boxes from the instantiated PROTO
# nodes, and mass / centre of mass / suspension from the .proto source file.  That file
# also cross-checks every value that has two sources, so a stale PROTO or a stale
# simulation is reported instead of silently corrupting the log.

# Vehicle frame: origin on the rear axle at wheel centre height, X forward, Y left.
# The per-wheel positions are derived from the detected wheelbase and tracks.
WHEEL_ORDER = ("fl", "fr", "rl", "rr")
WHEEL_INDEX = {
    "fl": Car.WHEEL_FRONT_LEFT,
    "fr": Car.WHEEL_FRONT_RIGHT,
    "rl": Car.WHEEL_REAR_LEFT,
    "rr": Car.WHEEL_REAR_RIGHT,
}
WHEEL_SENSOR_NAMES = {
    "fl": "left_front_sensor",
    "fr": "right_front_sensor",
    "rl": "left_rear_sensor",
    "rr": "right_rear_sensor",
}
WHEEL_CONTACT_RADIUS = 0.45   # [m] contact points closer than this belong to a wheel

# Fields of the 'customData' string, in the order car.c parses them.
CUSTOM_DATA_FIELDS = (
    "wheelbase", "track_front", "track_rear", "front_tyre_radius", "rear_tyre_radius",
    "brake_coefficient", "wheels_damping", "engine_type", "engine_max_torque",
    "engine_max_power", "engine_min_rpm", "engine_max_rpm", "engine_coefficient_0",
    "engine_coefficient_1", "engine_coefficient_2", "hybrid_power_split_ratio",
    "hybrid_power_split_rpm", "engine_sound_rpm_reference", "gear_number",
    "engine_sound_length",
)
ENGINE_TYPE_CODES = {"C": "combustion", "E": "electric", "P": "parallel hybrid",
                     "S": "power-split hybrid"}

# (label used in the report, attribute of VehicleParameters, key of the .proto file)
PROTO_CHECKS = (
    ("wheelbase", "wheelbase", "wheelbase", 1e-3),
    ("trackFront", "track_front", "trackFront", 1e-3),
    ("trackRear", "track_rear", "trackRear", 1e-3),
    ("brakeCoefficient", "brake_coefficient", "brakeCoefficient", 1e-3),
    ("engineMaxTorque", "engine_max_torque", "engineMaxTorque", 1e-3),
    ("engineMaxPower", "engine_max_power", "engineMaxPower", 1e-3),
    ("engineMinRPM", "engine_min_rpm", "engineMinRPM", 1e-3),
    ("engineMaxRPM", "engine_max_rpm", "engineMaxRPM", 1e-3),
    ("wheelsDampingConstant", "wheels_damping", "wheelsDampingConstant", 1e-3),
    ("gearRatio", "gear_ratios", "gearRatio", 1e-6),
)


def _fmt(value):
    """Format a number or a sequence of numbers for the printed report."""
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(item) for item in value) + ")"
    if isinstance(value, float):
        return "%.4g" % value
    return str(value)


def _same(runtime, declared, tolerance):
    """Compare two numbers or two equal-length number sequences."""
    if isinstance(runtime, (list, tuple)) and isinstance(declared, (list, tuple)):
        return (len(runtime) == len(declared) and
                all(abs(float(a) - float(b)) <= tolerance for a, b in zip(runtime, declared)))
    try:
        return abs(float(runtime) - float(declared)) <= tolerance
    except (TypeError, ValueError):
        return runtime == declared


def read_custom_data(text):
    """Parse the vehicle description string written by the Car PROTO."""
    tokens = text.split()
    if len(tokens) < len(CUSTOM_DATA_FIELDS) + 2:
        raise RuntimeError(
            "the robot provided only %d customData fields; attach this controller to a "
            "Car based PROTO (a kinematic '*Simple' node has no vehicle description)"
            % len(tokens))
    values = {}
    for name, token in zip(CUSTOM_DATA_FIELDS, tokens):
        if name == "engine_type":
            values[name] = ENGINE_TYPE_CODES.get(token, token)
        elif name in ("gear_number", "engine_sound_length"):
            values[name] = int(token)
        else:
            values[name] = float(token)
    gear_number = values["gear_number"]
    values["gear_ratios"] = [float(value) for value in
                             tokens[len(CUSTOM_DATA_FIELDS):
                                    len(CUSTOM_DATA_FIELDS) + gear_number]]
    values["engine_coefficients"] = (values.pop("engine_coefficient_0"),
                                     values.pop("engine_coefficient_1"),
                                     values.pop("engine_coefficient_2"))
    return values


def read_box(node, def_name, label):
    """Read one collision box from the instantiated PROTO; None when it is missing."""
    pose = node.getFromProtoDef(def_name)
    if pose is None:
        return None
    children = pose.getField("children")
    child = children.getMFNode(0) if children is not None else None
    if child is None:
        return None
    rotation = pose.getField("rotation").getSFRotation()
    return {"name": label,
            "center": tuple(pose.getField("translation").getSFVec3f()),
            "axis": tuple(rotation[:3]),
            "angle": float(rotation[3]),
            "size": tuple(child.getField("size").getSFVec3f())}


def find_proto_file(car, node):
    """Return the .proto source of the vehicle, or None when it cannot be located."""
    world = car.getWorldPath()
    if not world:
        return None
    world_directory = os.path.dirname(os.path.abspath(world))
    for folder in (world_directory,
                   os.path.join(world_directory, os.pardir, "protos"),
                   os.path.join(world_directory, os.pardir, os.pardir, "protos")):
        candidate = os.path.join(folder, node.getTypeName() + ".proto")
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


def _proto_numbers(text, pattern, count):
    """Return `count` numbers that follow `pattern` in `text`, or None."""
    token = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    match = re.search(pattern + r"\s+(" + token + (r"\s+" + token) * (count - 1) + ")", text)
    return [float(value) for value in match.group(1).split()] if match else None


def read_proto_file(path):
    """Read the numbers declared in the vehicle .proto source (best effort)."""
    with open(path, encoding="utf-8", errors="replace") as proto:
        # Drop the comments first: several fields are intentionally commented out.
        text = "\n".join(line.split("#", 1)[0] for line in proto.read().splitlines())
    declared = {}
    for key in ("wheelbase", "trackFront", "trackRear", "brakeCoefficient",
                "engineMaxTorque", "engineMaxPower", "engineMinRPM", "engineMaxRPM",
                "wheelsDampingConstant", "mass"):
        numbers = _proto_numbers(text, r"\b" + key, 1)
        declared[key] = numbers[0] if numbers else None
    match = re.search(r'\bengineType\s+"([^"]*)"', text)
    declared["engineType"] = match.group(1) if match else None
    match = re.search(r"\bgearRatio\s*\[([^\]]*)\]", text)
    declared["gearRatio"] = [float(value) for value in match.group(1).split()] if match else None
    match = re.search(r"\bcenterOfMass\s*\[([^\]]*)\]", text)
    declared["centerOfMass"] = ([float(value) for value in match.group(1).split()]
                                if match else None)
    for label, start_marker, end_marker in (("lower", "DEF LOWER_BOX", "DEF CENTRAL_BOX"),
                                            ("central", "DEF CENTRAL_BOX", "physics Physics")):
        start = text.find(start_marker)
        end = text.find(end_marker, start + 1) if start >= 0 else -1
        if start < 0 or end < 0:
            declared["box_" + label] = None
            continue
        chunk = text[start:end]
        declared["box_" + label] = {"center": _proto_numbers(chunk, "translation", 3),
                                    "rotation": _proto_numbers(chunk, "rotation", 4),
                                    "size": _proto_numbers(chunk, "size", 3)}
    return declared


class VehicleParameters:
    """Powertrain, chassis and collision geometry of the simulated vehicle."""

    def __init__(self, values, boxes, proto_path, proto_values):
        self.__dict__.update(values)
        self.boxes = boxes
        self.proto_path = proto_path
        self.proto_values = proto_values
        self.wheel_positions = {
            "fl": (self.wheelbase, 0.5 * self.track_front),
            "fr": (self.wheelbase, -0.5 * self.track_front),
            "rl": (0.0, 0.5 * self.track_rear),
            "rr": (0.0, -0.5 * self.track_rear),
        }
        self.box_corners = {
            box["name"]: (np.asarray(box["center"], dtype=float)
                          + (_rotation_matrix(box["axis"], box["angle"])
                             @ (np.asarray(box["size"], dtype=float) * BOX_SIGNS).T).T)
            for box in boxes
        }

    @property
    def tyre_radius(self):
        return self.front_tyre_radius

    def torque_at(self, rpm):
        """Motor torque [Nm] at this rpm, following the vehicle library (driver.c)."""
        if rpm >= self.engine_max_rpm:
            return 0.0
        if self.engine_type == "electric":
            if rpm <= 0.0:
                return self.engine_max_torque
            return min(self.engine_max_torque,
                       self.engine_max_power * 60.0 / (2.0 * math.pi * rpm))
        a, b, c = self.engine_coefficients
        effective_rpm = max(rpm, self.engine_min_rpm)
        return a + b * effective_rpm + c * effective_rpm * effective_rpm

    def gear_ratio(self, gear):
        """Total ratio of the engaged gear (0 when the gearbox is disengaged)."""
        if gear > 0:
            return self.gear_ratios[gear] if gear < len(self.gear_ratios) else 0.0
        if gear < 0:
            return self.gear_ratios[0]
        return 0.0


def detect_vehicle_parameters(car, node):
    """Assemble the vehicle parameters from the simulation and its PROTO source."""
    values = read_custom_data(car.getCustomData() or "")
    boxes = [box for box in (read_box(node, "LOWER_BOX", "lower"),
                             read_box(node, "CENTRAL_BOX", "central")) if box]
    proto_path = find_proto_file(car, node)
    proto_values = {}
    if proto_path is not None:
        try:
            proto_values = read_proto_file(proto_path)
        except OSError as error:
            print("longdyn_logger: cannot read %s (%s), proto check skipped"
                  % (proto_path, error), flush=True)
            proto_path = None
    return VehicleParameters(values, boxes, proto_path, proto_values)


def compare_with_proto(params):
    """Return [(label, runtime value, proto value, status)] for every declared value."""
    rows = []
    for label, attribute, key, tolerance in PROTO_CHECKS:
        runtime = getattr(params, attribute)
        declared = params.proto_values.get(key)
        if declared is None:
            status = "not set"
        elif _same(runtime, declared, tolerance):
            status = "PASS"
        else:
            status = "MISMATCH"
        rows.append((label, runtime, declared if declared is not None else "not set", status))
    for box in params.boxes:
        declared = params.proto_values.get("box_" + box["name"]) or {}
        checks = (("centre", box["center"], declared.get("center")),
                  ("size", box["size"], declared.get("size")),
                  ("rotation", tuple(box["axis"]) + (box["angle"],), declared.get("rotation")))
        for part, runtime, expected in checks:
            if expected is None:
                status = "not set"
            elif _same(runtime, expected, 1e-3):
                status = "PASS"
            else:
                status = "MISMATCH"
            rows.append(("box %s %s" % (box["name"], part), runtime,
                         expected if expected is not None else "not set", status))
    return rows


def describe_parameters(params):
    """Return the printable report of the detected vehicle parameters."""
    lines = ["longdyn_logger: vehicle parameters detected from the simulation",
             "  powertrain  %-12s max torque %8.4g Nm  max power %8.4g kW  rpm %g..%g" % (
                 params.engine_type, params.engine_max_torque,
                 params.engine_max_power / 1000.0,
                 params.engine_min_rpm, params.engine_max_rpm),
             "  gearbox     %d ratios %s  engaged gear %d (%.3f:1)" % (
                 len(params.gear_ratios), _fmt(params.gear_ratios), TARGET_GEAR,
                 params.gear_ratio(TARGET_GEAR)),
             "  brakes      coefficient %.4g  wheel damping %.4g" % (
                 params.brake_coefficient, params.wheels_damping),
             "  chassis     wheelbase %.3f m  track %.3f/%.3f m  tyre radius %.3f m" % (
                 params.wheelbase, params.track_front, params.track_rear,
                 params.tyre_radius)]
    for box in params.boxes:
        lines.append("  box %-8s centre %-20s axis %-12s %.3f rad  size %s" % (
            box["name"], _fmt(box["center"]), _fmt(box["axis"]), box["angle"],
            _fmt(box["size"])))
    if not params.boxes:
        lines.append("  box         none found (getFromProtoDef returned no LOWER_BOX/"
                     "CENTRAL_BOX); bounding box columns will be empty")
    if params.proto_values.get("mass") is not None:
        lines.append("  body        mass %.4g kg  centre of mass %s" % (
            params.proto_values["mass"], _fmt(params.proto_values.get("centerOfMass"))))
    lines.append("  proto file  %s" % (params.proto_path or "not found, check skipped"))
    rows = compare_with_proto(params) if params.proto_path else []
    if rows:
        passed = sum(1 for row in rows if row[3] == "PASS")
        lines.append("  proto check %d/%d values match the .proto source" % (passed, len(rows)))
        for label, runtime, declared, status in rows:
            if status != "PASS":
                lines.append("    %-22s %-8s runtime %-22s proto %s" % (
                    label, status, _fmt(runtime), _fmt(declared)))
    return lines

# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _rotation_matrix(axis, angle):
    """Return the 3x3 rotation matrix of an axis-angle rotation (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0 or angle == 0.0:
        return np.eye(3)
    x, y, z = axis / norm
    cosine, sine = np.cos(angle), np.sin(angle)
    one_minus = 1.0 - cosine
    return np.array([
        [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
        [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
        [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
    ])


# Sign combinations of the 8 half-extent offsets of a box.
BOX_SIGNS = np.array([(dx, dy, dz) for dx in (-1.0, 1.0)
                                     for dy in (-1.0, 1.0)
                                     for dz in (-1.0, 1.0)])

def _finite(value, default=0.0):
    """Return ``default`` when the value is not a finite number."""
    try:
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Tyre contact detection
# ---------------------------------------------------------------------------


class ContactClassifier:
    """Assign the contact points of the vehicle to its four wheels.

    Every contact point reports the id of the Solid it belongs to (the tyre, the
    chassis, ...), so the ids are learnt from the geometry on the first steps and
    reused afterwards.  Contact points of the chassis or of the crash barriers are
    ignored, which makes the wheel lift-off flag reliable.
    """

    def __init__(self, params):
        self._wheel_ids = {}
        self._params = params

    def classify(self, node, position, rotation):
        """Return {wheel: [contact heights]} for the current step."""
        contacts = {wheel: [] for wheel in WHEEL_ORDER}
        for contact_point in node.getContactPoints(True):
            point = contact_point.getPoint()
            node_id = contact_point.getNodeId()
            wheel = self._wheel_ids.get(node_id)
            if wheel is None:
                wheel = self._nearest_wheel(point, position, rotation)
                if wheel is not None and wheel not in self._wheel_ids.values():
                    self._wheel_ids[node_id] = wheel
            if wheel is not None:
                contacts[wheel].append(point[2])
        return contacts

    def _nearest_wheel(self, point, position, rotation):
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        local = rotation.T @ (np.asarray(point, dtype=float) - np.asarray(position, dtype=float))
        best_wheel = None
        best_distance = WHEEL_CONTACT_RADIUS ** 2
        for wheel in WHEEL_ORDER:
            dx = local[0] - self._params.wheel_positions[wheel][0]
            dy = local[1] - self._params.wheel_positions[wheel][1]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_wheel = wheel
        return best_wheel


# ---------------------------------------------------------------------------
# Open-loop input schedule
# ---------------------------------------------------------------------------


def _half_sine(t, duration):
    """Pulse shape 0 -> 1 -> 0 over ``duration`` seconds (raised half sine)."""
    return math.sin(math.pi * min(max(t, 0.0), duration) / duration)


def _is_pulse(phase):
    """True for the phases whose waveform period is their own duration."""
    return phase.get("type") in ("pulse", "accel_pulse", "brake_pulse")


def _square(cycles):
    """Return +1 for the first half of the cycle and -1 for the second half.

    ``cycles`` is the number of elapsed periods (``2*pi*f*t_phase`` divided by 2*pi),
    so the duty cycle stays exactly 50 % at any sampling rate.
    """
    return 1.0 if math.floor(2.0 * cycles) % 2 == 0 else -1.0


class Schedule:
    """Sequential open-loop throttle/brake phases."""

    def __init__(self, phases):
        self._phases = phases
        self._index = -1
        self._start_time = 0.0
        self.overlap_events = 0
        self.overlap_phases = set()
        self._start_throttle = 0.0
        self._start_brake = 0.0
        self._throttle = 0.0
        self._brake = 0.0

    @property
    def finished(self):
        return self._index >= len(self._phases)

    @property
    def phase_index(self):
        return max(self._index, 0)

    @property
    def phase_name(self):
        if self.finished or self._index < 0:
            return "finished"
        return self._phases[self._index].get("name", "phase_%d" % self._index)

    @property
    def phase_block(self):
        if self.finished or self._index < 0:
            return "finished"
        return self._phases[self._index].get("block", "other")

    @property
    def commands(self):
        return self._throttle, self._brake

    def time_in_phase(self, t):
        return t - self._start_time

    def begin(self, t, throttle, brake):
        self._index = -1
        self._throttle, self._brake = throttle, brake
        self._advance(t)

    def _advance(self, t):
        self._index += 1
        self._start_time = t
        self._start_throttle = self._throttle
        self._start_brake = self._brake

    def update(self, t, speed_kmh):
        """Return the (throttle, brake) commands for time t."""
        if self.finished:
            return self._throttle, self._brake

        phase = self._phases[self._index]
        t_phase = t - self._start_time
        duration = phase.get("duration", math.inf)
        end_speed = phase.get("end_speed_below")
        end_above = phase.get("end_speed_above")
        # The sampled period of a pulse is its duration, so the last step still belongs
        # to the phase and lands exactly on the end level.
        past_duration = t_phase > duration if _is_pulse(phase) else t_phase >= duration
        # A speed threshold is only honoured after 'min_duration' so a run-up does not
        # end on the very first step, while the speed is still below the target.
        settled = t_phase >= phase.get("min_duration", 0.0)
        reached_target = (settled and end_above is not None and speed_kmh > end_above)
        in_band = (settled and phase.get("type") == "regulate" and
                   abs(speed_kmh - phase.get("speed_kmh", 0.0)) <= phase.get("band", 2.0))
        if past_duration or reached_target or in_band or \
                (end_speed is not None and speed_kmh < end_speed):
            self._advance(t)
            if self.finished:
                return self._throttle, self._brake
            phase = self._phases[self._index]
            t_phase = 0.0

        if phase.get("type") == "regulate":
            # Closed loop, used only to place the car at a target speed before a measured
            # event.  These phases are labelled 'reg' in the log and are meant to be
            # filtered out of the identification data.
            target = float(phase.get("speed_kmh", 40.0))
            error = target - speed_kmh
            if error > phase.get("band", 2.0):
                throttle = min(1.0, max(0.15, abs(error) / 10.0))
                brake = 0.0
            elif error < -phase.get("band", 2.0):
                throttle = 0.0
                brake = min(1.0, max(0.05, abs(error) / 25.0))
            else:
                throttle, brake = 0.0, 0.0
        elif phase.get("type") == "sine":
            # One channel at a time, and never negative: the offset is clamped at zero
            # instead of letting the sine swing through it.
            channel = phase.get("channel", "throttle")
            mean = phase.get("mean", phase.get("throttle_mean", 0.0))
            amplitude = phase.get("amplitude", phase.get("throttle_amplitude", 0.0))
            frequency = phase.get("frequency", phase.get("throttle_frequency", 1.0))
            value = mean + amplitude * math.sin(2.0 * math.pi * frequency * t_phase)
            if channel == "brake":
                throttle, brake = 0.0, value
            else:
                throttle, brake = value, 0.0
        elif phase.get("type") == "pulse":
            # Raised half sine on a single channel, from 0 up to 'amplitude' and back.
            channel = phase.get("channel", "throttle")
            value = phase.get("amplitude", 1.0) * _half_sine(t_phase, duration)
            if channel == "brake":
                throttle, brake = 0.0, value
            else:
                throttle, brake = value, 0.0
        elif phase.get("type") == "square":
            # Square wave on a single channel, 50 % duty.
            channel = phase.get("channel", "throttle")
            value = phase.get("amplitude", 1.0) if _square(
                phase.get("frequency", 0.5) * t_phase) > 0 else 0.0
            if channel == "brake":
                throttle, brake = 0.0, value
            else:
                throttle, brake = value, 0.0
        else:
            ramp = phase.get("ramp", 0.0)
            ratio = 1.0 if ramp <= 0.0 else min(1.0, t_phase / ramp)
            throttle = self._start_throttle + \
                (phase.get("throttle", 0.0) - self._start_throttle) * ratio
            brake = self._start_brake + \
                (phase.get("brake", 0.0) - self._start_brake) * ratio

        self._throttle = min(1.0, max(0.0, throttle))
        self._brake = min(1.0, max(0.0, brake))
        # Invariant: the throttle and the brake are never applied together.  If a badly
        # authored phase asks for both, the weaker command is dropped and the event is
        # counted so it shows up in the run summary instead of silently corrupting the
        # data.
        if self._throttle > 0.0 and self._brake > 0.0:
            self.overlap_events += 1
            self.overlap_phases.add(self.phase_name)
            if self._throttle >= self._brake:
                self._brake = 0.0
            else:
                self._throttle = 0.0
        return self._throttle, self._brake


def describe_phase(index, phase, start_time):
    """Return one console line describing a phase (timing, inputs, end condition)."""
    name = phase.get("name", "phase_%d" % index)
    duration = phase.get("duration")
    if phase.get("type") == "sine":
        line = "%2d  %-10s t=%6.2f s  %s sine %.2f + %.2f*sin(2*pi*%.2f*t_phase)" % (
            index, name, start_time, phase.get("channel", "throttle"),
            phase.get("mean", 0.0), phase.get("amplitude", 0.0),
            phase.get("frequency", 1.0))
    elif phase.get("type") == "pulse":
        line = "%2d  %-10s t=%6.2f s  %s pulse 0 -> %.2f -> 0 over %.2f s" % (
            index, name, start_time, phase.get("channel", "throttle"),
            phase.get("amplitude", 1.0), duration)
    elif phase.get("type") == "square":
        line = "%2d  %-10s t=%6.2f s  %s square 0 <-> %.2f at %.2f Hz" % (
            index, name, start_time, phase.get("channel", "throttle"),
            phase.get("amplitude", 1.0), phase.get("frequency", 0.5))
    elif phase.get("type") == "regulate":
        line = "%2d  %-10s t=%6.2f s  closed loop hold %.1f km/h (+-%.1f)" % (
            index, name, start_time, phase.get("speed_kmh", 0.0), phase.get("band", 2.5))
    else:
        line = "%2d  %-10s t=%6.2f s  throttle %.2f -> %.2f, brake %.2f -> %.2f" % (
            index, name, start_time, phase.get("throttle_start", 0.0),
            phase.get("throttle", 0.0), phase.get("brake_start", 0.0),
            phase.get("brake", 0.0))
    ramp = phase.get("ramp", 0.0)
    if ramp > 0.0:
        line += ", ramp %.2f s" % ramp
    line += ", duration %s" % ("inf" if duration is None else "%.2f s" % duration)
    if phase.get("end_speed_below") is not None:
        line += ", ends below %.1f km/h" % phase["end_speed_below"]
    if phase.get("end_speed_above") is not None:
        line += ", ends above %.1f km/h" % phase["end_speed_above"]
    return line


def print_schedule(phases):
    """Print the open-loop schedule: one line per phase, or per block when it is long."""
    total = sum(phase.get("duration", 0.0) for phase in phases)
    print("longdyn_logger: schedule with %d phases (nominal length %.1f s)" % (
        len(phases), total))
    if len(phases) <= 24:
        start_time = 0.0
        throttle, brake = 0.0, 0.0
        for index, phase in enumerate(phases):
            described = dict(phase)
            described.setdefault("throttle_start", throttle)
            described.setdefault("brake_start", brake)
            print("longdyn_logger:   " + describe_phase(index, described, start_time))
            throttle = phase.get("throttle", 0.0)
            brake = phase.get("brake", 0.0)
            start_time += phase.get("duration", 0.0)
        return
    # Long schedules: summarise each block, otherwise the console is flooded.
    block, count, duration, throttle_levels, brake_levels = None, 0, 0.0, set(), set()
    def flush():
        if block is None:
            return
        print("longdyn_logger:   %-12s %3d phases  %6.1f s  throttle %s  brake %s" % (
            block, count, duration, _level_summary(throttle_levels),
            _level_summary(brake_levels)))
    for phase in phases:
        if phase.get("block", "core") != block:
            flush()
            block = phase.get("block", "core")
            count, duration, throttle_levels, brake_levels = 0, 0.0, set(), set()
        count += 1
        duration += phase.get("duration", 0.0) or 0.0
        if phase.get("throttle", 0.0) > 0.0:
            throttle_levels.add(round(phase["throttle"], 4))
        if phase.get("brake", 0.0) > 0.0:
            brake_levels.add(round(phase["brake"], 4))
    flush()


def _level_summary(levels):
    """Return a short description of the set of input levels of a block."""
    if not levels:
        return "-"
    ordered = sorted(levels)
    if len(ordered) == 1:
        return "%.2f" % ordered[0]
    return "%d levels %.2f..%.2f" % (len(ordered), ordered[0], ordered[-1])


def expand_phases(phases):
    """Expand the sweep / pulse_train / alternate macros into concrete phases."""
    expanded = []
    for phase in phases:
        kind = phase.get("type")
        block = phase.get("block", "core")
        if kind is None and "name" not in phase:
            continue      # documentation-only entry
        if kind == "sweep":
            channel = phase.get("channel", "throttle")
            prefix = phase.get("prefix", channel[:3])
            hold = phase.get("hold", 0.8)
            gap = phase.get("gap", 0.0)
            for level in phase.get("levels", []):
                entry = {"name": "%s_%03d" % (prefix, round(level * 100)),
                         "block": block, "duration": hold,
                         "throttle": 0.0, "brake": 0.0}
                entry[channel] = level
                expanded.append(entry)
                if gap > 0.0:
                    expanded.append({"name": prefix + "_gap", "block": block,
                                     "duration": gap, "throttle": 0.0, "brake": 0.0})
        elif kind == "regulated_sweep":
            # One pulse per level, each preceded by a reset that puts the car back at the
            # same speed: the only way to keep the acceleration events inside the usable
            # speed band (the vehicle saturates at 64 km/h and cannot break away at rest).
            channel = phase.get("channel", "throttle")
            prefix = phase.get("prefix", channel[:3])
            hold = phase.get("hold", 0.8)
            gap = phase.get("gap", 1.0)
            reset = phase.get("reset", {})
            mode = reset.get("mode", "hold")
            for level in phase.get("levels", []):
                if mode == "hold":
                    expanded.append({"type": "regulate", "name": prefix + "_reg",
                                     "block": block,
                                     "duration": reset.get("duration", 8.0),
                                     "speed_kmh": reset.get("speed_kmh", 40.0),
                                     "band": reset.get("band", 2.5),
                                     "min_duration": reset.get("min_duration", 0.3)})
                elif mode == "speed_up":
                    expanded.append({"name": prefix + "_up", "block": block,
                                     "duration": reset.get("duration", 12.0),
                                     "throttle": reset.get("throttle", 1.0), "brake": 0.0,
                                     "end_speed_above": reset.get("speed_kmh", 45.0),
                                     "min_duration": reset.get("min_duration", 0.3)})
                else:
                    expanded.append({"name": prefix + "_down", "block": block,
                                     "duration": reset.get("duration", 4.0),
                                     "throttle": 0.0, "brake": reset.get("brake", 0.15),
                                     "end_speed_below": reset.get("speed_kmh", 12.0)})
                expanded.append({"name": prefix + "_settle", "block": block,
                                 "duration": reset.get("settle", 0.5),
                                 "throttle": 0.0, "brake": 0.0})
                entry = {"name": "%s_%03d" % (prefix, round(level * 100)), "block": block,
                         "duration": hold, "throttle": 0.0, "brake": 0.0}
                entry[channel] = level
                # A brake pulse must never stop the car: the wheel joints lock at zero
                # speed and the vehicle cannot reliably pull away again (measured).
                if channel == "brake" and phase.get("end_speed_below"):
                    entry["end_speed_below"] = phase["end_speed_below"]
                expanded.append(entry)
                if gap > 0.0:
                    expanded.append({"name": prefix + "_gap", "block": block,
                                     "duration": gap, "throttle": 0.0, "brake": 0.0})
        elif kind == "pulse_train":
            channel = phase.get("channel", "throttle")
            level = phase.get("level", 0.3)
            pulse = phase.get("pulse", 0.5)
            gap = phase.get("gap", 1.0)
            prefix = phase.get("prefix", channel[:3])
            for index in range(int(phase.get("count", 3))):
                entry = {"name": "%s_p%d" % (prefix, index + 1), "block": block,
                         "duration": pulse, "throttle": 0.0, "brake": 0.0}
                entry[channel] = level
                if channel == "brake" and phase.get("end_speed_below"):
                    entry["end_speed_below"] = phase["end_speed_below"]
                expanded.append(entry)
                expanded.append({"name": prefix + "_gap", "block": block,
                                 "duration": gap, "throttle": 0.0, "brake": 0.0})
        elif kind == "alternate":
            pulse = phase.get("pulse", 0.5)
            gap = phase.get("gap", 1.0)
            prefix = phase.get("prefix", "alt")
            for index in range(int(phase.get("count", 6))):
                if index % 2 == 0:
                    entry = {"name": "%s_t%d" % (prefix, index // 2 + 1), "block": block,
                             "duration": pulse,
                             "throttle": phase.get("throttle_level", 0.3), "brake": 0.0}
                else:
                    entry = {"name": "%s_b%d" % (prefix, index // 2 + 1), "block": block,
                             "duration": pulse,
                             "throttle": 0.0, "brake": phase.get("brake_level", 0.3)}
                expanded.append(entry)
                expanded.append({"name": prefix + "_gap", "block": block,
                                 "duration": gap, "throttle": 0.0, "brake": 0.0})
        elif kind == "ramp_event":
            # Ramp up, hold, then die down: the attack and the release can have different
            # durations and the peak is the amplitude, on one channel only.
            name = phase["name"]
            channel = phase["channel"]
            amplitude = phase["amplitude"]
            up = {"name": name + "_up", "block": block, "duration": phase["attack"],
                  "throttle": 0.0, "brake": 0.0, "ramp": phase["attack"]}
            up[channel] = amplitude
            hold = {"name": name + "_hold", "block": block, "duration": phase["hold"],
                    "throttle": 0.0, "brake": 0.0}
            hold[channel] = amplitude
            for key in ("end_speed_above", "end_speed_below", "min_duration"):
                if key in phase:
                    hold[key] = phase[key]
            down = {"name": name + "_down", "block": block, "duration": phase["release"],
                    "throttle": 0.0, "brake": 0.0, "ramp": phase["release"]}
            expanded += [up, hold, down]
        elif kind == "square_alternate":
            # Throttle for t seconds, then brake for t seconds, back to back with no gap:
            # a true square wave of the two inputs, never overlapping.
            t_on = phase.get("t", 1.0)
            for index in range(int(phase.get("count", 6))):
                on_throttle = index % 2 == 0
                expanded.append({
                    "name": "%s_%s%d" % (phase.get("prefix", "sq"),
                                         "t" if on_throttle else "b", index // 2 + 1),
                    "block": block, "duration": t_on,
                    "throttle": phase.get("throttle_level", 0.3) if on_throttle else 0.0,
                    "brake": 0.0 if on_throttle else phase.get("brake_level", 0.1)})
        else:
            entry = dict(phase)
            entry["block"] = block
            expanded.append(entry)
    return expanded


def validate_phases(phases):
    """Report the phases that would produce unusable identification data.

    Checks, in this order: simultaneous throttle and brake, a sine that would be
    clipped flat at zero, and an acceleration phase that starts from a standstill at a
    level too low to break the tyre contact (the car then only creeps).
    """
    problems = []
    for index, phase in enumerate(phases):
        name = phase.get("name", "phase_%d" % index)
        throttle = phase.get("throttle", 0.0) or 0.0
        brake = phase.get("brake", 0.0) or 0.0
        if phase.get("type") not in ("sine", "pulse", "square") and \
                throttle > 0.0 and brake > 0.0:
            problems.append("%s: throttle %.2f and brake %.2f together" % (
                name, throttle, brake))
        if phase.get("type") == "sine":
            mean = phase.get("mean", phase.get("throttle_mean", 0.0))
            amplitude = phase.get("amplitude", phase.get("throttle_amplitude", 0.0))
            if mean < amplitude:
                problems.append("%s: sine mean %.2f < amplitude %.2f, it clips at zero" % (
                    name, mean, amplitude))
        restart = phase.get("restart_from_rest")
        if restart and throttle < 0.6:
            problems.append("%s: acceleration from rest at throttle %.2f only creeps" % (
                name, throttle))
    print("longdyn_logger: schedule validation: %d phases, %d warning(s)" % (
        len(phases), len(problems)), flush=True)
    for problem in problems:
        print("longdyn_logger:   WARNING " + problem, flush=True)
    return problems


def load_schedule():
    """Return SCHEDULE, or the JSON list given as first controller argument."""
    if len(sys.argv) < 2:
        return expand_phases(SCHEDULE)
    path = sys.argv[1]
    if not os.path.isabs(path) and not os.path.isfile(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, "r", encoding="utf-8") as file:
        phases = json.load(file)
    if not isinstance(phases, list) or not phases:
        raise ValueError("%s must contain a non-empty list of phases" % path)
    expanded = expand_phases(phases)
    print("longdyn_logger: %d entries loaded from %s, expanded to %d phases" % (
        len(phases), path, len(expanded)))
    return expanded


# ---------------------------------------------------------------------------
# Lateral controller
# ---------------------------------------------------------------------------


def lateral_control(lat_error, head_error, yaw_rate, speed_ms, previous_command, dt):
    """Return the steering angle [rad] keeping the car on its straight line.

    ``lat_error`` is positive on the left of the reference line and ``head_error``
    positive when the car points left of it, whereas a positive steering angle steers
    right, hence the positive signs.  Inside the deadband the command is smoothly
    released to zero so the controller stays silent on a perfectly straight run.
    """
    config = LATERAL_CONTROLLER
    if abs(lat_error) <= config["deadband_lateral"] and \
            abs(head_error) <= config["deadband_heading"]:
        target = 0.0
    else:
        speed = max(abs(speed_ms), config["min_speed"])
        target = (config["heading_gain"] * head_error +
                  config["yaw_rate_gain"] * yaw_rate +
                  config["cross_track_gain"] * lat_error / speed)
        limit = config["max_steering_angle"]
        target = max(-limit, min(limit, target))
    alpha = dt / (config["filter_time_constant"] + dt)
    return previous_command + alpha * (target - previous_command)


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


def build_columns(box_names):
    """Return the ordered list of CSV columns for the detected box names."""
    columns = [
        "t_s", "step", "phase_id", "block", "phase", "t_phase_s",
        "throttle_cmd", "brake_cmd", "throttle_applied", "brake_applied",
        "steer_cmd_rad", "steer_left_rad", "steer_right_rad",
        "control_mode", "gear", "engine_rpm",
        "engine_torque_nm", "wheel_torque_total_nm",
        "speed_kmh", "speed_ms", "speed_ground_ms", "speed_long_ms", "s_travelled_m",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "roll_rad", "pitch_rad", "yaw_rad",
        "gps_x", "gps_y", "gps_z", "gps_vx", "gps_vy", "gps_vz",
        "pos_x", "pos_y", "pos_z",
        "rot00", "rot01", "rot02", "rot10", "rot11", "rot12", "rot20", "rot21", "rot22",
        "fwd_x", "fwd_y", "fwd_z",
        "com_x", "com_y", "com_z",
        "lat_error", "head_error", "lat_error_rate",
        "body_min_x", "body_min_y", "body_min_z",
        "body_max_x", "body_max_y", "body_max_z",
    ]
    for name in box_names:
        columns += ["obb_%s_c%d_%s" % (name, index, axis)
                    for index in range(8) for axis in ("x", "y", "z")]
    for wheel in WHEEL_ORDER:
        columns += ["wheel_%s_%s" % (wheel, name) for name in
                    ("encoder_rad", "speed_rads", "speed_ms", "slip",
                     "contact", "contact_count", "contact_z")]
    return columns


def format_value(value):
    """Format a value for the CSV file."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return "%.9g" % value
    return value


class CsvLogger:
    """Write one row per control step, with a documented comment header."""

    def __init__(self, path, metadata, box_names):
        self._file = open(path, "w", newline="", encoding="utf-8")
        for key, value in metadata.items():
            self._file.write("# %s: %s\n" % (key, value))
        self._writer = csv.writer(self._file)
        self._columns = build_columns(box_names)
        self._writer.writerow(self._columns)
        self.rows = 0

    def write(self, row):
        self._writer.writerow([format_value(row.get(column)) for column in self._columns])
        self.rows += 1

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# State sampling
# ---------------------------------------------------------------------------


def build_metadata(car, node, timestep, schedule, run_path, params):
    """Return the comment header written at the top of the CSV file."""
    return {
        "logger": "longdyn_logger.py",
        "created": wall_clock.strftime("%Y-%m-%d %H:%M:%S"),
        "file": os.path.basename(run_path),
        "world": os.path.basename(car.getWorldPath() or ""),
        "vehicle": car.getModel(),
        "robot_name": car.getName(),
        "time_step_ms": timestep,
        "vehicle_mass_kg": _fmt(params.proto_values.get("mass", "not detected")),
        "wheelbase_m": params.wheelbase,
        "track_front_m": params.track_front,
        "track_rear_m": params.track_rear,
        "tyre_radius_m": params.tyre_radius,
        "sensors": "GPS at the vehicle frame origin (rear axle); InertialUnit, Gyro and "
                   "Accelerometer at the chassis centre of mass (1.6 0 0.2 m)",
        "sign_conventions": "steering positive = right; lat/head errors positive = left of "
                            "the reference line; accelerometer includes gravity",
        "bounding_boxes": "detected from the PROTO nodes: " + "; ".join(
            "%s centre %s axis %s %.4g rad size %s" % (box["name"], _fmt(box["center"]),
                                                       _fmt(box["axis"]), box["angle"],
                                                       _fmt(box["size"]))
            for box in params.boxes),
        "wheel_frame": "vehicle frame origin on the rear axle, X forward, Y left",
        "gearbox": "fixed gear %d, ratios %s (index 0 is reverse, 0 disengaged)"
                   % (TARGET_GEAR, params.gear_ratios),
        "powertrain": "%s, max torque %.6g Nm, max power %.6g kW, rpm %.6g..%.6g; the "
                      "engine_torque_nm and wheel_torque_total_nm columns follow the "
                      "library curve and ignore its acceleration limiter"
                      % (params.engine_type, params.engine_max_torque,
                         params.engine_max_power / 1000.0,
                         params.engine_min_rpm, params.engine_max_rpm),
        "parameters_from": params.proto_path or "simulation only",
        "schedule": json.dumps(schedule),
        "supervisor": "pose and contact points streamed at the time step",
    }


def log_bounding_boxes(row, position, rotation, params):
    """Store the world axis-aligned bound and the world corners of both boxes."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    position = np.asarray(position, dtype=float)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for box in params.boxes:
        corners = position + (rotation @ params.box_corners[box["name"]].T).T
        for index in range(8):
            prefix = "obb_%s_c%d_" % (box["name"], index)
            row[prefix + "x"] = float(corners[index, 0])
            row[prefix + "y"] = float(corners[index, 1])
            row[prefix + "z"] = float(corners[index, 2])
        minimum = np.minimum(minimum, corners.min(axis=0))
        maximum = np.maximum(maximum, corners.max(axis=0))
    row["body_min_x"], row["body_min_y"], row["body_min_z"] = (
        float(minimum[0]), float(minimum[1]), float(minimum[2]))
    row["body_max_x"], row["body_max_y"], row["body_max_z"] = (
        float(maximum[0]), float(maximum[1]), float(maximum[2]))


def log_wheels(row, car, classifier, node, position, rotation, longitudinal_speed_ms, params):
    """Store the wheel state; contact points give the wheel lift-off detection."""
    contacts = classifier.classify(node, position, rotation)
    for wheel in WHEEL_ORDER:
        angular_speed = _finite(car.getWheelSpeed(WHEEL_INDEX[wheel]), None)
        if angular_speed is None:
            row["wheel_%s_encoder_rad" % wheel] = None
            row["wheel_%s_speed_rads" % wheel] = None
            row["wheel_%s_speed_ms" % wheel] = None
            row["wheel_%s_slip" % wheel] = None
        else:
            wheel_speed = angular_speed * params.tyre_radius
            row["wheel_%s_encoder_rad" % wheel] = car.getWheelEncoder(WHEEL_INDEX[wheel])
            row["wheel_%s_speed_rads" % wheel] = angular_speed
            row["wheel_%s_speed_ms" % wheel] = wheel_speed
            row["wheel_%s_slip" % wheel] = ((wheel_speed - longitudinal_speed_ms) /
                                            max(abs(longitudinal_speed_ms), 0.5)
                                            if abs(longitudinal_speed_ms) > 0.5 else 0.0)
        row["wheel_%s_contact" % wheel] = 1 if contacts[wheel] else 0
        row["wheel_%s_contact_count" % wheel] = len(contacts[wheel])
        row["wheel_%s_contact_z" % wheel] = (min(contacts[wheel]) if contacts[wheel]
                                             else None)


def sample(car, node, devices, classifier, run, step_index, schedule, t, position,
           rotation, forward, lat_error, head_error, lat_error_rate, steering_command,
           speed_kmh, params):
    """Return one dictionary holding every logged channel for the current step."""
    gps_values = devices["gps"].getValues()
    gps_speed = devices["gps"].getSpeedVector()
    roll, pitch, yaw = devices["imu"].getRollPitchYaw()
    acceleration = devices["accelerometer"].getValues()
    angular_velocity = devices["gyro"].getValues()
    center_of_mass = node.getCenterOfMass()
    speed_long_ms = sum(forward[i] * gps_speed[i] for i in range(3))
    rpm = _finite(car.getRpm(), 0.0)
    gear = car.getGear()
    engine_torque = params.torque_at(rpm)

    row = {
        "t_s": t,
        "step": step_index,
        "phase_id": schedule.phase_index,
        "block": schedule.phase_block,
        "phase": schedule.phase_name,
        "t_phase_s": schedule.time_in_phase(t),
        "throttle_cmd": schedule.commands[0],
        "brake_cmd": schedule.commands[1],
        "throttle_applied": car.getThrottle(),
        "brake_applied": car.getBrakeIntensity(),
        "steer_cmd_rad": steering_command,
        "steer_left_rad": car.getLeftSteeringAngle(),
        "steer_right_rad": car.getRightSteeringAngle(),
        "control_mode": car.getControlMode(),
        "gear": gear,
        "engine_rpm": rpm,
        "engine_torque_nm": engine_torque,
        "wheel_torque_total_nm": (engine_torque * schedule.commands[0] *
                                  params.gear_ratio(gear)),
        "speed_kmh": speed_kmh,
        "speed_ms": speed_kmh / 3.6,
        "speed_ground_ms": math.sqrt(sum(component * component for component in gps_speed)),
        "speed_long_ms": speed_long_ms,
        "s_travelled_m": run["s_travelled"],
        "accel_x": acceleration[0],
        "accel_y": acceleration[1],
        "accel_z": acceleration[2],
        "gyro_x": angular_velocity[0],
        "gyro_y": angular_velocity[1],
        "gyro_z": angular_velocity[2],
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "gps_x": gps_values[0],
        "gps_y": gps_values[1],
        "gps_z": gps_values[2],
        "gps_vx": gps_speed[0],
        "gps_vy": gps_speed[1],
        "gps_vz": gps_speed[2],
        "pos_x": position[0],
        "pos_y": position[1],
        "pos_z": position[2],
        "fwd_x": forward[0],
        "fwd_y": forward[1],
        "fwd_z": forward[2],
        "com_x": center_of_mass[0],
        "com_y": center_of_mass[1],
        "com_z": center_of_mass[2],
        "lat_error": lat_error,
        "head_error": head_error,
        "lat_error_rate": lat_error_rate,
    }
    for index, name in enumerate(("rot00", "rot01", "rot02", "rot10", "rot11", "rot12",
                                  "rot20", "rot21", "rot22")):
        row[name] = rotation[index]

    log_bounding_boxes(row, position, rotation, params)
    log_wheels(row, car, classifier, node, position, rotation, speed_long_ms, params)
    return row


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def enable_device(robot, name, period):
    """Return an enabled device, failing loudly when it is missing."""
    device = robot.getDevice(name)
    if device is None:
        raise RuntimeError("device '%s' not found in the world file" % name)
    device.enable(period)
    return device


class Actuators:
    """Forward the throttle, brake and steering commands to the car.

    Only the changed commands are sent: the Car library keeps the last value, so
    re-sending an identical command every step is pure overhead.
    """

    def __init__(self, car):
        self._car = car
        self._throttle = None
        self._brake = None
        self._steering = None

    def apply(self, throttle, brake, steering):
        if self._throttle is None or abs(throttle - self._throttle) > 1e-9:
            self._throttle = throttle
            self._car.setThrottle(throttle)
        if self._brake is None or abs(brake - self._brake) > 1e-9:
            self._brake = brake
            self._car.setBrakeIntensity(brake)
        if self._steering is None or abs(steering - self._steering) > 1e-9:
            self._steering = steering
            self._car.setSteeringAngle(steering)


def main():
    car = Car()
    timestep = int(car.getBasicTimeStep())
    dt = timestep / 1000.0
    # The simulation may run on a finer step than the log: sample and log on a
    # CONTROL_PERIOD_MS grid (5 ms world step -> every step, 10 ms -> every other step).
    control_dt = CONTROL_PERIOD_MS / 1000.0
    log_period = CONTROL_PERIOD_MS
    log_steps = max(1, int(round(CONTROL_PERIOD_MS / float(timestep))))

    node = car.getSelf()
    if node is None:
        sys.stderr.write("The vehicle must be a supervisor: add 'supervisor TRUE' "
                         "to the robot node of the world file.\n")
        return 1
    node.enablePoseTracking(log_period, None)
    node.enableContactPointsTracking(log_period, True)

    devices = {
        "gps": enable_device(car, "gps", log_period),
        "imu": enable_device(car, "imu", log_period),
        "gyro": enable_device(car, "gyro", log_period),
        "accelerometer": enable_device(car, "accelerometer", log_period),
    }
    for name in WHEEL_SENSOR_NAMES.values():
        sensor = car.getDevice(name)
        if sensor is not None:
            sensor.enable(log_period)

    # Torque control: cruise control would bypass the engine model (and the RPM).
    # The Car library starts in neutral: a gear must be engaged to get any drive torque.
    car.setThrottle(0.0)
    car.setBrakeIntensity(0.0)
    car.setSteeringAngle(0.0)
    car.setGear(TARGET_GEAR)

    # Detect the vehicle from the running simulation and from its PROTO source, then
    # report it before a single row is logged.
    params = detect_vehicle_parameters(car, node)
    for line in describe_parameters(params):
        print(line, flush=True)

    phases = load_schedule()
    print_schedule(phases)
    validate_phases(phases)
    schedule = Schedule(phases)
    classifier = ContactClassifier(params)

    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    run_path = os.path.join(LOG_DIRECTORY,
                            "run_%s.csv" % wall_clock.strftime("%Y%m%d-%H%M%S"))
    logger = CsvLogger(run_path, build_metadata(car, node, timestep, phases, run_path,
                                                params),
                       [box["name"] for box in params.boxes])
    print("longdyn_logger: logging to %s" % run_path)

    # First step: initialise the simulation, then take the initial pose as reference.
    if car.step() == -1:
        logger.close()
        return 0
    position = node.getPosition()
    rotation = node.getOrientation()
    reference_point = (position[0], position[1])
    direction_norm = math.hypot(rotation[0], rotation[3])
    reference_direction = ((rotation[0] / direction_norm, rotation[3] / direction_norm)
                           if direction_norm > 1e-6 else (1.0, 0.0))
    print("longdyn_logger: reference line origin (%.3f, %.3f) direction (%.3f, %.3f)"
          % (reference_point[0], reference_point[1],
             reference_direction[0], reference_direction[1]))

    start_time = car.getTime()
    schedule.begin(start_time, 0.0, 0.0)
    actuators = Actuators(car)

    steering_command = 0.0
    previous_lat_error = None
    lat_error_rate = 0.0
    step_index = 1
    abort_reason = None
    max_speed_kmh = 0.0
    t = start_time
    run = {"s_travelled": 0.0}
    physics_step_index = 0

    try:
        while True:
            t = car.getTime()

            # The state below is only sampled (and the lateral command only updated) on
            # the log grid, so a finer physics step does not add rows to the CSV.
            if log_steps == 1 or physics_step_index % log_steps == 0:
                speed_kmh = _finite(car.getCurrentSpeed(), 0.0)
                speed_ms = speed_kmh / 3.6
                yaw_rate = _finite(devices["gyro"].getValues()[2], 0.0)
                max_speed_kmh = max(max_speed_kmh, speed_kmh)
                position = node.getPosition()
                rotation = node.getOrientation()

                forward = (rotation[0], rotation[3], rotation[6])
                delta = (position[0] - reference_point[0], position[1] - reference_point[1])
                lat_error = (reference_direction[0] * delta[1] -
                             reference_direction[1] * delta[0])
                run["s_travelled"] = (reference_direction[0] * delta[0] +
                                      reference_direction[1] * delta[1])
                head_error = math.atan2(
                    reference_direction[0] * forward[1] - reference_direction[1] * forward[0],
                    reference_direction[0] * forward[0] + reference_direction[1] * forward[1])
                lat_error_rate = (0.0 if previous_lat_error is None
                                  else (lat_error - previous_lat_error) / control_dt)
                previous_lat_error = lat_error

                previous_phase = schedule.phase_index
                throttle, brake = schedule.update(t, speed_kmh)
                if schedule.finished:
                    break
                if schedule.phase_index != previous_phase:
                    print("longdyn_logger: phase %d/%d '%s' entered at t=%.2f s, "
                          "%d samples so far" % (schedule.phase_index + 1, len(phases),
                                                 schedule.phase_name, t, logger.rows))

                steering_command = lateral_control(lat_error, head_error, yaw_rate,
                                                   speed_ms, steering_command, control_dt)
                actuators.apply(throttle, brake, steering_command)

                logger.write(sample(car, node, devices, classifier, run, step_index,
                                    schedule, t, position, rotation, forward, lat_error,
                                    head_error, lat_error_rate, steering_command,
                                    speed_kmh, params))
                if step_index % FLUSH_INTERVAL == 0:
                    logger.flush()
                step_index += 1

                # Safety net: never let an unexpected behaviour ruin a long run silently.
                if abort_reason is None:
                    if t - start_time > MAX_RUN_TIME:
                        abort_reason = "maximum run time reached"
                    elif abs(lat_error) > MAX_LATERAL_ERROR:
                        abort_reason = "lateral error %.2f m" % lat_error
                    elif position[0] < ROAD_END_X:
                        abort_reason = "end of the road reached"
                    if abort_reason:
                        print("longdyn_logger: aborting (%s), applying full brakes"
                              % abort_reason)
                        phases = [{"name": "abort_stop", "duration": 6.0,
                                   "throttle": 0.0, "brake": 1.0,
                                   "end_speed_below": 0.5}]
                        schedule = Schedule(phases)
                        schedule.begin(t, throttle, brake)

            physics_step_index += 1
            if car.step() == -1:
                break
    except KeyboardInterrupt:
        pass

    logger.flush()
    logger.close()
    print("longdyn_logger: %d samples in %.2f s, max speed %.1f km/h%s -> %s"
          % (logger.rows, t - start_time, max_speed_kmh,
             "" if abort_reason is None else " (aborted: %s)" % abort_reason, run_path))
    print("longdyn_logger: throttle/brake overlaps prevented: %d%s"
          % (schedule.overlap_events,
             "" if not schedule.overlap_phases
             else " in " + ", ".join(sorted(schedule.overlap_phases))))

    if EXIT_ON_FINISH:
        car.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
