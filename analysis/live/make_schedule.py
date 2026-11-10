"""Generate `schedule_postopt.json`, the driving script for the `postopt_viewer` controller.

The script is written as a story rather than as an identification grid: launch to full speed, a hard
die-down, another launch, then a train of alternating throttle/brake pulses, then a final die-down.

Why the pulses: at steady state the body sits within a couple of millimetres, so the heave and pitch
traces barely move and explain nothing at video scale. During a pulse the load transfer swings far
further, so the traces gain real amplitude and the body visibly rocks. A 1 s half-cycle is also about
one period of the pitch mode (~2 Hz), so the body follows instead of blurring.

Why the amplitudes are modest: the fitted half-car model describes a car whose four tyres stay on the
road. Full throttle from rest and full brake at speed instead lift an axle off the ground - measured
over a run at 1.0 throttle and 1.0 brake, the front axle was unloaded for 48 % of the throttle pulses
and the rear for 50 % of the die-downs - and no model of that class predicts a wheelie. At the pedal
levels below every phase stays planted, which is also what the model was fitted on.

Run:
    python analysis/live/make_schedule.py
"""

from __future__ import annotations

import json
import os

CONTROLLER = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "controllers", "postopt_viewer"))
OUTPUT = os.path.join(CONTROLLER, "schedule_postopt.json")

# Braking is wheel-speed-proportional damping, so it is strong at speed and fades as the car stops.
# `end_speed_below` ends a die-down when the car is essentially stopped, and the following hold keeps
# the brake applied so the trace settles at a flat zero instead of creeping.
STOPPED_KMH = 0.4

PULSES = 4

# Pedal levels that leave all four tyres on the road. `diedown` is also given a long duration
# because a softer brake takes longer to stop the car; `end_speed_below` cuts it short.
LAUNCH_THROTTLE = 0.55
ROLL_UP_THROTTLE = 0.35
PULSE_THROTTLE = 0.60
PULSE_BRAKE = 0.40
DIE_DOWN_BRAKE = 0.50


def build() -> list:
    phases = [
        {"name": "settle", "duration": 2.0, "throttle": 0.0, "brake": 0.0},
        # 0 -> motorway speed: squat, then a plateau at constant throttle.
        {"name": "launch_a", "duration": 3.0, "throttle": LAUNCH_THROTTLE, "brake": 0.0},
        {"name": "plateau_a", "duration": 1.0, "throttle": LAUNCH_THROTTLE, "brake": 0.0},
        {"name": "diedown_a", "duration": 8.0, "throttle": 0.0, "brake": DIE_DOWN_BRAKE,
         "end_speed_below": STOPPED_KMH},
        {"name": "rest_a", "duration": 0.6, "throttle": 0.0, "brake": DIE_DOWN_BRAKE},
        # The same input again, so the repeat is visible rather than asserted.
        {"name": "launch_b", "duration": 3.0, "throttle": LAUNCH_THROTTLE, "brake": 0.0},
        {"name": "diedown_b", "duration": 8.0, "throttle": 0.0, "brake": DIE_DOWN_BRAKE,
         "end_speed_below": STOPPED_KMH},
        {"name": "rest_b", "duration": 0.6, "throttle": 0.0, "brake": DIE_DOWN_BRAKE},
        # Moderate speed, where both squat and dive read clearly on the traces.
        {"name": "roll_up", "duration": 1.2, "throttle": ROLL_UP_THROTTLE, "brake": 0.0},
    ]
    for index in range(1, PULSES + 1):
        phases.append({"name": f"pulse_{index}_thr", "duration": 1.0,
                       "throttle": PULSE_THROTTLE, "brake": 0.0})
        phases.append({"name": f"pulse_{index}_brk", "duration": 1.0,
                       "throttle": 0.0, "brake": PULSE_BRAKE})
    phases.append({"name": "final_diedown", "duration": 8.0, "throttle": 0.0,
                   "brake": DIE_DOWN_BRAKE, "end_speed_below": STOPPED_KMH})
    phases.append({"name": "final_hold", "duration": 0.8, "throttle": 0.0,
                   "brake": DIE_DOWN_BRAKE})
    return phases


def main() -> int:
    phases = build()
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(phases, handle, indent=2)
        handle.write("\n")
    nominal = sum(float(phase["duration"]) for phase in phases)
    launches = sum(1 for phase in phases if phase["name"].startswith("launch"))
    diedowns = sum(1 for phase in phases if "diedown" in phase["name"])
    pulses = sum(1 for phase in phases if phase["name"].startswith("pulse"))
    print(f"wrote {OUTPUT}")
    print(f"  {len(phases)} phases, {nominal:.1f} s nominal "
          f"(die-downs end early once the car stops)")
    print(f"  {launches} launches, {diedowns} die-downs, {pulses} pulse half-cycles "
          f"({PULSES} full cycles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
