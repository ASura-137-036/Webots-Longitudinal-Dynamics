"""Assert that the Python half-car port matches the Julia model that produced the parameters.

Runs with no Webots involvement, so the port can be checked on its own:

    python analysis/live/check_parity.py

Two fixtures are produced by `analysis/live/export_params.jl`:

* `rhs_fixture.json` - 200 states with the derivative Julia computes for each. This is the strong
  check: it tests the equations themselves, term by term, at points spread across the state range the
  model actually visits. A single transcription error shows up immediately.
* `trajectory_fixture.json` - a 2 s integration at the same RK4 substep the controller uses. This
  tests the integrator: a scheme that is right per step but drifts, or a substep count that does not
  match, fails here even when every individual evaluation passes.

Tolerances are tight on purpose. The two implementations are the same arithmetic in a different
language, so the only difference should be the last bits of `tanh` and the summation order - far below
1e-10. If a check fails, regenerate the fixtures first: a stale fixture after a model edit is the most
likely cause, and that is exactly the failure this test exists to surface.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLLER = os.path.abspath(os.path.join(HERE, "..", "..", "controllers", "postopt_viewer"))
sys.path.insert(0, CONTROLLER)

import halfcar_model  # noqa: E402  (path has to be set up first)

RHS_TOLERANCE = 1e-10
TRAJECTORY_TOLERANCE = 1e-9

# Optional suffix so a second exported model can be checked without overwriting the reference set:
#   python analysis/live/check_parity.py _diagnosis
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""


def load(name):
    # The suffix belongs before the extension, not after it: `"rhs_fixture.json" + "_v22"` names a file
    # that cannot exist, so suffixed fixture sets were unreachable.
    stem, extension = os.path.splitext(name)
    with open(os.path.join(CONTROLLER, stem + SUFFIX + extension), "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_rhs(parameters) -> bool:
    fixture = load("rhs_fixture.json")
    worst = 0.0
    worst_case = -1
    worst_index = -1
    for case_index, case in enumerate(fixture["cases"]):
        state = [float(value) for value in case["u"]]
        expected = [float(value) for value in case["du"]]
        # The fixture stores the pedals it was evaluated with; using anything else would compare a
        # different operating point and report it as an equation error.
        actual = halfcar_model.rhs(state, parameters,
                                   throttle=float(case["throttle"]),
                                   brake=float(case["brake"]))
        for index in range(10):
            error = abs(actual[index] - expected[index])
            if error > worst:
                worst, worst_case, worst_index = error, case_index, index
    ok = worst <= RHS_TOLERANCE
    print(f"  RHS fixture      : {len(fixture['cases'])} cases, max |error| = {worst:.3e} "
          f"({'PASS' if ok else 'FAIL'}, tol {RHS_TOLERANCE:.0e})")
    if not ok:
        print(f"    worst at case {worst_case}, state component {worst_index}")
    return ok


def check_trajectory(parameters) -> bool:
    fixture = load("trajectory_fixture.json")
    sample_period = float(fixture["sample_period"])
    intervals = int(fixture["intervals"])
    throttle = [float(value) for value in fixture["throttle"]]
    brake = [float(value) for value in fixture["brake"]]
    state = [float(value) for value in fixture["u0"]]

    # Exactly the controller's loop: hold the pedals for one measurement interval, subdivide it. The
    # fixture was produced the same way, so agreement should be at rounding level.
    for interval in range(intervals):
        state = halfcar_model.advance(state, parameters,
                                      throttle[interval], brake[interval], sample_period)

    expected = [float(value) for value in fixture["final"]]
    worst = max(abs(state[index] - expected[index]) for index in range(10))
    ok = worst <= TRAJECTORY_TOLERANCE
    print(f"  Trajectory       : {intervals} intervals x {halfcar_model.SUBSTEPS} substeps, "
          f"max |error| = {worst:.3e} ({'PASS' if ok else 'FAIL'}, tol {TRAJECTORY_TOLERANCE:.0e})")
    if not ok:
        for index in range(10):
            print(f"    state[{index}]: got {state[index]:.12f} expected {expected[index]:.12f}")
    return ok


def main() -> int:
    parameter_path = os.path.join(CONTROLLER, "params_opt_v2" + SUFFIX + ".json")
    if not os.path.exists(parameter_path):
        print(f"missing {parameter_path}; run analysis/live/export_params.jl first")
        return 2
    parameters = halfcar_model.load_parameters(parameter_path)
    print(f"model: {parameters['_label']} (fitted on {parameters['_log_file']})")
    print(f"python port in {CONTROLLER}")

    results = [check_rhs(parameters), check_trajectory(parameters)]
    if all(results):
        print("parity OK")
        return 0
    print("parity FAILED - regenerate the fixtures, then bring the port back into line")
    return 1


if __name__ == "__main__":
    sys.exit(main())
