from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .test_config import (
    N,
    U_inf,
    alpha,
    c,
    impulse_gust,
    step_gust,
    suddenly_imposed_gust,
    v_0,
)
from src.airfoil import Airfoil
from src.plotter import Plotter
from src.solver import Solver
from src.util import ResultUtils
from src.vortex_strategy import WCStrategy

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

NUM_TIME_STEPS = 200

# Only d_min changes in this experiment.
DMIN_OVER_C_VALUES = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
MAX_VORTICES_TO_LUMP = 4
MAX_LUMPING_DISTANCE = 4.0 * (c / N)
NUM_REPEATS = 4  # repeats used to get a mean solve time

# Save plots relative to this script, not relative to the working directory.
PLOT_DIR = Path(__file__).resolve().parent / "plots" / "dmin"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Error metrics
# -----------------------------------------------------------------------------


def lift_nrmse(test_lift, reference_lift):
    """RMS lift error normalised by the range of the reference lift history."""
    test_lift = np.asarray(test_lift, dtype=float)
    reference_lift = np.asarray(reference_lift, dtype=float)

    reference_range = np.ptp(reference_lift)
    return np.sqrt(np.mean((test_lift - reference_lift) ** 2)) / reference_range


def frequency_response_error_db(test_response, reference_response):
    """RMS logarithmic error between two squared-magnitude FRFs, in dB.

    ResultUtils returns a dimensionless squared response R, so the pointwise
    error is 10*log10(R_test / R_reference).
    """
    test_response = np.asarray(test_response, dtype=float)
    reference_response = np.asarray(reference_response, dtype=float)

    pointwise_error_db = 10.0 * np.log10(test_response / reference_response)
    return np.sqrt(np.mean(pointwise_error_db**2))


def get_frequency_response(result, solver):
    """Extract the existing Lysak-normalised squared frequency response."""
    return ResultUtils.extract_lift_frequency_response(
        lift_history=result.lift_history,
        delta_t=solver.delta_t,
        U_inf=U_inf,
        chord=c,
        rho=solver.rho,
        v_0=v_0,
        max_normalised_frequency=N / 4,
    )


def get_normalised_lift(result, solver):
    """Return lift normalised by the steady thin-airfoil lift scale."""
    lift = np.asarray(result.lift_history, dtype=float)
    steady_lift = np.pi * solver.rho * c * U_inf * v_0
    return lift / steady_lift


def normalised_lift_rmse(test_result, test_solver, reference_result, reference_solver):
    """RMSE between two normalised lift histories.

    This is used for the Wagner and kussner test cases so that the error
    measures only the change introduced by wake lumping relative to the
    corresponding no-lumping solver solution.
    """
    test_lift = get_normalised_lift(test_result, test_solver)
    reference_lift = get_normalised_lift(reference_result, reference_solver)

    if test_lift.shape != reference_lift.shape:
        raise ValueError("Lift histories must have the same shape")

    return np.sqrt(np.mean((test_lift - reference_lift) ** 2))


# -----------------------------------------------------------------------------
# Reference solutions: no wake lumping
# -----------------------------------------------------------------------------


airfoil = Airfoil.flat_plate(c, N)

# Impulse reference: repeated because this is also used for timing.
reference_impulse_solver = Solver(
    airfoil=airfoil,
    strategy=None,
    num_time_steps=NUM_TIME_STEPS,
    Q_inf=(U_inf, impulse_gust),
    alpha=alpha,
)

reference_times = []
reference_impulse_result = None
for _ in range(NUM_REPEATS):
    reference_impulse_result = reference_impulse_solver.solve()
    reference_times.append(reference_impulse_result.time_taken)

reference_solve_time = np.mean(reference_times)
reference_frequency, reference_response = get_frequency_response(
    reference_impulse_result, reference_impulse_solver
)

# Wagner reference: suddenly imposed uniform gust, with no wake lumping.
reference_wagner_solver = Solver(
    airfoil=airfoil,
    strategy=None,
    num_time_steps=NUM_TIME_STEPS,
    Q_inf=(U_inf, suddenly_imposed_gust),
    alpha=alpha,
)

reference_wagner_times = []
reference_wagner_result = None
for _ in range(NUM_REPEATS):
    reference_wagner_result = reference_wagner_solver.solve()
    reference_wagner_times.append(reference_wagner_result.time_taken)

reference_wagner_solve_time = np.mean(reference_wagner_times)

# kussner reference: sharp-edged step gust, with no wake lumping.
reference_kussner_solver = Solver(
    airfoil=airfoil,
    strategy=None,
    num_time_steps=NUM_TIME_STEPS,
    Q_inf=(U_inf, step_gust),
    alpha=alpha,
)

reference_kussner_times = []
reference_kussner_result = None
for _ in range(NUM_REPEATS):
    reference_kussner_result = reference_kussner_solver.solve()
    reference_kussner_times.append(reference_kussner_result.time_taken)

reference_kussner_solve_time = np.mean(reference_kussner_times)


# -----------------------------------------------------------------------------
# d_min sweep
# -----------------------------------------------------------------------------


lift_errors = []
frequency_errors_db = []
wagner_errors = []
kussner_errors = []

impulse_solve_times = []
wagner_solve_times = []
kussner_solve_times = []

impulse_time_fractions = []
wagner_time_fractions = []
kussner_time_fractions = []

for dmin_over_c in DMIN_OVER_C_VALUES:
    strategy = WCStrategy(
        min_lumping_distance_from_af=dmin_over_c * c,
        max_vortices_to_lump=MAX_VORTICES_TO_LUMP,
        max_lumping_distance=MAX_LUMPING_DISTANCE,
    )

    # -------------------------------------------------------------------------
    # Impulse gust: FRF error, lift-history error, and timing
    # -------------------------------------------------------------------------

    impulse_solver = Solver(
        airfoil=airfoil,
        strategy=strategy,
        num_time_steps=NUM_TIME_STEPS,
        Q_inf=(U_inf, impulse_gust),
        alpha=alpha,
    )

    specific_solver_times = []
    impulse_result = None
    for _ in range(NUM_REPEATS):
        impulse_result = impulse_solver.solve()
        specific_solver_times.append(impulse_result.time_taken)

    specific_solve_time = np.mean(specific_solver_times)

    frequency, response = get_frequency_response(impulse_result, impulse_solver)

    # All cases have the same number of time steps and delta_t, so their FFT
    # frequency coordinates should be identical.
    if not np.allclose(frequency, reference_frequency):
        raise RuntimeError("Frequency grids do not match the reference solution")

    lift_error = lift_nrmse(
        impulse_result.lift_history,
        reference_impulse_result.lift_history,
    )
    frequency_error = frequency_response_error_db(response, reference_response)
    impulse_time_fraction = specific_solve_time / reference_solve_time

    # -------------------------------------------------------------------------
    # Wagner gust: compare with the no-lumping Wagner response
    # -------------------------------------------------------------------------

    wagner_solver = Solver(
        airfoil=airfoil,
        strategy=strategy,
        num_time_steps=NUM_TIME_STEPS,
        Q_inf=(U_inf, suddenly_imposed_gust),
        alpha=alpha,
    )
    specific_wagner_times = []
    wagner_result = None
    for _ in range(NUM_REPEATS):
        wagner_result = wagner_solver.solve()
        specific_wagner_times.append(wagner_result.time_taken)

    wagner_solve_time = np.mean(specific_wagner_times)
    wagner_time_fraction = wagner_solve_time / reference_wagner_solve_time

    wagner_error = normalised_lift_rmse(
        wagner_result,
        wagner_solver,
        reference_wagner_result,
        reference_wagner_solver,
    )

    # -------------------------------------------------------------------------
    # kussner gust: compare with the no-lumping kussner response
    # -------------------------------------------------------------------------

    kussner_solver = Solver(
        airfoil=airfoil,
        strategy=strategy,
        num_time_steps=NUM_TIME_STEPS,
        Q_inf=(U_inf, step_gust),
        alpha=alpha,
    )
    specific_kussner_times = []
    kussner_result = None
    for _ in range(NUM_REPEATS):
        kussner_result = kussner_solver.solve()
        specific_kussner_times.append(kussner_result.time_taken)

    kussner_solve_time = np.mean(specific_kussner_times)
    kussner_time_fraction = kussner_solve_time / reference_kussner_solve_time

    kussner_error = normalised_lift_rmse(
        kussner_result,
        kussner_solver,
        reference_kussner_result,
        reference_kussner_solver,
    )

    lift_errors.append(lift_error)
    frequency_errors_db.append(frequency_error)
    wagner_errors.append(wagner_error)
    kussner_errors.append(kussner_error)

    impulse_solve_times.append(specific_solve_time)
    wagner_solve_times.append(wagner_solve_time)
    kussner_solve_times.append(kussner_solve_time)

    impulse_time_fractions.append(impulse_time_fraction)
    wagner_time_fractions.append(wagner_time_fraction)
    kussner_time_fractions.append(kussner_time_fraction)

    print(
        f"d_min/c = {dmin_over_c:4.2f} | "
        f"lift NRMSE = {lift_error:.4e} | "
        f"FRF error = {frequency_error:.4f} dB | "
        f"Wagner RMSE = {wagner_error:.4e} | "
        f"kussner RMSE = {kussner_error:.4e}\n"
        f"    Impulse: {specific_solve_time:.3f} s, time fraction = {impulse_time_fraction:.3f} | "
        f"Wagner: {wagner_solve_time:.3f} s, time fraction = {wagner_time_fraction:.3f} | "
        f"kussner: {kussner_solve_time:.3f} s, time fraction = {kussner_time_fraction:.3f}"
    )


lift_errors = np.asarray(lift_errors)
frequency_errors_db = np.asarray(frequency_errors_db)
wagner_errors = np.asarray(wagner_errors)
kussner_errors = np.asarray(kussner_errors)

impulse_solve_times = np.asarray(impulse_solve_times)
wagner_solve_times = np.asarray(wagner_solve_times)
kussner_solve_times = np.asarray(kussner_solve_times)

impulse_time_fractions = np.asarray(impulse_time_fractions)
wagner_time_fractions = np.asarray(wagner_time_fractions)
kussner_time_fractions = np.asarray(kussner_time_fractions)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def save_plot(fig, filename):
    """Save a publication-quality PNG in the experiment plot directory."""
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")


Plotter.apply_publication_style()

# Impulse lift-history deviation from the no-lumping solution.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, lift_errors, marker="o")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel("Impulse lift NRMSE")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "impulse_lift_nrmse.png")

# Frequency-response deviation from the no-lumping solution.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, frequency_errors_db, marker="o")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel("Frequency-response RMS error [dB]")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "frequency_response_error.png")

# Wagner-response deviation from the corresponding no-lumping solution.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, wagner_errors, marker="o")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel("Wagner-response normalised lift RMSE")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "wagner_rmse.png")

# kussner-response deviation from the corresponding no-lumping solution.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, kussner_errors, marker="o")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel("Kussner-response normalised lift RMSE")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "kussner_rmse.png")

# Solver-time fraction for the impulse-gust test.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, impulse_time_fractions, marker="o")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label="No-lumping reference")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel(r"Impulse solver time fraction, $T_{WC}/T_{ref}$")
ax.legend()
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "impulse_time_fraction.png")

# Solver-time fraction for the Wagner test.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, wagner_time_fractions, marker="o")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label="No-lumping reference")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel(r"Wagner solver time fraction, $T_{WC}/T_{ref}$")
ax.legend()
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "wagner_time_fraction.png")

# Solver-time fraction for the kussner test.
fig, ax = plt.subplots()
ax.plot(DMIN_OVER_C_VALUES, kussner_time_fractions, marker="o")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label="No-lumping reference")
ax.set_xlabel(r"Minimum lumping distance, $d_{min}/c$")
ax.set_ylabel(r"Kussner solver time fraction, $T_{WC}/T_{ref}$")
ax.legend()
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "kussner_time_fraction.png")

# Accuracy-efficiency tradeoff for the FRF.
fig, ax = plt.subplots()
ax.plot(impulse_time_fractions, frequency_errors_db, marker="o")
for i, dmin_over_c in enumerate(DMIN_OVER_C_VALUES):
    ax.annotate(
        f"{dmin_over_c:g}",
        (impulse_time_fractions[i], frequency_errors_db[i]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )
ax.set_xlabel(r"Impulse solver time fraction, $T_{WC}/T_{ref}$")
ax.set_ylabel("Frequency-response RMS error [dB]")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "frequency_response_error_vs_time_fraction.png")

# Accuracy-efficiency tradeoff for the Wagner response.
fig, ax = plt.subplots()
ax.plot(wagner_time_fractions, wagner_errors, marker="o")
for i, dmin_over_c in enumerate(DMIN_OVER_C_VALUES):
    ax.annotate(
        f"{dmin_over_c:g}",
        (wagner_time_fractions[i], wagner_errors[i]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )
ax.set_xlabel(r"Wagner solver time fraction, $T_{WC}/T_{ref}$")
ax.set_ylabel("Wagner-response normalised lift RMSE")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "wagner_error_vs_time_fraction.png")

# Accuracy-efficiency tradeoff for the kussner response.
fig, ax = plt.subplots()
ax.plot(kussner_time_fractions, kussner_errors, marker="o")
for i, dmin_over_c in enumerate(DMIN_OVER_C_VALUES):
    ax.annotate(
        f"{dmin_over_c:g}",
        (kussner_time_fractions[i], kussner_errors[i]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )
ax.set_xlabel(r"Kussner solver time fraction, $T_{WC}/T_{ref}$")
ax.set_ylabel("Kussner-response normalised lift RMSE")
Plotter._style_axes(ax)
fig.tight_layout()
save_plot(fig, "kussner_error_vs_time_fraction.png")

plt.show()
