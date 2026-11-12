"""Compare the no-lumping solver response with classical unsteady-aero functions."""

from config import (
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

NUM_TIME_STEPS = 5 * N


def compare_no_lumping_functions() -> None:
    """Plot no-lumping responses against Wagner, Küssner, and Sears functions.

    The comparisons use a flat plate at zero angle of attack, which matches the
    assumptions of the classical reference functions:

    * Wagner: suddenly imposed uniform gust.
    * Küssner: sharp-edged step gust.
    * Sears: discrete impulse gust and its frequency response.
    """
    airfoil = Airfoil.flat_plate(c, N)

    wagner_solver = Solver(
        airfoil=airfoil,
        strategy=None,
        num_time_steps=NUM_TIME_STEPS,
        Q_inf=(U_inf, suddenly_imposed_gust),
        alpha=alpha,
    )
    kussner_solver = Solver(
        airfoil=airfoil,
        strategy=None,
        num_time_steps=NUM_TIME_STEPS,
        Q_inf=(U_inf, step_gust),
        alpha=alpha,
    )
    sears_solver = Solver(
        airfoil=airfoil,
        strategy=None,
        num_time_steps=NUM_TIME_STEPS,
        Q_inf=(U_inf, impulse_gust),
        alpha=alpha,
    )

    wagner_result = wagner_solver.solve()
    kussner_result = kussner_solver.solve()
    sears_result = sears_solver.solve()

    Plotter.plot_lift_against_wagner(
        wagner_result,
        wagner_solver,
        v_0,
        title="No-Lumping Solver Compared with Wagner's Function",
    )
    Plotter.plot_lift_against_kussner(
        kussner_result,
        kussner_solver,
        v_0,
        title="No-Lumping Solver Compared with Küssner's Function",
    )
    Plotter.plot_lift_frequency_spectrum(
        sears_result,
        sears_solver,
        v_0,
        title="No-Lumping Solver Compared with Sears' Function",
    )
    Plotter.plot_lift_history(
        sears_result,
        sears_solver,
        label="No-lumping solver",
        title="No-Lumping Solver Impulse Response",
    )


if __name__ == "__main__":
    compare_no_lumping_functions()
