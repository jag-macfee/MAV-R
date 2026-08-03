import numpy as np

from config import (
    N,
    U_inf,
    alpha,
    c,
    delta_t,
    impulse_gust,
    no_vertical_gust,
    suddenly_imposed_gust,
    step_gust,
    v_0,
)
from src.airfoil import Airfoil
from src.plotter import Plotter
from src.solver import Solver
from src.util import ResultUtils


def plot_cambered_response():
    af1 = Airfoil.karman_trefftz(0.053, 0.0, np.deg2rad(15), c, N)

    spinup_steps = 10 * N  # assume it takes 10*N time steps to hit steady state for now
    response_steps = 5 * N  # run for another 10*N steps
    total_steps = spinup_steps + response_steps

    impulse_delayed = lambda t: impulse_gust(t - delta_t * spinup_steps)

    # Let the cambered airfoil reacha a steady state first before applying impulse
    gust_simulation = Solver(af1, None, total_steps, (U_inf, impulse_delayed), alpha)

    # run sim
    gust_result = gust_simulation.solve()

    gust_lift = gust_result.lift_history

    start_index = spinup_steps
    incremental_lift = [gust_lift[i] for i in range(start_index, total_steps)]

    normalised_frequency, response_squared = (
        ResultUtils.extract_lift_frequency_response(
            lift_history=incremental_lift,
            delta_t=gust_simulation.delta_t,
            U_inf=U_inf,
            chord=af1.c,
            rho=1.225,
            v_0=v_0,
            max_normalised_frequency=N / 4.0,
        )
    )

    Plotter.plot_precalculated_lift_history(incremental_lift, gust_simulation)
    Plotter.plot_lift_frequency_spectrum_from_calculated_response(
        normalised_frequency, response_squared
    )
