"""
Entry point for the MAV-R unsteady airfoil solver.
"""

from src.airfoil import Airfoil
from src.vortex_strategy import WCStrategy
from src.solver import Solver
from src.plotter import Plotter
import matplotlib.pyplot as plt
from src.util import ResultUtils, SolverUtils
import numpy as np

# Q_inf setup
U_inf = 10
alpha = 0.0
N = 20  # panels

# Some gust functions worth trying
v_0 = 1.0
no_vertical_gust = lambda t: 0
suddenly_imposed_gust = lambda t: v_0
step_gust = lambda t: v_0 if t >= 0 else 0.0

c = 1.0
delta_t = c / (N * U_inf)
impulse_gust = lambda t: v_0 if 0.0 <= t < delta_t else 0.0


# Show shape of Karman Trefftz airfoil demo
def plot_karman_trefftz_example():
    af1 = Airfoil.karman_trefftz(0.053, 0.0, np.deg2rad(15.0), 1, N)
    af2 = Airfoil.karman_trefftz(0.053, 0.1, np.deg2rad(15.0), 1, N)
    joukowsky_af = Airfoil.karman_trefftz(0.1, 0.1, np.deg2rad(0.0), 1, N)

    Plotter.plot_points(af1.airfoil_points, airfoil=af1)
    Plotter.plot_points(af2.airfoil_points, airfoil=af2)
    Plotter.plot_points(joukowsky_af.airfoil_points, airfoil=joukowsky_af)


def main():
    af1 = Airfoil.karman_trefftz(0.053, 0.1, np.deg2rad(15), c, N)

    spinup_steps = 10 * N  # assume it takes 10*N time steps to hit steady state for now
    response_steps = 5 * N  # run for another 10*N steps
    total_steps = spinup_steps + response_steps

    impulse_delayed = lambda t: impulse_gust(t - delta_t * spinup_steps)

    # try 2 sims - one with no gust and one with an impulse gust ONLY after it has reached
    # a relatively steady-state
    no_gust_simulation = Solver(
        af1, None, total_steps, (U_inf, no_vertical_gust), alpha
    )
    gust_simulation = Solver(af1, None, total_steps, (U_inf, impulse_delayed), alpha)

    # run sim
    baseline_result = no_gust_simulation.solve()
    gust_result = gust_simulation.solve()

    baseline_lift = baseline_result.lift_history
    gust_lift = gust_result.lift_history

    start_index = spinup_steps
    incremental_lift = [
        gust_lift[i] - baseline_lift[i] for i in range(start_index, total_steps)
    ]

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


if __name__ == "__main__":
    main()
