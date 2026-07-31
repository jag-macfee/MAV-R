import numpy as np

from config import (
    N,
    U_inf,
    alpha,
    c,
    impulse_gust,
    no_vertical_gust,
    suddenly_imposed_gust,
    step_gust,
    v_0,
)
from src.airfoil import Airfoil
from src.plotter import Plotter
from src.solver import Solver


def compare_alpha_vs_imposed_gust():
    af1 = Airfoil.karman_trefftz(0.053, 0.0, np.deg2rad(15), c, N)

    sim1 = Solver(af1, None, 5 * N, (U_inf, suddenly_imposed_gust), 0.0)
    sim2 = Solver(
        af1, None, 5 * N, (U_inf, no_vertical_gust), np.rad2deg(np.asin(v_0 / U_inf))
    )

    res1 = sim1.solve()
    res2 = sim2.solve()

    Plotter.plot_lift_against_wagner(res1, sim1, v_0)
    Plotter.plot_lift_against_wagner(res2, sim2, v_0)
