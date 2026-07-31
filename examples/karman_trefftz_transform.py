# Show shape of Karman Trefftz airfoil demo
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


def plot_karman_trefftz_example():
    af1 = Airfoil.karman_trefftz(0.053, 0.1, np.deg2rad(15.0), 1, N)
    af2 = Airfoil.karman_trefftz(0.053, 0.1, np.deg2rad(15.0), 1, N)
    joukowsky_af = Airfoil.karman_trefftz(0.1, 0.1, np.deg2rad(0.0), 1, N)

    Plotter.plot_points(af1.airfoil_points, airfoil=af1)
    Plotter.plot_points(af2.airfoil_points, airfoil=af2)
    Plotter.plot_points(joukowsky_af.airfoil_points, airfoil=joukowsky_af)
