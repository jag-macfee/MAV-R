"""
Entry point for the MAV-R unsteady airfoil solver.
"""

from src.airfoil import Airfoil
from src.vortex_strategy import WCStrategy
from src.solver import Solver
from src.plotter import Plotter
import matplotlib.pyplot as plt
from src.util import SolverUtils
import numpy as np

# Q_inf setup
U_inf = 5
Wfunc = lambda t: 1
N = 100  # panels


def main():
    af1 = Airfoil.karman_trefftz(0.053, 0.1, np.deg2rad(10), 1, N)
    simulation = Solver(af1, None, 2 * N, (U_inf, Wfunc), 0.0)

    result = simulation.solve()
    Plotter.plot_flow_field(
        result=result,
        airfoil=simulation.airfoil,
        alpha=simulation.alpha,
        Q_inf=simulation.Q_inf,
        delta_t=simulation.delta_t,
    )
    Plotter.plot_history_2D(result, 5)
    Plotter.plot_history_3D(result)


if __name__ == "__main__":
    main()
