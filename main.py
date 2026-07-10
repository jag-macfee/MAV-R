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


def main():
    kt_airfoil = Airfoil.karman_trefftz(0.053, 0.0, np.deg2rad(15.0), 2.0, 20)
    Plotter.plot_camberline(kt_airfoil)
    Plotter.plot_points(kt_airfoil.airfoil_points)


if __name__ == "__main__":
    main()
