from typing import List, Optional
import numpy as np
from src.airfoil import Airfoil
import matplotlib.pyplot as plt


class Plotter:
    """
    Contains methods for visualising solver outputs.
    Does not contain solver logic.
    """

    @staticmethod
    def plot_history_3D(gamma: List[np.ndarray]):
        """
        Plots the full circulation or vortex strength history as a 3D visualisation.

        :param gamma: The circulation dimension (k x max_vortices x 2)
        """
        pass

    @staticmethod
    def plot_history_2D(gamma: List[np.ndarray], num_snapshots: int):
        """
        Plots selected 2D snapshots of the circulation or vortex strength distribution over time.

        :param gamma: The circulation dimension (k x max_vortices x 2)
        :param num_snapshots: Number of snapshots to select across the time history
        """
        pass

    @staticmethod
    def plot_lift_history(lift_history: List[float]):
        """
        Plots lift as a function of time.

        :param lift_history: Lift evaluated at each time step
        """
        pass

    @staticmethod
    def plot_lift_frequency_spectrum(lift_history: List[float]):
        """
        Performs a frequency-domain visualisation of the lift history (e.g. FFT).

        :param lift_history: Lift evaluated at each time step
        """
        pass

    @staticmethod
    def plot_camberline(airfoil: Airfoil, debug: bool = False):
        """Plots the camber line in Matplotlib.
        Note: This operation is blocking until the plot window is closed, and is primarily meant to be used for debugging purposes.
        If the `debug` flag is explicitly passed in as `True`, the actual points will be printed too.

        Args:
            airfoil (Airfoil): The airfoil to plot
        """
        camber_points = airfoil.camber()
        x = camber_points[:, 0]
        y = camber_points[:, 1]

        if debug:
            print(x)
            print(y)
            print(camber_points)

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Camber line for {airfoil.get_name()}")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_points(points: np.ndarray, debug: bool = False):
        """A more general method than `plot_camberline()`, this method simply plots
        a series of points which are passed in. Useful for visualising rotational transformations.

        Args:
            points (np.ndarray): (N, 2) size np.ndarray of points
            debug (bool, optional): If `True`, prints the points to terminal. Defaults to False.
        """
        x = points[:, 0]
        y = points[:, 1]

        if debug:
            print(x)
            print(y)
            print(points)

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Point plot")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
