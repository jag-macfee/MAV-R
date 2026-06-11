import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod


class Airfoil(ABC):
    """
    Parent class/interface defining what an airfoil should look like.
    """

    def __init__(self, c: float, n_panels: int):
        """
        :param c: chord length in metres
        :param n_panels: number of panels the airfoil is split into
        """
        self.c = c
        self.n_panels = n_panels

    @abstractmethod
    def camber(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Returns an array of 2D points representing the camber line evaluated
        at each point in x.

        :param x: x-coordinates along the camber line
        :return: A 2D numpy array of shape (N, 2) where each row is [x, y]
        """
        pass

    @classmethod
    def from_code(cls, code: str, c: float, n_panels: int) -> "Airfoil":
        """
        Takes in an airfoil code, such as a NACA 4-digit or 6-digit code,
        and returns an instance of the appropriate airfoil subclass.
        """
        # A simple parser for NACA 4-digit airfoils as an example
        if len(code) == 4 and code.isdigit():
            return Naca4Digit(code, c, n_panels)

        raise ValueError(f"Unsupported airfoil code: {code}")


class Naca4Digit(Airfoil):
    """
    Subclass of Airfoil representing the NACA 4-digit airfoil series.
    """

    def __init__(self, code: str, c: float, n_panels: int):
        super().__init__(c, n_panels)
        self.code = code
        self.m = float(code[0]) / 100.0  # Maximum camber
        self.p = float(code[1]) / 10.0  # Position of maximum camber

    def camber(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Evaluates the camber line of a NACA 4-digit airfoil using the official NACA definition.
        Returns a 2D array of shape (N, 2) containing the [x, y] coordinates.
        """
        x_dim = np.asarray(x)
        y = np.zeros_like(x_dim, dtype=float)

        # Calculate the camberline based on the given sections
        for i, val in enumerate(x_dim):
            x_c = val / self.c
            if 0 <= x_c < self.p:
                y[i] = self.c * (self.m / (self.p**2)) * (2 * self.p * x_c - x_c**2)
            elif self.p <= x_c <= 1:
                y[i] = (
                    self.c
                    * (self.m / ((1 - self.p) ** 2))
                    * ((1 - 2 * self.p) + 2 * self.p * x_c - x_c**2)
                )

        return np.column_stack((x_dim, y))
