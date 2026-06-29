import numpy as np
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
        self.delta_x = c / n_panels

    @abstractmethod
    def camber(self) -> np.ndarray:
        """
        Returns an array of 2D points representing the camber line (pre-rotational transform) at
        uniformly spaced panel endpoints from x=0 to x=c.

        :return: A 2D numpy array of shape (n_panels + 1, 2) where each row is [x, y]
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Gets the name of the airfoil, ie. a NACA2412 airfoil will return "NACA2412"

        Returns:
            str: The airfoil's name (for display purposes)
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

    @classmethod
    def flat_plate(cls, c: float, n_panels: int) -> "Airfoil":
        """
        Returns a flat plate Airfoil object

        Args:
            c (float): chord length (m)
            n_panels (int): number of panels

        Returns:
            Airfoil: Flat plate Airfoil class instance
        """
        return FlatPlate(c, n_panels)


class Naca4Digit(Airfoil):
    """
    Subclass of Airfoil representing the NACA 4-digit airfoil series.
    """

    def __init__(self, code: str, c: float, n_panels: int):
        super().__init__(c, n_panels)
        self.code = code
        self.m = float(code[0]) / 100.0  # Maximum camber
        self.p = float(code[1]) / 10.0  # Position of maximum camber
        self.camber_points = None

    def camber(self) -> np.ndarray:
        """
        Evaluates the camber line of a NACA 4-digit airfoil using the official NACA definition.
        Returns a 2D array of shape (n_panels + 1, 2) containing the [x, y] coordinates
        at uniformly spaced panel endpoints along the chord.

        If this method has not been called before, calculate the camber line and set `Airfoil.camber_points` to be the result.
        Otherwise, return the already-calculated points.
        """
        if self.camber_points is not None:
            return self.camber_points

        x_dim = np.linspace(0.0, self.c, self.n_panels + 1)
        y = np.zeros_like(x_dim, dtype=float)

        # Symmetric NACA airfoil, e.g. NACA 0012
        if self.m == 0 or self.p == 0:
            return np.column_stack((x_dim, y))

        for i, val in enumerate(x_dim):
            x_c = val / self.c

            if x_c < self.p:
                y[i] = self.c * (self.m / self.p**2) * (2 * self.p * x_c - x_c**2)
            else:
                y[i] = (
                    self.c
                    * (self.m / (1 - self.p) ** 2)
                    * ((1 - 2 * self.p) + 2 * self.p * x_c - x_c**2)
                )

        self.camber_points = np.column_stack((x_dim, y))
        return self.camber_points

    def get_name(self):
        return "NACA" + self.code


class FlatPlate(Airfoil):
    """
    Subclass of Airfoil representing a flat plate
    """

    def __init__(self, c: float, n_panels: int):
        super().__init__(c, n_panels)
        self.camber_points = None

    def camber(self) -> np.ndarray:
        """
        Evaluates camber line of a flat plate (all y-values are set to 0), and returns a list of
        points which represent the flat plate geometry.
        Returns a 2D array of shape (n_panels + 1, 2) containing the [x, y] coordinates
        at uniformly spaced panel endpoints along the chord.
        """
        if self.camber_points is not None:
            return self.camber_points

        x_dim = np.linspace(0.0, self.c, self.n_panels + 1)
        y = np.zeros_like(x_dim, dtype=float)

        self.camber_points = np.column_stack((x_dim, y))
        return self.camber_points

    def get_name(self):
        return "Flat plate"
