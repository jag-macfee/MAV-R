import numpy as np
from abc import ABC, abstractmethod


class AirfoilUtils:
    @classmethod
    def continuous_complex_power(cls, q, m):
        """Takes in a complex value (or array of values) `q` and raises them to the power `m`
        Necessary to avoid `numpy` argument domain wrapping

        Args:
            q: Complex value (or `np.ndarray` of values)
            m: Power

        Returns:
            Complex number representing `q ** m`
        """
        r = np.abs(q)
        theta = np.unwrap(np.angle(q))
        return r**m * np.exp(1j * m * theta)

    @classmethod
    def prepare_surface_for_interp(cls, surface: np.ndarray):
        """
        Sort a surface by x-coordinate and remove duplicate x-values
        so that it can be used with np.interp. Used to find the camberline of a more
        complex airfoil (eg. Karman-Trefftz)
        """
        x = surface.real
        y = surface.imag

        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # Remove exact duplicate x values.
        # This is mainly to avoid issues at LE/TE.
        x_unique, unique_indices = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_indices]

        return x_unique, y_unique


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

    @classmethod
    def karman_trefftz(
        cls,
        epsilon: float,
        eta: float,
        delta_te: float,
        chord: float,
        n_panels: int,
    ) -> "Airfoil":
        """Returns a Von Karman-Trefftz airfoil given a series of parameters defining the complex-planed circle

        Args:
            epsilon (float): Real part of circle origin (divided by sigma)
            eta (float): Imaginary part of circle origin (divided by sigma)
            delta_te (float): Trailing edge thickness (radians)
            chord (float): Chord length (m)
            n_panels (int): Number of panels

        Returns:
            Airfoil: A Karman-Trefftz Airfoil
        """
        return KarmanTrefftzAirfoil(epsilon, eta, delta_te, chord, n_panels)


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


class KarmanTrefftzAirfoil(Airfoil):
    """
    Subclass of Airfoil representing a Von Karman-Trefftz airfoil
    Note: Specifying a trailing edge angle of 0 radians reduces the airfoil to the Joukowsky transform
    """

    def __init__(
        self,
        epsilon: float,
        eta: float = 0.0,
        delta_te: float = np.deg2rad(15.0),
        chord: float = 1.0,
        n_panels: int = 10,
    ):
        if chord <= 0:
            raise ValueError("chord must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if not (0 <= delta_te < np.pi):
            raise ValueError("delta_te should be in [0, pi) radians.")

        # Von Karman-Trefftz exponent
        m = 2.0 - delta_te / np.pi

        # Solve for sigma from Lysak's chord relation:
        #
        # c = 2 m sigma * (1 + epsilon)^m /
        #     ((1 + epsilon)^m - epsilon^m)
        sigma = (
            chord
            / (2.0 * m)
            * (((1.0 + epsilon) ** m - epsilon**m) / ((1.0 + epsilon) ** m))
        )

        # Circle centre and radius in the zeta-plane
        mu = sigma * (-epsilon + 1j * eta)
        radius = abs(sigma - mu)

        # Angle from circle centre to the trailing-edge point zeta = sigma
        theta_te = np.arctan2(eta, 1.0 + epsilon)

        # Circle parameterisation
        # The theta shift makes theta = 0 correspond to the trailing edge zeta = sigma.
        n_points = 1000  # We will take a resolution of 1000 points around the circle to transform
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        zeta = mu + radius * np.exp(1j * (theta - theta_te))

        # Von Karman-Trefftz mapping
        q = (zeta - sigma) / (zeta + sigma)
        q_m = AirfoilUtils.continuous_complex_power(q, m)

        # Lysak's x-shift, placing the chord approximately from -c/2 to c/2
        delta_x = chord / (2.0 * m * sigma) - 1.0

        # Mapped airfoil before moving the leading edge to the origin
        Z_unshifted = m * sigma * ((1.0 + q_m) / (1.0 - q_m) + delta_x)

        # Identify leading edge as the minimum x-coordinate point
        le_index = np.argmin(Z_unshifted.real)
        Z_le = Z_unshifted[le_index]

        # Shift so that leading edge is at the origin
        Z = Z_unshifted - Z_le

        # The first point corresponds to zeta = sigma, i.e. the trailing edge
        te_index = 0
        Z_te = Z[te_index]

        # Set class's own params
        # Note: The karman-trefftz airfoil will store all relevant parameters that define the pre-transformation circle in the
        # complex plane, as specified by Lysak.
        super().__init__(chord, n_panels)
        params = {
            "epsilon": epsilon,
            "eta": eta,
            "delta_te_rad": delta_te,
            "delta_te_deg": np.rad2deg(delta_te),
            "m": m,
            "chord": chord,
            "sigma": sigma,
            "mu": mu,
            "radius": radius,
            "theta_te": theta_te,
            "delta_x": delta_x,
            "leading_edge_unshifted": Z_le,
            "leading_edge_index": le_index,
            "trailing_edge": Z_te,
            "trailing_edge_index": te_index,
        }
        self.params = params

        self.zeta = zeta  # Pre-transformed points
        self.Z = Z  # Post-transformed points
        self.airfoil_points = np.column_stack((Z.real, Z.imag))  # For plotting
        self.camber_points = None

    def camber(self) -> np.ndarray:
        """
        Evaluates camber line of the Karman-Trefftz airfoil, and returns a list of
        points which represent the appropriate geometry.
        Returns a 2D array of shape (n_panels + 1, 2) containing the [x, y] coordinates
        at uniformly spaced panel endpoints along the chord.
        """
        if self.camber_points is not None:
            return self.camber_points

        Z = self.Z
        if len(Z) < 4:
            raise ValueError("Need at least 4 airfoil points to extract a camber line.")

        n_camber_points = self.n_panels + 1
        te_index = self.params["trailing_edge_index"]
        le_index = self.params["leading_edge_index"]

        # Roll so TE is at index 0
        contour = np.roll(Z, -te_index)

        # Recompute LE index in rolled coordinates unless supplied
        if le_index is None:
            le_index_rolled = np.argmin(contour.real)
        else:
            le_index_rolled = (le_index - te_index) % len(Z)

        # Split into two curves connecting LE and TE
        surface_a = contour[: le_index_rolled + 1][::-1]  # LE -> TE
        surface_b = np.concatenate((contour[le_index_rolled:], contour[:1]))  # LE -> TE

        # Decide upper/lower by mean y-position
        if np.nanmean(surface_a.imag) >= np.nanmean(surface_b.imag):
            upper_surface = surface_a
            lower_surface = surface_b
        else:
            upper_surface = surface_b
            lower_surface = surface_a

        # Prepare for interpolation
        x_upper, y_upper = AirfoilUtils.prepare_surface_for_interp(upper_surface)
        x_lower, y_lower = AirfoilUtils.prepare_surface_for_interp(lower_surface)

        # Common x-range over which both surfaces are defined
        x_min = max(np.min(x_upper), np.min(x_lower))
        x_max = min(np.max(x_upper), np.max(x_lower))

        x_common = np.linspace(x_min, x_max, n_camber_points)

        y_upper_i = np.interp(x_common, x_upper, y_upper)
        y_lower_i = np.interp(x_common, x_lower, y_lower)

        y_camber = 0.5 * (y_upper_i + y_lower_i)
        # thickness = y_upper_i - y_lower_i

        camber = x_common + 1j * y_camber
        self.camber_points = np.column_stack((camber.real, camber.imag))
        return self.camber_points

    def get_name(self):
        return f"Von Karman-Trefftz Airfoil with chord {self.c} and TE thickness {self.params["delta_te_deg"]} deg"
