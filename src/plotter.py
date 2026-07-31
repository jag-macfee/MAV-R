from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider

from src.airfoil import Airfoil
from src.solver import Solver
from src.util import ResultUtils

if TYPE_CHECKING:
    from src.solver import SolveResult


class Plotter:
    """
    Contains methods for visualising solver outputs.
    Does not contain solver logic.
    """

    @staticmethod
    def _prepare_combined_gamma_history(
        result: "SolveResult",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert bound and wake histories into rectangular plotting arrays.

        Bound circulation is placed first in every row, followed by the wake
        circulation from the same time step. Wake rows are padded with zero
        circulation so that every row has the same length.

        A missing wake vortex has no physical position of its own. For plotting
        its zero value, the point is taken from the first time step at which that
        wake slot exists. This gives the zero-padded region a consistent location
        without modifying any solved vortex positions.

        Args:
            result: Result returned by ``Solver.solve()``.

        Returns:
            A tuple ``(x_history, z_history, gamma_history)``. Each array has
            shape ``(num_time_steps, num_bound_vortices +
            max_num_wake_vortices)``.

        Raises:
            ValueError: If the histories are empty, have different numbers of
                time steps, contain inconsistent bound-vortex counts, or contain
                no circulation values.
        """
        bound_history = result.bound_gamma_history
        wake_history = result.wake_gamma_history

        if len(bound_history) != len(wake_history):
            raise ValueError(
                "bound and wake histories must contain the same number of time steps"
            )
        if len(bound_history) == 0:
            raise ValueError("SolveResult contains no circulation history")

        num_bound_vortices = len(bound_history[0])
        if any(len(row) != num_bound_vortices for row in bound_history):
            raise ValueError(
                "the number of bound vortices must remain constant across time steps"
            )

        max_num_wake_vortices = max(len(row) for row in wake_history)
        total_num_vortices = num_bound_vortices + max_num_wake_vortices

        if total_num_vortices == 0:
            raise ValueError("SolveResult contains no circulation values")

        num_time_steps = len(bound_history)
        gamma_history = np.zeros(
            (num_time_steps, total_num_vortices),
            dtype=float,
        )
        x_history = np.full(
            (num_time_steps, total_num_vortices),
            np.nan,
            dtype=float,
        )
        z_history = np.full(
            (num_time_steps, total_num_vortices),
            np.nan,
            dtype=float,
        )

        for timestep_index, (bound_row, wake_row) in enumerate(
            zip(bound_history, wake_history)
        ):
            combined_row = [*bound_row, *wake_row]
            row_length = len(combined_row)

            gamma_history[timestep_index, :row_length] = [
                point_value.value for point_value in combined_row
            ]
            x_history[timestep_index, :row_length] = [
                point_value.x for point_value in combined_row
            ]
            z_history[timestep_index, :row_length] = [
                point_value.z for point_value in combined_row
            ]

        # Each padded wake entry has Gamma = 0. Use the first known point
        # for that wake slot so the padded zero can still be displayed.
        for column_index in range(total_num_vortices):
            known_mask = np.isfinite(x_history[:, column_index]) & np.isfinite(
                z_history[:, column_index]
            )

            if not np.any(known_mask):
                raise ValueError(
                    f"No position is available for circulation column {column_index}"
                )

            first_known_index = int(np.flatnonzero(known_mask)[0])
            missing_mask = ~known_mask
            x_history[missing_mask, column_index] = x_history[
                first_known_index, column_index
            ]
            z_history[missing_mask, column_index] = z_history[
                first_known_index, column_index
            ]

        return x_history, z_history, gamma_history

    @staticmethod
    def _normalise_gamma_history(
        result: "SolveResult",
        x_history: np.ndarray,
        z_history: np.ndarray,
        gamma_history: np.ndarray,
        airfoil: Airfoil,
        v_0: float,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert circulation to normalised circulation density.

        The plotted quantities are

        ``x_normalised = x_chord / c``

        and

        ``gamma_normalised = Gamma / (panel_length * v_0)``.

        ``x_chord`` is obtained by rotating each solved vortex position back
        into the pre-angle-of-attack airfoil frame and taking its chordwise
        coordinate. Actual camber-line panel lengths are used for the bound
        vortices. Each wake vortex is divided by ``airfoil.delta_x``, because one
        wake vortex is shed per time step and represents the circulation
        associated with one chordwise panel spacing.

        Args:
            result: Result returned by ``Solver.solve()``.
            x_history: Rectangular vortex x-position history in the rotated
                solver frame.
            z_history: Rectangular vortex z-position history in the rotated
                solver frame.
            gamma_history: Rectangular circulation history.
            airfoil: Airfoil used by the solver.
            v_0: Gust upwash strength used to normalise circulation density.
            alpha: Clockwise angle of attack in degrees used by the solver.

        Returns:
            A tuple ``(x_over_c, gamma_density_over_v_0)``.

        Raises:
            ValueError: If the airfoil geometry or ``v_0`` is invalid, or the
                number of airfoil panels does not match the bound history.
        """
        if not np.isfinite(v_0) or np.isclose(v_0, 0.0):
            raise ValueError("v_0 must be finite and non-zero when normalised=True")
        if not np.isfinite(airfoil.c) or airfoil.c <= 0.0:
            raise ValueError("airfoil chord must be finite and positive")
        if not np.isfinite(airfoil.delta_x) or airfoil.delta_x <= 0.0:
            raise ValueError("airfoil panel spacing must be finite and positive")

        num_bound_vortices = len(result.bound_gamma_history[0])
        camber_points = np.asarray(airfoil.camber(), dtype=float)

        if camber_points.shape != (num_bound_vortices + 1, 2):
            raise ValueError(
                "airfoil camber geometry must contain one more point than the "
                "number of bound vortices"
            )

        bound_panel_lengths = np.linalg.norm(
            np.diff(camber_points, axis=0),
            axis=1,
        )
        if np.any(~np.isfinite(bound_panel_lengths)) or np.any(
            bound_panel_lengths <= 0.0
        ):
            raise ValueError("all airfoil panel lengths must be finite and positive")

        num_wake_columns = gamma_history.shape[1] - num_bound_vortices
        wake_element_lengths = np.full(
            num_wake_columns,
            airfoil.delta_x,
            dtype=float,
        )
        element_lengths = np.concatenate([bound_panel_lengths, wake_element_lengths])

        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")

        # Solver geometry is rotated clockwise through alpha. Applying the
        # inverse rotation recovers the pre-rotation chordwise coordinate:
        # x_chord = x*cos(alpha) - z*sin(alpha).
        alpha_rad = np.deg2rad(alpha)
        x_chord = x_history * np.cos(alpha_rad) - z_history * np.sin(alpha_rad)
        x_over_c = x_chord / airfoil.c
        gamma_density_over_v_0 = gamma_history / (element_lengths[np.newaxis, :] * v_0)

        return x_over_c, gamma_density_over_v_0

    @staticmethod
    def plot_history_3D(
        result: "SolveResult",
        normalised: bool = False,
        v_0: Optional[float] = None,
        airfoil: Optional[Airfoil] = None,
        alpha: float = 0.0,
        title: Optional[str] = None,
    ) -> None:
        """Plot the complete bound-and-wake circulation history in 3D.

        Each time step is represented by two traces at the same time-step value:
        a solid line with circular markers for the bound vortices, and a dotted
        line with x-markers for the wake vortices. All bound traces use one
        consistent colour and all wake traces use another, avoiding a different
        colour for every time step. Missing wake entries remain zero-padded.

        When ``normalised`` is true, each vortex position is first rotated back
        into the pre-angle-of-attack airfoil frame. Its chordwise coordinate is
        then plotted as ``x_chord / c``, and circulation is converted to circulation density and normalised by the
        gust strength:

        ``gamma / v_0 = Gamma / (panel_length * v_0)``.

        Args:
            result: Result returned by ``Solver.solve()``.
            normalised: Plot normalised circulation density if true.
            v_0: Gust upwash strength. Required when ``normalised`` is true.
            airfoil: Airfoil used by the solver. Required when ``normalised`` is
                true so that chord and panel lengths are available.
            alpha: Clockwise angle of attack in degrees used by the solver. It is
                used to recover pre-rotation chordwise vortex positions.

        Raises:
            ValueError: If normalisation is requested without a valid ``v_0``
                and ``airfoil``.
        """
        x_history, z_history, gamma_history = Plotter._prepare_combined_gamma_history(
            result
        )

        if normalised:
            if airfoil is None:
                raise ValueError("airfoil must be supplied when normalised=True")
            if v_0 is None:
                raise ValueError("v_0 must be supplied when normalised=True")

            x_history, gamma_history = Plotter._normalise_gamma_history(
                result,
                x_history,
                z_history,
                gamma_history,
                airfoil,
                v_0,
                alpha,
            )

        num_bound_vortices = len(result.bound_gamma_history[0])

        bound_x = x_history[:, :num_bound_vortices]
        bound_gamma = gamma_history[:, :num_bound_vortices]
        wake_x = x_history[:, num_bound_vortices:]
        wake_gamma = gamma_history[:, num_bound_vortices:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for timestep_index in range(gamma_history.shape[0]):
            timestep_number = timestep_index + 1

            bound_timestep_values = np.full(
                bound_gamma.shape[1],
                timestep_number,
                dtype=float,
            )
            ax.plot(
                bound_x[timestep_index],
                bound_timestep_values,
                bound_gamma[timestep_index],
                color="tab:blue",
                linestyle="-",
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                alpha=0.65,
                label="Bound vortices" if timestep_index == 0 else None,
            )

            if wake_gamma.shape[1] > 0:
                wake_timestep_values = np.full(
                    wake_gamma.shape[1],
                    timestep_number,
                    dtype=float,
                )
                ax.plot(
                    wake_x[timestep_index],
                    wake_timestep_values,
                    wake_gamma[timestep_index],
                    color="tab:orange",
                    linestyle=":",
                    marker="x",
                    markersize=3.0,
                    linewidth=1.0,
                    alpha=0.65,
                    label="Wake vortices" if timestep_index == 0 else None,
                )

        if normalised:
            ax.set_xlabel(r"Normalised chordwise position, $x_{chord}/c$")
            ax.set_zlabel(r"Normalised circulation density, $\gamma/v_0$")
            ax.set_title(
                title
                if title is not None
                else "Normalised Bound and Wake Circulation Density over Time"
            )
        else:
            ax.set_xlabel("x position")
            ax.set_zlabel(r"Circulation, $\Gamma$")
            ax.set_title(
                title
                if title is not None
                else "Bound and Wake Circulation Distribution over Time"
            )

        ax.set_ylabel("Time step, k")
        ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_history_2D(
        result: "SolveResult",
        num_snapshots: int,
        normalised: bool = False,
        v_0: Optional[float] = None,
        airfoil: Optional[Airfoil] = None,
        alpha: float = 0.0,
        title: Optional[str] = None,
    ) -> None:
        """Plot evenly distributed bound-and-wake circulation snapshots.

        The bound and wake distributions at a given time step share the same
        colour. Bound vortices use a solid line with circular markers, while wake
        vortices use a dotted line with x-markers. Missing wake entries remain
        zero-padded.

        Snapshot time steps are selected as evenly as possible across the full
        solved time-step domain. The first and final time steps are included when
        more than one snapshot is requested. For example, four snapshots from a
        ten-step result use k = 1, 4, 7, and 10.

        When ``normalised`` is true, each vortex position is first rotated back
        into the pre-angle-of-attack airfoil frame. Its chordwise coordinate is
        then plotted as ``x_chord / c``, and circulation is converted to circulation density and normalised by the
        gust strength:

        ``gamma / v_0 = Gamma / (panel_length * v_0)``.

        Args:
            result: Result returned by ``Solver.solve()``.
            num_snapshots: Number of time-step snapshots to plot.
            normalised: Plot normalised circulation density if true.
            v_0: Gust upwash strength. Required when ``normalised`` is true.
            airfoil: Airfoil used by the solver. Required when ``normalised`` is
                true so that chord and panel lengths are available.
            alpha: Clockwise angle of attack in degrees used by the solver. It is
                used to recover pre-rotation chordwise vortex positions.

        Raises:
            ValueError: If ``num_snapshots`` is invalid, or normalisation is
                requested without a valid ``v_0`` and ``airfoil``.
        """
        x_history, z_history, gamma_history = Plotter._prepare_combined_gamma_history(
            result
        )

        if normalised:
            if airfoil is None:
                raise ValueError("airfoil must be supplied when normalised=True")
            if v_0 is None:
                raise ValueError("v_0 must be supplied when normalised=True")

            x_history, gamma_history = Plotter._normalise_gamma_history(
                result,
                x_history,
                z_history,
                gamma_history,
                airfoil,
                v_0,
                alpha,
            )

        num_time_steps = gamma_history.shape[0]
        num_bound_vortices = len(result.bound_gamma_history[0])

        if num_snapshots <= 0:
            raise ValueError("num_snapshots must be positive")
        if num_snapshots > num_time_steps:
            raise ValueError(
                "num_snapshots cannot exceed the number of time steps "
                f"({num_time_steps})"
            )

        snapshot_indices = np.rint(
            np.linspace(0, num_time_steps - 1, num_snapshots)
        ).astype(int)

        bound_x = x_history[:, :num_bound_vortices]
        bound_gamma = gamma_history[:, :num_bound_vortices]
        wake_x = x_history[:, num_bound_vortices:]
        wake_gamma = gamma_history[:, num_bound_vortices:]

        fig, ax = plt.subplots()
        timestep_handles = []

        for snapshot_index in snapshot_indices:
            timestep_number = snapshot_index + 1

            # Let Matplotlib choose one colour for this time step, then reuse it
            # for the corresponding wake trace.
            bound_line = ax.plot(
                bound_x[snapshot_index],
                bound_gamma[snapshot_index],
                linestyle="-",
                marker="o",
                markersize=4,
                linewidth=1.4,
                label=f"k = {timestep_number}",
            )[0]
            timestep_handles.append(bound_line)

            if wake_gamma.shape[1] > 0:
                ax.plot(
                    wake_x[snapshot_index],
                    wake_gamma[snapshot_index],
                    color=bound_line.get_color(),
                    linestyle=":",
                    marker="x",
                    markersize=4,
                    linewidth=1.4,
                )

        if normalised:
            ax.set_xlabel(r"Normalised chordwise position, $x_{chord}/c$")
            ax.set_ylabel(r"Normalised circulation density, $\gamma/v_0$")
            ax.set_title(
                title
                if title is not None
                else (
                    "Normalised Bound and Wake Circulation Density "
                    "at Selected Time Steps"
                )
            )
        else:
            ax.set_xlabel("x position")
            ax.set_ylabel(r"Circulation, $\Gamma$")
            ax.set_title(
                title
                if title is not None
                else "Bound and Wake Circulation at Selected Time Steps"
            )

        ax.grid(True)

        timestep_legend = ax.legend(
            handles=timestep_handles,
            title="Time step",
            loc="best",
        )
        ax.add_artist(timestep_legend)

        style_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                marker="o",
                markersize=4,
                linewidth=1.4,
                label="Bound vortices",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=":",
                marker="x",
                markersize=4,
                linewidth=1.4,
                label="Wake vortices",
            ),
        ]
        ax.legend(
            handles=style_handles,
            title="Vortex type",
            loc="upper right",
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_flow_field(
        result: "SolveResult",
        airfoil: Airfoil,
        alpha: float = 0.0,
        Q_inf: Optional[Tuple[float, Callable[[float], float]]] = None,
        delta_t: Optional[float] = None,
        grid_shape: Tuple[int, int] = (35, 25),
        x_limits: Optional[Tuple[float, float]] = None,
        z_limits: Optional[Tuple[float, float]] = None,
        core_radius: Optional[float] = None,
        animation_interval_ms: int = 150,
        title: Optional[str] = None,
    ) -> None:
        """Interactively display the velocity field over the airfoil and wake.

        The field at each time step is evaluated from every bound and wake vortex
        stored in ``result``. A slider selects the displayed time step and a
        Play/Pause button advances the slider automatically.

        The arrows use a visible head-tail design and are scaled according to
        the local velocity magnitude. To keep isolated near-vortex velocities from
        overwhelming the plot, arrow lengths are capped at the 95th percentile of
        the speed field within each displayed time step. A small visualisation-only
        vortex core is also added to avoid singular arrows directly at vortex
        locations.

        If ``Q_inf`` is supplied, the constant horizontal freestream and the
        travelling vertical gust are added to the vortex-induced velocity field.
        Otherwise, only the induced velocity from the bound and wake vortices is
        shown.

        Args:
            result:
                Result returned by ``Solver.solve()``.
            airfoil:
                Airfoil used for the simulation. Its camber line is rotated by
                ``alpha`` and drawn over the vector field.
            alpha:
                Clockwise airfoil angle of attack in degrees.
            Q_inf:
                Optional tuple ``(U_inf, W)`` matching the solver input, where
                ``U_inf`` is a positive scalar and ``W(t)`` is the vertical gust.
            delta_t:
                Solver time-step length. If omitted while ``Q_inf`` is supplied,
                it is inferred as ``airfoil.delta_x / U_inf``.
            grid_shape:
                Number of vector samples in the x and z directions respectively.
            x_limits:
                Optional fixed x-axis limits. By default the full camber line and
                all solved wake positions are included.
            z_limits:
                Optional fixed z-axis limits. By default the full camber line and
                all solved wake positions are included.
            core_radius:
                Visualisation-only point-vortex core radius. Defaults to five
                percent of the panel's chordwise spacing.
            animation_interval_ms:
                Delay between frames when the Play button is active.

        Raises:
            ValueError:
                If the result is empty, the histories do not align, plotting
                limits are invalid, or the supplied grid/freestream parameters
                are invalid.
        """
        bound_history = result.bound_gamma_history
        wake_history = result.wake_gamma_history

        if len(bound_history) != len(wake_history):
            raise ValueError(
                "bound and wake histories must contain the same number of time steps"
            )
        if len(bound_history) == 0:
            raise ValueError("SolveResult contains no circulation history")

        num_x, num_z = grid_shape
        if num_x < 2 or num_z < 2:
            raise ValueError("grid_shape entries must both be at least 2")
        if animation_interval_ms <= 0:
            raise ValueError("animation_interval_ms must be positive")

        # Rotate the original camber line using the same clockwise convention as
        # the solver geometry preprocessing.
        alpha_rad = np.deg2rad(alpha)
        rotation_matrix = np.array(
            [
                [np.cos(alpha_rad), np.sin(alpha_rad)],
                [-np.sin(alpha_rad), np.cos(alpha_rad)],
            ]
        )
        camber_points = (rotation_matrix @ airfoil.camber().T).T

        all_vortex_points = [
            point_value.point
            for row in [*bound_history, *wake_history]
            for point_value in row
        ]
        all_geometry_points = np.vstack(
            [camber_points, np.asarray(all_vortex_points, dtype=float)]
        )

        def automatic_limits(
            values: np.ndarray,
            requested_limits: Optional[Tuple[float, float]],
            minimum_span: float,
        ) -> Tuple[float, float]:
            if requested_limits is not None:
                lower, upper = requested_limits
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    raise ValueError("plotting limits must be finite and increasing")
                return float(lower), float(upper)

            lower = float(np.min(values))
            upper = float(np.max(values))
            span = upper - lower

            if span < minimum_span:
                centre = 0.5 * (lower + upper)
                lower = centre - 0.5 * minimum_span
                upper = centre + 0.5 * minimum_span
                span = minimum_span

            margin = 0.10 * span
            return lower - margin, upper + margin

        x_min, x_max = automatic_limits(
            all_geometry_points[:, 0],
            x_limits,
            minimum_span=airfoil.c,
        )
        z_min, z_max = automatic_limits(
            all_geometry_points[:, 1],
            z_limits,
            minimum_span=0.75 * airfoil.c,
        )

        x_values = np.linspace(x_min, x_max, num_x)
        z_values = np.linspace(z_min, z_max, num_z)
        X, Z = np.meshgrid(x_values, z_values)

        if core_radius is None:
            core_radius = 0.05 * airfoil.delta_x
        if core_radius <= 0 or not np.isfinite(core_radius):
            raise ValueError("core_radius must be finite and positive")

        if Q_inf is not None:
            U_inf, Wfunc = Q_inf
            if U_inf <= 0 or not np.isfinite(U_inf):
                raise ValueError("Q_inf[0] must be finite and strictly positive")
            if delta_t is None:
                delta_t = airfoil.delta_x / U_inf
            if delta_t <= 0 or not np.isfinite(delta_t):
                raise ValueError("delta_t must be finite and positive")
        elif delta_t is not None:
            if delta_t <= 0 or not np.isfinite(delta_t):
                raise ValueError("delta_t must be finite and positive")

        def evaluate_gust(argument: np.ndarray) -> np.ndarray:
            """Evaluate W(t) for scalar-only or NumPy-aware callables."""
            if Q_inf is None:
                return np.zeros_like(argument, dtype=float)

            try:
                values = np.asarray(Wfunc(argument), dtype=float)
                if values.shape == ():
                    return np.full_like(argument, float(values), dtype=float)
                return np.broadcast_to(values, argument.shape).astype(float, copy=False)
            except (TypeError, ValueError):
                vectorised = np.vectorize(Wfunc, otypes=[float])
                return vectorised(argument)

        def velocity_field(timestep_index: int) -> Tuple[np.ndarray, np.ndarray]:
            vortices = [
                *bound_history[timestep_index],
                *wake_history[timestep_index],
            ]

            vortex_points = np.asarray(
                [point_value.point for point_value in vortices],
                dtype=float,
            )
            gamma = np.asarray(
                [point_value.value for point_value in vortices],
                dtype=float,
            )

            dx = X[..., np.newaxis] - vortex_points[:, 0]
            dz = Z[..., np.newaxis] - vortex_points[:, 1]
            radius_squared = dx**2 + dz**2 + core_radius**2

            # This follows SolverUtils.vel_induced_by_vortex():
            # [u, w] = Gamma / (2*pi*r^2) * [dz, -dx].
            U = np.sum(
                gamma * dz / (2.0 * np.pi * radius_squared),
                axis=-1,
            )
            W = np.sum(
                -gamma * dx / (2.0 * np.pi * radius_squared),
                axis=-1,
            )

            if Q_inf is not None:
                time = delta_t * timestep_index
                U = U + U_inf
                W = W + evaluate_gust(time - X / U_inf)

            return U, W

        # Set the longest displayed arrow relative to the vector-grid spacing.
        # This keeps arrows readable without excessive overlap.
        grid_spacing_x = (x_max - x_min) / (num_x - 1)
        grid_spacing_z = (z_max - z_min) / (num_z - 1)
        maximum_arrow_length = 0.80 * min(grid_spacing_x, grid_spacing_z)

        def display_vectors(
            U: np.ndarray, W: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Scale arrow lengths with speed while preserving flow direction.

            Speeds up to the 95th percentile are mapped linearly to arrow length.
            Larger values are capped so that points close to a vortex do not make
            the rest of the field unreadable.
            """
            speed = np.hypot(U, W)
            finite_positive_speed = speed[np.isfinite(speed) & (speed > 0.0)]

            if finite_positive_speed.size == 0:
                return np.zeros_like(U), np.zeros_like(W)

            reference_speed = float(np.percentile(finite_positive_speed, 95.0))
            if reference_speed <= 0.0 or not np.isfinite(reference_speed):
                return np.zeros_like(U), np.zeros_like(W)

            clipped_speed = np.minimum(speed, reference_speed)
            direction_factor = np.divide(
                clipped_speed,
                speed,
                out=np.zeros_like(speed),
                where=speed > 0.0,
            )
            plot_scale = maximum_arrow_length / reference_speed

            return U * direction_factor * plot_scale, W * direction_factor * plot_scale

        def marker_sizes(row) -> np.ndarray:
            if len(row) == 0:
                return np.empty((0,), dtype=float)

            strengths = np.abs(
                np.asarray([point_value.value for point_value in row], dtype=float)
            )
            largest = float(np.max(strengths))
            if largest == 0.0:
                return np.full(len(row), 25.0)
            return 25.0 + 75.0 * strengths / largest

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.subplots_adjust(bottom=0.20)

        initial_U, initial_W = velocity_field(0)
        initial_U_display, initial_W_display = display_vectors(initial_U, initial_W)

        quiver = ax.quiver(
            X,
            Z,
            initial_U_display,
            initial_W_display,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="tail",
            color="green",
            width=0.004,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.0,
            minshaft=1.5,
            minlength=0.1,
            zorder=2,
        )

        ax.plot(
            camber_points[:, 0],
            camber_points[:, 1],
            linewidth=2.0,
            label="Camber line",
        )

        initial_bound = bound_history[0]
        initial_wake = wake_history[0]

        bound_scatter = ax.scatter(
            [point_value.x for point_value in initial_bound],
            [point_value.z for point_value in initial_bound],
            s=marker_sizes(initial_bound),
            marker="o",
            label="Bound vortices",
            zorder=3,
        )
        wake_scatter = ax.scatter(
            [point_value.x for point_value in initial_wake],
            [point_value.z for point_value in initial_wake],
            s=marker_sizes(initial_wake),
            marker="x",
            label="Wake vortices",
            zorder=3,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x position")
        ax.set_ylabel("z position")
        ax.grid(True)
        ax.legend(loc="upper right")

        slider_axis = fig.add_axes([0.16, 0.075, 0.62, 0.035])
        timestep_slider = Slider(
            slider_axis,
            "Time step k",
            1,
            len(bound_history),
            valinit=1,
            valstep=1,
        )

        button_axis = fig.add_axes([0.82, 0.06, 0.10, 0.06])
        play_button = Button(button_axis, "Play")

        def update_plot(timestep_number: float) -> None:
            timestep_index = int(timestep_number) - 1
            U, W = velocity_field(timestep_index)
            U_display, W_display = display_vectors(U, W)
            quiver.set_UVC(U_display, W_display)

            bound_row = bound_history[timestep_index]
            wake_row = wake_history[timestep_index]

            bound_scatter.set_offsets(
                np.asarray(
                    [point_value.point for point_value in bound_row], dtype=float
                )
            )
            bound_scatter.set_sizes(marker_sizes(bound_row))

            wake_scatter.set_offsets(
                np.asarray([point_value.point for point_value in wake_row], dtype=float)
            )
            wake_scatter.set_sizes(marker_sizes(wake_row))

            time_text = ""
            if delta_t is not None:
                time_text = f", t = {delta_t * timestep_index:.4g} s"

            field_description = "total" if Q_inf is not None else "vortex-induced"
            ax.set_title(
                title
                if title is not None
                else (
                    f"{field_description.capitalize()} flow field at "
                    f"k = {timestep_index + 1}{time_text}"
                )
            )
            fig.canvas.draw_idle()

        timestep_slider.on_changed(update_plot)

        timer = fig.canvas.new_timer(interval=animation_interval_ms)
        playback_state = {"playing": False}

        def advance_frame() -> None:
            next_timestep = int(timestep_slider.val) + 1
            if next_timestep > len(bound_history):
                next_timestep = 1
            timestep_slider.set_val(next_timestep)

        timer.add_callback(advance_frame)

        def toggle_playback(_event) -> None:
            if playback_state["playing"]:
                timer.stop()
                play_button.label.set_text("Play")
            else:
                timer.start()
                play_button.label.set_text("Pause")
            playback_state["playing"] = not playback_state["playing"]

        play_button.on_clicked(toggle_playback)
        update_plot(1)
        plt.show()

    @staticmethod
    def plot_results_against_reference_data(
        reference_data: Sequence[Tuple[Union[str, Path], str]],
        calculated_data: Sequence[
            Union[
                Line2D,
                Tuple[Line2D, str],
                Tuple[Sequence[float], Sequence[float]],
                Tuple[Sequence[float], Sequence[float], str],
            ]
        ],
        x_label: str,
        y_label: str,
        title: Optional[str] = None,
        link_colours_by_index: bool = True,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot calculated curves against reference curves stored in CSV files.

        Each reference entry is ``(csv_path, label)``. The CSV file must contain
        columns named ``x`` and ``y`` and represents one reference curve.

        Each calculated entry may be one of:

        - a Matplotlib ``Line2D`` object;
        - ``(Line2D, label)``;
        - ``(x_values, y_values)``; or
        - ``(x_values, y_values, label)``.

        Reference curves use dashed lines and calculated curves use solid lines.
        When ``link_colours_by_index`` is true, reference curve ``i`` and
        calculated curve ``i`` share a colour.

        Args:
            reference_data:
                One or more ``(csv_path, label)`` tuples. Every CSV must contain
                columns named ``x`` and ``y``.
            calculated_data:
                Any number of calculated curves supplied as ``Line2D`` objects
                or x/y data tuples.
            x_label:
                Label for the horizontal axis.
            y_label:
                Label for the vertical axis.
            title:
                Plot title.
            link_colours_by_index:
                Match the colours of reference and calculated curves by their
                positions in the two lists.
            show:
                Display the figure immediately when true.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axes.

        Raises:
            ValueError:
                If reference or calculated curve definitions are invalid, or a
                CSV does not contain valid ``x`` and ``y`` columns.
            FileNotFoundError:
                If a reference CSV file does not exist.
        """
        references = list(reference_data)
        calculated_curves = list(calculated_data)

        if len(references) == 0:
            raise ValueError("at least one reference curve must be supplied")

        colour_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(colour_cycle) == 0:
            colour_cycle = [None]

        def colour_for(index: int, calculated: bool) -> Optional[str]:
            if link_colours_by_index:
                colour_index = index
            elif calculated:
                colour_index = len(references) + index
            else:
                colour_index = index

            return colour_cycle[colour_index % len(colour_cycle)]

        def validate_curve(
            x_values,
            y_values,
            description: str,
        ) -> Tuple[np.ndarray, np.ndarray]:
            x = np.atleast_1d(np.asarray(x_values, dtype=float))
            y = np.atleast_1d(np.asarray(y_values, dtype=float))

            if x.ndim != 1 or y.ndim != 1:
                raise ValueError(
                    f"{description} x and y values must be one-dimensional"
                )
            if x.size == 0:
                raise ValueError(f"{description} must contain at least one point")
            if x.size != y.size:
                raise ValueError(
                    f"{description} x and y values must have the same length"
                )
            if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
                raise ValueError(f"{description} must contain only finite values")

            return x, y

        fig, ax = plt.subplots()

        for index, reference in enumerate(references):
            if not isinstance(reference, tuple) or len(reference) != 2:
                raise ValueError(
                    "each reference curve must be supplied as (csv_path, label)"
                )

            path = Path(reference[0])
            label = str(reference[1])

            if not path.is_file():
                raise FileNotFoundError(f"reference CSV file not found: {path}")

            table = np.genfromtxt(
                path,
                delimiter=",",
                names=True,
                dtype=float,
                encoding="utf-8",
            )

            column_names = table.dtype.names
            if (
                column_names is None
                or "x" not in column_names
                or "y" not in column_names
            ):
                raise ValueError(
                    f"reference CSV file '{path}' must contain columns named 'x' and 'y'"
                )

            x, y = validate_curve(
                table["x"],
                table["y"],
                f"reference curve '{path}'",
            )

            ax.plot(
                x,
                y,
                color=colour_for(index, calculated=False),
                linestyle="--",
                linewidth=1.5,
                label=label,
            )

        for index, curve in enumerate(calculated_curves):
            if isinstance(curve, Line2D):
                x_values = curve.get_xdata()
                y_values = curve.get_ydata()
                label = curve.get_label()
                if not label or label.startswith("_"):
                    label = f"Calculated {index + 1}"
            elif (
                isinstance(curve, tuple)
                and len(curve) == 2
                and isinstance(curve[0], Line2D)
            ):
                x_values = curve[0].get_xdata()
                y_values = curve[0].get_ydata()
                label = str(curve[1])
            else:
                if not isinstance(curve, tuple) or len(curve) not in (2, 3):
                    raise ValueError(
                        "each calculated curve must be a Line2D object, "
                        "(Line2D, label), (x_values, y_values), or "
                        "(x_values, y_values, label)"
                    )

                x_values = curve[0]
                y_values = curve[1]
                label = str(curve[2]) if len(curve) == 3 else f"Calculated {index + 1}"

            x, y = validate_curve(
                x_values,
                y_values,
                f"calculated curve {index + 1}",
            )

            ax.plot(
                x,
                y,
                color=colour_for(index, calculated=True),
                linestyle="-",
                linewidth=1.5,
                label=label,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def _plot_lift_series(
        lift_history: List[float],
        solver: "Solver",
        title: str,
        ylabel: str,
        label: Optional[str] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a lift history against normalised convective time."""
        lift = np.asarray(lift_history, dtype=float)

        if lift.ndim != 1 or lift.size == 0:
            raise ValueError("lift history must be a non-empty one-dimensional array")

        # k = 1 corresponds to t = 0 in the solver/report convention.
        time = float(solver.delta_t) * np.arange(lift.size, dtype=float)
        normalised_time = time * float(solver.Q_inf[0]) / float(solver.airfoil.c)

        fig, ax = plt.subplots()
        ax.plot(normalised_time, lift, linewidth=1.5, label=label)
        ax.axhline(0.0, linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_xlabel(r"Normalised time, $tU_\infty/c$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        if label is not None:
            ax.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_lift_history(
        result: "SolveResult",
        solver: "Solver",
        use_coefficient: bool = False,
        label: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot total lift or lift coefficient against normalised convective time."""
        if result.lift_history is None:
            raise ValueError("SolveResult does not contain a total lift history")

        lift_history = result.lift_history
        default_title = "Unsteady Total Lift History"
        ylabel = r"Total lift per unit span, $L$ [N/m]"

        if use_coefficient:
            dynamic_pressure_chord = (
                0.5
                * float(solver.rho)
                * float(solver.Q_inf[0]) ** 2
                * float(solver.airfoil.c)
            )
            lift_history = (
                np.asarray(lift_history, dtype=float) / dynamic_pressure_chord
            )
            default_title = "Unsteady Total Lift Coefficient History"
            ylabel = r"Total lift coefficient, $C_L$"

        return Plotter._plot_lift_series(
            lift_history,
            solver,
            title=title if title is not None else default_title,
            ylabel=ylabel,
            label=label,
            show=show,
        )

    @staticmethod
    def plot_precalculated_lift_history(
        history: List[float],
        solver: "Solver",
        label: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot total lift against normalised convective time."""
        return Plotter._plot_lift_series(
            history,
            solver,
            title=title if title is not None else "Unsteady Total Lift History",
            ylabel=r"Total lift per unit span, $L$ [N/m]",
            label=label,
            show=show,
        )

    @staticmethod
    def plot_circulatory_lift_history(
        result: "SolveResult",
        solver: "Solver",
        label: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot circulatory lift against normalised convective time."""
        if result.circulatory_lift_history is None:
            raise ValueError("SolveResult does not contain a circulatory lift history")

        return Plotter._plot_lift_series(
            result.circulatory_lift_history,
            solver,
            title=title if title is not None else "Circulatory Lift History",
            ylabel=r"Circulatory lift per unit span, $L_{circ}$ [N/m]",
            label=label,
            show=show,
        )

    @staticmethod
    def plot_noncirculatory_lift_history(
        result: "SolveResult",
        solver: "Solver",
        label: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot non-circulatory lift against normalised convective time."""
        if result.non_circulatory_lift_history is None:
            raise ValueError(
                "SolveResult does not contain a non-circulatory lift history"
            )

        return Plotter._plot_lift_series(
            result.non_circulatory_lift_history,
            solver,
            title=title if title is not None else "Non-Circulatory Lift History",
            ylabel=r"Non-circulatory lift per unit span, $L_{noncirc}$ [N/m]",
            label=label,
            show=show,
        )

    @staticmethod
    def plot_lift_against_wagner(
        result: "SolveResult",
        solver: "Solver",
        v_0: float,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        r"""Compare normalised circulatory lift against Jones' Wagner approximation.

        Numerical lift is normalised by the thin-airfoil steady-state value

            L_steady = pi * rho * c * U_inf * v_0

        and time is normalised as

            t_star = t * U_inf / c.

        Args:
            result:
                Result returned by ``solver.solve()``.
            solver:
                Solver instance used to generate ``result``.
            v_0:
                Imposed uniform upwash magnitude.

        Returns:
            Matplotlib figure and axes.
        """
        lift = np.asarray(
            result.lift_history,
            dtype=float,
        )

        U_inf = float(solver.Q_inf[0])
        chord = float(solver.airfoil.c)
        rho = float(solver.rho)
        delta_t = float(solver.delta_t)

        # k = 1 corresponds to t = 0.
        time = delta_t * np.arange(lift.size, dtype=float)
        normalised_time = time * U_inf / chord

        steady_lift = np.pi * rho * chord * U_inf * v_0
        normalised_lift = lift / steady_lift

        # Jones uses s = 2 U_inf t / c = 2 t_star.
        dense_time = np.linspace(
            normalised_time[0],
            normalised_time[-1],
            1000,
        )
        wagner_jones = (
            1.0
            - 0.165 * np.exp(-0.091 * dense_time)
            - 0.335 * np.exp(-0.600 * dense_time)
        )

        fig, ax = plt.subplots()

        ax.plot(
            normalised_time,
            normalised_lift,
            label="Panel-method solution",
            linewidth=1.5,
        )
        ax.plot(
            dense_time,
            wagner_jones,
            label="Jones' Wagner approximation",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel(r"Normalised time, $tU_\infty/c$")
        ax.set_ylabel(
            r"Normalised lift, " r"$L_{\mathrm{circ}}/(\pi\rho cU_\infty v_0)$"
        )
        ax.set_ylim(top=1.0)
        ax.set_title(
            title if title is not None else "Lift Compared with Wagner's Function"
        )
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig, ax

    @staticmethod
    def plot_lift_against_kussner(
        result: "SolveResult",
        solver: "Solver",
        v_0: float,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        r"""Compare a sharp-edged step-gust response with Küssner's function.

        The numerical lift is normalised by

        ``L_steady = pi * rho * c * U_inf * v_0``

        and time is plotted as ``t_star = t * U_inf / c``. Kier's reduced time
        is ``tau = 2 * U_inf * t / c = 2 * t_star``, and the approximation used is

        ``Psi(tau) = 1 - 0.5 exp(-0.13 tau) - 0.5 exp(-tau)``.

        The supplied simulation should use a sharp-edged gust such as
        ``W(t) = v_0`` for ``t >= 0`` and zero otherwise. For direct comparison
        with classical Küssner theory, use a flat plate at zero angle of attack.
        """
        if result.lift_history is None:
            raise ValueError("SolveResult does not contain a lift history")

        lift = np.asarray(
            result.lift_history,
            dtype=float,
        )
        if lift.ndim != 1 or lift.size == 0:
            raise ValueError(
                "result.lift_history must be a non-empty " "one-dimensional array"
            )

        delta_t = float(solver.delta_t)
        U_inf = float(solver.Q_inf[0])
        chord = float(solver.airfoil.c)
        rho = float(solver.rho)
        v_0 = float(v_0)

        steady_lift = np.pi * rho * chord * U_inf * v_0
        if not np.isfinite(steady_lift) or np.isclose(steady_lift, 0.0):
            raise ValueError("pi * rho * c * U_inf * v_0 must be finite and non-zero")

        # k = 1 corresponds to t = 0 in the solver/report convention.
        time = delta_t * np.arange(lift.size, dtype=float)
        normalised_time = time * U_inf / chord
        normalised_lift = lift / steady_lift

        comparison_time = np.linspace(
            normalised_time[0],
            normalised_time[-1],
            1000,
        )

        # Kier (2005) uses tau = 2 U_inf t / c.
        tau = 2.0 * comparison_time
        kussner = 1.0 - 0.5 * np.exp(-0.13 * tau) - 0.5 * np.exp(-1.0 * tau)

        fig, ax = plt.subplots()
        ax.plot(
            normalised_time,
            normalised_lift,
            linewidth=1.5,
            label="Panel-method step-gust response",
        )
        ax.plot(
            comparison_time,
            kussner,
            linestyle="--",
            linewidth=1.5,
            label="Kier's Küssner approximation",
        )

        ax.set_xlabel(r"Normalised time, $tU_\infty/c$")
        ax.set_ylabel(
            r"Normalised lift, " r"$L_{\mathrm{circ}}/(\pi\rho cU_\infty v_0)$"
        )
        ax.set_title(
            title
            if title is not None
            else "Step-Gust Lift Compared with Küssner's Function"
        )
        ax.set_ylim(top=1.0)
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig, ax

    @staticmethod
    def plot_lift_frequency_spectrum(
        result: "SolveResult",
        solver: "Solver",
        v_0: float,
        apply_lysak_accuracy_limit: bool = True,
        x_limits: Tuple[float, float] = (1.0e-2, 1.0e2),
        y_limits: Tuple[float, float] = (1.0e-4, 1.0e1),
        show: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        r"""Plot the impulse-derived lift response against the Sears approximation.

        The extraction itself is delegated to
        :meth:`ResultUtils.extract_lift_frequency_response`. Both axes are
        logarithmic. The horizontal coordinate is ``f*c/U_inf`` and the
        vertical coordinate is the paper-equivalent form

        ``|DFT[L_k]|^2 / (pi*rho*c*v_0*U_inf)^2``.

        Equivalently, after dividing by the discrete impulse transform, this is
        ``|H(f)|^2 / (pi*rho*c*U_inf)^2``.

        for the existing discrete impulse ``v_0*delta(k - 1)``.

        Lysak notes that an ``N``-panel discrete-vortex solution begins to lose
        accuracy above ``f*c/U_inf = N/4`` despite its Nyquist limit of ``N/2``.
        That recommended limit is applied by default.

        Args:
            result:
                Result returned by ``solver.solve()`` for the impulse gust.
            solver:
                Solver instance used to generate ``result``.
            v_0:
                Amplitude multiplying the user's existing discrete impulse.
            apply_lysak_accuracy_limit:
                Restrict the numerical response to ``N/4`` when true.
            x_limits:
                Lower and upper limits for normalised frequency. Defaults to
                ``(1e-2, 1e2)``.
            y_limits:
                Lower and upper limits for the normalised squared response.
                Defaults to ``(1e-4, 1e1)``.
            show:
                Display the figure immediately when true.

        Returns:
            A tuple ``(fig, ax)`` containing the Matplotlib figure and axes.
        """
        if result.lift_history is None:
            raise ValueError("SolveResult does not contain a total lift history")

        maximum_frequency = None
        if apply_lysak_accuracy_limit:
            maximum_frequency = float(solver.airfoil.n_panels) / 4.0

        normalised_frequency, normalised_response_squared = (
            ResultUtils.extract_lift_frequency_response(
                lift_history=result.lift_history,
                delta_t=solver.delta_t,
                U_inf=solver.Q_inf[0],
                chord=solver.airfoil.c,
                rho=solver.rho,
                v_0=v_0,
                max_normalised_frequency=maximum_frequency,
            )
        )

        x_min, x_max = map(float, x_limits)
        y_min, y_max = map(float, y_limits)

        if x_min <= 0.0 or x_max <= x_min:
            raise ValueError("x_limits must be positive and increasing")
        if y_min <= 0.0 or y_max <= y_min:
            raise ValueError("y_limits must be positive and increasing")

        # Draw the analytical approximation across the complete requested
        # plotting range, even if the numerical N/4 accuracy limit is lower.
        sears_frequency = np.logspace(
            np.log10(x_min),
            np.log10(x_max),
            500,
        )
        sears_response_squared = ResultUtils.sears_response_squared_approximation(
            sears_frequency
        )

        fig, ax = plt.subplots()
        ax.loglog(
            normalised_frequency,
            normalised_response_squared,
            linestyle="none",
            marker="o",
            markersize=3.5,
            label="Panel-method impulse response",
        )
        ax.loglog(
            sears_frequency,
            sears_response_squared,
            linestyle="--",
            linewidth=1.5,
            label="Sears approximation",
        )

        ax.set_xlabel(r"Normalised frequency, $fc/U_\infty$")
        ax.set_ylabel(
            r"Normalised lift response, "
            r"$|\mathrm{DFT}(L)|^2/(\pi\rho c v_0 U_\infty)^2$"
        )
        ax.set_title(
            title
            if title is not None
            else "Lift Frequency Response Compared with the Sears Function"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both")
        ax.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_lift_frequency_spectrum_from_calculated_response(
        normalised_frequency,
        normalised_response_squared,
        x_limits: Tuple[float, float] = (1.0e-2, 1.0e2),
        y_limits: Tuple[float, float] = (1.0e-4, 1.0e1),
        show: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        r"""Plot the impulse-derived lift response against the Sears approximation, but this time with given axes values"""
        x_min, x_max = map(float, x_limits)
        y_min, y_max = map(float, y_limits)

        if x_min <= 0.0 or x_max <= x_min:
            raise ValueError("x_limits must be positive and increasing")
        if y_min <= 0.0 or y_max <= y_min:
            raise ValueError("y_limits must be positive and increasing")

        # Draw the analytical approximation across the complete requested
        # plotting range, even if the numerical N/4 accuracy limit is lower.
        sears_frequency = np.logspace(
            np.log10(x_min),
            np.log10(x_max),
            500,
        )
        sears_response_squared = ResultUtils.sears_response_squared_approximation(
            sears_frequency
        )

        fig, ax = plt.subplots()
        ax.loglog(
            normalised_frequency,
            normalised_response_squared,
            linestyle="none",
            marker="o",
            markersize=3.5,
            label="Panel-method impulse response",
        )
        ax.loglog(
            sears_frequency,
            sears_response_squared,
            linestyle="--",
            linewidth=1.5,
            label="Sears approximation",
        )

        ax.set_xlabel(r"Normalised frequency, $fc/U_\infty$")
        ax.set_ylabel(
            r"Normalised lift response, "
            r"$|\mathrm{DFT}(L)|^2/(\pi\rho c v_0 U_\infty)^2$"
        )
        ax.set_title(
            title
            if title is not None
            else "Lift Frequency Response Compared with the Sears Function"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both")
        ax.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_camberline(
        airfoil: Airfoil,
        debug: bool = False,
        title: Optional[str] = None,
    ):
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
        plt.title(
            title if title is not None else f"Camber line for {airfoil.get_name()}"
        )
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_points(
        points: np.ndarray,
        debug: bool = False,
        airfoil: Optional[Airfoil] = None,
        title: Optional[str] = None,
    ):
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

        if title is None:
            title = airfoil.get_name() if airfoil else "Point plot"

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.axis("equal")
        plt.grid(True)
        plt.show()
