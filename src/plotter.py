from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider

from src.airfoil import Airfoil

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert bound and wake histories into rectangular plotting arrays.

        Bound circulation is placed first in every row, followed by the wake
        circulation from the same time step. Wake rows are padded with zero
        circulation so that every row has the same length.

        A missing wake vortex has no physical position of its own. For plotting
        its zero value, the x-position is taken from the first time step at which
        that wake slot exists. This gives the zero-padded region a consistent
        horizontal location without modifying any solved vortex positions.

        Args:
            result: Result returned by ``Solver.solve()``.

        Returns:
            A tuple ``(x_history, gamma_history)``. Both arrays have shape
            ``(num_time_steps, num_bound_vortices + max_num_wake_vortices)``.

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

        # Each padded wake entry has Gamma = 0. Use the first known x-position
        # for that wake slot so the padded zero can still be displayed.
        for column_index in range(total_num_vortices):
            known_positions = x_history[
                np.isfinite(x_history[:, column_index]),
                column_index,
            ]

            if len(known_positions) == 0:
                raise ValueError(
                    f"No x-position is available for circulation column {column_index}"
                )

            x_history[
                ~np.isfinite(x_history[:, column_index]),
                column_index,
            ] = known_positions[0]

        return x_history, gamma_history

    @staticmethod
    def plot_history_3D(result: "SolveResult") -> None:
        """Plot the complete bound-and-wake circulation history in 3D.

        Each time step is represented by two traces at the same time-step value:
        a solid line with circular markers for the bound vortices, and a dotted
        line with x-markers for the wake vortices. All bound traces use one
        consistent colour and all wake traces use another, avoiding a different
        colour for every time step. Missing wake entries remain zero-padded.

        Args:
            result: Result returned by ``Solver.solve()``.
        """
        x_history, gamma_history = Plotter._prepare_combined_gamma_history(result)
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

        ax.set_xlabel("x position")
        ax.set_ylabel("Time step, k")
        ax.set_zlabel("Circulation, Gamma")
        ax.set_title("Bound and Wake Circulation Distribution over Time")
        ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_history_2D(
        result: "SolveResult",
        num_snapshots: int,
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

        Args:
            result: Result returned by ``Solver.solve()``.
            num_snapshots: Number of time-step snapshots to plot.

        Raises:
            ValueError: If ``num_snapshots`` is not positive or exceeds the
                number of solved time steps.
        """
        x_history, gamma_history = Plotter._prepare_combined_gamma_history(result)
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

        ax.set_xlabel("x position")
        ax.set_ylabel("Circulation, Gamma")
        ax.set_title("Bound and Wake Circulation at Selected Time Steps")
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
                f"{field_description.capitalize()} flow field at "
                f"k = {timestep_index + 1}{time_text}"
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
        plt.title("Point plot")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
