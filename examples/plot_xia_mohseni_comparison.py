from matplotlib import pyplot as plt
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


def plot_xia_mohseni_comparison():
    af1 = Airfoil.karman_trefftz(0.053, 0.0, np.deg2rad(15), c, N)

    sim_2deg = Solver(af1, None, 5 * N, (U_inf, no_vertical_gust), 2.0)
    sim_4deg = Solver(af1, None, 5 * N, (U_inf, no_vertical_gust), 4.0)
    sim_6deg = Solver(af1, None, 5 * N, (U_inf, no_vertical_gust), 6.0)
    sim_8deg = Solver(af1, None, 5 * N, (U_inf, no_vertical_gust), 8.0)
    sim_10deg = Solver(af1, None, 5 * N, (U_inf, no_vertical_gust), 10.0)

    result_2deg = sim_2deg.solve()
    result_4deg = sim_4deg.solve()
    result_6deg = sim_6deg.solve()
    result_8deg = sim_8deg.solve()
    result_10deg = sim_10deg.solve()

    f1, a1 = Plotter.plot_lift_history(
        result_2deg, sim_2deg, use_coefficient=True, show=False
    )
    f2, a2 = Plotter.plot_lift_history(
        result_4deg, sim_4deg, use_coefficient=True, show=False
    )
    f3, a3 = Plotter.plot_lift_history(
        result_6deg, sim_6deg, use_coefficient=True, show=False
    )
    f4, a4 = Plotter.plot_lift_history(
        result_8deg, sim_8deg, use_coefficient=True, show=False
    )
    f5, a5 = Plotter.plot_lift_history(
        result_10deg, sim_10deg, use_coefficient=True, show=False
    )

    reference_data = [
        ("static/data/xia_mohseni_cl_alpha_2deg.csv", r"Reference, $\alpha=2^\circ$"),
        ("static/data/xia_mohseni_cl_alpha_4deg.csv", r"Reference, $\alpha=2^\circ$"),
        ("static/data/xia_mohseni_cl_alpha_6deg.csv", r"Reference, $\alpha=2^\circ$"),
        ("static/data/xia_mohseni_cl_alpha_8deg.csv", r"Reference, $\alpha=2^\circ$"),
        ("static/data/xia_mohseni_cl_alpha_10deg.csv", r"Reference, $\alpha=2^\circ$"),
    ]

    calculated_data = [
        (a1.lines[0], r"Calculated, $\alpha=2^\circ$"),
        (a2.lines[0], r"Calculated, $\alpha=4^\circ$"),
        (a3.lines[0], r"Calculated, $\alpha=6^\circ$"),
        (a4.lines[0], r"Calculated, $\alpha=8^\circ$"),
        (a5.lines[0], r"Calculated, $\alpha=10^\circ$"),
    ]

    fig, ax = Plotter.plot_results_against_reference_data(
        reference_data,
        calculated_data,
        x_label=r"Normalised time, $tU_\infty/c$",
        y_label=r"Lift coefficient, $C_l$",
        title="Solver lift response compared with reference (Xia & Mohseni, 2017)",
        link_colours_by_index=True,
        show=False,
    )

    ax.set_xlim(0.0, 5)
    ax.set_ylim(bottom=0.0, top=1.3)

    plt.tight_layout()
    plt.show()
