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
from src.vortex_strategy import WCStrategy


def compare_strategy_time_wagner_and_sears():
    af1 = Airfoil.karman_trefftz(0.053, 0.0, np.deg2rad(15), c, N)

    strategy1 = WCStrategy(0.3 * c, 5, 0.25 * c)
    sim = Solver(af1, strategy1, 5 * N, (U_inf, suddenly_imposed_gust), 0.0)
    sim_impulse = Solver(af1, strategy1, 5 * N, (U_inf, impulse_gust), 0.0)
    sim_no_strategy = Solver(af1, None, 5 * N, (U_inf, suddenly_imposed_gust), 0.0)

    result = sim.solve()
    result_impulse = sim_impulse.solve()
    result_no_strategy = sim_no_strategy.solve()

    print(f"Strategy time: {result.time_taken}")
    print(f"Impulse time taken with strategy: {result_impulse.time_taken}")
    print(f"No strategy time: {result_no_strategy.time_taken}")

    Plotter.plot_lift_against_wagner(result, sim, v_0)
    Plotter.plot_lift_frequency_spectrum(result_impulse, sim_impulse, v_0)
    Plotter.plot_lift_against_wagner(result_no_strategy, sim_no_strategy, v_0)

    Plotter.plot_flow_field(result, af1, sim.alpha, sim.Q_inf, sim.delta_t)
    Plotter.plot_flow_field(
        result_no_strategy,
        af1,
        sim_no_strategy.alpha,
        sim_no_strategy.Q_inf,
        sim_no_strategy.delta_t,
    )
