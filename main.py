"""
Entry point for the MAV-R unsteady airfoil solver.
"""

# Used config values
from config import (
    N,
    U_inf,
    alpha,
    c,
    delta_t,
    impulse_gust,
    no_vertical_gust,
    suddenly_imposed_gust,
    step_gust,
    v_0,
)

from examples.alpha_vs_imposed_gust import compare_alpha_vs_imposed_gust
from examples.cambered_frequency_response import plot_cambered_response
from examples.compare_strategy_time import compare_strategy_time_wagner_and_sears


## RUN CODE HERE
def main():
    compare_strategy_time_wagner_and_sears()


if __name__ == "__main__":
    main()
