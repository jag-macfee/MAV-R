# Put universally occuring values here
# Values which are used here are imported into main.py, as well as each example file

# AIRFOIL PARAMETERS
alpha = 0.0
N = 20  # panels
c = 1.0

# FREESTREAM PARAMETERS

# Horizontal freestream (must be a constant)
U_inf = 15
delta_t = c / (N * U_inf)  # timestep len - for use in impulse gust formulation

# Defined time-dependent functions for vertical gust
v_0 = 1.0

no_vertical_gust = lambda t: 0
suddenly_imposed_gust = lambda t: v_0
step_gust = lambda t: v_0 if t >= 0 else 0.0
impulse_gust = lambda t: v_0 if 0.0 <= t < delta_t else 0.0
