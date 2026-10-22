# MAV-R

**MAV-R (Modular Airfoil Vortex Response)** is a research-oriented Python solver for the unsteady **two-dimensional lift and lift frequency response** of thin airfoils. It combines a discrete vortex-panel representation of the airfoil camber line with a time-marching vortex wake and optional wake-vortex lumping.

## Overview

MAV-R is intended for rapid, low-order unsteady aerodynamic analysis and for
investigating the **accuracy–runtime trade-off associated with reducing the
number of discrete wake vortices**. Unlike a closed-contour panel method, the
airfoil is represented by its _camber line_, rather than separate pressure and
suction surfaces. Airfoil families with thickness parameters can be used to
_construct_ a camber line, but the solver does **not** resolve thickness-related
flow effects explicitly.

The codebase supports:

- Flat plates, NACA 4-digit camber lines, and Kármán–Trefftz camber lines.
- A constant streamwise freestream, optional time-varying vertical gust, and a
  fixed airfoil angle of attack.
- Bound and shed wake circulation solved at every discrete time step.
- Total, circulatory, and non-circulatory **lift per unit span**.
- Lift-history, circulation-history, wake and flow-field visualisations.
- Impulse-derived lift frequency response and comparison with a Sears-function
  approximation.
- Full-wake (unlumped) simulations or optional weighted-centroid wake lumping.
- Standalone examples, CSV-driven batch execution, and parameter studies.

**Scope:** This is a two-dimensional incompressible, inviscid potential-flow
model. It does not directly model viscosity, boundary-layer separation, stall,
compressibility, three-dimensional effects, or the surface pressure field of a
finite-thickness airfoil. Think of it as a smaller more experimental version of XFOIL. The current `Solver` takes a _constant_ horizontal
freestream and a callable vertical gust; it does not explicitly integrate
pitching motion or a time-varying streamwise freestream. It is important to not interpret its
camber-line calculations as thick-airfoil CFD, or its lift output as a validated
drag prediction.

## Contents

- [Overview](#overview)
- [Getting started](#getting-started)
- [Repository layout](#repository-layout)
    - [Core modules](#core-modules)
    - [Numerical method](#numerical-method)
    - [Wake lumping and optimisation studies](#wake-lumping-and-optimisation-studies)
    - [Results and post-processing](#results-and-post-processing)
    - [Minimal Python example](#minimal-python-example)
- [Resources / References](#resources--references)
- [Batch Solver](#batch-solver)
    - [Output Behaviour](#output-behaviour)
- [Input CSV Format](#input-csv-format)
- [General Parameters](#general-parameters)
- [Airfoil Definition](#airfoil-definition)
    - [Common Airfoil Parameters](#common-airfoil-parameters)
    - [Flat Plate](#flat-plate)
    - [NACA 4-Digit Airfoil](#naca-4-digit-airfoil)
    - [Kármán–Trefftz Airfoil](#kármántrefftz-airfoil)
- [Flow Conditions](#flow-conditions)
- [Gust Definition](#gust-definition)
    - [No Gust](#no-gust)
    - [Constant Vertical Velocity](#constant-vertical-velocity)
    - [Step Gust](#step-gust)
    - [Impulse Gust](#impulse-gust)
    - [Sinusoidal Gust](#sinusoidal-gust)
- [Wake-Vortex Lumping](#wake-vortex-lumping)
    - [No Lumping](#no-lumping)
    - [Weighted-Centroid Lumping](#weighted-centroid-lumping)
- [Conditional Parameter Requirements](#conditional-parameter-requirements)
- [Input Validation](#input-validation)
- [Example Batch](#example-batch)

## Getting started

From the repository root, install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the main example or use the convenience launcher:

```bash
python3 main.py
# or, if the launcher is executable:
./run-MAVR
```

To run a validated CSV batch:

```bash
chmod +x batch-runner       # needed once on Unix-like systems
./batch-runner batch-cases/example.csv
```

Replace `batch-cases/example.csv` with the path to an existing input file. For
batch flags (`--jobs`, `--clear`) and the complete CSV schema, see
[Batch Solver](#batch-solver) below. The `run-parameter-tests` script is the
convenience entry point for the analysis scripts in `parameter_testing/`;
inspect that launcher for its accepted parameters.

## Repository layout

```text
MAV-R/
├── src/                       # Reusable solver implementation
│   ├── airfoil.py             # Airfoil factories and camber-line geometry
│   ├── solver.py              # Time-marching simulation and SolveResult
│   ├── vortex_strategy.py     # Wake-vortex lumping strategy interface and WCStrategy
│   ├── types.py               # Panel and PointValue data structures
│   ├── util.py                # Vortex influences, wake evolution, lift FRF utilities
│   ├── plotter.py             # Publication-styled plots and interactive visualisations
│   └── __init__.py
├── examples/                  # Standalone research/demo simulations
├── parameter_testing/         # Parameter sweeps, accuracy/runtime comparisons
│   └── plots/                 # Figures grouped by parameter
├── batch-cases/              # Batch CSV input files
├── logs/                     # Generated batch-case outputs
├── static/data/              # Digitised or reference datasets for comparison
├── theory/                   # Scientific papers and theoretical material
├── docs/                     # Additional project documentation
├── main.py                   # Main demo entry point
├── batch_run.py              # CSV validation, execution, export and parallel jobs
├── config.py                 # Shared settings and gust definitions
├── batch-runner              # Shell wrapper around batch_run.py
├── run-MAVR                  # Convenience launcher for a standard run
├── run-parameter-tests       # Launcher for parameter studies
├── requirements.txt          # Python dependencies
└── README.md                 # Architecture, usage and batch CSV schema
```

### Core modules

| Module                   | Responsibility                                                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/airfoil.py`         | Defines the `Airfoil` interface and implementations `FlatPlate`, `Naca4Digit`, and `KarmanTrefftzAirfoil`. Airfoils supply uniformly spaced camber-line endpoints.                   |
| `src/types.py`           | Defines `Panel`, including its tangent, normal, length, quarter-panel bound vortex, and three-quarter-panel control point; and `PointValue`, a point plus scalar vortex circulation. |
| `src/solver.py`          | Defines `Solver`, which preprocesses the airfoil, advances the unsteady solution, calculates lift, and returns a `SolveResult`.                                                      |
| `src/util.py`            | Provides low-level induced-velocity and influence-matrix operations (`SolverUtils`) and frequency-response/benchmark calculations (`ResultUtils`).                                   |
| `src/vortex_strategy.py` | Defines the `VortexLumpingStrategy` interface and the weighted-centroid implementation `WCStrategy`.                                                                                 |
| `src/plotter.py`         | Visualises circulation history, the wake/flow field, lift histories, and frequency responses, with Matplotlib styles suitable for academic figures.                                  |

### Numerical method

1. **Discretise geometry.** `Airfoil.camber()` returns `N + 1` points. After
   rotation by the fixed angle of attack (specified in **degrees**), adjacent
   points define `N` panels. Each panel has a bound vortex at one-quarter of
   its length and a collocation point at three-quarters of its length.
2. **Assemble the system.** The no-penetration boundary condition contributes
   `N` equations. An additional equation enforcing Kelvin circulation
   conservation closes the system for the `N` bound vortices and the newest
   shed vortex. With stationary geometry, the `(N + 1) × (N + 1)` coefficient
   matrix is assembled **once**, before time marching.
3. **March in time.** At each time step, the right-hand side includes the
   freestream, the chordwise-convecting gust, and the influence of previously
   shed wake vortices. `numpy.linalg.solve` computes the new bound and shed
   circulation, after which the wake vortices are convected by the evaluated
   flow field. The baseline time step is
   `Δt = (c/N) / U∞`.
4. **Calculate lift.** Panel pressure jumps are evaluated using the unsteady
   Bernoulli formulation, separating a circulation-related term from the time
   derivative of cumulative bound circulation. Their vertically projected
   forces give `L_total = L_circulatory + L_non_circulatory` (N/m).
5. **Optionally simplify the wake.** A configured strategy is applied to the
   convected wake after each step. Lumping reduces the number of vortices
   participating in later steps, trading model fidelity for lower execution
   cost. With `strategy=None`, no vortices are lumped.

The imposed gust is specified at the leading edge by `W(t)`. For a panel
control point at chordwise position `x_i`, the local gust is evaluated with a
convection delay `W(t - x_i/U∞)`. The horizontal freestream `U∞` remains
constant. The discretisation and lift formulas are discussed in the theoretical
material in `theory/` and the references below.

### Wake lumping and optimisation studies

The available `WCStrategy` is a **weighted-centroid** strategy with three
parameters:

| Python constructor argument    | Meaning                                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `min_lumping_distance_from_af` | Minimum Euclidean distance from the trailing edge before a wake vortex becomes eligible for lumping. |
| `max_lumping_distance`         | Maximum Euclidean separation allowed between vortices in the same lump.                              |
| `max_vortices_to_lump`         | Maximum number of vortices to combine into one lump (at least 2).                                    |

Lumping is performed from the _far wake toward the airfoil_, separately for
positive and negative vortex circulation. Only same-signed eligible vortices
are grouped. A resulting vortex has the group's **sum of circulation** and a
**circulation-weighted centroid** position. Vortices too close to the trailing
edge, or with zero circulation, are retained. The distances supplied directly
to `WCStrategy` are dimensional; the batch CSV accepts chord-normalised
versions and converts them to metres.

### Results and post-processing

`Solver.solve()` returns a `SolveResult` containing:

| Field                          | Description                                                                                                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bound_gamma_history`          | Bound-vortex positions and circulations at each step.                                                                                                       |
| `wake_gamma_history`           | Wake-vortex positions and circulations at each step; the newly shed vortex is included. The recorded snapshot precedes that step's wake convection/lumping. |
| `lift_history`                 | Total lift per unit span, in N/m.                                                                                                                           |
| `circulatory_lift_history`     | Circulatory contribution to lift, in N/m.                                                                                                                   |
| `non_circulatory_lift_history` | Non-circulatory (unsteady) contribution to lift, in N/m.                                                                                                    |
| `time_taken`                   | Elapsed time of the solver operation in seconds, excluding separate plotting/export work.                                                                   |

`ResultUtils.extract_lift_frequency_response()` transforms the **total lift
history from a discrete impulse-gust simulation** and returns the reduced
frequency `fc/U∞` and squared, dimensionless response magnitude:

```math
\frac{|H(f)|^2}{(\pi \rho c U_\infty)^2}
= \frac{|\operatorname{DFT}[L_k]|^2}
       {(\pi \rho c U_\infty v_0)^2}.
```

The `ResultUtils.sears_response_squared_approximation()` helper provides the
corresponding thin-flat-plate reference. The batch frequency-response export
limits the displayed `fc/U∞` range to `N/4`, following the accuracy criterion
used in Lysak's study, rather than the higher theoretical Nyquist limit.

### Minimal Python example

```python
from src.airfoil import Airfoil
from src.plotter import Plotter
from src.solver import Solver

# Airfoil geometry is given in metres and the angle of attack in degrees.
airfoil = Airfoil.flat_plate(c=1.0, n_panels=40)

def step_gust(t: float) -> float:
    return 1.0 if t >= 0.5 else 0.0  # m/s, defined at the leading edge

solver = Solver(
    airfoil=airfoil,
    strategy=None,             # Keep every shed wake vortex
    num_time_steps=600,
    Q_inf=(10.0, step_gust),   # U_infinity (m/s), W(t) (m/s)
    alpha=0.0,                # degrees
    rho=1.225,                # kg/m^3
)

result = solver.solve()
print(f"Completed in {result.time_taken:.3f} s")
print(f"Final lift: {result.lift_history[-1]:.6g} N/m")

fig, _ = Plotter.plot_lift_history(result, solver, show=False)
fig.savefig("lift_history.png", dpi=300)
```

For batch execution, prefer the CSV runner below rather than reimplementing
case validation and output conventions. To extend the solver, add an airfoil
subclass in `src/airfoil.py`, a `VortexLumpingStrategy` in
`src/vortex_strategy.py`, or new post-processing methods in `src/util.py` and
`src/plotter.py`, while keeping experimental drivers in `examples/` or
`parameter_testing/`.

---

## Resources / References

- NACA airfoil geometry: <https://web.stanford.edu/~cantwell/AA200_Course_Material/The%20NACA%20airfoil%20series.pdf>
- Katz & Plotkin, _Low-Speed Aerodynamics_ (discrete-vortex and unsteady panel formulations).
- Peter D. Lysak (2011), _Unsteady Lift of Thick Airfoils in Incompressible Turbulent Flow_, especially Chapter 2 (discrete vortex method and impulse-based frequency response).

## Batch Solver

The batch solver provides an interface for running one or more solver
configurations from a CSV input file.

Each row of the CSV represents one independent solver case and defines its
airfoil geometry, flow conditions, gust input, simulation length, and optional
wake-vortex lumping strategy.

Place `batch_run.py` and the `batch-runner` launcher in the project root.
The launcher runs `batch_run.py` from the project root so that project-relative
paths and imports resolve consistently.

On first use, make the launcher executable:

```bash
chmod +x batch-runner
```

Run a batch CSV from the project root using:

```bash
./batch-runner path/to/cases.csv
```

For example:

```bash
./batch-runner batch-cases/lumping_cases.csv
```

By default, cases are executed serially. To run multiple solver cases in
parallel, use `--jobs` (or `-j`):

```bash
./batch-runner --jobs 4 batch-cases/lumping_cases.csv
```

Use `--jobs 0` to use all available CPU cores:

```bash
./batch-runner --jobs 0 batch-cases/lumping_cases.csv
```

Parallel execution may affect measured solver run times because multiple cases
compete for CPU resources. Use serial execution (`--jobs 1`, the default) when
comparing solver timing performance.

To delete all previous batch outputs before starting a new batch, use
`--clear`:

```bash
./batch-runner --clear batch-cases/lumping_cases.csv
```

Options may be combined:

```bash
./batch-runner --clear --jobs 4 batch-cases/lumping_cases.csv
```

The complete command-line interface can be displayed with:

```bash
./batch-runner --help
```

The batch runner validates the complete CSV before executing any solver cases.
If validation succeeds, it constructs the required airfoil, gust and
wake-lumping strategy for each enabled row and runs the corresponding solver.

Results are written beneath the project-root `logs/` directory. Each case gets
its own directory named using its `case_id`:

```text
logs/
└── <case_id>/
    ├── lift_history.csv
    └── figures/
        ├── lift_history.pdf
        └── lift_history.png
```

For `gust_type=impulse` cases, the directory additionally contains:

```text
logs/
└── <case_id>/
    ├── frequency_response.csv
    └── figures/
        ├── frequency_response.pdf
        └── frequency_response.png
```

If a case directory with the same `case_id` already exists, that case
directory is replaced when the case is rerun. The `--clear` option instead
removes the entire `logs/` directory before the batch starts.

The `--clear` action occurs only after the CSV has passed validation, so an
invalid input file will not delete existing logs.

---

### Output Behaviour

Output quantities are determined automatically and are not specified in the
input CSV.

For every successfully completed case, the batch runner outputs:

- a CSV containing the unsteady lift history; and
- a publication-formatted lift-history figure.

If the case uses `gust_type=impulse`, the runner additionally outputs:

- a CSV containing the calculated lift frequency response; and
- a publication-formatted frequency-response figure.

Frequency-response output is only generated for impulse cases because the
current frequency-response calculation is based on the solver's discrete
impulse response.

---

## Input CSV Format

The input file must contain one header row followed by one or more solver
cases.

The full CSV schema is:

```csv
case_id,enabled,airfoil_type,chord_m,n_panels,naca_code,kt_epsilon,kt_eta,kt_delta_te_deg,u_inf_m_s,rho_kg_m3,alpha_deg,num_time_steps,gust_type,gust_amplitude_m_s,gust_start_step,gust_frequency_hz,gust_phase_deg,lumping_type,lump_min_distance_c,lump_max_distance_c,lump_max_vortices
```

For example:

```csv
case_id,enabled,airfoil_type,chord_m,n_panels,naca_code,kt_epsilon,kt_eta,kt_delta_te_deg,u_inf_m_s,rho_kg_m3,alpha_deg,num_time_steps,gust_type,gust_amplitude_m_s,gust_start_step,gust_frequency_hz,gust_phase_deg,lumping_type,lump_min_distance_c,lump_max_distance_c,lump_max_vortices
flat_impulse,true,flat,1.0,40,,,,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
naca_step,true,naca4,1.0,40,2412,,,10.0,1.225,2.0,600,step,1.0,1,,,none,,,
kt_reference,true,karman_trefftz,1.0,40,,0.053,0.0,15.0,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
kt_lumped,true,karman_trefftz,1.0,40,,0.053,0.0,15.0,10.0,1.225,0.0,600,impulse,1.0,1,,,wc,2.0,0.5,8
```

The same fixed set of columns is used for every case. Parameters which do not
apply to a particular configuration should be left blank.

Do not use `0` as a placeholder for an unused parameter, since zero may itself
be a valid physical value.

---

## General Parameters

| Column           | Requirement | Description                                                                                                                           |
| ---------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `case_id`        | Required    | Unique identifier for the solver case. This should be a short, descriptive name suitable for use in output filenames and directories. |
| `enabled`        | Optional    | Determines whether the case is executed. Accepted values are `true` and `false`. If blank, `true` is assumed.                         |
| `num_time_steps` | Required    | Number of discrete solver time steps to calculate. Must be a positive integer.                                                        |

Each row is an independent simulation. A single input file may therefore
contain different airfoils, panel counts, flow conditions, gusts and
wake-lumping configurations.

---

## Airfoil Definition

### Common Airfoil Parameters

| Column         | Requirement | Description                                                                                          |
| -------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `airfoil_type` | Required    | Airfoil geometry type. Supported values are `flat`, `naca4`, and `karman_trefftz`.                   |
| `chord_m`      | Required    | Airfoil chord length `c`, in metres. Must be positive.                                               |
| `n_panels`     | Required    | Number of discrete panels `N` used to represent the airfoil camber line. Must be a positive integer. |

For an airfoil of chord `c` represented by `N` panels, the chordwise panel
spacing is:

```math
\Delta x = \frac{c}{N}
```

### Flat Plate

For a flat plate, use:

```text
airfoil_type = flat
```

No additional airfoil parameters are required.

The following columns should be left blank:

```text
naca_code
kt_epsilon
kt_eta
kt_delta_te_deg
```

Example:

```csv
flat_case,true,flat,1.0,40,,,,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
```

### NACA 4-Digit Airfoil

For a NACA 4-digit airfoil, use:

```text
airfoil_type = naca4
```

and specify:

| Column      | Requirement | Description                                                    |
| ----------- | ----------- | -------------------------------------------------------------- |
| `naca_code` | Required    | Four-character NACA designation, for example `2412` or `0012`. |

The current solver uses the NACA airfoil's camber line.

Example:

```csv
naca2412_case,true,naca4,1.0,40,2412,,,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
```

`naca_code` must be read as a string rather than an integer so that leading
zeroes are preserved. For example, `0012` must remain `0012`.

The Kármán–Trefftz parameter columns should be left blank.

### Kármán–Trefftz Airfoil

For a Kármán–Trefftz airfoil, use:

```text
airfoil_type = karman_trefftz
```

and specify:

| Column            | Requirement | Description                                                                                                                      |
| ----------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `kt_epsilon`      | Required    | Dimensionless `epsilon` parameter defining the real component of the generating-circle offset. Must be non-negative.             |
| `kt_eta`          | Required    | Dimensionless `eta` parameter defining the imaginary component of the generating-circle offset and therefore the airfoil camber. |
| `kt_delta_te_deg` | Required    | Kármán–Trefftz trailing-edge angle, in degrees.                                                                                  |

The CSV uses degrees for the trailing-edge angle for convenience. The batch
runner converts this value to radians before constructing the airfoil.

Example:

```csv
kt_case,true,karman_trefftz,1.0,40,,0.053,0.0,15.0,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
```

The `naca_code` column should be left blank.

---

## Flow Conditions

| Column      | Requirement | Description                                                                           |
| ----------- | ----------- | ------------------------------------------------------------------------------------- |
| `u_inf_m_s` | Required    | Constant horizontal freestream velocity `U_inf`, in m/s. Must be strictly positive.   |
| `rho_kg_m3` | Optional    | Fluid density `rho`, in kg/m³. If blank, the solver default of `1.225 kg/m³` is used. |
| `alpha_deg` | Required    | Constant airfoil angle of attack `alpha`, in degrees.                                 |

The solver represents the freestream using a constant horizontal velocity and
a time-dependent vertical gust velocity:

```text
Q_inf = (U_inf, W(t))
```

The time-step size is **not** specified independently in the input CSV.

It is calculated automatically from the airfoil discretisation and freestream
velocity:

```math
\Delta t
=
\frac{\Delta x}{U_\infty}
=
\frac{c}{N U_\infty}
```

This prevents inconsistent combinations of `chord_m`, `n_panels`,
`u_inf_m_s`, and a separately supplied time step.

---

## Gust Definition

The `gust_type` column determines the time-dependent vertical component `W(t)`
of the freestream.

| Column               | Requirement    | Description                                                                                                       |
| -------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `gust_type`          | Required       | Gust definition. Supported values are `none`, `constant`, `step`, `impulse`, and `sine`.                          |
| `gust_amplitude_m_s` | Gust-dependent | Vertical gust amplitude `v0`, in m/s.                                                                             |
| `gust_start_step`    | Optional       | Solver time step at which the gust begins. Defaults to `1`. Solver time steps use the convention `k = 1, 2, ...`. |
| `gust_frequency_hz`  | Sine only      | Frequency of a sinusoidal gust, in Hz.                                                                            |
| `gust_phase_deg`     | Sine only      | Phase offset of a sinusoidal gust, in degrees. Defaults to `0`.                                                   |

Parameters which do not apply to the selected gust type should be left blank.

### No Gust

```text
gust_type = none
```

The vertical gust velocity is zero:

```text
W(t) = 0
```

No additional gust parameters are required.

### Constant Vertical Velocity

```text
gust_type = constant
```

The vertical velocity remains equal to the specified gust amplitude:

```text
W(t) = v0
```

Required:

```text
gust_amplitude_m_s
```

### Step Gust

```text
gust_type = step
```

A sharp-edged gust begins at `gust_start_step`.

Before the specified start step, the vertical gust velocity is zero. From the
start step onwards, it is equal to `gust_amplitude_m_s`.

Required:

```text
gust_amplitude_m_s
```

Optional:

```text
gust_start_step
```

### Impulse Gust

```text
gust_type = impulse
```

A discrete impulse is applied for one solver time step.

At `gust_start_step`:

```text
W = v0
```

At all other time steps:

```text
W = 0
```

Required:

```text
gust_amplitude_m_s
```

Optional:

```text
gust_start_step
```

Impulse cases receive additional frequency-response post-processing after the
solver completes.

### Sinusoidal Gust

```text
gust_type = sine
```

A sinusoidal vertical gust begins at `gust_start_step`.

Required:

```text
gust_amplitude_m_s
gust_frequency_hz
```

Optional:

```text
gust_start_step
gust_phase_deg
```

The sinusoidal gust is defined using amplitude `v0`, frequency `f`, phase
`phi`, and start time `t0`:

```math
W(t)
=
v_0
\sin\left(2\pi f(t-t_0)+\phi\right)
```

---

## Wake-Vortex Lumping

Wake-vortex lumping is optional and is selected using `lumping_type`.

| Column                | Requirement | Description                                                                                                      |
| --------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `lumping_type`        | Required    | Wake-lumping strategy. Supported values are `none` and `wc`.                                                     |
| `lump_min_distance_c` | WC only     | Minimum trailing-edge distance at which a wake vortex becomes eligible for lumping, expressed relative to chord. |
| `lump_max_distance_c` | WC only     | Maximum distance permitted between vortices belonging to the same lump, expressed relative to chord.             |
| `lump_max_vortices`   | WC only     | Maximum number of wake vortices which may be combined into one lump. Must be at least `2`.                       |

### No Lumping

To disable wake-vortex lumping, use:

```text
lumping_type = none
```

and leave the following fields blank:

```text
lump_min_distance_c
lump_max_distance_c
lump_max_vortices
```

### Weighted-Centroid Lumping

To use the weighted-centroid strategy, use:

```text
lumping_type = wc
```

and specify all three lumping parameters.

For example:

```text
lumping_type        = wc
lump_min_distance_c = 2.0
lump_max_distance_c = 0.5
lump_max_vortices   = 8
```

The two distance parameters are specified relative to the airfoil chord.

For a case with chord `c`, the dimensional values passed to the solver are:

```text
min_lumping_distance = chord_m × lump_min_distance_c
max_lumping_distance = chord_m × lump_max_distance_c
```

For example, if:

```text
chord_m            = 1.5
lump_min_distance_c = 2.0
```

then:

```text
min_lumping_distance = 3.0 m
```

Using chord-normalised distances allows lumping configurations to be compared
directly between airfoils with different chord lengths.

The weighted-centroid strategy groups eligible same-signed wake vortices. Each
group is replaced by one vortex whose circulation equals the total circulation
of the group and whose position is its circulation-weighted centroid.

---

## Conditional Parameter Requirements

| Selection                     | Additional required parameters                                    |
| ----------------------------- | ----------------------------------------------------------------- |
| `airfoil_type=flat`           | None                                                              |
| `airfoil_type=naca4`          | `naca_code`                                                       |
| `airfoil_type=karman_trefftz` | `kt_epsilon`, `kt_eta`, `kt_delta_te_deg`                         |
| `gust_type=none`              | None                                                              |
| `gust_type=constant`          | `gust_amplitude_m_s`                                              |
| `gust_type=step`              | `gust_amplitude_m_s`                                              |
| `gust_type=impulse`           | `gust_amplitude_m_s`                                              |
| `gust_type=sine`              | `gust_amplitude_m_s`, `gust_frequency_hz`                         |
| `lumping_type=none`           | None                                                              |
| `lumping_type=wc`             | `lump_min_distance_c`, `lump_max_distance_c`, `lump_max_vortices` |

---

## Input Validation

The batch runner validates the input before executing solver cases.

Invalid configurations include:

- missing or duplicate `case_id` values;
- unsupported `airfoil_type`, `gust_type`, or `lumping_type` values;
- non-positive `chord_m`;
- non-positive `n_panels`;
- non-positive `u_inf_m_s`;
- non-positive `num_time_steps`;
- missing `naca_code` for `airfoil_type=naca4`;
- missing Kármán–Trefftz parameters for
  `airfoil_type=karman_trefftz`;
- missing parameters required by the selected gust type;
- missing weighted-centroid parameters for `lumping_type=wc`;
- `lump_max_vortices` less than `2`; or
- parameters populated for an incompatible airfoil, gust, or lumping type.

Unused parameters should always be represented using an empty CSV field.

For example, this is valid:

```csv
flat_case,true,flat,1.0,40,,,,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
```

whereas a flat-plate case containing values for `naca_code` or the
Kármán–Trefftz parameters should be treated as invalid input.

---

## Example Batch

The following CSV defines four independent simulations:

```csv
case_id,enabled,airfoil_type,chord_m,n_panels,naca_code,kt_epsilon,kt_eta,kt_delta_te_deg,u_inf_m_s,rho_kg_m3,alpha_deg,num_time_steps,gust_type,gust_amplitude_m_s,gust_start_step,gust_frequency_hz,gust_phase_deg,lumping_type,lump_min_distance_c,lump_max_distance_c,lump_max_vortices
flat_reference,true,flat,1.0,40,,,,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
naca2412,true,naca4,1.0,40,2412,,,10.0,1.225,2.0,600,impulse,1.0,1,,,none,,,
kt_reference,true,karman_trefftz,1.0,40,,0.053,0.0,15.0,10.0,1.225,0.0,600,impulse,1.0,1,,,none,,,
kt_lumped,true,karman_trefftz,1.0,40,,0.053,0.0,15.0,10.0,1.225,0.0,600,impulse,1.0,1,,,wc,2.0,0.5,8
```

Run it using:

```bash
./batch-runner batch-cases/example.csv
```

To discard all previous batch logs first:

```bash
./batch-runner --clear batch-cases/example.csv
```

To execute up to four cases concurrently:

```bash
./batch-runner --jobs 4 batch-cases/example.csv
```

The first three cases use the full discrete wake. The fourth uses
weighted-centroid vortex lumping.

Because all four cases use `gust_type=impulse`, all four produce both
lift-history and frequency-response outputs.
