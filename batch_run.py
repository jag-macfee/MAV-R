#!/usr/bin/env python3
"""Batch runner for the unsteady vortex-panel solver.

Each row of the supplied CSV file defines one solver case. Results are written
under ``<project-root>/logs/<case_id>/``.

The runner always exports lift history data and a lift-history figure. For
``gust_type=impulse`` cases it also exports the impulse-derived frequency
response and its figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional

# FontTools emits harmless warnings for legacy font timestamps while Matplotlib
# loads fonts for exported figures. Avoids loads of timestamp warnings getting printed
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

# The batch runner only writes figures; it never opens an interactive window.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.airfoil import Airfoil
from src.plotter import Plotter
from src.solver import Solver
from src.util import ResultUtils
from src.vortex_strategy import WCStrategy, VortexLumpingStrategy

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_RHO = 1.225

CSV_COLUMNS = (
    "case_id",
    "enabled",
    "airfoil_type",
    "chord_m",
    "n_panels",
    "naca_code",
    "kt_epsilon",
    "kt_eta",
    "kt_delta_te_deg",
    "u_inf_m_s",
    "rho_kg_m3",
    "alpha_deg",
    "num_time_steps",
    "gust_type",
    "gust_amplitude_m_s",
    "gust_start_step",
    "gust_frequency_hz",
    "gust_phase_deg",
    "lumping_type",
    "lump_min_distance_c",
    "lump_max_distance_c",
    "lump_max_vortices",
)

CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CSVValidationError(ValueError):
    """Raised when one or more rows of a batch CSV are invalid."""


@dataclass(frozen=True)
class CaseConfig:
    """Validated configuration for one solver case."""

    case_id: str
    enabled: bool

    airfoil_type: str
    chord_m: float
    n_panels: int
    naca_code: Optional[str]
    kt_epsilon: Optional[float]
    kt_eta: Optional[float]
    kt_delta_te_deg: Optional[float]

    u_inf_m_s: float
    rho_kg_m3: float
    alpha_deg: float
    num_time_steps: int

    gust_type: str
    gust_amplitude_m_s: Optional[float]
    gust_start_step: Optional[int]
    gust_frequency_hz: Optional[float]
    gust_phase_deg: Optional[float]

    lumping_type: str
    lump_min_distance_c: Optional[float]
    lump_max_distance_c: Optional[float]
    lump_max_vortices: Optional[int]


# ---------------------------------------------------------------------------
# Gust functions
# ---------------------------------------------------------------------------
# These functions are deliberately defined at module scope and parameterised
# with functools.partial by make_gust_function(). This keeps the construction
# explicit and makes the resulting definitions suitable for Solver(Q_inf=...).


def zero_gust(t: float) -> float:
    """No vertical gust."""
    return 0.0


def constant_gust(t: float, *, amplitude: float) -> float:
    """Constant vertical gust velocity."""
    del t
    return amplitude


def step_gust(t: float, *, amplitude: float, start_time: float) -> float:
    """Sharp-edged step gust beginning at ``start_time`` at x=0."""
    return amplitude if t >= start_time else 0.0


def impulse_gust(
    t: float,
    *,
    amplitude: float,
    start_time: float,
    duration: float,
) -> float:
    """One-solver-time-step rectangular impulse beginning at ``start_time``.

    The solver evaluates W(t - x/U_inf), so giving the impulse a duration of
    one solver time step creates a travelling gust pulse one panel spacing long.
    An interval is used instead of floating-point equality at one instant.
    """
    return amplitude if start_time <= t < start_time + duration else 0.0


def sine_gust(
    t: float,
    *,
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    start_time: float,
) -> float:
    """Sinusoidal gust beginning at ``start_time`` at x=0."""
    if t < start_time:
        return 0.0
    return amplitude * math.sin(
        2.0 * math.pi * frequency_hz * (t - start_time) + phase_rad
    )


def make_gust_function(case: CaseConfig, delta_t: float) -> Callable[[float], float]:
    """Construct the vertical gust function W(t) for a validated case."""
    gust_type = case.gust_type

    if gust_type == "none":
        return zero_gust

    if gust_type == "constant":
        assert case.gust_amplitude_m_s is not None
        return partial(constant_gust, amplitude=case.gust_amplitude_m_s)

    # Step numbering follows Solver: k=1 corresponds to t=0.
    start_step = case.gust_start_step if case.gust_start_step is not None else 1
    start_time = (start_step - 1) * delta_t

    if gust_type == "step":
        assert case.gust_amplitude_m_s is not None
        return partial(
            step_gust,
            amplitude=case.gust_amplitude_m_s,
            start_time=start_time,
        )

    if gust_type == "impulse":
        assert case.gust_amplitude_m_s is not None
        return partial(
            impulse_gust,
            amplitude=case.gust_amplitude_m_s,
            start_time=start_time,
            duration=delta_t,
        )

    if gust_type == "sine":
        assert case.gust_amplitude_m_s is not None
        assert case.gust_frequency_hz is not None
        phase_deg = case.gust_phase_deg if case.gust_phase_deg is not None else 0.0
        return partial(
            sine_gust,
            amplitude=case.gust_amplitude_m_s,
            frequency_hz=case.gust_frequency_hz,
            phase_rad=math.radians(phase_deg),
            start_time=start_time,
        )

    # This should be unreachable after validation.
    raise ValueError(f"Unsupported gust_type: {gust_type}")


# ---------------------------------------------------------------------------
# Solver-object factories
# ---------------------------------------------------------------------------


def make_airfoil(case: CaseConfig) -> Airfoil:
    """Construct the airfoil selected by a validated case."""
    if case.airfoil_type == "flat":
        return Airfoil.flat_plate(case.chord_m, case.n_panels)

    if case.airfoil_type == "naca4":
        assert case.naca_code is not None
        return Airfoil.from_code(case.naca_code, case.chord_m, case.n_panels)

    if case.airfoil_type == "karman_trefftz":
        assert case.kt_epsilon is not None
        assert case.kt_eta is not None
        assert case.kt_delta_te_deg is not None
        return Airfoil.karman_trefftz(
            epsilon=case.kt_epsilon,
            eta=case.kt_eta,
            delta_te=math.radians(case.kt_delta_te_deg),
            chord=case.chord_m,
            n_panels=case.n_panels,
        )

    raise ValueError(f"Unsupported airfoil_type: {case.airfoil_type}")


def make_lumping_strategy(case: CaseConfig) -> Optional[VortexLumpingStrategy]:
    """Construct the wake-lumping strategy selected by a validated case."""
    if case.lumping_type == "none":
        return None

    if case.lumping_type == "wc":
        assert case.lump_min_distance_c is not None
        assert case.lump_max_distance_c is not None
        assert case.lump_max_vortices is not None
        return WCStrategy(
            min_lumping_distance_from_af=(case.chord_m * case.lump_min_distance_c),
            max_vortices_to_lump=case.lump_max_vortices,
            max_lumping_distance=(case.chord_m * case.lump_max_distance_c),
        )

    raise ValueError(f"Unsupported lumping_type: {case.lumping_type}")


def make_solver(case: CaseConfig) -> Solver:
    """Construct a Solver instance from one validated CSV case."""
    airfoil = make_airfoil(case)
    delta_t = airfoil.delta_x / case.u_inf_m_s
    gust = make_gust_function(case, delta_t)
    strategy = make_lumping_strategy(case)

    return Solver(
        airfoil=airfoil,
        strategy=strategy,
        num_time_steps=case.num_time_steps,
        Q_inf=(case.u_inf_m_s, gust),
        alpha=case.alpha_deg,
        rho=case.rho_kg_m3,
    )


# ---------------------------------------------------------------------------
# CSV parsing and validation
# ---------------------------------------------------------------------------


def _blank(value: Optional[str]) -> bool:
    return value is None or value.strip() == ""


def _text(row: dict[str, str], name: str) -> Optional[str]:
    value = row.get(name)
    if _blank(value):
        return None
    return value.strip()


def _required_text(row: dict[str, str], name: str) -> str:
    value = _text(row, name)
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _float_value(
    row: dict[str, str],
    name: str,
    *,
    required: bool = False,
    default: Optional[float] = None,
) -> Optional[float]:
    raw = _text(row, name)
    if raw is None:
        if required:
            raise ValueError(f"{name} is required")
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _int_value(
    row: dict[str, str],
    name: str,
    *,
    required: bool = False,
    default: Optional[int] = None,
) -> Optional[int]:
    raw = _text(row, name)
    if raw is None:
        if required:
            raise ValueError(f"{name} is required")
        return default
    try:
        # Reject values such as 4.5 and 4.0 rather than silently truncating.
        if not re.fullmatch(r"[+-]?\d+", raw):
            raise ValueError
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _bool_value(row: dict[str, str], name: str, *, default: bool = True) -> bool:
    raw = _text(row, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false', got {raw!r}")


def _require_blank(row: dict[str, str], names: Iterable[str], context: str) -> None:
    populated = [name for name in names if not _blank(row.get(name))]
    if populated:
        joined = ", ".join(populated)
        raise ValueError(f"{joined} must be blank when {context}")


def validate_row(row: dict[str, str], row_number: int) -> CaseConfig:
    """Validate and convert one CSV row into CaseConfig."""
    case_id = _required_text(row, "case_id")
    if not CASE_ID_PATTERN.fullmatch(case_id) or case_id in {".", ".."}:
        raise ValueError(
            "case_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )

    enabled = _bool_value(row, "enabled", default=True)

    # Airfoil ----------------------------------------------------------------
    airfoil_type = _required_text(row, "airfoil_type").lower()
    if airfoil_type not in {"flat", "naca4", "karman_trefftz"}:
        raise ValueError("airfoil_type must be one of: flat, naca4, karman_trefftz")

    chord_m = _float_value(row, "chord_m", required=True)
    n_panels = _int_value(row, "n_panels", required=True)
    assert chord_m is not None and n_panels is not None
    if chord_m <= 0.0:
        raise ValueError("chord_m must be strictly positive")
    if n_panels <= 0:
        raise ValueError("n_panels must be strictly positive")

    naca_code: Optional[str] = None
    kt_epsilon: Optional[float] = None
    kt_eta: Optional[float] = None
    kt_delta_te_deg: Optional[float] = None

    if airfoil_type == "flat":
        _require_blank(
            row,
            ("naca_code", "kt_epsilon", "kt_eta", "kt_delta_te_deg"),
            "airfoil_type=flat",
        )

    elif airfoil_type == "naca4":
        naca_code = _required_text(row, "naca_code")
        if not re.fullmatch(r"\d{4}", naca_code):
            raise ValueError("naca_code must contain exactly four digits")
        _require_blank(
            row,
            ("kt_epsilon", "kt_eta", "kt_delta_te_deg"),
            "airfoil_type=naca4",
        )

    else:  # karman_trefftz
        _require_blank(row, ("naca_code",), "airfoil_type=karman_trefftz")
        kt_epsilon = _float_value(row, "kt_epsilon", required=True)
        kt_eta = _float_value(row, "kt_eta", required=True)
        kt_delta_te_deg = _float_value(row, "kt_delta_te_deg", required=True)
        assert (
            kt_epsilon is not None
            and kt_eta is not None
            and kt_delta_te_deg is not None
        )
        if kt_epsilon < 0.0:
            raise ValueError("kt_epsilon must be non-negative")
        if not 0.0 <= kt_delta_te_deg < 180.0:
            raise ValueError("kt_delta_te_deg must lie in [0, 180)")

    # Flow -------------------------------------------------------------------
    u_inf_m_s = _float_value(row, "u_inf_m_s", required=True)
    rho_kg_m3 = _float_value(row, "rho_kg_m3", default=DEFAULT_RHO)
    alpha_deg = _float_value(row, "alpha_deg", required=True)
    num_time_steps = _int_value(row, "num_time_steps", required=True)
    assert (
        u_inf_m_s is not None
        and rho_kg_m3 is not None
        and alpha_deg is not None
        and num_time_steps is not None
    )
    if u_inf_m_s <= 0.0:
        raise ValueError("u_inf_m_s must be strictly positive")
    if rho_kg_m3 <= 0.0:
        raise ValueError("rho_kg_m3 must be strictly positive")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be strictly positive")

    # Gust -------------------------------------------------------------------
    gust_type = _required_text(row, "gust_type").lower()
    if gust_type not in {"none", "constant", "step", "impulse", "sine"}:
        raise ValueError(
            "gust_type must be one of: none, constant, step, impulse, sine"
        )

    gust_amplitude_m_s: Optional[float] = None
    gust_start_step: Optional[int] = None
    gust_frequency_hz: Optional[float] = None
    gust_phase_deg: Optional[float] = None

    if gust_type == "none":
        _require_blank(
            row,
            (
                "gust_amplitude_m_s",
                "gust_start_step",
                "gust_frequency_hz",
                "gust_phase_deg",
            ),
            "gust_type=none",
        )

    elif gust_type == "constant":
        gust_amplitude_m_s = _float_value(row, "gust_amplitude_m_s", required=True)
        _require_blank(
            row,
            ("gust_start_step", "gust_frequency_hz", "gust_phase_deg"),
            "gust_type=constant",
        )

    elif gust_type in {"step", "impulse"}:
        gust_amplitude_m_s = _float_value(row, "gust_amplitude_m_s", required=True)
        gust_start_step = _int_value(row, "gust_start_step", default=1)
        _require_blank(
            row,
            ("gust_frequency_hz", "gust_phase_deg"),
            f"gust_type={gust_type}",
        )

    else:  # sine
        gust_amplitude_m_s = _float_value(row, "gust_amplitude_m_s", required=True)
        gust_start_step = _int_value(row, "gust_start_step", default=1)
        gust_frequency_hz = _float_value(row, "gust_frequency_hz", required=True)
        gust_phase_deg = _float_value(row, "gust_phase_deg", default=0.0)
        assert gust_frequency_hz is not None
        if gust_frequency_hz <= 0.0:
            raise ValueError("gust_frequency_hz must be strictly positive")

    if gust_start_step is not None:
        if gust_start_step < 1:
            raise ValueError("gust_start_step must be at least 1")
        if gust_start_step > num_time_steps:
            raise ValueError("gust_start_step cannot exceed num_time_steps")

    if gust_type == "impulse":
        assert gust_amplitude_m_s is not None
        if math.isclose(gust_amplitude_m_s, 0.0, abs_tol=0.0):
            raise ValueError(
                "gust_amplitude_m_s must be non-zero for an impulse case "
                "because it is used to normalise the frequency response"
            )

    # Lumping ----------------------------------------------------------------
    lumping_type = _required_text(row, "lumping_type").lower()
    if lumping_type not in {"none", "wc"}:
        raise ValueError("lumping_type must be one of: none, wc")

    lump_min_distance_c: Optional[float] = None
    lump_max_distance_c: Optional[float] = None
    lump_max_vortices: Optional[int] = None

    if lumping_type == "none":
        _require_blank(
            row,
            (
                "lump_min_distance_c",
                "lump_max_distance_c",
                "lump_max_vortices",
            ),
            "lumping_type=none",
        )
    else:
        lump_min_distance_c = _float_value(row, "lump_min_distance_c", required=True)
        lump_max_distance_c = _float_value(row, "lump_max_distance_c", required=True)
        lump_max_vortices = _int_value(row, "lump_max_vortices", required=True)
        assert (
            lump_min_distance_c is not None
            and lump_max_distance_c is not None
            and lump_max_vortices is not None
        )
        if lump_min_distance_c < 0.0:
            raise ValueError("lump_min_distance_c must be non-negative")
        if lump_max_distance_c <= 0.0:
            raise ValueError("lump_max_distance_c must be strictly positive")
        if lump_max_vortices < 2:
            raise ValueError("lump_max_vortices must be at least 2")

    return CaseConfig(
        case_id=case_id,
        enabled=enabled,
        airfoil_type=airfoil_type,
        chord_m=chord_m,
        n_panels=n_panels,
        naca_code=naca_code,
        kt_epsilon=kt_epsilon,
        kt_eta=kt_eta,
        kt_delta_te_deg=kt_delta_te_deg,
        u_inf_m_s=u_inf_m_s,
        rho_kg_m3=rho_kg_m3,
        alpha_deg=alpha_deg,
        num_time_steps=num_time_steps,
        gust_type=gust_type,
        gust_amplitude_m_s=gust_amplitude_m_s,
        gust_start_step=gust_start_step,
        gust_frequency_hz=gust_frequency_hz,
        gust_phase_deg=gust_phase_deg,
        lumping_type=lumping_type,
        lump_min_distance_c=lump_min_distance_c,
        lump_max_distance_c=lump_max_distance_c,
        lump_max_vortices=lump_max_vortices,
    )


def load_cases(csv_path: Path) -> list[CaseConfig]:
    """Read and validate an entire batch CSV before any solve is started."""
    if not csv_path.is_file():
        raise CSVValidationError(f"CSV file does not exist: {csv_path}")

    errors: list[str] = []
    cases: list[CaseConfig] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise CSVValidationError("CSV is empty or has no header row")

        actual_columns = [name.strip() for name in reader.fieldnames]
        expected = set(CSV_COLUMNS)
        actual = set(actual_columns)

        duplicate_headers = sorted(
            {name for name in actual_columns if actual_columns.count(name) > 1}
        )
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if duplicate_headers or missing or extra:
            messages = []
            if duplicate_headers:
                messages.append("duplicate columns: " + ", ".join(duplicate_headers))
            if missing:
                messages.append("missing columns: " + ", ".join(missing))
            if extra:
                messages.append("unknown columns: " + ", ".join(extra))
            raise CSVValidationError("Invalid CSV header (" + "; ".join(messages) + ")")

        # Normalise header whitespace while retaining csv.DictReader behaviour.
        reader.fieldnames = actual_columns

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                errors.append(
                    f"row {row_number}: too many values for the declared CSV columns"
                )
                continue

            # Permit fully blank lines.
            if all(_blank(value) for value in row.values()):
                continue

            try:
                cases.append(validate_row(row, row_number))
            except ValueError as exc:
                case_hint = _text(row, "case_id") or "<unknown case>"
                errors.append(f"row {row_number} ({case_hint}): {exc}")

    if not cases and not errors:
        errors.append("CSV contains no solver cases")

    duplicates = sorted(
        case_id
        for case_id in {case.case_id for case in cases}
        if sum(case.case_id == case_id for case in cases) > 1
    )
    if duplicates:
        errors.append("duplicate case_id values: " + ", ".join(duplicates))

    if errors:
        formatted = "\n  - ".join(errors)
        raise CSVValidationError(f"Batch CSV validation failed:\n  - {formatted}")

    return cases


# ---------------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------------


def _write_case_config(case_dir: Path, case: CaseConfig) -> None:
    """Write the validated parameters used for this solver case to JSON.

    Defaults applied during CSV parsing are included, so this file records the
    effective configuration used by the runner rather than the raw CSV cells.
    The file is written before solving so it is also available for failed cases.
    """
    path = case_dir / "case_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(case), handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_lift_history(case_dir: Path, solver: Solver, result) -> None:
    """Write total/circulatory/non-circulatory lift history to CSV."""
    total = np.asarray(result.lift_history, dtype=float)
    circulatory = np.asarray(result.circulatory_lift_history, dtype=float)
    non_circulatory = np.asarray(result.non_circulatory_lift_history, dtype=float)

    if not (len(total) == len(circulatory) == len(non_circulatory)):
        raise RuntimeError("Lift-history arrays have inconsistent lengths")

    time_s = solver.delta_t * np.arange(len(total), dtype=float)
    normalised_time = time_s * solver.Q_inf[0] / solver.airfoil.c

    path = case_dir / "lift_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time_step",
                "time_s",
                "normalised_time_tU_over_c",
                "lift_total_N_per_m",
                "lift_circulatory_N_per_m",
                "lift_non_circulatory_N_per_m",
            )
        )
        for index in range(len(total)):
            writer.writerow(
                (
                    index + 1,
                    time_s[index],
                    normalised_time[index],
                    total[index],
                    circulatory[index],
                    non_circulatory[index],
                )
            )


def _write_frequency_response(
    case_dir: Path, case: CaseConfig, solver: Solver, result
) -> None:
    """Write impulse-derived lift frequency response and Sears reference to CSV."""
    if case.gust_type != "impulse":
        return

    assert case.gust_amplitude_m_s is not None
    normalised_frequency, response_squared = (
        ResultUtils.extract_lift_frequency_response(
            lift_history=result.lift_history,
            delta_t=solver.delta_t,
            U_inf=case.u_inf_m_s,
            chord=case.chord_m,
            rho=case.rho_kg_m3,
            v_0=case.gust_amplitude_m_s,
            max_normalised_frequency=case.n_panels / 4.0,
        )
    )
    sears_squared = ResultUtils.sears_response_squared_approximation(
        normalised_frequency
    )

    path = case_dir / "frequency_response.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "normalised_frequency_fc_over_U",
                "normalised_lift_response_squared",
                "sears_response_squared",
            )
        )
        writer.writerows(zip(normalised_frequency, response_squared, sears_squared))


def _save_figure(fig, output_stem: Path) -> None:
    """Save publication PDF plus convenient PNG copy, then close the figure."""
    try:
        fig.savefig(output_stem.with_suffix(".pdf"))
        fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    finally:
        plt.close(fig)


def _write_figures(case_dir: Path, case: CaseConfig, solver: Solver, result) -> None:
    figures_dir = case_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = Plotter.plot_lift_history(result, solver, show=False)
    _save_figure(fig, figures_dir / "lift_history")

    if case.gust_type == "impulse":
        assert case.gust_amplitude_m_s is not None
        fig, _ = Plotter.plot_lift_frequency_spectrum(
            result,
            solver,
            v_0=case.gust_amplitude_m_s,
            apply_lysak_accuracy_limit=True,
            show=False,
        )
        _save_figure(fig, figures_dir / "frequency_response")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_case(case: CaseConfig, logs_dir: Path = LOGS_DIR) -> tuple[str, bool, str]:
    """Run one solver case and export its results.

    Returns ``(case_id, success, message)`` so the function can be used directly
    by ProcessPoolExecutor workers.
    """
    case_dir = logs_dir / case.case_id

    # Replace an existing case folder so stale frequency-response files cannot
    # survive if, for example, a case changes from impulse to step input.
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # Record the effective validated input before solving. This is useful both
    # for provenance and for diagnosing cases which subsequently fail.
    _write_case_config(case_dir, case)

    try:
        solver = make_solver(case)
        result = solver.solve()

        _write_lift_history(case_dir, solver, result)
        _write_frequency_response(case_dir, case, solver, result)
        _write_figures(case_dir, case, solver, result)

        return case.case_id, True, f"completed in {solver.time():.6g} s"

    except Exception as exc:  # Preserve a useful per-case failure record.
        error_text = (
            f"Case {case.case_id!r} failed: {type(exc).__name__}: {exc}\n\n"
            + traceback.format_exc()
        )
        (case_dir / "error.txt").write_text(error_text, encoding="utf-8")
        return case.case_id, False, f"{type(exc).__name__}: {exc}"


def clear_logs(logs_dir: Path = LOGS_DIR) -> None:
    """Delete the batch log directory and all previous case outputs."""
    if logs_dir.exists():
        shutil.rmtree(logs_dir)


def _normalise_jobs(value: int) -> int:
    if value < 0:
        raise ValueError("--jobs must be 0 or a positive integer")
    if value == 0:
        return os.cpu_count() or 1
    return value


def run_batch(cases: list[CaseConfig], jobs: int, logs_dir: Path = LOGS_DIR) -> bool:
    """Execute all enabled cases, optionally using multiple worker processes."""
    enabled_cases = [case for case in cases if case.enabled]
    disabled_count = len(cases) - len(enabled_cases)

    logs_dir.mkdir(parents=True, exist_ok=True)

    if disabled_count:
        print(f"Skipping {disabled_count} disabled case(s).")
    if not enabled_cases:
        print("No enabled cases to run.")
        return True

    print(
        f"Running {len(enabled_cases)} case(s) with {jobs} worker"
        f"{'s' if jobs != 1 else ''}."
    )
    if jobs > 1:
        print(
            "Note: parallel execution can affect individual solver timing due to CPU "
            "contention."
        )

    all_successful = True

    if jobs == 1:
        for case in enabled_cases:
            case_id, success, message = run_case(case, logs_dir)
            status = "OK" if success else "FAILED"
            print(f"[{status}] {case_id}: {message}")
            all_successful &= success
        return all_successful

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_case = {
            executor.submit(run_case, case, logs_dir): case.case_id
            for case in enabled_cases
        }
        for future in as_completed(future_to_case):
            case_id = future_to_case[future]
            try:
                _, success, message = future.result()
            except Exception as exc:
                # This catches worker/process failures outside run_case itself.
                success = False
                message = f"worker failure: {type(exc).__name__}: {exc}"
            status = "OK" if success else "FAILED"
            print(f"[{status}] {case_id}: {message}")
            all_successful &= success

    return all_successful


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or more unsteady panel-solver cases from a CSV file."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the batch input CSV.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing project logs before starting this batch.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of solver processes to run in parallel. "
            "Use 0 to use all available CPUs. Default: 1."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        jobs = _normalise_jobs(args.jobs)
        csv_path = args.csv_path.expanduser().resolve()
        cases = load_cases(csv_path)
    except (CSVValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Validate first so a malformed input file cannot accidentally wipe logs.
    if args.clear:
        clear_logs(LOGS_DIR)
        print(f"Cleared previous batch logs: {LOGS_DIR}")

    successful = run_batch(cases, jobs=jobs, logs_dir=LOGS_DIR)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
