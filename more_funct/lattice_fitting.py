"""Lattice parameter fitting from experimental peak positions.

This module lets the user pick one uploaded experimental diffraction pattern
and one uploaded structure, obtain a list of peak positions (either by automatic
peak detection that can be edited in a table, or by uploading/pasting a column
of 2θ positions), and then least-squares refine the lattice parameters of the
structure against those peak positions.

The refinement is restricted to the lattice parameters only (a, b, c, α, β, γ),
which is the cell-refinement subset of a full Rietveld refinement.  By default
only the symmetry-independent parameters of the crystal system are refinable,
but the user may override this and treat the cell as triclinic.  Optional
zero-shift (2θ zero error) and sample-displacement corrections are also offered,
mirroring the common instrumental parameters used in Rietveld refinement.
"""

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from scipy.signal import find_peaks
from scipy.optimize import least_squares, differential_evolution, dual_annealing
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from helpers import load_structure, get_full_conventional_structure_diffra
from more_funct.xrd_nd_section import (
    _load_exp_xy,
    PRESET_OPTIONS,
    PRESET_WAVELENGTHS,
)

# Symmetry-independent lattice parameters per crystal system.  The remaining
# parameters are tied to these (or fixed at 90/120°) inside ``_assemble_cell``.
CRYSTAL_SYSTEM_FREE = {
    "cubic": ["a"],
    "tetragonal": ["a", "c"],
    "hexagonal": ["a", "c"],
    "trigonal": ["a", "c"],          # treated in the hexagonal setting
    "rhombohedral": ["a", "alpha"],
    "orthorhombic": ["a", "b", "c"],
    "monoclinic": ["a", "b", "c", "beta"],
    "triclinic": ["a", "b", "c", "alpha", "beta", "gamma"],
}

# Maximum number of peaks allowed for refinement on the shared online server.
# This is mainly a safety guard against pathological inputs — the dominant cost
# of the global search is the number of allowed reflections and optimizer
# iterations, not the peak count — so the cap can be generous. Unlimited locally.
ONLINE_MAX_PEAKS = 200

# Single-line (Kα1) presets only. The multi-line (Kα1+Kα2, +Kβ) presets are
# dropped — peak-position fitting uses a single wavelength. The λ field below the
# preset can still be edited freely for any other wavelength.
WAVELENGTH_PRESETS = [p for p in PRESET_OPTIONS if "+" not in p]

ALL_PARAMS = ["a", "b", "c", "alpha", "beta", "gamma"]
PARAM_LABELS = {
    "a": "a (Å)", "b": "b (Å)", "c": "c (Å)",
    "alpha": "α (°)", "beta": "β (°)", "gamma": "γ (°)",
}


def _assemble_cell(system, independent_vals, base):
    """Expand a set of independent parameters into the full (a,b,c,α,β,γ).

    ``independent_vals`` maps the symmetry-independent parameter names to their
    current values; ``base`` provides defaults for any parameter not given.
    The crystal-system ties are then applied.
    """
    g = dict(base)
    g.update(independent_vals)
    a, b, c = g["a"], g["b"], g["c"]
    al, be, ga = g["alpha"], g["beta"], g["gamma"]

    if system == "cubic":
        b = c = a
        al = be = ga = 90.0
    elif system == "tetragonal":
        b = a
        al = be = ga = 90.0
    elif system in ("hexagonal", "trigonal"):
        b = a
        al = be = 90.0
        ga = 120.0
    elif system == "rhombohedral":
        b = c = a
        be = ga = al
    elif system == "orthorhombic":
        al = be = ga = 90.0
    elif system == "monoclinic":
        al = ga = 90.0
    # triclinic: everything stays free

    return a, b, c, al, be, ga


def _style_chart(fig, tt_min, tt_max, height=580):
    """Apply the large-font chart styling used across the module."""
    fig.update_layout(
        height=height, hovermode="x unified",
        xaxis=dict(title=dict(text="2θ (°)", font=dict(size=30)),
                   tickfont=dict(size=24), range=[tt_min, tt_max]),
        yaxis=dict(title=dict(text="Intensity", font=dict(size=30)),
                   tickfont=dict(size=24)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=24)),
        hoverlabel=dict(font=dict(size=22)),
        font=dict(size=22),
        margin=dict(l=90, r=30, t=60, b=75))
    return fig


def _add_observed_peak_traces(fig, peaks, y_top, name="Selected peaks"):
    """Observed peaks as ONE toggleable trace: a faint full-height line per peak
    plus a triangle marker at the top, so the legend hides both together."""
    if not peaks:
        return
    xs, ys, sizes = [], [], []
    for p in peaks:
        xs += [p, p, None]
        ys += [0.0, y_top, None]
        sizes += [0, 13, 0]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=name,
        line=dict(color="rgba(214,39,40,0.30)", width=1),
        marker=dict(color="red", symbol="triangle-down", size=sizes),
        hovertemplate="peak 2θ = %{x:.3f}°<extra></extra>"))


def _add_stick_trace(fig, positions, heights, hovers, name, color, width=2):
    """Add vertical sticks (0 → height) for each peak, as in Powder Diffraction."""
    vx, vy, vh = [], [], []
    for x, h, txt in zip(positions, heights, hovers):
        vx.extend([x, x, None])
        vy.extend([0.0, h, None])
        vh.extend([txt, txt, None])
    fig.add_trace(go.Scatter(
        x=vx, y=vy, mode="lines", name=name,
        line=dict(color=color, width=width),
        text=vh, hoverinfo="text",
        hovertemplate="%{text}<extra></extra>"))


def _two_theta_from_d(d_arr, wavelength_A):
    """Bragg's law: 2θ (deg) for an array of d-spacings (Å)."""
    d_arr = np.asarray(d_arr, dtype=float)
    sin_theta = np.clip(wavelength_A / (2.0 * d_arr), -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(sin_theta))


def _displacement_2theta(two_theta_deg, displacement_mm, radius_mm):
    """Sample-displacement contribution to 2θ (deg)."""
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    return np.degrees(-2.0 * displacement_mm * np.cos(theta) / radius_mm)


def _predict_reflections(structure, wavelength_A, tt_min, tt_max):
    """Allowed (hkl) reflections of ``structure`` within the 2θ window.

    Returns a list of dicts with the representative ``hkl`` tuple and the
    calculated 2θ / d for the *initial* lattice (used only for indexing the
    observed peaks).
    """
    calc = XRDCalculator(wavelength=wavelength_A)
    pat = calc.get_pattern(
        structure,
        two_theta_range=(max(0.01, tt_min), min(179.9, tt_max)),
        scaled=False,
    )
    refl = []
    for x, y, d, hg in zip(pat.x, pat.y, pat.d_hkls, pat.hkls):
        if not hg:
            continue
        hkl = tuple(int(round(v)) for v in hg[0]["hkl"])
        if hkl == (0, 0, 0):
            continue
        refl.append({"hkl": hkl, "two_theta": float(x), "d": float(d),
                     "intensity": float(y)})
    # Normalise intensities to 100 for display.
    imax = max((r["intensity"] for r in refl), default=0.0)
    if imax > 0:
        for r in refl:
            r["intensity"] = r["intensity"] / imax * 100.0
    return refl


def _index_peaks(obs_two_theta, reflections, tol_deg):
    """Assign the nearest predicted reflection to each observed peak."""
    rows = []
    pred_tt = np.array([r["two_theta"] for r in reflections]) if reflections \
        else np.array([])
    for tt in obs_two_theta:
        if pred_tt.size == 0:
            rows.append({"obs": tt, "hkl": None, "pred": np.nan,
                         "intensity": 0.0, "matched": False})
            continue
        idx = int(np.argmin(np.abs(pred_tt - tt)))
        delta = abs(pred_tt[idx] - tt)
        rows.append({
            "obs": float(tt),
            "hkl": reflections[idx]["hkl"],
            "pred": float(pred_tt[idx]),
            "intensity": float(reflections[idx]["intensity"]),
            "matched": bool(delta <= tol_deg),
        })
    return rows


ALGORITHMS = [
    "Differential evolution (global)",
    "Dual annealing (global)",
    "Least squares (local)",
]


def _refine(matched, base_cell, system, wavelength_A,
            refine_params, fit_zero, fit_disp, radius_mm,
            max_change_pct=5.0, algorithm="Differential evolution (global)",
            all_hkls=None, progress_cb=None):
    """Refine the selected lattice parameters against the peak positions.

    ``matched`` is a list of dicts with keys ``obs`` (observed 2θ) and ``hkl``.
    ``max_change_pct`` constrains every refined parameter to stay within
    ±``max_change_pct`` % of its initial value, which keeps the fit physically
    reasonable and prevents the solver from running away on poorly indexed peaks.

    ``algorithm`` selects the optimiser. The global methods (differential
    evolution, dual annealing/simulated annealing) explore the whole bounded
    space and are far less likely to get stuck in a local minimum when the peaks
    are far apart; they are then polished with a local least-squares step so that
    1σ errors can still be estimated. ``"Least squares (local)"`` runs the local
    solver alone from the initial cell.

    Returns a dictionary with refined values, 1σ errors and per-peak residuals.
    """
    hkls = [m["hkl"] for m in matched]
    obs = np.array([m["obs"] for m in matched], dtype=float)
    frac = max(1e-4, max_change_pct / 100.0)

    # Build the variable vector: the refined lattice parameters first, then the
    # optional zero-shift and displacement corrections.
    var_names = list(refine_params)
    x0 = [base_cell[p] for p in var_names]
    lower = []
    upper = []
    for p in var_names:
        v = base_cell[p]
        lo = v * (1.0 - frac)
        hi = v * (1.0 + frac)
        if hi <= lo:
            hi = lo + 1e-6
        lower.append(lo)
        upper.append(hi)
    if fit_zero:
        var_names.append("_zero")
        x0.append(0.0)
        lower.append(-2.0)
        upper.append(2.0)
    if fit_disp:
        var_names.append("_disp")
        x0.append(0.0)
        lower.append(-2.0)
        upper.append(2.0)

    zero_idx = var_names.index("_zero") if "_zero" in var_names else None
    disp_idx = var_names.index("_disp") if "_disp" in var_names else None

    def _two_theta_for_hkls(x, hkl_list):
        independent = {}
        for name, val in zip(var_names, x):
            if name in ALL_PARAMS:
                independent[name] = val
        a, b, c, al, be, ga = _assemble_cell(system, independent, base_cell)
        latt = Lattice.from_parameters(a, b, c, al, be, ga)
        d_calc = np.array([latt.d_hkl(h) for h in hkl_list], dtype=float)
        tt = _two_theta_from_d(d_calc, wavelength_A)
        if zero_idx is not None:
            tt = tt + x[zero_idx]
        if disp_idx is not None:
            tt = tt + _displacement_2theta(tt, x[disp_idx], radius_mm)
        return tt

    # Candidate reflections used for *dynamic re-indexing* during the global
    # search: each observed peak is matched to the nearest predicted reflection
    # of the candidate cell, so the fit can recover even when the peaks have
    # shifted far from their initial positions.
    cand_hkls = list(all_hkls) if all_hkls else list(hkls)

    def _resid(x):
        return _two_theta_for_hkls(x, hkls) - obs

    def _obj_fixed(x):
        r = _resid(x)
        return float(np.dot(r, r))

    def _obj_reindex(x):
        pred = _two_theta_for_hkls(x, cand_hkls)
        diffs = np.abs(pred[None, :] - obs[:, None])
        return float(np.sum(np.min(diffs, axis=1) ** 2))

    lower_a = np.asarray(lower, dtype=float)
    upper_a = np.asarray(upper, dtype=float)
    n_data = len(obs)
    _t0 = time.time()

    # Global search (if requested) to escape local minima, then re-index every
    # observed peak to the nearest reflection of the global cell, then a local
    # least-squares polish for the final fit and its Jacobian. The optimisers
    # call ``progress_cb`` between iterations so the UI can report progress.
    DE_MAXITER, DA_MAXITER = 2000, 2000
    if algorithm.startswith("Differential evolution"):
        _state = {"it": 0}

        def _cb_de(xk, convergence=0.0, *args, **kwargs):
            _state["it"] += 1
            if progress_cb is not None:
                frac = min(max(float(convergence), 0.0), 1.0)
                rms = (max(_obj_reindex(xk), 0.0) / max(1, n_data)) ** 0.5
                progress_cb(iteration=_state["it"], frac=frac, rms=rms,
                            elapsed=time.time() - _t0)

        gres = differential_evolution(
            _obj_reindex, list(zip(lower, upper)), tol=1e-12, seed=0,
            polish=False, maxiter=DE_MAXITER, mutation=(0.5, 1.0),
            recombination=0.7, callback=_cb_de)
        x_start = np.clip(gres.x, lower_a, upper_a)
    elif algorithm.startswith("Dual annealing"):
        _state = {"it": 0, "best": np.inf}

        def _cb_da(x, f=np.inf, context=0, *args, **kwargs):
            _state["it"] += 1
            _state["best"] = min(_state["best"], float(f))
            if progress_cb is not None:
                rms = (max(_state["best"], 0.0) / max(1, n_data)) ** 0.5
                progress_cb(iteration=_state["it"], frac=None, rms=rms,
                            elapsed=time.time() - _t0)

        gres = dual_annealing(
            _obj_reindex, list(zip(lower, upper)), x0=x0, seed=0,
            maxiter=DA_MAXITER, callback=_cb_da)
        x_start = np.clip(gres.x, lower_a, upper_a)
    else:
        x_start = np.clip(np.asarray(x0, dtype=float), lower_a, upper_a)

    # Re-assign the (hkl) of every observed peak from the global solution.
    if not algorithm.startswith("Least squares"):
        pred = _two_theta_for_hkls(x_start, cand_hkls)
        hkls = [cand_hkls[int(np.argmin(np.abs(pred - o)))] for o in obs]

    res = least_squares(
        _resid, x_start, bounds=(lower, upper), method="trf",
        xtol=1e-12, ftol=1e-12,
    )

    residuals = res.fun
    n_data = len(obs)
    n_var = len(var_names)
    dof = max(1, n_data - n_var)
    mse = 2.0 * res.cost / dof  # res.cost = 0.5 * sum(resid**2)

    # Parameter standard errors from the Gauss-Newton covariance estimate.
    perr = np.full(n_var, np.nan)
    try:
        J = res.jac
        cov = np.linalg.inv(J.T @ J) * mse
        perr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        pass

    final_independent = {}
    errors = {}
    for name, val, err in zip(var_names, res.x, perr):
        if name in ALL_PARAMS:
            final_independent[name] = val
            errors[name] = err
    a, b, c, al, be, ga = _assemble_cell(system, final_independent, base_cell)
    latt = Lattice.from_parameters(a, b, c, al, be, ga)

    calc_tt = _two_theta_for_hkls(res.x, hkls)

    return {
        "cell": {"a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga},
        "errors": errors,
        "volume": latt.volume,
        "calc_two_theta": calc_tt,
        "hkls_used": hkls,
        "residuals": residuals,
        "rms": float(np.sqrt(np.mean(residuals ** 2))),
        "max_abs": float(np.max(np.abs(residuals))),
        "zero": res.x[var_names.index("_zero")] if "_zero" in var_names else None,
        "disp": res.x[var_names.index("_disp")] if "_disp" in var_names else None,
        "n_peaks": n_data,
        "success": res.success,
    }


def run_lattice_fitting_section(uploaded_files, user_pattern_file,
                                is_local=False):
    st.subheader("🎯 Fitting Lattice Parameters")
    st.markdown("Fitting lattice parameters on experimental peak positions.")

    pattern_files = (user_pattern_file
                     if isinstance(user_pattern_file, list)
                     else ([user_pattern_file] if user_pattern_file else []))

    if not pattern_files:
        st.info("⬅️ Upload at least one **experimental pattern** "
                "(.xy / .xrdml / .ras / two-column text) in the sidebar.")
        return
    if not uploaded_files:
        st.info("⬅️ Upload at least one **structure file** "
                "(CIF, POSCAR, …) in the sidebar.")
        return

    # ---------------------------------------------------------------- inputs
    col_a, col_b = st.columns(2)
    with col_a:
        exp_names = [f.name for f in pattern_files]
        sel_exp_name = st.selectbox(
            "Experimental pattern", exp_names, key="latfit_exp")
        sel_exp = pattern_files[exp_names.index(sel_exp_name)]
    with col_b:
        struct_names = [f.name for f in uploaded_files]
        sel_struct_name = st.selectbox(
            "Structure", struct_names, key="latfit_struct")
        sel_struct = uploaded_files[struct_names.index(sel_struct_name)]

    # Load experimental data.
    try:
        x_exp, y_exp = _load_exp_xy(sel_exp)
        x_exp = np.asarray(x_exp, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)
    except Exception as exc:
        st.error(f"Could not read experimental pattern '{sel_exp_name}': {exc}")
        return

    # Load structure + symmetry.
    try:
        struct = get_full_conventional_structure_diffra(
            load_structure(sel_struct))
        sga = SpacegroupAnalyzer(struct)
        crystal_system = sga.get_crystal_system()
        # Distinguish rhombohedral cells (a=b=c, α=β=γ≠90) from hexagonal-set
        # trigonal cells so that the right parameters are made independent.
        if crystal_system == "trigonal":
            lat = struct.lattice
            if (abs(lat.a - lat.b) < 1e-3 and abs(lat.b - lat.c) < 1e-3
                    and abs(lat.alpha - 90.0) > 1e-2):
                crystal_system = "rhombohedral"
        space_group = sga.get_space_group_symbol()
        lat = struct.lattice
        base_cell = {"a": lat.a, "b": lat.b, "c": lat.c,
                     "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma}
    except Exception as exc:
        st.error(f"Could not read structure '{sel_struct_name}': {exc}")
        return

    st.markdown(
        f"**Crystal system:** {crystal_system.capitalize()} &nbsp;&nbsp; "
        f"**Space group:** {space_group} &nbsp;&nbsp; "
        f"**Initial cell:** a={lat.a:.4f}, b={lat.b:.4f}, c={lat.c:.4f} Å, "
        f"α={lat.alpha:.3f}, β={lat.beta:.3f}, γ={lat.gamma:.3f}°"
    )

    # The fit button and the running/finished status live ABOVE the tabs, in
    # containers created first and filled later (once all settings are read).
    fit_button_area = st.container()
    fit_status_area = st.container()

    # Tabs (in display order): radiation/peaks/initial pattern first, then the
    # refinement options, the fitted result, and the refined-structure download.
    set_tab_peak, set_tab_rad, tab_fit, tab_dl = st.tabs(
        ["☢️ Radiation, peaks & initial pattern",
         "🎯 Refinement options",
         "✅ After fitting",
         "💾 Download refined structure"])

    # ----------------------------------------------------------- peak source
    # The peak list always lives in one editable text field — one 2θ value per
    # line — so a peak can be removed simply by deleting its line.
    txt_key = f"latfit_peaktext::{sel_exp_name}"

    obs_peaks = []
    with set_tab_peak:
        # ------------------------------------------- radiation & 2θ range
        # Wavelength and 2θ window are the settings placed above the chart.
        st.markdown("##### Radiation & 2θ range")
        cw1, cw2, cw3, cw4 = st.columns([1.4, 1, 1, 1])
        with cw1:
            cu_idx = WAVELENGTH_PRESETS.index("Copper (CuKa1)") \
                if "Copper (CuKa1)" in WAVELENGTH_PRESETS else 0
            preset = st.selectbox("Wavelength preset", WAVELENGTH_PRESETS,
                                  index=cu_idx, key="latfit_preset")
            # Sync the λ field to the preset: when the preset changes, push the
            # new wavelength into the number field's state (before the widget is
            # instantiated) so the theoretical pattern recalculates. The λ field
            # can still be edited freely afterwards for any other wavelength.
            if st.session_state.get("latfit_preset_prev") != preset:
                st.session_state["latfit_wl"] = float(
                    round(PRESET_WAVELENGTHS.get(preset, 0.15406) * 10.0, 5))
            st.session_state["latfit_preset_prev"] = preset
        with cw2:
            wavelength_A = st.number_input(
                "Wavelength λ (Å)", min_value=0.05, max_value=5.0,
                step=0.0001, format="%.5f", key="latfit_wl")
        data_min = float(np.nanmin(x_exp)) if x_exp.size else 5.0
        data_max = float(np.nanmax(x_exp)) if x_exp.size else 90.0
        with cw3:
            tt_min = st.number_input("2θ min (°)", value=round(data_min, 2),
                                     step=1.0, key="latfit_ttmin")
        with cw4:
            tt_max = st.number_input("2θ max (°)", value=round(data_max, 2),
                                     step=1.0, key="latfit_ttmax")

        mask = (x_exp >= tt_min) & (x_exp <= tt_max)
        x_win, y_win = x_exp[mask], y_exp[mask]
        y_top = float(np.max(y_win)) if y_win.size else 1.0

        # Theoretical reflections of the *initial* (unrefined) cell — used for
        # indexing the observed peaks and for the calculated pattern.
        reflections = _predict_reflections(struct, wavelength_A, tt_min, tt_max)

        # The initial chart is rendered here — directly below the wavelength
        # field and above the peak-detection controls.
        init_chart_area = st.container()

        # Peak source controls on the left, editable peak list on the right, so
        # the layout stays compact in height.
        col_mode, col_field = st.columns([1.1, 1])
        detected = []
        with col_mode:
            peak_mode = st.radio(
                "How to obtain the peak positions",
                ["Automatic peak detection", "Enter / paste 2θ positions",
                 "Upload a 2θ column file"],
                key="latfit_peakmode")

            if peak_mode == "Automatic peak detection":
                if y_win.size == 0:
                    st.warning("No experimental data in the selected 2θ "
                               "window.")
                    return
                y_norm = (y_win / np.max(y_win) * 100.0
                          if np.max(y_win) > 0 else y_win)
                min_prom = st.slider("Min. prominence (% of max)", 0.1, 50.0,
                                     2.0, 0.1, key="latfit_prom")
                min_height = st.slider("Min. height (% of max)", 0.0, 50.0,
                                       1.0, 0.5, key="latfit_height")
                min_sep = st.slider("Min. peak separation (°)", 0.05, 5.0, 0.3,
                                    0.05, key="latfit_sep")
                step = (float(np.median(np.diff(x_win)))
                        if x_win.size > 1 else 0.02)
                dist_pts = max(1, int(round(min_sep / step))) if step > 0 else 1
                peaks, _ = find_peaks(
                    y_norm, prominence=min_prom, height=min_height,
                    distance=dist_pts)
                detected = [round(float(x_win[i]), 4) for i in peaks]
                if txt_key not in st.session_state:
                    st.session_state[txt_key] = "\n".join(
                        f"{p:.4f}" for p in detected)
                if st.button(
                        f"🔄 Detect peaks ({len(detected)} found) — reset list",
                        key="latfit_detect", width="stretch"):
                    st.session_state[txt_key] = "\n".join(
                        f"{p:.4f}" for p in detected)
                    st.rerun()
            elif peak_mode == "Upload a 2θ column file":
                up = st.file_uploader(
                    "Upload a text file with one column of 2θ positions",
                    type=["txt", "dat", "csv", "xy"], key="latfit_upload")
            else:
                st.caption("Type or paste one 2θ value per line in the field "
                           "on the right.")

        with col_field:
            if peak_mode == "Upload a 2θ column file":
                obs_peaks = []
                if up is not None:
                    raw = up.read()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    obs_peaks = _parse_number_blob(raw)
                st.caption(f"Parsed **{len(obs_peaks)}** peak positions.")
            else:
                if txt_key not in st.session_state:
                    st.session_state[txt_key] = ""
                st.caption("One 2θ value per line — delete a line to drop a "
                           "peak, add a line for a new one.")
                text = st.text_area(
                    "Peak positions (2θ, °)", key=txt_key, height=260,
                    placeholder="28.4400\n47.3000\n56.1200\n...")
                obs_peaks = _parse_number_blob(text)

        obs_peaks = [p for p in obs_peaks if tt_min <= p <= tt_max]

        # Render the initial (experimental + theoretical) pattern into the
        # container created above the settings.
        with init_chart_area:
            st.markdown("##### Initial pattern (experimental + theoretical)")
            fig = go.Figure()
            if x_win.size:
                fig.add_trace(go.Scatter(
                    x=x_win, y=y_win, mode="lines", name=sel_exp_name,
                    line=dict(color="black", width=1.5)))
            # Theoretical calculated pattern from the selected CIF (initial
            # cell), as intensity-scaled vertical sticks in a faint colour so
            # the experimental data stays dominant.
            if reflections:
                _add_stick_trace(
                    fig,
                    [r["two_theta"] for r in reflections],
                    [r["intensity"] / 100.0 * y_top for r in reflections],
                    [f"(hkl) {''.join(str(v) for v in r['hkl'])}<br>"
                     f"2θ = {r['two_theta']:.3f}°<br>"
                     f"I = {r['intensity']:.1f}" for r in reflections],
                    name="Theoretical (initial cell)",
                    color="rgba(46,139,87,0.40)", width=2)
            _add_observed_peak_traces(fig, obs_peaks, y_top)
            _style_chart(fig, tt_min, tt_max)
            st.plotly_chart(fig, width="stretch", key="latfit_init_plot")
            st.caption(
                f"Initial cell: a={base_cell['a']:.4f}, b={base_cell['b']:.4f}, "
                f"c={base_cell['c']:.4f} Å, α={base_cell['alpha']:.3f}, "
                f"β={base_cell['beta']:.3f}, γ={base_cell['gamma']:.3f}° — "
                f"{len(obs_peaks)} selected peak(s), "
                f"{len(reflections)} theoretical reflection(s).")

    # ---------------------------------------------------- refinement options
    # Rendered into the radiation tab (it executes after the peak tab so that
    # obs_peaks is already known).
    with set_tab_rad:
        st.markdown("##### Refinement options")
        free = CRYSTAL_SYSTEM_FREE.get(crystal_system, ALL_PARAMS)

        ignore_sym = st.checkbox(
            "Ignore symmetry constraints (refine all six parameters, "
            "triclinic)", value=False, key="latfit_ignoresym")
        fit_system = "triclinic" if ignore_sym else crystal_system
        refinable = ALL_PARAMS if ignore_sym \
            else CRYSTAL_SYSTEM_FREE.get(crystal_system, ALL_PARAMS)

        st.caption("Select which lattice parameters to refine "
                   f"(symmetry-independent for **{crystal_system}**: "
                   f"{', '.join(PARAM_LABELS[p] for p in free)}).")
        refine_params = []
        pcols = st.columns(len(refinable))
        for i, p in enumerate(refinable):
            with pcols[i]:
                if st.checkbox(PARAM_LABELS[p], value=True,
                               key=f"latfit_ref_{p}"):
                    refine_params.append(p)

        algorithm = st.selectbox(
            "Optimization algorithm", ALGORITHMS, index=0, key="latfit_algo",
            help="Global methods (differential evolution, dual annealing / "
                 "simulated annealing) search the whole bounded space and are "
                 "much more likely to reach the global minimum when peaks are "
                 "far apart; they are then polished with a local least-squares "
                 "step. 'Least squares (local)' uses only the local solver.")

        max_change_pct = st.slider(
            "Max. allowed change per parameter (± %)", 0.1, 50.0, 10.0, 0.1,
            key="latfit_maxchange",
            help="Each refined lattice parameter is constrained to stay within "
                 "this percentage of its initial value, keeping the fit "
                 "physically reasonable.")

        co1, co2, co3 = st.columns(3)
        with co1:
            fit_zero = st.checkbox("Refine zero-shift (2θ offset)", value=True,
                                   key="latfit_zero")
        with co2:
            fit_disp = st.checkbox("Refine sample displacement", value=False,
                                   key="latfit_disp")
        with co3:
            radius_mm = st.number_input(
                "Goniometer radius (mm)", min_value=10.0, max_value=1000.0,
                value=173.0, step=1.0, key="latfit_radius",
                disabled=not fit_disp)

        tol_deg = st.slider(
            "Peak ↔ reflection matching tolerance (°)", 0.05, 2.0, 0.5, 0.05,
            key="latfit_tol",
            help="Used only by 'Least squares (local)': observed peaks farther "
                 "than this from an allowed reflection of the initial cell are "
                 "excluded. The global methods re-index every peak themselves "
                 "and ignore this tolerance.")

    # The fit button and the running/finished status are rendered into the
    # containers defined ABOVE the tabs, now that every setting is known.
    sig = (sel_exp_name, sel_struct_name, round(wavelength_A, 5))
    with fit_button_area:
        run = st.button("🎯 Fit lattice parameters", type="primary",
                        width="stretch", key="latfit_run")
        st.caption("Configure the wavelength, peaks and refinement options in "
                   "the tabs below, then press the button.")
        if is_local:
            st.caption("🖥️ Local version: unlimited number of peaks for the "
                       "refinement.")
        else:
            st.caption(f"☁️ Online version: at most **{ONLINE_MAX_PEAKS}** "
                       "peaks can be used for the refinement. Run the app "
                       "locally for an unlimited number.")

    if run:
        with fit_status_area:
            if not reflections:
                st.error("The structure has no allowed reflections in this 2θ "
                         "range for the chosen wavelength.")
            elif not obs_peaks:
                st.error("No peak positions available. Add peaks first.")
            elif not refine_params and not fit_zero and not fit_disp:
                st.error("Select at least one parameter to refine — a lattice "
                         "parameter, the zero-shift, or the sample displacement.")
            elif (not is_local) and len(obs_peaks) > ONLINE_MAX_PEAKS:
                st.error(
                    f"The online version allows at most {ONLINE_MAX_PEAKS} "
                    f"peaks for refinement (you selected {len(obs_peaks)}). "
                    "Please reduce the number of peaks, or run the app locally "
                    "for an unlimited number.")
            else:
                indexed = _index_peaks(obs_peaks, reflections, tol_deg)
                is_global = not algorithm.startswith("Least squares")
                if is_global:
                    # The global search re-indexes every peak against the
                    # candidate cell, so do not drop peaks by the initial-cell
                    # tolerance — all observed peaks take part in the fit.
                    matched = list(indexed)
                    unmatched = []
                else:
                    matched = [r for r in indexed if r["matched"]]
                    unmatched = [r for r in indexed if not r["matched"]]
                n_var = len(refine_params) + int(fit_zero) + int(fit_disp)
                if len(matched) < n_var:
                    st.error(
                        f"Only {len(matched)} peak(s) are available for the "
                        f"fit, but {n_var} parameter(s) are being refined. "
                        "Add more peaks or refine fewer parameters.")
                else:
                    with st.status("🔄 Fitting lattice parameters…",
                                   expanded=True) as status:
                        st.write("Fitting lattice parameters, please wait…")
                        st.write(f"Using **{len(matched)}** of "
                                 f"**{len(obs_peaks)}** peaks for the fit.")
                        _what = []
                        if refine_params:
                            _what.append(f"{len(refine_params)} lattice "
                                         "parameter(s)")
                        if fit_zero:
                            _what.append("zero-shift")
                        if fit_disp:
                            _what.append("sample displacement")
                        st.write(f"Refining {', '.join(_what)} — "
                                 f"{algorithm}…")
                        # Live progress: a bar (convergence) for differential
                        # evolution, and a text line for both global methods.
                        _prog_bar = (st.progress(0.0)
                                     if algorithm.startswith("Differential")
                                     else None)
                        _prog_txt = st.empty()

                        def _progress(iteration, frac, rms, elapsed):
                            if frac is not None and _prog_bar is not None:
                                _prog_bar.progress(min(max(frac, 0.0), 1.0))
                            _prog_txt.write(
                                f"Iteration {iteration} · best RMS Δ2θ ≈ "
                                f"{rms:.4f}°")

                        result = _refine(
                            matched, base_cell, fit_system, wavelength_A,
                            refine_params, fit_zero, fit_disp, radius_mm,
                            max_change_pct, algorithm,
                            all_hkls=[r["hkl"] for r in reflections],
                            progress_cb=_progress)
                        # Re-assign each peak's (hkl) / intensity from the fit
                        # (the global search may have re-indexed them).
                        _int_by_hkl = {r["hkl"]: r["intensity"]
                                       for r in reflections}
                        for _m, _h in zip(matched, result["hkls_used"]):
                            _m["hkl"] = _h
                            _m["intensity"] = _int_by_hkl.get(
                                _h, _m.get("intensity", 0.0))
                        # Initial (unrefined) calc 2θ for the matched peaks, and
                        # the RMS Δ2θ of that initial cell for a before/after
                        # comparison against the refined RMS.
                        init_calc = _two_theta_from_d(
                            np.array([Lattice.from_parameters(
                                base_cell["a"], base_cell["b"], base_cell["c"],
                                base_cell["alpha"], base_cell["beta"],
                                base_cell["gamma"]).d_hkl(m["hkl"])
                                for m in matched]), wavelength_A)
                        init_rms = float(np.sqrt(np.mean(
                            (np.asarray(init_calc, dtype=float)
                             - np.array([m["obs"] for m in matched],
                                        dtype=float)) ** 2)))
                        # Full theoretical pattern of the refined cell (all
                        # reflections), for the before/after comparison and for
                        # exporting the refined structure.
                        c = result["cell"]
                        refined_lat = Lattice.from_parameters(
                            c["a"], c["b"], c["c"],
                            c["alpha"], c["beta"], c["gamma"])
                        # Use per-site species (Composition) so disordered /
                        # partially-occupied structures are handled too.
                        refined_struct = Structure(
                            refined_lat,
                            [site.species for site in struct],
                            struct.frac_coords,
                            site_properties=struct.site_properties)
                        refined_reflections = _predict_reflections(
                            refined_struct, wavelength_A, tt_min, tt_max)
                        st.session_state["latfit_result"] = {
                            "sig": sig, "result": result, "matched": matched,
                            "unmatched": unmatched, "base_cell": base_cell,
                            "refine_params": refine_params, "tol_deg": tol_deg,
                            "radius_mm": radius_mm, "wavelength_A": wavelength_A,
                            "init_calc": list(map(float, init_calc)),
                            "init_rms": init_rms,
                            "struct_name": sel_struct_name,
                            "crystal_system": crystal_system,
                            "ignore_sym": ignore_sym,
                            "tt_min": float(tt_min), "tt_max": float(tt_max),
                            "reflections_init": reflections,
                            "reflections_refined": refined_reflections,
                            "refined_struct": refined_struct,
                        }
                        st.write(
                            f"Done — RMS Δ2θ: {init_rms:.4f}° (initial) → "
                            f"{result['rms']:.4f}° (refined).")
                        status.update(
                            label="✅ Refinement finished — see the "
                                  "“After fitting” tab.",
                            state="complete", expanded=True)
                    st.toast("Lattice parameter fit complete ✅")

    stored = st.session_state.get("latfit_result")
    has_fit = stored is not None and stored.get("sig") == sig

    with tab_fit:
        if not has_fit:
            st.info("Set up the radiation, peaks and refinement options, "
                    "then press **🎯 Fit lattice parameters** below.")
        else:
            _render_fit_results(stored, x_win, y_win, y_top, tt_min, tt_max)

    with tab_dl:
        if not has_fit:
            st.info("Run a fit first — the refined structure can then be "
                    "downloaded here.")
        else:
            _render_download_tab(stored)


def _render_download_tab(stored):
    """Export the structure with the refined lattice in the editor's formats."""
    refined_struct = stored.get("refined_struct")
    struct_name = stored["struct_name"]
    cell = stored["result"]["cell"]
    base_cell = stored["base_cell"]
    if refined_struct is None:
        st.warning("Refined structure is unavailable — please re-run the fit.")
        return

    st.markdown("#### Download structure with refined lattice parameters")
    st.caption(
        f"Initial cell:  a={base_cell['a']:.5f}, b={base_cell['b']:.5f}, "
        f"c={base_cell['c']:.5f} Å, α={base_cell['alpha']:.4f}, "
        f"β={base_cell['beta']:.4f}, γ={base_cell['gamma']:.4f}°")
    st.caption(
        f"Refined cell: a={cell['a']:.5f}, b={cell['b']:.5f}, "
        f"c={cell['c']:.5f} Å, α={cell['alpha']:.4f}, β={cell['beta']:.4f}, "
        f"γ={cell['gamma']:.4f}°, V={refined_struct.lattice.volume:.3f} Å³")
    st.info("The fractional coordinates are kept from the selected structure; "
            "only the lattice (cell metric) is replaced by the refined values.")

    # Reuse the Crystal Structure Editor's exporter for identical format options.
    from more_funct.structure_editor import _build_download_content

    base = struct_name.rsplit(".", 1)[0]
    dl_format = st.selectbox(
        "File format", ["CIF", "VASP (POSCAR)", "LAMMPS", "XYZ"],
        key="latfit_dl_format")
    try:
        content, default_name, mime = _build_download_content(
            refined_struct, dl_format, f"latfit_{base}")
    except Exception as exc:
        st.error(f"Could not build the {dl_format} file: {exc}")
        return

    if default_name == "POSCAR":
        fname = f"POSCAR_{base}_refined"
    else:
        ext = default_name.rsplit(".", 1)[-1]
        fname = f"{base}_refined.{ext}"
    data = content.encode("utf-8") if isinstance(content, str) else content
    st.download_button(
        f"⬇️ Download {dl_format}", data=data, file_name=fname, mime=mime,
        key="latfit_dl_struct", width="stretch", type="primary")


def _param_status(p, system, refine_params, ignore_sym):
    """Human-readable status of a lattice parameter in the refinement."""
    if p in refine_params:
        return "✔ refined"
    if ignore_sym:
        return "fixed (not refined)"
    free = CRYSTAL_SYSTEM_FREE.get(system, ALL_PARAMS)
    if p in free:
        return "fixed (not refined)"
    # Otherwise the parameter is determined by the crystal symmetry.
    ties = {
        "cubic": {"b": "= a (symmetry)", "c": "= a (symmetry)",
                  "alpha": "90° (symmetry)", "beta": "90° (symmetry)",
                  "gamma": "90° (symmetry)"},
        "tetragonal": {"b": "= a (symmetry)", "alpha": "90° (symmetry)",
                       "beta": "90° (symmetry)", "gamma": "90° (symmetry)"},
        "hexagonal": {"b": "= a (symmetry)", "alpha": "90° (symmetry)",
                      "beta": "90° (symmetry)", "gamma": "120° (symmetry)"},
        "trigonal": {"b": "= a (symmetry)", "alpha": "90° (symmetry)",
                     "beta": "90° (symmetry)", "gamma": "120° (symmetry)"},
        "rhombohedral": {"b": "= a (symmetry)", "c": "= a (symmetry)",
                         "beta": "= α (symmetry)", "gamma": "= α (symmetry)"},
        "orthorhombic": {"alpha": "90° (symmetry)", "beta": "90° (symmetry)",
                         "gamma": "90° (symmetry)"},
        "monoclinic": {"alpha": "90° (symmetry)", "gamma": "90° (symmetry)"},
    }
    return ties.get(system, {}).get(p, "constrained by symmetry")


def _render_fit_results(stored, x_win, y_win, y_top, tt_min, tt_max):
    result = stored["result"]
    matched = stored["matched"]
    unmatched = stored["unmatched"]
    base_cell = stored["base_cell"]
    refine_params = stored["refine_params"]
    wavelength_A = stored["wavelength_A"]
    init_calc = stored["init_calc"]
    struct_name = stored["struct_name"]
    crystal_system = stored.get("crystal_system", "triclinic")
    ignore_sym = stored.get("ignore_sym", False)

    if not result["success"]:
        st.warning("The least-squares solver did not fully converge — "
                   "inspect the residuals below.")

    # Fit overlay chart shown first: experimental curve, the observed peaks, and
    # the FULL theoretical pattern of both the initial and the refined cell, so
    # every peak is shown before (initial) and after (refined).
    refl_init = stored.get("reflections_init", [])
    refl_refined = stored.get("reflections_refined", [])
    fig_fit = go.Figure()
    if x_win.size:
        fig_fit.add_trace(go.Scatter(
            x=x_win, y=y_win, mode="lines", name="Experimental",
            line=dict(color="black", width=1.5)))
    _add_observed_peak_traces(fig_fit, [m["obs"] for m in matched], y_top,
                              name="Observed")
    if refl_init:
        _add_stick_trace(
            fig_fit,
            [r["two_theta"] for r in refl_init],
            [r["intensity"] / 100.0 * y_top for r in refl_init],
            [f"(hkl) {''.join(str(v) for v in r['hkl'])}<br>"
             f"initial 2θ = {r['two_theta']:.3f}°" for r in refl_init],
            name="Calc (initial cell)", color="rgba(255,165,0,0.55)", width=1.5)
    if refl_refined:
        _add_stick_trace(
            fig_fit,
            [r["two_theta"] for r in refl_refined],
            [r["intensity"] / 100.0 * y_top for r in refl_refined],
            [f"(hkl) {''.join(str(v) for v in r['hkl'])}<br>"
             f"refined 2θ = {r['two_theta']:.3f}°" for r in refl_refined],
            name="Calc (refined cell)", color="green", width=2)
    _style_chart(fig_fit, tt_min, tt_max)
    st.plotly_chart(fig_fit, width="stretch", key="latfit_result_plot")

    # Clean "a → new a" summary for every refined parameter.
    refined_now = [p for p in ALL_PARAMS if p in refine_params]
    if refined_now:
        st.markdown("#### Refined lattice parameters")
        mcols = st.columns(len(refined_now))
        for col, p in zip(mcols, refined_now):
            old = base_cell[p]
            new = result["cell"][p]
            pct = (new - old) / old * 100.0 if old else 0.0
            col.metric(
                PARAM_LABELS[p], f"{new:.5f}",
                delta=f"{new - old:+.5f} ({pct:+.2f}%)")
        summary_lines = []
        for p in refined_now:
            old, new = base_cell[p], result["cell"][p]
            err = result["errors"].get(p)
            err_txt = "" if (err is None or np.isnan(err)) else f" ± {err:.5f}"
            unit = "Å" if p in ("a", "b", "c") else "°"
            summary_lines.append(
                f"- **{PARAM_LABELS[p]}**: {old:.5f} → "
                f"**{new:.5f}{err_txt}** {unit}")
        st.markdown("\n".join(summary_lines))
    else:
        # No lattice parameter was refined — only instrumental corrections.
        # Show those as the headline result so the refined value is clear.
        st.markdown("#### Refined instrumental parameters")
        st.caption("No lattice parameter was refined — only the instrumental "
                   "correction(s) below were fitted; the cell is unchanged.")
        instr = []
        if result["zero"] is not None:
            instr.append(("Zero-shift (°)", f"{result['zero']:.4f}"))
        if result["disp"] is not None:
            instr.append(
                (f"Sample displacement (mm, R={stored['radius_mm']:.0f} mm)",
                 f"{result['disp']:.4f}"))
        if instr:
            icols = st.columns(len(instr))
            for col, (label, val) in zip(icols, instr):
                col.metric(label, val, delta=f"{float(val):+.4f} from 0")

    init_rms = stored.get("init_rms")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Peaks used", f"{result['n_peaks']}")
    m2.metric("Initial RMS Δ2θ (°)",
              "—" if init_rms is None else f"{init_rms:.4f}")
    m3.metric("Refined RMS Δ2θ (°)", f"{result['rms']:.4f}",
              delta=(None if init_rms is None
                     else f"{result['rms'] - init_rms:+.4f}"),
              delta_color="inverse")
    m4.metric("Max |Δ2θ| (°)", f"{result['max_abs']:.4f}")
    m5.metric("Cell volume (Å³)", f"{result['volume']:.3f}")

    # When lattice parameters were refined, list the instrumental corrections
    # here too; in the instrumental-only case they are already shown above as
    # metrics, so this line is skipped to avoid duplication.
    if refined_now:
        extra = []
        if result["zero"] is not None:
            extra.append(f"zero-shift = {result['zero']:.4f}°")
        if result["disp"] is not None:
            extra.append(f"sample displacement = {result['disp']:.4f} mm "
                         f"(R = {stored['radius_mm']:.0f} mm)")
        if extra:
            st.markdown("**Instrumental corrections:** " + ", ".join(extra))

    # Full initial-vs-refined table, with the status of every parameter
    # (refined / fixed / constrained by symmetry).
    rows = []
    for p in ALL_PARAMS:
        refined = result["cell"][p]
        err = result["errors"].get(p)
        rows.append({
            "Parameter": PARAM_LABELS[p],
            "Initial": round(base_cell[p], 5),
            "Refined": round(refined, 5),
            "1σ error": ("—" if (p not in refine_params or err is None
                                 or np.isnan(err)) else round(err, 5)),
            "Δ": round(refined - base_cell[p], 5),
            "Status": _param_status(p, crystal_system, refine_params,
                                    ignore_sym),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Parameter"), width="stretch")
    st.caption("**Status** — *refined*: optimised against the peaks; "
               "*fixed (not refined)*: independent but left at its initial "
               "value; *…(symmetry)*: determined by the crystal system.")

    # Observed vs calculated peak positions (initial and refined).
    obs_table = []
    for m, calc_tt, init_tt, resid in zip(
            matched, result["calc_two_theta"], init_calc, result["residuals"]):
        d_obs = wavelength_A / (2.0 * np.sin(np.radians(m["obs"] / 2.0)))
        obs_table.append({
            "hkl": "".join(str(v) if v >= 0 else f"-{abs(v)}" for v in m["hkl"]),
            "2θ obs (°)": round(m["obs"], 4),
            "2θ calc initial (°)": round(float(init_tt), 4),
            "2θ calc refined (°)": round(float(calc_tt), 4),
            "Δ2θ refined (°)": round(float(resid), 4),
            "d obs (Å)": round(float(d_obs), 5),
        })
    df_obs = pd.DataFrame(obs_table)
    st.markdown("#### Peak positions: observed vs calculated")
    st.dataframe(df_obs, width="stretch", hide_index=True)

    if unmatched:
        st.warning(
            f"{len(unmatched)} peak(s) were **not indexed** within "
            f"{stored['tol_deg']:.2f}° and were excluded: "
            + ", ".join(f"{r['obs']:.3f}°" for r in unmatched))

    # Downloads.
    summary = pd.DataFrame(rows)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "💾 Download refined parameters (CSV)",
            data=summary.to_csv(index=False),
            file_name=f"{struct_name.rsplit('.', 1)[0]}_refined_cell.csv",
            mime="text/csv", key="latfit_dl_params")
    with dl2:
        st.download_button(
            "💾 Download observed vs calculated (CSV)",
            data=df_obs.to_csv(index=False),
            file_name=f"{struct_name.rsplit('.', 1)[0]}_obs_calc.csv",
            mime="text/csv", key="latfit_dl_obs")


def _parse_number_blob(text):
    """Extract a sorted list of floats from free-form text."""
    if not text:
        return []
    import re
    vals = []
    for tok in re.split(r"[\s,;]+", str(text).strip()):
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return sorted(vals)
