import io
import os
import re
import tempfile

import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from streamlit_plotly_events import plotly_events

try:
    from xrd_rust_calculator import XRDCalculatorRust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from helpers import (
    load_structure,
    get_full_conventional_structure_diffra,
    rgb_color,
    convert_intensity_scale,
)

try:
    from helpers import (
        DEFAULT_TWO_THETA_MAX_FOR_PRESET,
        DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET,
    )
except ImportError:
    DEFAULT_TWO_THETA_MAX_FOR_PRESET = {}
    DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET = {}


MULTI_COMPONENT_PRESETS = {
    "Cu(Ka1+Ka2)": {
        "wavelengths": [0.15406, 0.15444],
        "factors":     [1.0,     1 / 2.0],
        "labels":      ["Kα1",   "Kα2"],
    },
    "Cu(Ka1+Ka2+Kb1)": {
        "wavelengths": [0.15406, 0.15444, 0.13922],
        "factors":     [1.0,     1 / 2.0, 1 / 9.0],
        "labels":      ["Kα1",   "Kα2",   "Kβ"],
    },
    "Mo(Ka1+Ka2)": {
        "wavelengths": [0.07093, 0.0711],
        "factors":     [1.0,     1 / 2.0],
        "labels":      ["Kα1",   "Kα2"],
    },
    "Mo(Ka1+Ka2+Kb1)": {
        "wavelengths": [0.07093, 0.0711, 0.064],
        "factors":     [1.0,     1 / 2.0, 1 / 9.0],
        "labels":      ["Kα1",   "Kα2",  "Kβ"],
    },
    "Cr(Ka1+Ka2)": {
        "wavelengths": [0.22897, 0.22888],
        "factors":     [1.0,     1 / 2.0],
        "labels":      ["Kα1",   "Kα2"],
    },
    "Cr(Ka1+Ka2+Kb1)": {
        "wavelengths": [0.22897, 0.22888, 0.208],
        "factors":     [1.0,     1 / 2.0, 1 / 9.0],
        "labels":      ["Kα1",   "Kα2",  "Kβ"],
    },
    "Fe(Ka1+Ka2)": {
        "wavelengths": [0.19360, 0.194],
        "factors":     [1.0,     1 / 2.0],
        "labels":      ["Kα1",   "Kα2"],
    },
    "Fe(Ka1+Ka2+Kb1)": {
        "wavelengths": [0.19360, 0.194, 0.176],
        "factors":     [1.0,     1 / 2.0, 1 / 9.0],
        "labels":      ["Kα1",   "Kα2",  "Kβ"],
    },
    "Co(Ka1+Ka2)": {
        "wavelengths": [0.17889, 0.17927],
        "factors":     [1.0,     1 / 2.0],
        "labels":      ["Kα1",   "Kα2"],
    },
    "Co(Ka1+Ka2+Kb1)": {
        "wavelengths": [0.17889, 0.17927, 0.163],
        "factors":     [1.0,     1 / 2.0, 1 / 9.0],
        "labels":      ["Kα1",   "Kα2",  "Kβ"],
    },
    "Ag(Ka1+Ka2)": {
        "wavelengths": [0.0561, 0.05634],
        "factors":     [1.0,    1 / 2.0],
        "labels":      ["Kα1",  "Kα2"],
    },
    "Ag(Ka1+Ka2+Kb1)": {
        "wavelengths": [0.0561, 0.05634, 0.0496],
        "factors":     [1.0,    1 / 2.0, 1 / 9.0],
        "labels":      ["Kα1",  "Kα2",  "Kβ"],
    },
}

PRESET_OPTIONS = [
    "Cobalt (CoKa1)", "Copper (CuKa1)", "Molybdenum (MoKa1)",
    "Chromium (CrKa1)", "Iron (FeKa1)", "Silver (AgKa1)",
    "Co(Ka1+Ka2)", "Co(Ka1+Ka2+Kb1)",
    "Mo(Ka1+Ka2)", "Mo(Ka1+Ka2+Kb1)",
    "Cu(Ka1+Ka2)", "Cu(Ka1+Ka2+Kb1)",
    "Cr(Ka1+Ka2)", "Cr(Ka1+Ka2+Kb1)",
    "Fe(Ka1+Ka2)", "Fe(Ka1+Ka2+Kb1)",
    "Ag(Ka1+Ka2)", "Ag(Ka1+Ka2+Kb1)",
]

PRESET_WAVELENGTHS = {
    "Copper (CuKa1)":      0.15406,
    "Cu(Ka1+Ka2)":         0.154,
    "CuKa2":               0.15444,
    "Cu(Ka1+Ka2+Kb1)":     0.153339,
    "CuKb1":               0.13922,
    "Molybdenum (MoKa1)":  0.07093,
    "Mo(Ka1+Ka2)":         0.071,
    "MoKa2":               0.0711,
    "Mo(Ka1+Ka2+Kb1)":     0.07059119,
    "MoKb1":               0.064,
    "Chromium (CrKa1)":    0.22897,
    "Cr(Ka1+Ka2)":         0.229,
    "CrKa2":               0.22888,
    "Cr(Ka1+Ka2+Kb1)":     0.22775471,
    "CrKb1":               0.208,
    "Iron (FeKa1)":        0.19360,
    "Fe(Ka1+Ka2)":         0.194,
    "FeKa2":               0.194,
    "Fe(Ka1+Ka2+Kb1)":     0.1927295,
    "FeKb1":               0.176,
    "Cobalt (CoKa1)":      0.17889,
    "Co(Ka1+Ka2)":         0.179,
    "CoKa2":               0.17927,
    "Co(Ka1+Ka2+Kb1)":     0.1781100,
    "CoKb1":               0.163,
    "Silver (AgKa1)":      0.0561,
    "AgKa2":               0.05634,
    "Ag(Ka1+Ka2)":         0.0561,
    "AgKb1":               0.0496,
    "Ag(Ka1+Ka2+Kb1)":     0.0557006,
}

PRESET_OPTIONS_NEUTRON = ["Thermal Neutrons", "Cold Neutrons", "Hot Neutrons"]
PRESET_WAVELENGTHS_NEUTRON = {
    "Thermal Neutrons": 0.154,
    "Cold Neutrons":    0.475,
    "Hot Neutrons":     0.087,
}

X_AXIS_OPTIONS = [
    "2θ (°)", "2θ (rad)", "θ (°)", "θ (rad)",
    "q (1/Å)", "q (1/nm)",
    "d (Å)", "d (nm)",
    "energy (keV)", "frequency (PHz)",
]

X_AXIS_OPTIONS_NEUTRON = [
    "2θ (°)", "2θ (rad)", "θ (°)", "θ (rad)",
    "q (1/Å)", "q (1/nm)",
    "d (Å)", "d (nm)",
]


def format_index(index, first=False, last=False):
    s = str(index)
    if s.startswith("-") and len(s) == 2:
        return s
    elif first and len(s) == 2:
        return s + " "
    elif last and len(s) == 2:
        return " " + s + " "
    elif len(s) >= 2:
        return " " + s + " "
    return s


def twotheta_to_metric(twotheta_deg, metric, wavelength_A, wavelength_nm,
                       diffraction_choice):
    twotheta_deg = np.asarray(twotheta_deg, dtype=float)
    theta = np.deg2rad(twotheta_deg / 2.0)
    if metric == "2θ (°)":
        result = twotheta_deg
    elif metric == "2θ (rad)":
        result = np.deg2rad(twotheta_deg)
    elif metric == "θ (°)":
        result = twotheta_deg / 2.0
    elif metric == "θ (rad)":
        result = np.deg2rad(twotheta_deg / 2.0)
    elif metric == "q (1/Å)":
        result = (4 * np.pi / wavelength_A) * np.sin(theta)
    elif metric == "q (1/nm)":
        result = (4 * np.pi / wavelength_nm) * np.sin(theta)
    elif metric == "d (Å)":
        result = np.where(np.sin(theta) == 0, np.inf,
                          wavelength_A / (2 * np.sin(theta)))
    elif metric == "d (nm)":
        result = np.where(np.sin(theta) == 0, np.inf,
                          wavelength_nm / (2 * np.sin(theta)))
    elif metric == "energy (keV)":
        if diffraction_choice == "ND (Neutron)":
            return float(0.003956 / (wavelength_nm ** 2))
        result = (24.796 * np.sin(theta)) / wavelength_A
    elif metric == "frequency (PHz)":
        result = ((24.796 * np.sin(theta)) / wavelength_A * 2.418e17) / 1e15
    else:
        result = twotheta_deg
    return (float(result) if np.ndim(twotheta_deg) == 0 else result)


def metric_to_twotheta(metric_value, metric, wavelength_A, wavelength_nm,
                       diffraction_choice):
    if metric == "2θ (°)":
        return metric_value
    elif metric == "2θ (rad)":
        return np.rad2deg(metric_value)
    elif metric == "q (1/Å)":
        theta = np.arcsin(np.clip(
            metric_value * wavelength_A / (4 * np.pi), 0, 1))
        return np.rad2deg(2 * theta)
    elif metric == "θ (°)":
        return 2 * metric_value
    elif metric == "θ (rad)":
        return 2 * np.rad2deg(metric_value)
    elif metric == "q (1/nm)":
        theta = np.arcsin(np.clip(
            metric_value * wavelength_nm / (4 * np.pi), 0, 1))
        return np.rad2deg(2 * theta)
    elif metric == "d (Å)":
        sin_theta = np.clip(wavelength_A / (2 * metric_value), 0, 1)
        return np.rad2deg(2 * np.arcsin(sin_theta))
    elif metric == "d (nm)":
        sin_theta = np.clip(wavelength_nm / (2 * metric_value), 0, 1)
        return np.rad2deg(2 * np.arcsin(sin_theta))
    elif metric == "energy (keV)":
        if diffraction_choice == "ND (Neutron)":
            lam_nm = np.sqrt(0.003956 / metric_value)
            sin_t  = np.clip(lam_nm / (2 * wavelength_nm), 0, 1)
        else:
            sin_t = np.clip(metric_value * wavelength_A / 24.796, 0, 1)
        return np.rad2deg(2 * np.arcsin(np.clip(sin_t, 0, 1)))
    elif metric == "frequency (PHz)":
        f_Hz  = metric_value * 1e15
        E_keV = f_Hz / 2.418e17
        sin_t = np.clip(E_keV * wavelength_A / 24.796, 0, 1)
        return np.rad2deg(2 * np.arcsin(sin_t))
    return metric_value


def energy_to_wavelength(energy_kev):
    return 1.2398 / energy_kev


def wavelength_to_energy(wavelength_nm):
    return 1.2398 / wavelength_nm


def hkl_str(hkl_group):
    if len(hkl_group[0]["hkl"]) == 3:
        return ", ".join(
            f"({format_index(h['hkl'][0], first=True)}"
            f"{format_index(h['hkl'][1])}"
            f"{format_index(h['hkl'][2], last=True)})"
            for h in hkl_group
        )
    else:
        return ", ".join(
            f"({format_index(h['hkl'][0], first=True)}"
            f"{format_index(h['hkl'][1])}"
            f"{format_index(h['hkl'][3], last=True)})"
            for h in hkl_group
        )


def _is_zero_hkl(hkl_group):
    for h in hkl_group:
        if len(h["hkl"]) == 3 and tuple(h["hkl"][:3]) == (0, 0, 0):
            return True
        if len(h["hkl"]) == 4 and tuple(h["hkl"][:4]) == (0, 0, 0, 0):
            return True
    return False



def _file_fingerprint(f):
    try:
        size = f.size
    except AttributeError:
        try:
            pos = f.tell()
            f.seek(0, 2)
            size = f.tell()
            f.seek(pos)
        except Exception:
            size = 0
    return (f.name, size)


def _cache_key(uploaded_files, wavelength_A, diffraction_choice,
               use_debye_waller, debye_waller_factors_per_file, preset_choice):
    file_key = tuple(sorted(_file_fingerprint(f) for f in uploaded_files))
    dw_key   = None
    if use_debye_waller and debye_waller_factors_per_file:
        dw_key = tuple(
            (fn, tuple(sorted(fac.items())))
            for fn, fac in sorted(debye_waller_factors_per_file.items())
        )
    return (file_key, round(wavelength_A, 7), diffraction_choice,
            use_debye_waller, dw_key, preset_choice)



def _get_calculator(diffraction_choice, wavelength_A, dw_dict, use_rust):
    if diffraction_choice == "ND (Neutron)":
        return NDCalculator(wavelength=wavelength_A,
                            debye_waller_factors=dw_dict)
    if use_rust and HAS_RUST:
        return XRDCalculatorRust(wavelength=wavelength_A,
                                 debye_waller_factors=dw_dict)
    return XRDCalculator(wavelength=wavelength_A,
                         debye_waller_factors=dw_dict)


def _load_mg_structure(file):
    from pymatgen.core import Structure as PmgStructure

    if file.name.lower().endswith(".cif"):
        mg = load_structure(file)
    else:
        mg = load_structure(file)
        cif_writer = CifWriter(mg, symprec=0.01)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif",
                                         delete=False) as tmp:
            cif_writer.write_file(tmp.name)
            tmp_path = tmp.name
        try:
            mg = PmgStructure.from_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return get_full_conventional_structure_diffra(mg)


def _calculate_raw_patterns(uploaded_files, wavelength_A, diffraction_choice,
                             use_debye_waller, debye_waller_factors_per_file,
                             use_rust, preset_choice):
    full_range   = (0.01, 179.9)
    is_multi     = preset_choice in MULTI_COMPONENT_PRESETS
    raw_patterns = {}

    for file in uploaded_files:
        dw_dict = None
        if use_debye_waller and debye_waller_factors_per_file:
            dw_dict = debye_waller_factors_per_file.get(file.name)

        try:
            mg = _load_mg_structure(file)
        except Exception as exc:
            st.warning(f"Could not load '{file.name}': {exc}")
            raw_patterns[file.name] = dict(raw_x=np.array([]), raw_y=np.array([]),
                                           raw_hkls=[], peak_types=[])
            continue

        all_x, all_y, all_hkls, all_types = [], [], [], []

        if is_multi:
            comp   = MULTI_COMPONENT_PRESETS[preset_choice]
            labels = comp.get("labels", ["Kα1"] * len(comp["wavelengths"]))
            for ci, (wl_nm, factor) in enumerate(
                    zip(comp["wavelengths"], comp["factors"])):
                wl_A = wl_nm * 10.0
                calc = _get_calculator(diffraction_choice, wl_A, dw_dict,
                                       use_rust)
                try:
                    pat = calc.get_pattern(mg, two_theta_range=full_range,
                                           scaled=False)
                except Exception:
                    continue
                lbl = labels[ci] if ci < len(labels) else "Kα1"
                for xv, yv, hg in zip(pat.x, pat.y, pat.hkls):
                    if _is_zero_hkl(hg):
                        continue
                    all_x.append(xv)
                    all_y.append(yv * factor)
                    all_hkls.append(hg)
                    all_types.append(lbl)
        else:
            calc = _get_calculator(diffraction_choice, wavelength_A, dw_dict,
                                   use_rust)
            try:
                pat = calc.get_pattern(mg, two_theta_range=full_range,
                                       scaled=False)
            except Exception:
                raw_patterns[file.name] = dict(raw_x=np.array([]),
                                               raw_y=np.array([]),
                                               raw_hkls=[], peak_types=[])
                continue
            for xv, yv, hg in zip(pat.x, pat.y, pat.hkls):
                if _is_zero_hkl(hg):
                    continue
                all_x.append(xv)
                all_y.append(yv)
                all_hkls.append(hg)
                all_types.append("Kα1")

        raw_patterns[file.name] = dict(
            raw_x=np.array(all_x), raw_y=np.array(all_y),
            raw_hkls=all_hkls,    peak_types=all_types,
        )

    return raw_patterns



_DENSE_X = np.linspace(0.01, 179.9, 8000)
_DENSE_DX = _DENSE_X[1] - _DENSE_X[0]

PROFILE_OPTIONS = [
    "Delta (stick)",
    "Gaussian",
    "Lorentzian",
    "Pseudo-Voigt",
    "Pearson VII",
]

SCHERRER_K = 0.9


def _fwhm_at_twotheta(two_theta_deg, wavelength_A, crystallite_size_nm,
                       instrumental_fwhm_deg):
    theta_rad = np.deg2rad(np.asarray(two_theta_deg, dtype=float) / 2.0)
    cos_theta = np.cos(theta_rad)
    L_A = crystallite_size_nm * 10.0
    beta_rad = (SCHERRER_K * wavelength_A) / (L_A * cos_theta)
    beta_deg = np.rad2deg(beta_rad)
    return np.sqrt(beta_deg**2 + instrumental_fwhm_deg**2)


def _sigma_from_fwhm(fwhm_deg):
    return fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _profile_curve(x_arr, peak_pos, fwhm, profile, eta=0.5, m=1.5,
                   norm="area"):
    dx = x_arr - peak_pos

    if profile == "Gaussian":
        sigma = _sigma_from_fwhm(fwhm)
        y = np.exp(-dx**2 / (2.0 * sigma**2))

    elif profile == "Lorentzian":
        gamma = fwhm / 2.0
        y = 1.0 / (1.0 + (dx / gamma)**2)

    elif profile == "Pseudo-Voigt":
        sigma = _sigma_from_fwhm(fwhm)
        gamma = fwhm / 2.0
        G = np.exp(-dx**2 / (2.0 * sigma**2))
        L = 1.0 / (1.0 + (dx / gamma)**2)
        y = eta * L + (1.0 - eta) * G

    elif profile == "Pearson VII":
        denom = 2.0 * np.sqrt(2.0**(1.0 / m) - 1.0)
        w = fwhm / denom if denom > 0 else fwhm
        y = (1.0 + (dx / w)**2) ** (-m)

    else:
        sigma = _sigma_from_fwhm(fwhm)
        y = np.exp(-dx**2 / (2.0 * sigma**2))

    if norm == "height":
        peak_max = float(np.max(y))
        return y / peak_max if peak_max > 0 else y
    else:
        dx_step = x_arr[1] - x_arr[0] if len(x_arr) > 1 else 1.0
        area = np.sum(y) * dx_step
        return y / area if area > 0 else y


def _process_for_display(raw_patterns, two_theta_min, two_theta_max,
                         intensity_filter, peak_representation, sigma,
                         intensity_scale_option, y_axis_scale, x_axis_metric,
                         wavelength_A, wavelength_nm, diffraction_choice,
                         num_annotate,
                         use_scherrer=False, crystallite_size_nm=100.0,
                         pseudo_voigt_eta=0.5, pearson_m=1.5,
                         profile_norm="area"):
    instrumental_fwhm = sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))
    pattern_details = {}

    for file_name, raw in raw_patterns.items():
        raw_x, raw_y = raw["raw_x"], raw["raw_y"]
        raw_hkls, raw_types = raw["raw_hkls"], raw["peak_types"]

        if len(raw_x) == 0:
            pattern_details[file_name] = dict(
                peak_vals=np.array([]), intensities=np.array([]),
                hkls=[], peak_types=[], annotate_indices=set(),
                x_dense_full=_DENSE_X,
                y_dense=np.zeros_like(_DENSE_X),
            )
            continue

        in_range = (raw_x >= two_theta_min) & (raw_x <= two_theta_max)
        y_in_rng = raw_y[in_range]
        max_rng  = float(np.max(y_in_rng)) if len(y_in_rng) > 0 else 1.0
        thresh   = (intensity_filter / 100.0) * max_rng if intensity_filter > 0 else 0.0

        fx, fy, fhkls, ftypes = [], [], [], []
        for i in range(len(raw_x)):
            if raw_x[i] < two_theta_min or raw_x[i] > two_theta_max:
                continue
            if intensity_filter > 0 and raw_y[i] < thresh:
                continue
            fx.append(raw_x[i])
            fy.append(raw_y[i])
            fhkls.append(raw_hkls[i])
            ftypes.append(raw_types[i])

        fx = np.array(fx)
        fy = np.array(fy)

        y_dense = np.zeros_like(_DENSE_X)
        if peak_representation == "Delta (stick)":
            for peak, inten in zip(fx, fy):
                idx = int(np.argmin(np.abs(_DENSE_X - peak)))
                y_dense[idx] += inten
        else:
            for peak, inten in zip(fx, fy):
                if use_scherrer:
                    fwhm = _fwhm_at_twotheta(peak, wavelength_A,
                                             crystallite_size_nm,
                                             instrumental_fwhm)
                else:
                    fwhm = instrumental_fwhm
                curve = _profile_curve(_DENSE_X, peak, fwhm,
                                       peak_representation,
                                       eta=pseudo_voigt_eta, m=pearson_m,
                                       norm=profile_norm)
                y_dense += inten * curve

        if y_axis_scale != "Linear":
            y_dense    = convert_intensity_scale(y_dense, y_axis_scale)
            fy_display = convert_intensity_scale(fy,      y_axis_scale)
        else:
            fy_display = fy.copy()

        if intensity_scale_option == "Normalized":
            pk_max = float(np.max(fy_display)) if len(fy_display) > 0 else 1.0
            if pk_max > 0:
                fy_display = (fy_display / pk_max) * 100.0
            d_max = float(np.max(y_dense))
            if d_max > 0:
                y_dense = (y_dense / d_max) * 100.0

        peak_vals = (
            twotheta_to_metric(fx, x_axis_metric, wavelength_A,
                               wavelength_nm, diffraction_choice)
            if len(fx) > 0 else np.array([])
        )

        ka1_idx  = [i for i, pt in enumerate(ftypes) if pt == "Kα1"]
        if ka1_idx and len(fy_display) > 0:
            sorted_ka1 = sorted(
                ((i, float(fy_display[i])) for i in ka1_idx),
                key=lambda t: t[1], reverse=True,
            )
            annotate_indices = {i for i, _ in sorted_ka1[:num_annotate]}
        else:
            annotate_indices = set()

        pattern_details[file_name] = dict(
            peak_vals=peak_vals,
            intensities=fy_display,
            hkls=fhkls,
            peak_types=ftypes,
            annotate_indices=annotate_indices,
            x_dense_full=_DENSE_X,
            y_dense=y_dense,
        )

    return pattern_details




def _tab_background_subtraction(user_pattern_file, x_axis_metric):
    st.subheader("📉 Background Subtraction")
    if not user_pattern_file:
        st.info("Upload experimental data files to use background subtraction.")
        return

    files = user_pattern_file if isinstance(user_pattern_file, list) \
        else [user_pattern_file]

    if len(files) > 1:
        selected_exp_name = st.selectbox(
            "Select file for background subtraction",
            [f.name for f in files],
            key="bg_file_select",
        )
        selected_file_obj = next(f for f in files if f.name == selected_exp_name)
    else:
        selected_file_obj  = files[0]
        selected_exp_name  = selected_file_obj.name

    try:
        df      = pd.read_csv(selected_file_obj, sep=r"\s+|,|;",
                              engine="python", header=None, skiprows=1)
        x_exp   = df.iloc[:, 0].values
        y_exp   = df.iloc[:, 1].values

        if "original_exp_data" not in st.session_state:
            st.session_state.original_exp_data = {}
        if selected_exp_name not in st.session_state.original_exp_data:
            st.session_state.original_exp_data[selected_exp_name] = {
                "x": x_exp.copy(), "y": y_exp.copy()
            }

        bg_method = st.radio(
            "Background Estimation Method",
            ["None", "Polynomial Fit", "SNIP Algorithm",
             "Rolling Ball Algorithm", "airPLS (Adaptive Baseline)"],
            index=0, key="bg_method", horizontal=True,
        )

        if bg_method == "None":
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scatter(x=x_exp, y=y_exp, mode="lines",
                                         name="Data", line=dict(color="black", width=2)))
            fig_raw.update_layout(
                title="Experimental Data Preview",
                xaxis_title=x_axis_metric, yaxis_title="Intensity (a.u.)",
                height=450, hovermode="x unified",
            )
            st.plotly_chart(fig_raw, use_container_width=True)
            return

        from scipy.signal import savgol_filter as _savgol

        if bg_method == "Polynomial Fit":
            poly_degree = st.slider("Polynomial Degree", 1, 15, 6, 1,
                                    help="Higher degree follows the baseline more closely.")
            n_iters = st.slider("Iterations", 1, 200, 40, 1,
                                help="More iterations suppress peaks more aggressively before fitting.")
            sort_idx = np.argsort(x_exp)
            xs, ys_sorted = x_exp[sort_idx], y_exp[sort_idx]
            ywork = ys_sorted.copy().astype(float)
            for _ in range(n_iters):
                try:
                    coeffs = np.polyfit(xs, ywork, poly_degree)
                    fitted = np.polyval(coeffs, xs)
                except Exception:
                    fitted = ywork
                ywork = np.minimum(ywork, fitted)
            coeffs     = np.polyfit(xs, ywork, poly_degree)
            bg_sorted  = np.polyval(coeffs, xs)
            unsort     = np.argsort(sort_idx)
            background = bg_sorted[unsort]

        elif bg_method == "SNIP Algorithm":
            snip_iters  = st.slider("SNIP Iterations", 1, 100, 20, 1)
            snip_window = st.slider("Window Size", 3, 101, 21, 2)
            if snip_window % 2 == 0:
                snip_window += 1
            half = snip_window // 2
            y_bg = y_exp.copy()
            for _ in range(snip_iters):
                for i in range(half, len(y_exp) - half):
                    y_bg[i] = min(y_bg[i],
                                  min(y_bg[i - half], y_bg[i + half]))
            background = y_bg

        elif bg_method == "Rolling Ball Algorithm":
            ball_radius   = st.slider("Ball Radius", 1, 100, 30, 1)
            ball_smoothing = st.slider("Smoothing Passes", 0, 10, 3, 1)
            ys = y_exp.copy()
            if ball_smoothing > 0:
                wlen = min(len(ys) // 10, 20)
                if wlen % 2 == 0:
                    wlen += 1
                if wlen >= 3:
                    for _ in range(ball_smoothing):
                        ys = _savgol(ys, wlen, 2)
            y_bg = np.array([
                np.min(ys[max(0, i - ball_radius):
                          min(len(ys), i + ball_radius + 1)])
                for i in range(len(ys))
            ])
            background = y_bg

        else:
            from scipy.sparse import diags as _sp_diags, eye as _sp_eye
            from scipy.sparse.linalg import spsolve as _spsolve

            lam = st.select_slider(
                "Smoothness (λ)",
                options=[1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 1e8],
                value=1e6,
                format_func=lambda v: f"{v:.0e}",
                help="Higher = smoother/flatter baseline. 1e5–1e7 suits most diffraction data.",
            )
            p = st.slider(
                "Asymmetry (p)", 0.001, 0.05, 0.01, 0.001, format="%.3f",
                help="Fraction of points considered background. "
                     "Lower = baseline hugs the valley more tightly.",
            )
            n_iter = st.slider("Iterations", 5, 50, 15, 1)

            y = y_exp.astype(float)
            n = len(y)
            D = _sp_diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
            H = lam * D.T.dot(D)
            w = np.ones(n)
            z = y.copy()
            for _ in range(n_iter):
                W = _sp_diags(w, 0, shape=(n, n))
                z = _spsolve(W + H, w * y)
                w = p * (y > z) + (1 - p) * (y <= z)
            background = z

        y_bg_sub = np.maximum(0, y_exp - background)

        if "bg_subtracted_data" not in st.session_state:
            st.session_state.bg_subtracted_data = {}
        st.session_state.bg_subtracted_data[selected_exp_name] = {
            "x": x_exp, "y": y_bg_sub, "background": background,
        }

        if st.button("Permanently Apply This Background Subtraction",
                     type="primary"):
            if "permanent_exp_data" not in st.session_state:
                st.session_state.permanent_exp_data = {}
            st.session_state.permanent_exp_data[selected_exp_name] = {
                "x": x_exp.copy(), "y": y_bg_sub.copy(),
                "background": background.copy(),
            }
            st.success(f"Background subtraction applied to {selected_exp_name}!")

            col1, col2 = st.columns(2)
            with col1:
                dl_data = "\n".join(
                    f"{r[0]:.6f}  {r[1]:.6f}"
                    for r in np.column_stack((x_exp, y_bg_sub))
                )
                st.download_button(
                    "💾 Download Background-Subtracted Data (.xy)",
                    data="# X-axis  Intensity (Background Subtracted)\n" + dl_data,
                    file_name=f"{selected_exp_name.rsplit('.', 1)[0]}_bg_subtracted.xy",
                    mime="text/plain", type="primary",
                )
            with col2:
                bg_data = "\n".join(
                    f"{r[0]:.6f}  {r[1]:.6f}"
                    for r in np.column_stack((x_exp, background))
                )
                st.download_button(
                    "💾 Download Background Curve (.xy)",
                    data="# X-axis  Background\n" + bg_data,
                    file_name=f"{selected_exp_name.rsplit('.', 1)[0]}_background.xy",
                    mime="text/plain", type="primary",
                )

        fig_bg = go.Figure()
        for trace_y, name, color in [
            (y_exp,      "Original Data",        "black"),
            (background, "Estimated Background", "red"),
            (y_bg_sub,   "After Subtraction",    "blue"),
        ]:
            fig_bg.add_trace(go.Scatter(
                x=x_exp, y=trace_y, mode="lines", name=name,
                line=dict(color=color, width=3),
            ))
        fig_bg.update_layout(
            title=dict(text="Background Subtraction", font=dict(size=32)),
            xaxis=dict(title=dict(text=x_axis_metric, font=dict(size=28)),
                       tickfont=dict(size=24)),
            yaxis=dict(title=dict(text="Intensity (a.u.)", font=dict(size=28)),
                       tickfont=dict(size=24)),
            legend=dict(font=dict(size=24), orientation="h",
                        yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=600, hovermode="x unified",
            hoverlabel=dict(font=dict(size=20)),
        )
        st.plotly_chart(fig_bg, use_container_width=True)

        use_bg = st.checkbox("Use background-subtracted data for visualization",
                             value=True)
        st.session_state.use_bg_subtracted       = use_bg
        st.session_state.active_bg_subtracted_file = selected_exp_name if use_bg \
            else None

    except Exception as exc:
        st.error(f"Error processing {selected_exp_name}: {exc}")



def _load_exp_xy(file_obj):
    fname = file_obj.name

    if ("permanent_exp_data" in st.session_state
            and fname in st.session_state.permanent_exp_data):
        d = st.session_state.permanent_exp_data[fname]
        return d["x"], d["y"]

    if (st.session_state.get("use_bg_subtracted")
            and st.session_state.get("active_bg_subtracted_file") == fname
            and "bg_subtracted_data" in st.session_state
            and fname in st.session_state.bg_subtracted_data):
        d = st.session_state.bg_subtracted_data[fname]
        return d["x"], d["y"]

    file_obj.seek(0)
    raw = file_obj.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+|,|;",
                     engine="python", header=None, skiprows=1)
    file_obj.seek(0)
    return df.iloc[:, 0].values, df.iloc[:, 1].values



def _add_exp_traces(fig, user_pattern_file, intensity_scale_option,
                    y_axis_scale, two_theta_min, two_theta_max, x_axis_metric,
                    wavelength_A, wavelength_nm, diffraction_choice):
    if not user_pattern_file:
        return
    files  = user_pattern_file if isinstance(user_pattern_file, list) \
        else [user_pattern_file]
    colors = ["black", "brown", "grey", "purple"]

    for i, fobj in enumerate(files):
        try:
            x_u, y_u = _load_exp_xy(fobj)
        except Exception:
            continue
        if y_axis_scale != "Linear":
            y_u = convert_intensity_scale(y_u, y_axis_scale)
        if intensity_scale_option == "Normalized" and np.max(y_u) > 0:
            y_u = (y_u / np.max(y_u)) * 100.0

        mask = (x_u >= two_theta_min) & (x_u <= two_theta_max)
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_u[mask], y=y_u[mask],
            mode="lines", name=fobj.name,
            line=dict(dash="solid", width=1, color=color),
            hovertemplate=(
                f"<span style='color:{color};'><b>{fobj.name}:</b><br>"
                "x = %{x:.2f}<br>Intensity = %{y:.2f}</span><extra></extra>"
            ),
        ))


def _plot_patterns(fig, pattern_details, uploaded_files, preset_choice,
                   peak_representation, line_thickness, x_axis_metric,
                   wavelength_A, wavelength_nm, diffraction_choice,
                   two_theta_min, two_theta_max,
                   show_Kalpha1_hover=True,
                   show_Kalpha2_hover=False,
                   show_Kbeta_hover=False,
                   show_delta_ref=True):
    tab10 = plt.cm.tab10.colors

    for idx, file in enumerate(uploaded_files):
        fname   = file.name
        if fname not in pattern_details:
            continue
        details = pattern_details[fname]

        base_color = rgb_color(tab10[idx % len(tab10)], opacity=0.8)
        mask = ((details["x_dense_full"] >= two_theta_min) &
                (details["x_dense_full"] <= two_theta_max))
        x_rng = twotheta_to_metric(
            details["x_dense_full"][mask], x_axis_metric,
            wavelength_A, wavelength_nm, diffraction_choice)
        y_rng = details["y_dense"][mask]

        if peak_representation != "Delta (stick)" and show_delta_ref:
            stem_groups = {}
            for i, pv in enumerate(details["peak_vals"]):
                pt = details["peak_types"][i]
                can = metric_to_twotheta(pv, x_axis_metric, wavelength_A,
                                         wavelength_nm, diffraction_choice)
                if not (two_theta_min <= can <= two_theta_max):
                    continue
                hover_txt = f"(hkl): {hkl_str(details['hkls'][i])}"
                g = stem_groups.setdefault(pt, {"vx": [], "vy": [], "vh": []})
                g["vx"].extend([pv, pv, None])
                g["vy"].extend([0, details["intensities"][i], None])
                g["vh"].extend([hover_txt, hover_txt, None])

            pt_style = {
                "Kα1": dict(opacity=0.18, width=1.2, dash="solid"),
                "Kα2": dict(opacity=0.12, width=0.9, dash="dot"),
                "Kβ":  dict(opacity=0.10, width=0.8, dash="dash"),
            }
            for pt, g in stem_groups.items():
                style = pt_style.get(pt, dict(opacity=0.10, width=0.8, dash="dot"))
                ref_color = rgb_color(tab10[idx % len(tab10)], opacity=style["opacity"])
                fig.add_trace(go.Scatter(
                    x=g["vx"], y=g["vy"], mode="lines",
                    name=f"{fname} ({pt} ref)",
                    showlegend=False,
                    line=dict(color=ref_color, width=style["width"],
                              dash=style["dash"]),
                    text=g["vh"],
                    hovertemplate=(
                        f"<b>{fname} {pt}</b><br>"
                        f"<b>{x_axis_metric}: %{{x:.4f}}</b><br>"
                        "Intensity: %{y:.2f}<br>"
                        "<b>%{text}</b><extra></extra>"
                    ),
                    hoverlabel=dict(
                        bgcolor=rgb_color(tab10[idx % len(tab10)], opacity=0.7),
                        font=dict(color="white", size=20)),
                ))

        if peak_representation == "Delta (stick)":
            if "peak_types" in details:
                groups = {}
                for i, pv in enumerate(details["peak_vals"]):
                    can = metric_to_twotheta(pv, x_axis_metric, wavelength_A,
                                            wavelength_nm, diffraction_choice)
                    if not (two_theta_min <= can <= two_theta_max):
                        continue
                    pt = details["peak_types"][i]
                    groups.setdefault(pt, {"x": [], "y": [], "hover": []})
                    groups[pt]["x"].append(pv)
                    groups[pt]["y"].append(details["intensities"][i])
                    groups[pt]["hover"].append(
                        f"(hkl): {hkl_str(details['hkls'][i])}")

                for pt, data in groups.items():
                    if pt == "Kα1":
                        pt_color   = base_color
                        dash_type  = "solid"
                        skip_hover = False
                    elif pt == "Kα2":
                        pt_color   = rgb_color(tab10[idx % len(tab10)],
                                               opacity=0.6)
                        dash_type  = "dot"
                        skip_hover = True
                    else:
                        pt_color   = rgb_color(tab10[idx % len(tab10)],
                                               opacity=0.4)
                        dash_type  = "dash"
                        skip_hover = True

                    vx, vy, vh = [], [], []
                    for j in range(len(data["x"])):
                        vx.extend([data["x"][j], data["x"][j], None])
                        vy.extend([0,             data["y"][j], None])
                        vh.extend([data["hover"][j], data["hover"][j], None])

                    fig.add_trace(go.Scatter(
                        x=vx, y=vy, mode="lines",
                        name=f"{fname} - {pt}",
                        line=dict(color=pt_color, width=line_thickness,
                                  dash=dash_type),
                        hoverinfo="skip" if skip_hover else "text",
                        text=vh,
                        hovertemplate=(
                            None if skip_hover else
                            f"<br>{fname} - {pt}<br>"
                            f"<b>{x_axis_metric}: %{{x:.2f}}</b><br>"
                            f"Intensity: %{{y:.2f}}<br><b>%{{text}}</b>"
                            "<extra></extra>"
                        ),
                        hoverlabel=dict(
                            bgcolor=pt_color,
                            font=dict(color="white", size=24)),
                    ))
            else:
                vx, vy, vh = [], [], []
                for i, pv in enumerate(details["peak_vals"]):
                    can = metric_to_twotheta(pv, x_axis_metric, wavelength_A,
                                            wavelength_nm, diffraction_choice)
                    if not (two_theta_min <= can <= two_theta_max):
                        continue
                    vx.extend([pv, pv, None])
                    vy.extend([0, details["intensities"][i], None])
                    txt = f"(hkl): {hkl_str(details['hkls'][i])}"
                    vh.extend([txt, txt, None])
                fig.add_trace(go.Scatter(
                    x=vx, y=vy, mode="lines", name=fname,
                    line=dict(color=base_color, width=line_thickness),
                    hoverinfo="text", text=vh,
                    hovertemplate=(
                        f"<br>{fname}<br>"
                        f"<b>{x_axis_metric}: %{{x:.2f}}</b><br>"
                        "Intensity: %{y:.2f}<br><b>%{text}</b><extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor=base_color,
                                    font=dict(color="white", size=24)),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x_rng, y=y_rng, mode="lines", name=fname,
                line=dict(color=base_color, width=line_thickness),
                hoverinfo="skip",
            ))



def _tab_annotation(pattern_details, uploaded_files, fig_interactive,
                    x_axis_metric, wavelength_A, wavelength_nm,
                    diffraction_choice, two_theta_min, two_theta_max):
    st.subheader("🏷️ Annotate Specific Planes")
    c1, c2, c3, c4 = st.columns(4)
    h_idx = c1.number_input("h index", -10, 10, 0, 1, key="h_plane")
    k_idx = c2.number_input("k index", -10, 10, 0, 1, key="k_plane")
    l_idx = c3.number_input("l index", -10, 10, 1, 1, key="l_plane")
    max_m = c4.number_input("Max multiple", 1, 20, 5, 1, key="max_mult")

    plane_hkl = (h_idx, k_idx, l_idx)

    if not st.button("Generate Annotated Plot", type="primary"):
        return

    def _equiv(hkl_base, sg_ops, crystal_system):
        h, k, l = hkl_base
        equivs = set()
        if sg_ops:
            try:
                for op in sg_ops:
                    r = op.rotation_matrix
                    if crystal_system in ("hexagonal", "trigonal"):
                        he, ke, le = [int(v) for v in r.dot([h, k, l])]
                        equivs.add((he, ke, -(he + ke), le))
                        equivs.add((-he, -ke, -((-he) + (-ke)), -le))
                    else:
                        equivs.add(tuple(int(v) for v in r.dot([h, k, l])))
                        equivs.add(tuple(-int(v) for v in r.dot([h, k, l])))
            except Exception:
                pass
        equivs.add((h, k, l))
        equivs.add((-h, -k, -l))
        return equivs

    def _multiples(hkl_base, max_n, sg_ops, crystal_system):
        all_m = set()
        for eh in _equiv(hkl_base, sg_ops, crystal_system):
            for n in range(1, max_n + 1):
                all_m.add(tuple(v * n for v in eh))
                all_m.add(tuple(-v * n for v in eh))
        return all_m

    sg_infos = {}
    for file in uploaded_files:
        try:
            struct = load_structure(file)
            sga    = SpacegroupAnalyzer(struct)
            sg_infos[file.name] = {
                "symbol":        sga.get_space_group_symbol(),
                "operations":    sga.get_space_group_operations(),
                "crystal_system": sga.get_crystal_system(),
            }
        except Exception:
            sg_infos[file.name] = {
                "symbol":        "Unknown",
                "operations":    None,
                "crystal_system": "unknown",
            }

    st.write("**Space Groups:**")
    for fn, info in sg_infos.items():
        st.write(f"• {fn}: {info['symbol']} ({info['crystal_system']})")

    fig_ann = go.Figure(fig_interactive)
    total   = 0

    for file in uploaded_files:
        fname   = file.name
        if fname not in pattern_details:
            continue
        details = pattern_details[fname]
        info    = sg_infos.get(fname, {})
        multis  = _multiples(
            plane_hkl, max_m,
            info.get("operations"),
            info.get("crystal_system", "unknown"),
        )

        ann_x, ann_y, ann_txt = [], [], []
        for i, (pv, inten, hg) in enumerate(
                zip(details["peak_vals"], details["intensities"],
                    details["hkls"])):
            for hd in hg:
                hkl = hd["hkl"]
                cmp = tuple(hkl[:4]) if len(hkl) == 4 else tuple(hkl[:3])
                if cmp in multis:
                    can = metric_to_twotheta(
                        pv, x_axis_metric, wavelength_A,
                        wavelength_nm, diffraction_choice)
                    if two_theta_min <= can <= two_theta_max:
                        ann_x.append(pv)
                        ann_y.append(inten)
                        ann_txt.append(
                            f"({' '.join(str(v) for v in hkl)})")
                        total += 1

        if ann_x:
            fig_ann.add_trace(go.Scatter(
                x=ann_x, y=ann_y, mode="markers+text",
                text=ann_txt, textposition="top center",
                textfont=dict(size=20, color="red", family="Arial Black"),
                marker=dict(size=12, color="red", symbol="diamond",
                            line=dict(width=2, color="darkred")),
                name=f"{fname} – {plane_hkl} family "
                     f"({info.get('symbol','?')})",
            ))

    fig_ann.update_layout(
        title=f"Diffraction Pattern — {plane_hkl} Family Annotations",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_ann, use_container_width=True, key="annotated_plot")
    st.success(
        f"Annotated {total} peak(s) for the {plane_hkl} family "
        "(symmetry-aware)."
    )



def _tab2_quantitative(pattern_details, uploaded_files, x_axis_metric):
    import base64

    st.subheader("Quantitative Data for Calculated Diffraction Patterns")

    for file in uploaded_files:
        fname   = file.name
        if fname not in pattern_details:
            continue
        details = pattern_details[fname]

        with st.expander(f"View All Peak Data: **{fname}**"):
            lines = ["#X-axis    Intensity    hkl"]
            for pv, inten, hg in zip(details["peak_vals"],
                                     details["intensities"],
                                     details["hkls"]):
                lines.append(
                    f"{float(pv):<12.3f} {float(inten):<12.3f} "
                    f"{hkl_str(hg)}"
                )
            st.code("\n".join(lines), language="text")

        with st.expander(
                f"View Highest Intensity Peaks: **{fname}**", expanded=True):
            lines = ["#X-axis    Intensity    hkl"]
            for i, (pv, inten, hg) in enumerate(
                    zip(details["peak_vals"],
                        details["intensities"],
                        details["hkls"])):
                if i in details["annotate_indices"]:
                    lines.append(
                        f"{float(pv):<12.3f} {float(inten):<12.3f} "
                        f"{hkl_str(hg)}"
                    )
            st.code("\n".join(lines), language="text")

        btn_key = f"prepare_download_{fname}"
        if btn_key not in st.session_state:
            st.session_state[btn_key] = False

        st.button(
            f"Download Continuous Curve Data for {fname}",
            key=f"button_{fname}",
            on_click=lambda k=btn_key: st.session_state.update({k: True}),
        )
        if st.session_state[btn_key]:
            df = pd.DataFrame({
                "X-axis":  details["x_dense_full"],
                "Y-value": details["y_dense"],
            })
            csv  = df.to_csv(index=False)
            b64  = base64.b64encode(csv.encode()).decode()
            link = (f'<a href="data:file/csv;base64,{b64}" '
                    f'download="curve_{fname.replace(".", "_")}.csv">'
                    f'Download Continuous Curve Data for {fname}</a>')
            st.markdown(link, unsafe_allow_html=True)

    view_combined = st.checkbox(
        "📈 View peak data across all structures in an interactive table")
    if view_combined:
        with st.expander("📊 Combined Peak Data", expanded=True):
            rows = []
            for file in uploaded_files:
                fname   = file.name
                if fname not in pattern_details:
                    continue
                details = pattern_details[fname]
                for pv, inten, hg in zip(details["peak_vals"],
                                          details["intensities"],
                                          details["hkls"]):
                    rows.append([float(pv), float(inten), hkl_str(hg), fname])
            df_comb = pd.DataFrame(
                rows, columns=[x_axis_metric, "Intensity", "(hkl)", "Phase"])
            st.dataframe(df_comb)



def _diffraction_settings_ui():

    defaults = dict(
        peak_representation="Delta (stick)",
        intensity_scale_option="Normalized",
        diffraction_choice="XRD (X-ray)",
        _prev_diffraction_choice="XRD (X-ray)",
        line_thickness=2.0,
        use_debye_waller=False,
        wavelength_value=0.17889,
        sigma=0.050,
        profile_norm="area",
        x_axis_metric="2θ (°)",
        y_axis_scale="Linear",
        intensity_filter=0.0,
        num_annotate=5,
        two_theta_min=5.0,
        two_theta_max=165.0,
        input_mode="Preset",
        preset_choice="Cobalt (CoKa1)",
        energy_kev=8.048,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.expander("Diffraction Settings", icon="⚙️",
                     expanded=st.session_state.get(
                         "expander_diff_settings", True)):

        _h_col, _rust_col = st.columns([5, 1])
        with _h_col:
            st.subheader("⚙️ Diffraction Settings")
        with _rust_col:
            if HAS_RUST:
                use_rust = st.checkbox(
                    "⚡",
                    value=st.session_state.get("use_rust_cb", True),
                    key="use_rust_cb",
                    help="⚡ checked = Rust-accelerated calculator (fast)\n\n"
                         "Unchecked = original pymatgen implementation\n\n"
                        "[📄 Details (arXiv paper)](https://arxiv.org/abs/2602.11709)",

                )
            else:
                use_rust = False

        set_tab1, set_tab2, set_tab3 = st.tabs(["⚙️ General", "📐 Axes", "🔔 Broadening"])

        with set_tab1:

            diffraction_choice = st.radio(
                "Diffraction Calculator",
                ["XRD (X-ray)", "ND (Neutron)"],
                key="diffraction_choice",
                horizontal=True,
            )

            col_c, col_d = st.columns(2)
            with col_c:
                intensity_scale_option = st.radio(
                    "Intensity scale",
                    ["Normalized", "Absolute"],
                    key="intensity_scale_option",
                )
            with col_d:
                line_thickness = st.slider(
                    "⚙️ Line thickness:", 0.5, 6.0, step=0.1,
                    key="line_thickness",
                )

            use_debye_waller = st.checkbox(
                "✓ Apply Debye-Waller temperature factors",
                key="use_debye_waller",
            )
            if use_debye_waller:
                _debye_waller_ui()

            prev_diff = st.session_state.get("_prev_diffraction_choice")
            if prev_diff != diffraction_choice:
                if diffraction_choice == "XRD (X-ray)":
                    st.session_state.wavelength_value  = PRESET_WAVELENGTHS["Cobalt (CoKa1)"]
                    st.session_state.preset_choice     = "Cobalt (CoKa1)"
                    st.session_state._prev_preset      = "Cobalt (CoKa1)"
                    st.session_state.two_theta_max     = 165.0
                    st.session_state.two_theta_min     = 5.0
                else:
                    st.session_state.wavelength_value       = PRESET_WAVELENGTHS_NEUTRON["Thermal Neutrons"]
                    st.session_state.preset_choice_neutron  = "Thermal Neutrons"
                    st.session_state._prev_preset_nd        = "Thermal Neutrons"
                    st.session_state.two_theta_max          = 165.0
                    st.session_state.two_theta_min          = 5.0
                st.session_state.input_mode      = "Preset"
                st.session_state.last_input_mode = "Preset"
                st.session_state.raw_patterns_cache_key = None
                st.session_state["_prev_diffraction_choice"] = diffraction_choice

            if diffraction_choice == "XRD (X-ray)":
                wavelength_value, preset_choice = _xrd_wavelength_ui()
            else:
                wavelength_value, preset_choice = _nd_wavelength_ui()

            wavelength_A  = wavelength_value * 10.0
            wavelength_nm = wavelength_value

            _, col_ann = st.columns(2)
            with col_ann:
                num_annotate = st.number_input(
                    "⚙️ Peaks to annotate in table:",
                    min_value=0, max_value=30,
                    value=st.session_state.num_annotate,
                    step=1, key="num_annotate_widget")
                st.session_state.num_annotate = num_annotate

            intensity_filter = st.slider(
                "⚙️ Filter peaks (% of max intensity):",
                min_value=0.0, max_value=50.0, step=0.1,
                key="intensity_filter_widget")
            st.session_state.intensity_filter = intensity_filter

            if diffraction_choice == "ND (Neutron)":
                if st.button("Calculate ND", type="primary"):
                    st.session_state.calc_xrd = True
                    st.session_state.raw_patterns_cache_key = None

            else:
                if st.button("Calculate XRD", type="primary"):
                    st.session_state.calc_xrd = True
                    st.session_state.raw_patterns_cache_key = None


            _calc_status = st.empty()

            _calc_t = st.session_state.get("last_calc_time_s")
            if _calc_t is not None:
                if _calc_t < 1.0:
                    _t_str = f"{_calc_t * 1000:.0f} ms"
                else:
                    _t_str = f"{_calc_t:.2f} s"
                _calc_status.caption(f"⏱ Last calculation: {_t_str}")
            st.session_state["_calc_status_placeholder"] = _calc_status

        with set_tab2:

            col_x, col_y = st.columns(2)
            with col_x:
                opts = (X_AXIS_OPTIONS_NEUTRON if diffraction_choice == "ND (Neutron)"
                        else X_AXIS_OPTIONS)
                x_axis_metric = st.selectbox("X-axis metric", opts,
                                              key="x_axis_metric")
            with col_y:
                y_axis_scale = st.selectbox(
                    "Y-axis scale",
                    ["Linear", "Square Root", "Logarithmic"],
                    key="y_axis_scale",
                )

            y_axis_title = {
                "Linear":      "Intensity (a.u.)",
                "Square Root": "√Intensity (a.u.)",
                "Logarithmic": "log₁₀(Intensity) (a.u.)",
            }[y_axis_scale]

            disp_min = twotheta_to_metric(
                st.session_state.two_theta_min, x_axis_metric,
                wavelength_A, wavelength_nm, diffraction_choice)
            disp_max = twotheta_to_metric(
                st.session_state.two_theta_max, x_axis_metric,
                wavelength_A, wavelength_nm, diffraction_choice)

            step = 0.0174533 if x_axis_metric == "2θ (rad)" else 0.1
            if x_axis_metric == "2θ (°)":
                step = 1.0

            col_rmin, col_rmax = st.columns(2)
            if x_axis_metric in ("d (Å)", "d (nm)"):
                raw_max = col_rmin.number_input(
                    f"Maximum {x_axis_metric}", value=disp_min, step=step,
                    key=f"min_val_{x_axis_metric}")
                raw_min = col_rmax.number_input(
                    f"Minimum {x_axis_metric}", value=disp_max, step=step,
                    key=f"max_val_{x_axis_metric}")
            else:
                raw_min = col_rmin.number_input(
                    f"Minimum {x_axis_metric}", value=disp_min, step=step,
                    key=f"min_val_{x_axis_metric}")
                raw_max = col_rmax.number_input(
                    f"Maximum {x_axis_metric}", value=disp_max, step=step,
                    key=f"max_val_{x_axis_metric}")

            if x_axis_metric in ("d (Å)", "d (nm)"):
                st.session_state.two_theta_min = metric_to_twotheta(
                    raw_max, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
                st.session_state.two_theta_max = metric_to_twotheta(
                    raw_min, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
            else:
                st.session_state.two_theta_min = metric_to_twotheta(
                    raw_min, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
                st.session_state.two_theta_max = metric_to_twotheta(
                    raw_max, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)

        with set_tab3:

            _wl_A = wavelength_A

            peak_representation = st.selectbox(
                "Peak Profile",
                PROFILE_OPTIONS,
                index=PROFILE_OPTIONS.index(
                    st.session_state.get("peak_representation", "Delta (stick)")),
                key="peak_representation",
                help=(
                    "**Delta (stick):** Infinitely sharp peaks — fastest, best for "
                    "phase identification.\n\n"
                    "**Gaussian:** Symmetric bell curve; good approximation for "
                    "instrumental broadening.\n\n"
                    "**Lorentzian:** Heavier tails than Gaussian; typical for "
                    "size-broadened peaks.\n\n"
                    "**Pseudo-Voigt:** Linear mix of Gaussian + Lorentzian — "
                    "standard in Rietveld refinement (set η below).\n\n"
                    "**Pearson VII:** Generalised profile controlled by exponent m "
                    "(m=1 → Lorentzian, m→∞ → Gaussian)."
                ),
            )

            is_delta = (peak_representation == "Delta (stick)")

            if is_delta:
                st.info("Delta (stick) uses no broadening — switch to another "
                        "profile to enable peak shape and broadening parameters.")
                sigma               = st.session_state.get("sigma", 0.050)
                pseudo_voigt_eta    = st.session_state.get("pseudo_voigt_eta", 0.5)
                pearson_m           = st.session_state.get("pearson_m", 1.5)
                use_scherrer        = False
                crystallite_size_nm = st.session_state.get("crystallite_size_nm", 100.0)
                show_delta_ref      = True
                profile_norm        = st.session_state.get("profile_norm", "area")
            else:
                show_delta_ref = st.checkbox(
                    "Show reference delta lines",
                    value=st.session_state.get("show_delta_ref", True),
                    key="show_delta_ref",
                    help="Display faint vertical stems at each peak position "
                         "for easy identification when a shaped profile is active.",
                )
                st.markdown("#### Instrumental broadening")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    sigma = st.number_input(
                        "Instrumental σ (°)",
                        min_value=0.01, max_value=5.0,
                        value = 0.05,
                        step=0.01, format="%.3f", key="sigma",
                        help="Gaussian σ for the instrumental broadening contribution. "
                             "FWHM = σ × 2√(2 ln 2) = 2.354820 × σ",
                        on_change=lambda: None)
                with col_s2:
                    fwhm_instr = sigma * 2.3548200450309493
                    st.metric("Instrumental FWHM", f"{fwhm_instr:.4f} °")

                if peak_representation == "Pseudo-Voigt":
                    pseudo_voigt_eta = st.slider(
                        "η — mixing (0 = pure Gaussian, 1 = pure Lorentzian)",
                        min_value=0.0, max_value=1.0,
                        value=st.session_state.get("pseudo_voigt_eta", 0.5),
                        step=0.01, key="pseudo_voigt_eta")
                else:
                    pseudo_voigt_eta = st.session_state.get("pseudo_voigt_eta", 0.5)

                if peak_representation == "Pearson VII":
                    pearson_m = st.slider(
                        "m — shape exponent (1 = Lorentzian, >10 ≈ Gaussian)",
                        min_value=0.5, max_value=20.0,
                        value=st.session_state.get("pearson_m", 1.5),
                        step=0.1, key="pearson_m")
                else:
                    pearson_m = st.session_state.get("pearson_m", 1.5)

                st.markdown("#### 📊 Intensity scaling")
                profile_norm = st.radio(
                    "Profile intensity scaling",
                    ["area", "height"],
                    index=0,
                    horizontal=True,
                    key="profile_norm",
                    help=(
                        "**area** — Unit-area normalisation: integral under each peak = "
                        "theoretical intensity. Narrow peaks (small σ) are taller than "
                        "the delta sticks; physically correct for integrated intensity.\n\n"
                        "**height** — Unit-height normalisation: peak apex always equals "
                        "the theoretical intensity (= delta-stick height). Broader peaks "
                        "are shorter. Convention used by VESTA, FullProf, GSAS-II."
                    ),
                )

                st.markdown("#### 🔬 Scherrer size broadening")
                use_scherrer = st.checkbox(
                    "Enable Scherrer peak broadening (crystallite size)",
                    value=st.session_state.get("use_scherrer", False),
                    key="use_scherrer",
                    help="Adds angle-dependent broadening via the Scherrer equation: "
                         "β = Kλ / (L cos θ),  K = 0.9.  "
                         "Combined with instrumental σ in quadrature.")
                if use_scherrer:
                    col_sc1, col_sc2 = st.columns(2)
                    with col_sc1:
                        crystallite_size_nm = st.number_input(
                            "Crystallite size L (nm)",
                            min_value=1.0, max_value=10000.0,
                            value=st.session_state.get("crystallite_size_nm", 100.0),
                            step=1.0, format="%.1f", key="crystallite_size_nm",
                            help="Volume-weighted mean column length. "
                                 "Smaller → broader peaks.")
                    with col_sc2:
                        import math as _math
                        fwhm_guide = _math.degrees(
                            SCHERRER_K * (_wl_A * 1e-10) /
                            ((crystallite_size_nm * 1e-9) *
                             _math.cos(_math.radians(22.5)))
                        )
                        st.metric("Expected FWHM at 2θ = 45°",
                                  f"{fwhm_guide:.3f} °")
                    st.caption(
                        "Total FWHM²(2θ) = β_Scherrer²(2θ) + FWHM_instrumental²  "
                        "→ broader at low angles, narrower at high angles.")
                else:
                    crystallite_size_nm = st.session_state.get(
                        "crystallite_size_nm", 100.0)

    return (wavelength_A, wavelength_nm, diffraction_choice, preset_choice,
            peak_representation, intensity_scale_option, line_thickness,
            use_debye_waller, sigma, x_axis_metric, y_axis_scale,
            intensity_filter, num_annotate,
            st.session_state.two_theta_min,
            st.session_state.two_theta_max,
            use_rust, y_axis_title,
            use_scherrer, crystallite_size_nm, pseudo_voigt_eta, pearson_m,
            show_delta_ref, profile_norm)



def _debye_waller_ui():
    st.markdown("### 🔥 Debye-Waller B-factors")
    if "debye_waller_factors_per_file" not in st.session_state:
        st.session_state.debye_waller_factors_per_file = {}

    preset_map = {
        "Room Temperature (300K)": {
            "H":1.1,"C":0.8,"N":0.9,"O":0.7,"F":0.8,"Na":1.1,"Mg":0.5,
            "Al":0.6,"Si":0.5,"P":0.7,"S":0.6,"Cl":0.8,"K":1.2,"Ca":0.6,
            "Ti":0.4,"V":0.4,"Cr":0.4,"Mn":0.5,"Fe":0.4,"Co":0.4,"Ni":0.4,
            "Cu":0.5,"Zn":0.6,"Ga":0.7,"Ge":0.6,
        },
        "Low Temperature (100K)": {
            "H":0.6,"C":0.4,"N":0.5,"O":0.3,"F":0.4,"Na":0.6,"Mg":0.3,
            "Al":0.3,"Si":0.2,"P":0.3,"S":0.3,"Cl":0.4,"K":0.7,"Ca":0.3,
            "Ti":0.2,"V":0.2,"Cr":0.2,"Mn":0.3,"Fe":0.2,"Co":0.2,"Ni":0.2,
            "Cu":0.3,"Zn":0.3,"Ga":0.4,"Ge":0.3,
        },
        "High Temperature (500K)": {
            "H":1.8,"C":1.3,"N":1.5,"O":1.2,"F":1.3,"Na":1.8,"Mg":0.9,
            "Al":1.0,"Si":0.8,"P":1.2,"S":1.0,"Cl":1.4,"K":2.0,"Ca":1.0,
            "Ti":0.7,"V":0.7,"Cr":0.7,"Mn":0.8,"Fe":0.7,"Co":0.7,"Ni":0.7,
            "Cu":0.8,"Zn":1.0,"Ga":1.2,"Ge":1.0,
        },
    }

    pc1, pc2 = st.columns([1, 3])
    with pc1:
        apply_preset = st.selectbox(
            "Apply preset to all files",
            ["Custom (No Preset)"] + list(preset_map.keys()),
            key="dw_preset")
    with pc2:
        apply_btn = False
        if apply_preset != "Custom (No Preset)":
            st.write(f"Example: Si={preset_map[apply_preset].get('Si','N/A')} Å², "
                     f"O={preset_map[apply_preset].get('O','N/A')} Å², "
                     f"Fe={preset_map[apply_preset].get('Fe','N/A')} Å²")
            apply_btn = st.button("Apply preset to all files")

    if "uploaded_files" not in st.session_state:
        return

    for file in st.session_state.uploaded_files:
        fkey = file.name
        if fkey not in st.session_state.debye_waller_factors_per_file:
            st.session_state.debye_waller_factors_per_file[fkey] = {}
        st.markdown(f"**{fkey}**")
        try:
            struct = load_structure(fkey)
            elems  = sorted({
                sp.symbol
                for site in struct
                for sp in (site.species if not site.is_ordered else [site.specie])
            })
        except Exception:
            continue
        cols = st.columns(min(4, len(elems)))
        for j, el in enumerate(elems):
            dfl = (preset_map[apply_preset].get(el, 1.0)
                   if (apply_preset != "Custom (No Preset)" and apply_btn)
                   else st.session_state.debye_waller_factors_per_file[fkey]
                   .get(el, 1.0))
            with cols[j % len(cols)]:
                val = st.number_input(
                    f"B ({el}) Å²", 0.0, 10.0, value=dfl, step=0.1,
                    format="%.2f", key=f"b_{fkey}_{el}")
            st.session_state.debye_waller_factors_per_file[fkey][el] = val



def _xrd_wavelength_ui():
    input_mode = st.radio(
        "Wavelength input",
        ["Preset", "Custom λ", "Energy"],
        key="input_mode",
        horizontal=True,
    )

    if "last_input_mode" not in st.session_state:
        st.session_state.last_input_mode = input_mode
    elif st.session_state.last_input_mode != input_mode:
        if input_mode == "Preset":
            st.session_state.wavelength_value = PRESET_WAVELENGTHS.get(
                "Cobalt (CoKa1)", 0.17889)
        st.session_state.last_input_mode = input_mode

    if input_mode == "Preset":
        col_p, col_wl = st.columns([3, 2])
        with col_p:
            preset = st.selectbox("Preset", PRESET_OPTIONS,
                                   key="preset_choice", index=0,
                                   label_visibility="collapsed")
        if (st.session_state.get("preset_choice") !=
                st.session_state.get("_prev_preset")):
            st.session_state.wavelength_value = PRESET_WAVELENGTHS.get(
                preset, 0.17889)
            st.session_state["_prev_preset"] = preset
            if preset in DEFAULT_TWO_THETA_MAX_FOR_PRESET:
                new_max = DEFAULT_TWO_THETA_MAX_FOR_PRESET[preset]
                st.session_state.two_theta_max = new_max
                if st.session_state.two_theta_min >= new_max:
                    st.session_state.two_theta_min = max(0.1, new_max * 0.5)

        hide_input_for = ["Cu(Ka1+Ka2+Kb1)", "Cu(Ka1+Ka2)"]
        if preset not in hide_input_for:
            with col_wl:
                wl = st.number_input("λ (nm)", min_value=0.001,
                                     step=0.001, format="%.5f",
                                     key="wavelength_value",
                                     label_visibility="collapsed")
        else:
            wl = PRESET_WAVELENGTHS[preset]
            st.session_state.wavelength_value = wl
            with col_wl:
                st.metric("λ (nm)", f"{wl:.5f}")
        return wl, preset

    elif input_mode == "Custom λ":
        col_wl, col_e = st.columns(2)
        with col_wl:
            wl = st.number_input("λ (nm)", 0.01, 5.0,
                                  value=st.session_state.wavelength_value,
                                  step=0.001, format="%.5f",
                                  key="wavelength_value")
        with col_e:
            st.metric("Energy", f"{wavelength_to_energy(wl):.3f} keV")
        return wl, "Custom"

    else:
        if st.session_state.get("last_input_mode_xrd") != "Energy":
            st.session_state.energy_kev = 8.048
        st.session_state["last_input_mode_xrd"] = "Energy"
        col_e, col_wl = st.columns(2)
        with col_e:
            ekev = st.number_input("Energy (keV)", 6.0, 65.0, step=0.1,
                                   format="%.3f", key="energy_kev")
        with col_wl:
            wl = energy_to_wavelength(ekev)
            st.session_state.wavelength_value = wl
            st.metric("λ (nm)", f"{wl:.5f}")
        return wl, "Custom"



def _nd_wavelength_ui():
    col_p, col_wl = st.columns([3, 2])
    with col_p:
        preset = st.selectbox("Neutron preset", PRESET_OPTIONS_NEUTRON,
                               index=0, key="preset_choice_neutron",
                               label_visibility="collapsed")
    if (st.session_state.get("preset_choice_neutron") !=
            st.session_state.get("_prev_preset_nd")):
        st.session_state.wavelength_value = PRESET_WAVELENGTHS_NEUTRON.get(
            preset, 0.154)
        st.session_state["_prev_preset_nd"] = preset
        if preset in DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET:
            new_max = DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET[preset]
            st.session_state.two_theta_max = new_max
            if st.session_state.two_theta_min >= new_max:
                st.session_state.two_theta_min = max(0.1, new_max * 0.5)

    with col_wl:
        wl = st.number_input("λ (nm)", min_value=0.001,
                             step=0.001, format="%.5f",
                             key="wavelength_value",
                             label_visibility="collapsed")
    return wl, preset



def _tab_lattice(uploaded_files):
    st.subheader("🔷 Lattice Parameters Overview")

    if not uploaded_files:
        st.info("Upload structure files to see their lattice parameters.")
        return

    rows = []
    errors = []
    for file in uploaded_files:
        try:
            struct = load_structure(file)
            conv   = get_full_conventional_structure_diffra(struct)
            sga    = SpacegroupAnalyzer(conv)
            lat    = conv.lattice
            rows.append({
                "File":           file.name,
                "Space group":    sga.get_space_group_symbol(),
                "Crystal system": sga.get_crystal_system().capitalize(),
                "a (Å)":          round(lat.a,   5),
                "b (Å)":          round(lat.b,   5),
                "c (Å)":          round(lat.c,   5),
                "α (°)":          round(lat.alpha, 4),
                "β (°)":          round(lat.beta,  4),
                "γ (°)":          round(lat.gamma, 4),
                "V (Å³)":         round(lat.volume, 3),
                "Sites":          len(conv),
            })
        except Exception as exc:
            errors.append((file.name, str(exc)))

    if rows:
        df = pd.DataFrame(rows).set_index("File")
        st.dataframe(df, use_container_width=True)

        csv = df.reset_index().to_csv(index=False)
        st.download_button(
            "💾 Download as CSV",
            data=csv,
            file_name="lattice_parameters.csv",
            mime="text/csv",
            type = 'primary'
        )

    for fname, err in errors:
        st.warning(f"Could not read '{fname}': {err}")



def run_diffraction_section(uploaded_files, user_pattern_file):
    if "calc_xrd" not in st.session_state:
        st.session_state.calc_xrd = False

    colmain_1, colmain_2 = st.columns([0.5, 1])

    with colmain_1:
        (wavelength_A, wavelength_nm, diffraction_choice, preset_choice,
         peak_representation, intensity_scale_option, line_thickness,
         use_debye_waller, sigma, x_axis_metric, y_axis_scale,
         intensity_filter, num_annotate,
         two_theta_min, two_theta_max,
         use_rust, y_axis_title,
         use_scherrer, crystallite_size_nm,
         pseudo_voigt_eta, pearson_m,
         show_delta_ref, profile_norm) = _diffraction_settings_ui()

    with colmain_2:
        if user_pattern_file:
            _tab_labels = ["➡️ Patterns chart", "📉 Experimental Data",
                           "🖥️ Quantitative peak data",
                           "🏷️ Annotate Planes", "🔷 Lattice Parameters"]
        else:
            _tab_labels = ["➡️ Patterns chart", "🖥️ Quantitative peak data",
                           "🏷️ Annotate Planes", "🔷 Lattice Parameters"]
        _tabs = st.tabs(_tab_labels)
        tab1 = _tabs[0]
        if user_pattern_file:
            tab_exp = _tabs[1]
            tab2    = _tabs[2]
            tab_ann = _tabs[3]
            tab_lat = _tabs[4]
        else:
            tab_exp = None
            tab2    = _tabs[1]
            tab_ann = _tabs[2]
            tab_lat = _tabs[3]

        if tab_exp is not None:
            with tab_exp:
                _tab_background_subtraction(user_pattern_file, x_axis_metric)

        if not st.session_state.calc_xrd:
            with tab1:
                st.warning("Upload structure files and press 'Calculate XRD / ND'.")

                fig_exp = go.Figure()
                _add_exp_traces(fig_exp, user_pattern_file,
                                intensity_scale_option, y_axis_scale,
                                two_theta_min, two_theta_max, x_axis_metric,
                                wavelength_A, wavelength_nm, diffraction_choice)
                if fig_exp.data:
                    _apply_layout(fig_exp, x_axis_metric, y_axis_title,
                                  two_theta_min, two_theta_max,
                                  intensity_scale_option, peak_representation,
                                  wavelength_A, wavelength_nm, diffraction_choice)
                    st.plotly_chart(fig_exp, use_container_width=True)
            return

        if not uploaded_files:
            with tab1:
                st.warning("No structure files loaded.")
            return

        current_key = _cache_key(
            uploaded_files, wavelength_A, diffraction_choice,
            use_debye_waller,
            st.session_state.get("debye_waller_factors_per_file", {}),
            preset_choice,
        )
        _stored_patterns = st.session_state.get("raw_patterns") or {}
        _expected_files  = {f.name for f in uploaded_files}
        _stored_files    = set(_stored_patterns.keys())
        needs_recalc = (
            st.session_state.get("raw_patterns_cache_key") != current_key
            or "raw_patterns" not in st.session_state
            or _expected_files != _stored_files
        )

        if needs_recalc:
            _status_ph = st.session_state.get("_calc_status_placeholder")
            if _status_ph is not None:
                _status_ph.info(
                    "⏳ Calculating the powder diffraction pattern(s), "
                    "please wait…")
            _t0 = time.perf_counter()
            st.session_state.raw_patterns = _calculate_raw_patterns(
                uploaded_files, wavelength_A, diffraction_choice,
                use_debye_waller,
                st.session_state.get("debye_waller_factors_per_file", {}),
                use_rust, preset_choice,
            )
            st.session_state.raw_patterns_cache_key = current_key
            _elapsed = time.perf_counter() - _t0
            st.session_state.last_calc_time_s = _elapsed
            if _status_ph is not None:
                _t_str = (f"{_elapsed * 1000:.0f} ms"
                          if _elapsed < 1.0 else f"{_elapsed:.2f} s")
                _status_ph.caption(f"⏱ Last calculation: {_t_str}")

        pattern_details = _process_for_display(
            st.session_state.raw_patterns,
            two_theta_min, two_theta_max,
            intensity_filter, peak_representation, sigma,
            intensity_scale_option, y_axis_scale, x_axis_metric,
            wavelength_A, wavelength_nm, diffraction_choice, num_annotate,
            use_scherrer=use_scherrer,
            crystallite_size_nm=crystallite_size_nm,
            pseudo_voigt_eta=pseudo_voigt_eta,
            pearson_m=pearson_m,
            profile_norm=profile_norm,
        )

        show_Ka1 = show_Ka2 = show_Kb = True
        if peak_representation != "Delta (stick)":
            if preset_choice in MULTI_COMPONENT_PRESETS:
                st.sidebar.subheader("Include Kα/Kβ for hovering:")
                n_comp = len(MULTI_COMPONENT_PRESETS[preset_choice]["wavelengths"])
                show_Ka1 = st.sidebar.checkbox("Include Kα1 hover", value=True,
                                                key="hover_ka1")
                if n_comp >= 2:
                    show_Ka2 = st.sidebar.checkbox("Include Kα2 hover", value=False,
                                                    key="hover_ka2")
                if n_comp >= 3:
                    show_Kb  = st.sidebar.checkbox("Include Kβ hover", value=False,
                                                    key="hover_kb")
            else:
                st.sidebar.subheader("Include Kα1 for hovering:")
                show_Ka1 = st.sidebar.checkbox("Include Kα1 hover", value=True,
                                                key="hover_ka1_single")

        with tab1:
            if ("new_structure_added" in st.session_state
                    and st.session_state.new_structure_added):
                if "new_structure_name" in st.session_state:
                    st.success(
                        f"Modified structure "
                        f"'{st.session_state.new_structure_name}' added.")
                st.session_state.new_structure_added = False

            fig = go.Figure()

            if use_scherrer and peak_representation != "Delta (stick)":
                pattern_details_no_scherrer = _process_for_display(
                    st.session_state.raw_patterns,
                    two_theta_min, two_theta_max,
                    intensity_filter, peak_representation, sigma,
                    intensity_scale_option, y_axis_scale, x_axis_metric,
                    wavelength_A, wavelength_nm, diffraction_choice, num_annotate,
                    use_scherrer=False,
                    crystallite_size_nm=crystallite_size_nm,
                    pseudo_voigt_eta=pseudo_voigt_eta,
                    pearson_m=pearson_m,
                    profile_norm=profile_norm,
                )
                for idx, file in enumerate(uploaded_files):
                    fname   = file.name
                    if fname not in pattern_details_no_scherrer:
                        continue
                    d_orig  = pattern_details_no_scherrer[fname]
                    mask    = ((d_orig["x_dense_full"] >= two_theta_min) &
                               (d_orig["x_dense_full"] <= two_theta_max))
                    x_orig  = twotheta_to_metric(
                        d_orig["x_dense_full"][mask], x_axis_metric,
                        wavelength_A, wavelength_nm, diffraction_choice)
                    y_orig  = d_orig["y_dense"][mask]
                    tab10   = plt.cm.tab10.colors
                    c_orig  = rgb_color(tab10[idx % len(tab10)], opacity=0.6)
                    fig.add_trace(go.Scatter(
                        x=x_orig, y=y_orig, mode="lines",
                        name=f"{fname} — instr. only",
                        line=dict(color=c_orig, width=line_thickness + 0.5,
                                  dash="dash"),
                        hoverinfo="skip",
                    ))
                _plot_patterns(
                    fig, pattern_details, uploaded_files,
                    preset_choice, peak_representation,
                    line_thickness,
                    x_axis_metric, wavelength_A, wavelength_nm,
                    diffraction_choice, two_theta_min, two_theta_max,
                    show_Kalpha1_hover=show_Ka1,
                    show_Kalpha2_hover=show_Ka2,
                    show_Kbeta_hover=show_Kb,
                    show_delta_ref=show_delta_ref,
                )
            else:
                _plot_patterns(
                    fig, pattern_details, uploaded_files, preset_choice,
                    peak_representation, line_thickness, x_axis_metric,
                    wavelength_A, wavelength_nm, diffraction_choice,
                    two_theta_min, two_theta_max,
                    show_Kalpha1_hover=show_Ka1,
                    show_Kalpha2_hover=show_Ka2,
                    show_Kbeta_hover=show_Kb,
                    show_delta_ref=show_delta_ref,
                )

            _add_exp_traces(
                fig, user_pattern_file, intensity_scale_option, y_axis_scale,
                two_theta_min, two_theta_max, x_axis_metric,
                wavelength_A, wavelength_nm, diffraction_choice,
            )
            _apply_layout(fig, x_axis_metric, y_axis_title,
                          two_theta_min, two_theta_max,
                          intensity_scale_option, peak_representation,
                          wavelength_A, wavelength_nm, diffraction_choice)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            _tab2_quantitative(pattern_details, uploaded_files, x_axis_metric)

        with tab_ann:
            if pattern_details:
                _tab_annotation(
                    pattern_details, uploaded_files, fig,
                    x_axis_metric, wavelength_A, wavelength_nm,
                    diffraction_choice, two_theta_min, two_theta_max,
                )
            else:
                st.info("Calculate a diffraction pattern first to enable annotation.")

        with tab_lat:
            _tab_lattice(uploaded_files)





def _apply_layout(fig, x_axis_metric, y_axis_title,
                  two_theta_min, two_theta_max,
                  intensity_scale_option, peak_representation,
                  wavelength_A=1.789, wavelength_nm=0.17889,
                  diffraction_choice="XRD (X-ray)"):
    disp_min = twotheta_to_metric(
        two_theta_min, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
    disp_max = twotheta_to_metric(
        two_theta_max, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)

    if x_axis_metric in ("d (Å)", "d (nm)"):
        x_range = [disp_max, disp_min]
    else:
        x_range = [disp_min, disp_max]

    y_range = [0, 125] if (
        peak_representation == "Delta (stick)"
        and intensity_scale_option != "Absolute"
    ) else None

    fig.update_layout(
        height=800 if peak_representation == "Delta (stick)" else 1000,
        margin=dict(t=80, b=80, l=60, r=30),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="top", y=-0.2,
            xanchor="center", x=0.5, font=dict(size=24),
        ),
        xaxis=dict(
            title=dict(text=x_axis_metric,
                       font=dict(size=36, color="black"), standoff=20),
            tickfont=dict(size=36, color="black"),
            range=x_range,
        ),
        yaxis=dict(
            title=dict(text=y_axis_title,
                       font=dict(size=36, color="black")),
            tickfont=dict(size=36, color="black"),
            range=y_range,
        ),
        hoverlabel=dict(font=dict(size=24)),
        font=dict(size=18),
        autosize=True,
    )
