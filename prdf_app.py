import warnings
import io
from fractions import Fraction
from math import gcd
from functools import reduce
import streamlit as st

st.set_page_config(page_title="XRDlicious – (P)RDF Calculator", layout="wide")
st.markdown(
    "\n    <style>\n    .block-container { padding-top: 0rem; }\n    #MainMenu {visibility: hidden;}\n    footer     {visibility: hidden;}\n    header     {visibility: hidden;}\n    </style>\n",
    unsafe_allow_html=True,
)
css = '\n<style>\n.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {\n    font-size: 1.15rem !important;\n    color: #1e3a8a !important;\n    font-weight: 600 !important;\n    margin: 0 !important;\n}\n\n.stTabs [data-baseweb="tab-list"] {\n    gap: 20px !important;\n}\n\n.stTabs [data-baseweb="tab-list"] button {\n    background-color: #f0f4ff !important;\n    border-radius: 12px !important;\n    padding: 8px 16px !important;\n    transition: all 0.3s ease !important;\n    border: none !important;\n    color: #1e3a8a !important;\n}\n\n.stTabs [data-baseweb="tab-list"] button:hover {\n    background-color: #dbe5ff !important;\n    cursor: pointer;\n}\n\n.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {\n    background-color: #e0e7ff !important;\n    color: #1e3a8a !important;\n    font-weight: 700 !important;\n    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;\n}\n\n.stTabs [data-baseweb="tab-list"] button:focus {\n    outline: none !important;\n}\n</style>\n'
st.markdown(css, unsafe_allow_html=True)
import os, tempfile
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from matminer.featurizers.structure import PartialRadialDistributionFunction

MAX_ATOMS = 1000
MAX_SUPERCELL_ATOMS = 1000
MAX_SPECIES = 6


def rgb_to_hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


COLORS = [rgb_to_hex(c) for c in plt.cm.tab10.colors]
FONT = dict(size=22, color="black")


def load_structure_from_file(uploaded_file) -> Structure:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    if suffix == "cif":
        try:
            parser = CifParser.from_str(raw.decode("utf-8", errors="replace"))
            structs = parser.parse_structures(primitive=False)
            if structs:
                return structs[0]
        except Exception:
            pass
    fmt_map = {
        "poscar": "vasp",
        "vasp": "vasp",
        "lmp": "lammps-data",
        "data": "lammps-data",
        "xsf": "xsf",
        "xyz": "extxyz",
        "cfg": "cfg",
        "pw": "espresso-in",
    }
    ase_fmt = fmt_map.get(suffix, None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        atoms = ase_read(tmp_path, format=ase_fmt) if ase_fmt else ase_read(tmp_path)
        return AseAtomsAdaptor.get_structure(atoms)
    finally:
        os.remove(tmp_path)


def validate_structure(struct: Structure):
    n_atoms = len(struct)
    species = set()
    for site in struct:
        if site.is_ordered:
            species.add(site.specie.symbol)
        else:
            for sp in site.species:
                species.add(sp.symbol)
    n_sp = len(species)
    if n_atoms > MAX_ATOMS:
        return (
            False,
            f"Structure has **{n_atoms} atoms**, which exceeds the limit of **{MAX_ATOMS}**. Please use a smaller supercell or a primitive cell.",
        )
    if n_sp > MAX_SPECIES:
        return (
            False,
            f"Structure has **{n_sp} element types**, which exceeds the limit of **{MAX_SPECIES}**. Please use a structure with fewer species.",
        )
    return (True, "")


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _supercell_scale(struct: Structure) -> int:
    denoms = []
    for site in struct:
        if not site.is_ordered:
            for _sp, occ in site.species.items():
                f = float(occ)
                if 0.0 < f < 1.0:
                    denoms.append(Fraction(f).limit_denominator(20).denominator)
    if not denoms:
        return 1
    ideal = reduce(_lcm, denoms, 1)
    for s in range(ideal, 0, -1):
        if s**3 * len(struct) <= MAX_SUPERCELL_ATOMS:
            return s
    return 1


def _round_supercell_occupancies(supercell: Structure) -> Structure:
    from pymatgen.core import PeriodicSite, DummySpecies
    from pymatgen.core.composition import Composition

    new_sites = []
    for site in supercell:
        if site.is_ordered:
            new_sites.append(site)
            continue
        real = {
            sp: float(occ)
            for sp, occ in site.species.items()
            if not isinstance(sp, DummySpecies) and str(sp).lower() not in ("x0+", "x")
        }
        if not real:
            continue
        snapped = {sp: round(occ, 4) for sp, occ in real.items()}
        snapped = {sp: occ for sp, occ in snapped.items() if occ > 0.0001}
        if not snapped:
            continue
        total = sum(snapped.values())
        if total > 1.0 + 0.001:
            snapped = {sp: occ / total for sp, occ in snapped.items()}
        new_sites.append(
            PeriodicSite(
                Composition(snapped), site.frac_coords, site.lattice, properties=site.properties
            )
        )
    return Structure.from_sites(new_sites)


def _proportional_order_fallback(supercell: Structure, seed: int = 42):
    import random
    from pymatgen.core import PeriodicSite

    rng = random.Random(seed)
    ordered_sites = []
    disordered_by_comp = defaultdict(list)
    for site in supercell:
        if site.is_ordered:
            ordered_sites.append(site)
        else:
            comp_key = tuple(
                sorted(((str(sp), round(float(occ), 4)) for sp, occ in site.species.items()))
            )
            disordered_by_comp[comp_key].append(site)
    extra_sites = []
    for comp_key, sites in disordered_by_comp.items():
        n = len(sites)
        sp_names = [k for k, _ in comp_key]
        occs = [v for _, v in comp_key]
        assignment = []
        for sp_name, occ in zip(sp_names, occs):
            cnt = round(float(occ) * n)
            assignment.extend([sp_name] * max(cnt, 0))
        assignment = assignment[:n]
        assignment += ["__vacancy__"] * (n - len(assignment))
        rng.shuffle(assignment)
        for site, sp_name in zip(sites, assignment):
            if sp_name == "__vacancy__":
                continue
            extra_sites.append(
                PeriodicSite(sp_name, site.frac_coords, site.lattice, properties=site.properties)
            )
    all_sites = ordered_sites + extra_sites
    if not all_sites:
        raise ValueError("No sites remain after ordering -- check occupancy values.")
    return Structure.from_sites(all_sites)


def make_ordered_supercell(struct: Structure):
    scale = _supercell_scale(struct)
    warnings_out = []
    supercell = struct.copy()
    supercell.make_supercell([scale, scale, scale])
    n_super = len(supercell)
    if n_super > MAX_SUPERCELL_ATOMS:
        raise ValueError(
            f"The smallest possible supercell ({scale}×{scale}×{scale}) already contains **{n_super} sites**, which exceeds the supercell limit of **{MAX_SUPERCELL_ATOMS} atoms**. Please provide a smaller primitive cell, or pre-order the structure externally and upload it directly."
        )
    supercell_clean = _round_supercell_occupancies(supercell)
    try:
        ordered = _proportional_order_fallback(supercell_clean)
    except Exception as exc:
        raise ValueError(f"Could not build ordered supercell: {exc}") from exc
    n_ordered = len(ordered)
    n_sites = len(supercell_clean)
    n_vac = n_sites - n_ordered
    n_copies = scale**3
    occ_rows = []
    seen_comps = {}
    for i, site in enumerate(struct):
        if site.is_ordered:
            continue
        for sp, occ in site.species.items():
            from pymatgen.core import DummySpecies as _DS

            if isinstance(sp, _DS) or str(sp).lower() in ("x0+", "x"):
                continue
            occ_f = round(float(occ), 6)
            assigned = round(occ_f * n_copies)
            actual = assigned / n_copies if n_copies > 0 else 0.0
            fc = site.frac_coords
            occ_rows.append(
                {
                    "site_idx": i,
                    "frac": f"({fc[0]:.4f}, {fc[1]:.4f}, {fc[2]:.4f})",
                    "element": str(sp),
                    "target_occ": occ_f,
                    "n_copies": n_copies,
                    "n_assigned": assigned,
                    "actual_occ": actual,
                }
            )
    msg = f"Disordered structure converted to a **{scale}×{scale}×{scale} vacancy supercell** ({n_ordered} atoms + {n_vac} vacant sites out of {n_sites} total). Atom counts approximate the CIF occupancies; vacant sites are assigned randomly (seed=42). Download this structure from the **🧩 Vacancy Supercells** tab."
    return (ordered, scale, msg, warnings_out, occ_rows)


def load_experimental_data(uploaded_file):
    content = uploaded_file.read().decode("utf-8", errors="replace")
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(io.StringIO(content), sep=None, engine="python", comment="#", header=None)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns.")
        df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 5:
            raise ValueError("Fewer than 5 valid numeric rows found.")
        return (df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float))
    except Exception as exc:
        raise ValueError(f"Could not parse experimental data file: {exc}") from exc


def _norm(y):
    y = np.asarray(y, dtype=float)
    m = float(np.max(y))
    return y / m if m > 0 else y


def smooth_gaussian(y, sigma=1.5):
    return gaussian_filter1d(y, sigma=sigma)


def smooth_savgol(y, window=11, polyorder=3):
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(y, window, polyorder)


def smooth_spline(x, y, n_pts=300):
    xs = np.linspace(x[0], x[-1], n_pts)
    return (xs, np.maximum(0, make_interp_spline(x, y, k=3)(xs)))


st.markdown(
    "\n    <h3 style='color:#8b0000;'>\n        <strong>XRDlicious</strong> – <em>(P)RDF Calculator</em>\n    </h3>\n    <hr style=\"border:none;height:5px;background-color:#8b0000;\n               border-radius:6px;margin:0 0 8px 0;\">\n",
    unsafe_allow_html=True,
)
st.info(
    f"Upload one or more crystal structure files in the **sidebar** and press **▶️ Calculate RDF**. Each structure is processed **individually**. Use the **🔀 Comparison** tab to overlay PRDFs from different structures.  \n⚠️ Limits: **max {MAX_ATOMS} atoms** and **max {MAX_SPECIES} element types** per structure. Disordered CIFs are auto-converted to vacancy supercells capped at **{MAX_SUPERCELL_ATOMS} sites** (vacancies placed randomly)."
)
with st.expander("📖 How to **Cite**", expanded=False):
    st.markdown(
        "\n### How to cite\n\nIf you use **XRDlicious**, please cite:\n\n- **XRDlicious**:   \n  [LEBEDA, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)\n\n\n- **Matminer**:  \n  [WARD, Logan, et al. *Matminer: An open source toolkit for materials data mining.* Computational Materials Science, 2018, 152: 60–69.](https://www.sciencedirect.com/science/article/pii/S0927025618303252)\n    "
    )
st.sidebar.header("📁 Upload Structure Files")
uploaded_files = st.sidebar.file_uploader(
    "CIF, POSCAR, LMP, XSF, XYZ (with cell), CFG, PW …",
    type=["cif", "xyz", "vasp", "poscar", "lmp", "data", "xsf", "pw", "cfg"],
    accept_multiple_files=True,
    key="prdf_uploader",
)
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Experimental Data")
st.sidebar.caption(
    "Two-column files (r, intensity) in any text format – CSV, TSV, space-separated, etc. Comment lines starting with **#** are skipped."
)
uploaded_exp_files = st.sidebar.file_uploader(
    "Select experimental data file(s)",
    type=["csv", "txt", "dat", "xy", "gr", "xye"],
    accept_multiple_files=True,
    key="exp_uploader",
)
if "prdf_structures" not in st.session_state:
    st.session_state.prdf_structures = {}
if "prdf_rejected" not in st.session_state:
    st.session_state.prdf_rejected = {}
if "ordered_structures" not in st.session_state:
    st.session_state.ordered_structures = {}
if "disorder_messages" not in st.session_state:
    st.session_state.disorder_messages = {}
if "disorder_occ_info" not in st.session_state:
    st.session_state.disorder_occ_info = {}
if "prdf_cif_ready" not in st.session_state:
    st.session_state.prdf_cif_ready = {}
if "prdf_experimental" not in st.session_state:
    st.session_state.prdf_experimental = {}
active_names = {f.name for f in uploaded_files} if uploaded_files else set()
if uploaded_files:
    for f in uploaded_files:
        if (
            f.name not in st.session_state.prdf_structures
            and f.name not in st.session_state.prdf_rejected
        ):
            try:
                struct = load_structure_from_file(f)
                if not struct.is_ordered:
                    with st.spinner(f"⚙️ Ordering disordered structure: {f.name} …"):
                        try:
                            ordered, scale, msg, order_warns, occ_rows = make_ordered_supercell(
                                struct
                            )
                        except Exception as order_exc:
                            st.session_state.prdf_rejected[f.name] = str(order_exc)
                            st.sidebar.error(f"❌ {f.name} rejected")
                            continue
                    st.session_state.ordered_structures[f.name] = ordered
                    st.session_state.disorder_occ_info[f.name] = occ_rows
                    st.session_state.prdf_cif_ready[f.name] = False
                    full_msg = msg
                    if order_warns:
                        full_msg += "  \n⚠️ " + "  \n⚠️ ".join(order_warns)
                    st.session_state.disorder_messages[f.name] = full_msg
                    struct = ordered
                ok, reason = validate_structure(struct)
                if ok:
                    st.session_state.prdf_structures[f.name] = struct
                    st.sidebar.success(f"✅ {f.name}")
                else:
                    st.session_state.prdf_rejected[f.name] = reason
                    st.sidebar.error(f"❌ {f.name} rejected")
            except Exception as e:
                st.session_state.prdf_rejected[f.name] = str(e)
                st.sidebar.error(f"❌ {f.name}: {e}")
for name in list(st.session_state.prdf_structures.keys()):
    if name not in active_names:
        del st.session_state.prdf_structures[name]
        st.session_state.ordered_structures.pop(name, None)
        st.session_state.disorder_messages.pop(name, None)
        st.session_state.disorder_occ_info.pop(name, None)
        st.session_state.prdf_cif_ready.pop(name, None)
for name in list(st.session_state.prdf_rejected.keys()):
    if name not in active_names:
        del st.session_state.prdf_rejected[name]
active_exp_names = {f.name for f in uploaded_exp_files} if uploaded_exp_files else set()
if uploaded_exp_files:
    for ef in uploaded_exp_files:
        if ef.name not in st.session_state.prdf_experimental:
            try:
                xd, yd = load_experimental_data(ef)
                st.session_state.prdf_experimental[ef.name] = (xd, yd)
                st.sidebar.success(f"✅ Exp: {ef.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Exp {ef.name}: {e}")
for name in list(st.session_state.prdf_experimental.keys()):
    if name not in active_exp_names:
        del st.session_state.prdf_experimental[name]
structures: dict = st.session_state.prdf_structures
for name, reason in st.session_state.prdf_rejected.items():
    st.error(f"**{name}** was not loaded: {reason}")
for fname, msg in st.session_state.disorder_messages.items():
    if fname in structures:
        st.info(f"🔀 **{fname}**: {msg}")
if structures:
    st.sidebar.markdown(f"**{len(structures)} valid structure(s):**")
    for sname, s in structures.items():
        sp_set = set()
        for site in s:
            if site.is_ordered:
                sp_set.add(site.specie.symbol)
            else:
                for sp in site.species:
                    sp_set.add(sp.symbol)
        st.sidebar.caption(f"• {sname}  ({len(s)} atoms, {len(sp_set)} species)")
st.subheader(
    "⚙️ Settings",
    help="PRDF describes atom-pair distance distributions, providing insight into local environments and structural disorder. Values are unitless relative intensities. Peaks = preferred bonding distances; peak width = disorder.",
)
col_cut, col_bin = st.columns(2)
cutoff = col_cut.number_input(
    "Cutoff (Å)", min_value=1.0, max_value=50.0, value=10.0, step=0.5, format="%.1f"
)
bin_size = col_bin.number_input(
    "Bin size (Å)", min_value=0.005, max_value=2.0, value=0.1, step=0.005, format="%.3f"
)
st.markdown("#### Plot options")
col_ps, col_ls = st.columns(2)
plot_style = col_ps.radio(
    "Plot style", ["Smooth Curve", "Raw Data Points", "Bars (Histogram)"], index=0
)
line_style = col_ls.radio("Marker style", ["Lines Only", "Lines + Markers"], index=0)
sigma, sg_win, sg_ord, spline_pts, bar_width_factor = (1.5, 11, 3, 300, 0.8)
smoothing_method = "Gaussian"
if plot_style == "Smooth Curve":
    col_sm, col_sp = st.columns([2, 2])
    smoothing_method = col_sm.radio(
        "Smoothing method", ["Gaussian", "Savitzky-Golay", "Cubic Spline"], index=0, horizontal=True
    )
    if smoothing_method == "Gaussian":
        sigma = col_sp.slider("Gaussian σ", 0.5, 5.0, 1.5, 0.1)
    elif smoothing_method == "Savitzky-Golay":
        sg_win = col_sp.slider("Window length (odd)", 5, 21, 11, 2)
        sg_ord = col_sp.slider("Polynomial order", 2, 5, 3, 1)
    else:
        spline_pts = col_sp.slider("Interpolation points", 100, 600, 300, 50)
if plot_style == "Bars (Histogram)":
    bar_width_factor = st.slider("Bar width factor", 0.1, 2.0, 0.8, 0.1)
normalize_to_max = st.checkbox(
    "📐 Normalize each trace to its maximum (max = 1)",
    value=False,
    help="When enabled, every calculated PRDF/RDF trace and every experimental data set is independently divided by its own maximum value, so all profiles share the same [0, 1] y-scale for easy shape comparison.",
)


def apply_smoothing(x_arr, y_arr):
    x_arr, y_arr = (np.array(x_arr), np.array(y_arr))
    if plot_style != "Smooth Curve":
        return (x_arr, y_arr)
    if smoothing_method == "Gaussian":
        return (x_arr, smooth_gaussian(y_arr, sigma))
    elif smoothing_method == "Savitzky-Golay":
        return (x_arr, smooth_savgol(y_arr, sg_win, sg_ord))
    else:
        return smooth_spline(x_arr, y_arr, spline_pts)


def add_trace(fig, x, y, name, color, dash="solid"):
    if normalize_to_max:
        y = _norm(y)
    mode = "lines" if line_style == "Lines Only" else "lines+markers"
    if plot_style == "Bars (Histogram)":
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=name,
                marker=dict(color=color, line=dict(width=0)),
                width=bin_size * bar_width_factor,
                opacity=0.75,
            )
        )
    else:
        xp, yp = apply_smoothing(x, y)
        if plot_style == "Smooth Curve":
            x_arr_raw, y_arr_raw = (np.array(x), np.array(y))
            x_stems, y_stems = ([], [])
            for xi, yi in zip(x_arr_raw, y_arr_raw):
                x_stems.extend([xi, xi, None])
                y_stems.extend([0, yi, None])
            fig.add_trace(
                go.Scatter(
                    x=x_stems,
                    y=y_stems,
                    mode="lines",
                    name=f"{name} (raw)",
                    line=dict(color=color, width=1),
                    opacity=0.35,
                    showlegend=True,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xp,
                y=yp,
                mode=mode,
                name=name,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7) if "markers" in mode else dict(),
            )
        )


def add_experimental_traces(fig, selected_exp_names, color_offset=0):
    dash_styles_exp = ["dot", "dashdot", "longdash", "longdashdot"]
    for ei, ename in enumerate(selected_exp_names):
        xd, yd = st.session_state.prdf_experimental[ename]
        if normalize_to_max:
            yd = _norm(yd)
        color_idx = (color_offset + ei) % len(COLORS)
        dash_exp = dash_styles_exp[ei % len(dash_styles_exp)]
        short = ename if len(ename) <= 35 else ename[:32] + "…"
        fig.add_trace(
            go.Scatter(
                x=xd,
                y=yd,
                mode="lines+markers" if line_style == "Lines + Markers" else "lines",
                name=f"Exp: {short}",
                line=dict(color=COLORS[color_idx], width=2, dash=dash_exp),
                marker=(
                    dict(symbol="circle-open", size=6)
                    if line_style == "Lines + Markers"
                    else dict()
                ),
            )
        )


def make_layout(title, barmode=None):
    ylabel = "Pair correlation function g(r)" if not normalize_to_max else "Normalized intensity"
    d = dict(
        title=dict(text=title, font=FONT),
        xaxis=dict(title=dict(text="Distance (Å)", font=FONT), tickfont=FONT),
        yaxis=dict(title=dict(text=ylabel, font=FONT), tickfont=FONT, range=[0, None]),
        hovermode="x",
        font=FONT,
        hoverlabel=dict(font=FONT),
        legend=dict(
            orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=18)
        ),
    )
    if barmode:
        d["barmode"] = barmode
    return d


if "prdf_results" not in st.session_state:
    st.session_state.prdf_results = {}
if "prdf_do_calc" not in st.session_state:
    st.session_state.prdf_do_calc = False
if "prdf_download_ready" not in st.session_state:
    st.session_state.prdf_download_ready = False


def trigger_calculation():
    st.session_state.prdf_do_calc = True
    st.session_state.prdf_results = {}
    st.session_state.prdf_download_ready = False


def prepare_downloads():
    st.session_state.prdf_download_ready = True


st.button(
    "▶️  Calculate RDF", on_click=trigger_calculation, type="primary", disabled=len(structures) == 0
)
if not structures:
    st.warning("⬆️ Please upload at least one valid structure file in the sidebar.")
if st.session_state.prdf_do_calc and structures:
    struct_items = list(structures.items())
    progress_bar = st.progress(0, text="Starting …")
    calc_errors = []
    for s_idx, (fname, mg_struct) in enumerate(struct_items):
        progress_bar.progress(s_idx / len(struct_items), text=f"Processing {fname} …")
        try:
            featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
            featurizer.fit([mg_struct])
            prdf_vals = featurizer.featurize(mg_struct)
            labels = featurizer.feature_labels()
            prdf_dict = {}
            dist_dict = {}
            global_rdf = {}
            for j, label in enumerate(labels):
                pair_str, rng = label.split(" PRDF r=")
                pair = tuple(pair_str.split("-"))
                lo, hi = map(float, rng.split("-"))
                bc = (lo + hi) / 2.0
                prdf_dict.setdefault(pair, []).append(prdf_vals[j])
                dist_dict.setdefault(pair, []).append(bc)
                global_rdf[bc] = global_rdf.get(bc, 0.0) + prdf_vals[j]
            prdf_dict = {p: np.array(v) for p, v in prdf_dict.items()}
            st.session_state.prdf_results[fname] = {
                "prdf_dict": prdf_dict,
                "dist_dict": dist_dict,
                "global_rdf": global_rdf,
            }
        except Exception as e:
            calc_errors.append((fname, str(e)))
    progress_bar.progress(1.0, text="Done!")
    st.session_state.prdf_do_calc = False
    if calc_errors:
        for fname, msg in calc_errors:
            st.error(f"Error processing **{fname}**: {msg}")
    else:
        n = len(st.session_state.prdf_results)
        st.success(f"✅ Calculated PRDF for {n} structure(s).")
results: dict = st.session_state.prdf_results
if results:
    st.divider()
    all_pairs = sorted(
        {pair for r in results.values() for pair in r["prdf_dict"]}, key=lambda p: (p[0], p[1])
    )
    has_ordered = bool(st.session_state.ordered_structures)
    _tab_labels = ["📊 Individual PRDFs", "🔀 Comparison", "📈 Total RDF", "💾 Download"]
    if has_ordered:
        _tab_labels.append("🧩 Vacancy Supercells")
    _tabs = st.tabs(_tab_labels)
    tab_indiv = _tabs[0]
    tab_comp = _tabs[1]
    tab_total = _tabs[2]
    tab_dl = _tabs[3]
    tab_ord = _tabs[4] if has_ordered else None
    with tab_indiv:
        st.markdown("### Per-structure (P)RDF plots")
        fname_list = list(results.keys())
        selected_struct = (
            fname_list[0]
            if len(fname_list) == 1
            else st.selectbox("Select structure:", fname_list, key="indiv_select")
        )
        if selected_struct in st.session_state.disorder_messages:
            st.info(f"ℹ️ {st.session_state.disorder_messages[selected_struct]}")
        res = results[selected_struct]
        pairs = list(res["prdf_dict"].keys())
        st.markdown(f"**{selected_struct}** — {len(pairs)} element pair(s)")
        layout_mode = st.radio(
            "Layout",
            ["Separate plot per pair", "All pairs in one plot"],
            horizontal=True,
            key="indiv_layout",
        )
        if layout_mode == "All pairs in one plot":
            fig = go.Figure()
            for idx, pair in enumerate(pairs):
                add_trace(
                    fig,
                    res["dist_dict"][pair],
                    res["prdf_dict"][pair],
                    f"{pair[0]}–{pair[1]}",
                    COLORS[idx % len(COLORS)],
                )
            fig.update_layout(
                **make_layout(
                    f"PRDF – {selected_struct}: all pairs",
                    barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                )
            )
            st.plotly_chart(fig, width="stretch")
        else:
            for idx, pair in enumerate(pairs):
                fig = go.Figure()
                add_trace(
                    fig,
                    res["dist_dict"][pair],
                    res["prdf_dict"][pair],
                    f"{pair[0]}–{pair[1]}",
                    COLORS[idx % len(COLORS)],
                )
                fig.update_layout(**make_layout(f"PRDF: {pair[0]}–{pair[1]}  |  {selected_struct}"))
                st.plotly_chart(fig, width="stretch")
    with tab_comp:
        st.markdown("### Compare PRDFs across structures")
        if len(results) < 2:
            st.info("Upload and calculate at least **2 structures** to use the comparison view.")
        else:
            col_pair, col_structs = st.columns([1, 2])
            with col_pair:
                pair_labels = [f"{p[0]}–{p[1]}" for p in all_pairs]
                chosen_label = st.selectbox("Element pair:", pair_labels, key="comp_pair")
                chosen_pair = all_pairs[pair_labels.index(chosen_label)]
            with col_structs:
                all_names = list(results.keys())
                chosen_structs = st.multiselect(
                    "Structures to overlay:", all_names, default=all_names, key="comp_structs"
                )
            if not chosen_structs:
                st.warning("Select at least one structure.")
            else:
                dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
                fig_comp = go.Figure()
                skipped = []
                for s_idx, sname in enumerate(chosen_structs):
                    r = results[sname]
                    if chosen_pair not in r["prdf_dict"]:
                        skipped.append(sname)
                        continue
                    short = sname if len(sname) <= 35 else sname[:32] + "…"
                    add_trace(
                        fig_comp,
                        r["dist_dict"][chosen_pair],
                        r["prdf_dict"][chosen_pair],
                        short,
                        COLORS[s_idx % len(COLORS)],
                        dash=dash_styles[s_idx % len(dash_styles)],
                    )
                if skipped:
                    st.warning(
                        f"Pair **{chosen_label}** not present in: "
                        + ", ".join((f"*{s}*" for s in skipped))
                        + " — skipped."
                    )
                if st.session_state.prdf_experimental:
                    add_experimental_traces(
                        fig_comp,
                        list(st.session_state.prdf_experimental.keys()),
                        color_offset=len(chosen_structs),
                    )
                fig_comp.update_layout(
                    **make_layout(
                        f"PRDF comparison: {chosen_label}",
                        barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                    )
                )
                st.plotly_chart(fig_comp, width="stretch")
                st.markdown("#### Total RDF comparison")
                fig_tot = go.Figure()
                for s_idx, sname in enumerate(chosen_structs):
                    r = results[sname]
                    bins = sorted(r["global_rdf"].keys())
                    vals = [r["global_rdf"][b] for b in bins]
                    short = sname if len(sname) <= 35 else sname[:32] + "…"
                    add_trace(
                        fig_tot,
                        bins,
                        vals,
                        short,
                        COLORS[s_idx % len(COLORS)],
                        dash=dash_styles[s_idx % len(dash_styles)],
                    )
                if st.session_state.prdf_experimental:
                    add_experimental_traces(
                        fig_tot,
                        list(st.session_state.prdf_experimental.keys()),
                        color_offset=len(chosen_structs),
                    )
                fig_tot.update_layout(
                    **make_layout(
                        "Total RDF comparison",
                        barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                    )
                )
                st.plotly_chart(fig_tot, width="stretch")
    with tab_total:
        st.markdown("### Total RDF – individual structures")
        for s_idx, (fname, r) in enumerate(results.items()):
            bins = sorted(r["global_rdf"].keys())
            vals = [r["global_rdf"][b] for b in bins]
            fig_g = go.Figure()
            add_trace(fig_g, bins, vals, fname, COLORS[s_idx % len(COLORS)])
            if st.session_state.prdf_experimental:
                add_experimental_traces(
                    fig_g, list(st.session_state.prdf_experimental.keys()), color_offset=s_idx + 1
                )
            fig_g.update_layout(**make_layout(f"Total RDF – {fname}"))
            st.plotly_chart(fig_g, width="stretch")
    with tab_dl:
        st.markdown("### Download results as CSV")
        st.button("Prepare CSV files", on_click=prepare_downloads, type="secondary")
        if st.session_state.prdf_download_ready:
            for fname, r in results.items():
                safe = fname.rsplit(".", 1)[0].replace(" ", "_")
                st.markdown(f"#### {fname}")
                for pair, intensities in r["prdf_dict"].items():
                    pair_label = f"{pair[0]}_{pair[1]}"
                    df = pd.DataFrame(
                        {"Distance_Ang": r["dist_dict"][pair], "PRDF_Intensity": intensities}
                    )
                    st.download_button(
                        label=f"⬇️  {pair[0]}–{pair[1]} PRDF",
                        data=df.to_csv(index=False).encode(),
                        file_name=f"{safe}_PRDF_{pair_label}.csv",
                        mime="text/csv",
                        key=f"dl_{safe}_{pair_label}",
                    )
                bins = sorted(r["global_rdf"].keys())
                df_g = pd.DataFrame(
                    {"Distance_Ang": bins, "Total_RDF": [r["global_rdf"][b] for b in bins]}
                )
                st.download_button(
                    label="⬇️  Total RDF",
                    data=df_g.to_csv(index=False).encode(),
                    file_name=f"{safe}_Total_RDF.csv",
                    mime="text/csv",
                    key=f"dl_total_{safe}",
                )
    if has_ordered and tab_ord is not None:
        with tab_ord:
            st.markdown("### Vacancy supercells (random site assignment)")
            st.info(
                f"These supercells were generated from disordered CIF files. The number of atoms per species approximates the CIF occupancies; vacant sites are chosen **randomly** (fixed seed for reproducibility). This is **not** a crystallographically ordered structure — it is a single randomised realisation of the disordered occupancy. Atom cap: **{MAX_SUPERCELL_ATOMS} sites**."
            )
            for fname, ordered_struct in st.session_state.ordered_structures.items():
                if fname not in results:
                    continue
                safe = fname.rsplit(".", 1)[0].replace(" ", "_")
                st.markdown(f"#### {fname}")
                if fname in st.session_state.disorder_messages:
                    st.success(st.session_state.disorder_messages[fname])
                occ_rows = st.session_state.disorder_occ_info.get(fname, [])
                if occ_rows:
                    scale_used = occ_rows[0]["n_copies"]
                    cbrt = round(scale_used ** (1 / 3))
                    st.markdown("**Occupancy approximation — CIF vs. vacancy supercell:**")
                    table_md = "| Site | Fractional coords | Element | CIF occupancy | Sites in supercell | Atoms assigned | Actual occupancy | Delta |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
                    for row in occ_rows:
                        delta = row["actual_occ"] - row["target_occ"]
                        sign = "+" if delta >= 0 else ""
                        flag = "OK" if abs(delta) < 0.02 else "~" if abs(delta) < 0.05 else "(!)"
                        table_md += f"| {row['site_idx']} | {row['frac']} | **{row['element']}** | {row['target_occ']:.4f} | {row['n_copies']} | {row['n_assigned']} | {row['actual_occ']:.4f} | {sign}{delta:.4f} {flag} |\n"
                    st.markdown(table_md)
                    st.caption(
                        f"Supercell scale: {cbrt}x{cbrt}x{cbrt} ({scale_used} copies of each original site). Atom cap: {MAX_SUPERCELL_ATOMS}. OK = |d| < 0.02  ~= |d| < 0.05  (!) = |d| >= 0.05"
                    )
                st.markdown("**Download vacancy supercell:**")
                try:
                    cif_str = str(CifWriter(ordered_struct))
                    st.download_button(
                        label="⬇️  Download vacancy supercell (CIF)",
                        data=cif_str.encode("utf-8"),
                        file_name=f"{safe}_vacancy_supercell.cif",
                        mime="chemical/x-cif",
                        key=f"dl_ordered_tab_{safe}",
                        type="primary",
                    )
                except Exception as e:
                    st.warning(f"Could not export CIF for {fname}: {e}")
                st.markdown("---")
                st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<hr style="border:none;height:5px;background-color:#8b0000;\n                 border-radius:6px;margin:0 0 12px 0;">',
    unsafe_allow_html=True,
)
st.sidebar.info(
    "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. Spot a bug or have a feature idea? Let us know at: **lebedmi2@cvut.cz**. To compile the full app locally, visit our **[GitHub page](https://github.com/bracerino/xrdlicious)**. If you use this tool, please cite the **[article in IUCr](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)**. ❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**"
)
st.markdown(
    "\n### Acknowledgments\n\nThis module uses several open-source tools. We gratefully acknowledge their authors:\n\n- **[Matminer](https://github.com/hackingmaterials/matminer)**\n- **[Pymatgen](https://github.com/materialsproject/pymatgen)**\n- **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)**\n- **[Plotly](https://plotly.com)**\n- **[SciPy](https://scipy.org)**\n\n**XRDlicious (P)RDF module** is open-source and released under the\n[MIT License](https://github.com/bracerino/xrdlicious/blob/main/LICENSE).\n"
)
