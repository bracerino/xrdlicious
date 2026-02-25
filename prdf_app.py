import warnings

import streamlit as st

st.set_page_config(
    page_title="XRDlicious – (P)RDF Calculator",
    layout="wide",
)

st.markdown("""
    <style>
    .block-container { padding-top: 0rem; }
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

css = '''
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px !important;
}

.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f4ff !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    color: #1e3a8a !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #dbe5ff !important;
    cursor: pointer;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #e0e7ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] button:focus {
    outline: none !important;
}
</style>
'''

st.markdown(css, unsafe_allow_html=True)

import os, tempfile
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.ndimage     import gaussian_filter1d
from scipy.signal      import savgol_filter
from scipy.interpolate import make_interp_spline


from ase.io              import read as ase_read
from pymatgen.io.ase     import AseAtomsAdaptor
from pymatgen.core       import Structure
from pymatgen.io.cif     import CifParser
from matminer.featurizers.structure import PartialRadialDistributionFunction


# Limits
MAX_ATOMS   = 500
MAX_SPECIES = 6


def rgb_to_hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255))


COLORS = [rgb_to_hex(c) for c in plt.cm.tab10.colors]
FONT   = dict(size=22, color="black")


def load_structure_from_file(uploaded_file) -> Structure:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw    = uploaded_file.read()
    uploaded_file.seek(0)

    if suffix == "cif":
        try:
            parser  = CifParser.from_str(raw.decode("utf-8", errors="replace"))
            structs = parser.parse_structures(primitive=False)
            if structs:
                return structs[0]
        except Exception:
            pass
    fmt_map = {
        "poscar": "vasp",          "vasp": "vasp",
        "lmp":    "lammps-data",   "data": "lammps-data",
        "xsf":    "xsf",           "xyz":  "extxyz",
        "cfg":    "cfg",           "pw":   "espresso-in",
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
        return False, (
            f"Structure has **{n_atoms} atoms**, which exceeds the limit of "
            f"**{MAX_ATOMS}**. Please use a smaller supercell or a primitive cell."
        )
    if n_sp > MAX_SPECIES:
        return False, (
            f"Structure has **{n_sp} element types**, which exceeds the limit of "
            f"**{MAX_SPECIES}**. Please use a structure with fewer species."
        )
    return True, ""



def smooth_gaussian(y, sigma=1.5):
    return gaussian_filter1d(y, sigma=sigma)


def smooth_savgol(y, window=11, polyorder=3):
    if window % 2 == 0:
        window += 1
    window    = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(y, window, polyorder)


def smooth_spline(x, y, n_pts=300):
    xs = np.linspace(x[0], x[-1], n_pts)
    return xs, np.maximum(0, make_interp_spline(x, y, k=3)(xs))


st.markdown("""
    <h3 style='color:#8b0000;'>
        <strong>XRDlicious</strong> – <em>(P)RDF Calculator</em>
    </h3>
    <hr style="border:none;height:5px;background-color:#8b0000;
               border-radius:6px;margin:0 0 8px 0;">
""", unsafe_allow_html=True)

st.info(
    "Upload one or more crystal structure files in the **sidebar** and press "
    "**▶️ Calculate RDF**. Each structure is processed **individually**. "
    "Use the **🔀 Comparison** tab to overlay PRDFs from different structures.  \n"
    f"⚠️ Limits: **max {MAX_ATOMS} atoms** and **max {MAX_SPECIES} element types** per structure."
)

with st.expander("📖 How to **Cite**", expanded=False):
    st.markdown("""
### How to cite

If you use **XRDlicious**, please cite:

- **XRDlicious**:   
  [LEBEDA, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)


- **Matminer**:  
  [WARD, Logan, et al. *Matminer: An open source toolkit for materials data mining.* Computational Materials Science, 2018, 152: 60–69.](https://www.sciencedirect.com/science/article/pii/S0927025618303252)
    """)


st.sidebar.header("📁 Upload Structure Files")
uploaded_files = st.sidebar.file_uploader(
    "CIF, POSCAR, LMP, XSF, XYZ (with cell), CFG, PW …",
    type=["cif", "xyz", "vasp", "poscar", "lmp", "data", "xsf", "pw", "cfg"],
    accept_multiple_files=True,
    key="prdf_uploader",
)

if "prdf_structures" not in st.session_state:
    st.session_state.prdf_structures = {}
if "prdf_rejected"   not in st.session_state:
    st.session_state.prdf_rejected   = {}

active_names = {f.name for f in uploaded_files} if uploaded_files else set()

if uploaded_files:
    for f in uploaded_files:
        if (f.name not in st.session_state.prdf_structures and
                f.name not in st.session_state.prdf_rejected):
            try:
                struct       = load_structure_from_file(f)
                ok, reason   = validate_structure(struct)
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
for name in list(st.session_state.prdf_rejected.keys()):
    if name not in active_names:
        del st.session_state.prdf_rejected[name]

structures: dict = st.session_state.prdf_structures

for name, reason in st.session_state.prdf_rejected.items():
    st.error(f"**{name}** was not loaded: {reason}")

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


st.subheader("⚙️ Settings", help=(
    "PRDF describes atom-pair distance distributions, providing insight into "
    "local environments and structural disorder. Values are unitless relative "
    "intensities. Peaks = preferred bonding distances; peak width = disorder."
))

col_cut, col_bin = st.columns(2)
cutoff   = col_cut.number_input("Cutoff (Å)",   min_value=1.0,   max_value=50.0, value=10.0, step=0.5,   format="%.1f")
bin_size = col_bin.number_input("Bin size (Å)", min_value=0.005, max_value=2.0,  value=0.10, step=0.005, format="%.3f")

st.markdown("#### Plot options")
col_ps, col_ls = st.columns(2)
plot_style = col_ps.radio(
    "Plot style",
    ["Smooth Curve", "Raw Data Points", "Bars (Histogram)"],
    index=0,
)
line_style = col_ls.radio(
    "Marker style",
    ["Lines Only", "Lines + Markers"],
    index=0,
)

sigma, sg_win, sg_ord, spline_pts, bar_width_factor = 1.5, 11, 3, 300, 0.8
smoothing_method = "Gaussian"

if plot_style == "Smooth Curve":
    col_sm, col_sp = st.columns([2, 2])
    smoothing_method = col_sm.radio(
        "Smoothing method",
        ["Gaussian", "Savitzky-Golay", "Cubic Spline"],
        index=0, horizontal=True,
    )
    if smoothing_method == "Gaussian":
        sigma = col_sp.slider("Gaussian σ", 0.5, 5.0, 1.5, 0.1)
    elif smoothing_method == "Savitzky-Golay":
        sg_win = col_sp.slider("Window length (odd)", 5, 21, 11, 2)
        sg_ord = col_sp.slider("Polynomial order",    2,  5,  3, 1)
    else:
        spline_pts = col_sp.slider("Interpolation points", 100, 600, 300, 50)

if plot_style == "Bars (Histogram)":
    bar_width_factor = st.slider("Bar width factor", 0.1, 2.0, 0.8, 0.1)



def apply_smoothing(x_arr, y_arr):
    x_arr, y_arr = np.array(x_arr), np.array(y_arr)
    if plot_style != "Smooth Curve":
        return x_arr, y_arr
    if smoothing_method == "Gaussian":
        return x_arr, smooth_gaussian(y_arr, sigma)
    elif smoothing_method == "Savitzky-Golay":
        return x_arr, smooth_savgol(y_arr, sg_win, sg_ord)
    else:
        return smooth_spline(x_arr, y_arr, spline_pts)


def add_trace(fig, x, y, name, color, dash="solid"):
    mode = "lines" if line_style == "Lines Only" else "lines+markers"
    if plot_style == "Bars (Histogram)":
        fig.add_trace(go.Bar(
            x=x, y=y, name=name,
            marker=dict(color=color, line=dict(width=0)),
            width=bin_size * bar_width_factor,
            opacity=0.75,
        ))
    else:
        xp, yp = apply_smoothing(x, y)
        if plot_style == "Smooth Curve":
            x_arr_raw, y_arr_raw = np.array(x), np.array(y)
            x_stems, y_stems = [], []
            for xi, yi in zip(x_arr_raw, y_arr_raw):
                x_stems.extend([xi, xi, None])
                y_stems.extend([0, yi, None])
            fig.add_trace(go.Scatter(
                x=x_stems, y=y_stems,
                mode="lines",
                name=f"{name} (raw)",
                line=dict(color=color, width=1),
                opacity=0.35,
                showlegend=True,
            ))
        fig.add_trace(go.Scatter(
            x=xp, y=yp, mode=mode, name=name,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=7) if "markers" in mode else dict(),
        ))


def make_layout(title, barmode=None):
    d = dict(
        title=dict(text=title, font=FONT),
        xaxis=dict(title=dict(text="Distance (Å)", font=FONT), tickfont=FONT),
        yaxis=dict(title=dict(text="(P)RDF Intensity", font=FONT),
                   tickfont=FONT, range=[0, None]),
        hovermode="x",
        font=FONT,
        hoverlabel=dict(font=FONT),
        legend=dict(
            orientation="h", yanchor="top", y=-0.28,
            xanchor="center", x=0.5, font=dict(size=18),
        ),
    )
    if barmode:
        d["barmode"] = barmode
    return d

if "prdf_results"        not in st.session_state:
    st.session_state.prdf_results        = {}
if "prdf_do_calc"        not in st.session_state:
    st.session_state.prdf_do_calc        = False
if "prdf_download_ready" not in st.session_state:
    st.session_state.prdf_download_ready = False


def trigger_calculation():
    st.session_state.prdf_do_calc        = True
    st.session_state.prdf_results        = {}
    st.session_state.prdf_download_ready = False


def prepare_downloads():
    st.session_state.prdf_download_ready = True


st.button(
    "▶️  Calculate RDF",
    on_click=trigger_calculation,
    type="primary",
    disabled=len(structures) == 0,
)

if not structures:
    st.warning("⬆️ Please upload at least one valid structure file in the sidebar.")

if st.session_state.prdf_do_calc and structures:
    struct_items = list(structures.items())
    progress_bar = st.progress(0, text="Starting …")
    calc_errors  = []

    for s_idx, (fname, mg_struct) in enumerate(struct_items):
        progress_bar.progress(s_idx / len(struct_items),
                              text=f"Processing {fname} …")
        try:
            featurizer = PartialRadialDistributionFunction(
                cutoff=cutoff, bin_size=bin_size
            )
            featurizer.fit([mg_struct])
            prdf_vals = featurizer.featurize(mg_struct)
            labels    = featurizer.feature_labels()

            prdf_dict  = {}
            dist_dict  = {}
            global_rdf = {}

            for j, label in enumerate(labels):
                pair_str, rng = label.split(" PRDF r=")
                pair          = tuple(pair_str.split("-"))
                lo, hi        = map(float, rng.split("-"))
                bc            = (lo + hi) / 2.0

                prdf_dict.setdefault(pair, []).append(prdf_vals[j])
                dist_dict.setdefault(pair, []).append(bc)
                global_rdf[bc] = global_rdf.get(bc, 0.0) + prdf_vals[j]

            prdf_dict = {p: np.array(v) for p, v in prdf_dict.items()}

            st.session_state.prdf_results[fname] = {
                "prdf_dict":  prdf_dict,
                "dist_dict":  dist_dict,
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
        {pair for r in results.values() for pair in r["prdf_dict"]},
        key=lambda p: (p[0], p[1]),
    )

    tab_indiv, tab_comp, tab_total, tab_dl = st.tabs([
        "📊 Individual PRDFs",
        "🔀 Comparison",
        "📈 Total RDF",
        "💾 Download",
    ])

    with tab_indiv:
        st.markdown("### Per-structure (P)RDF plots")
        fname_list = list(results.keys())

        selected_struct = (
            fname_list[0] if len(fname_list) == 1
            else st.selectbox("Select structure:", fname_list, key="indiv_select")
        )

        res     = results[selected_struct]
        pairs   = list(res["prdf_dict"].keys())
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
                add_trace(fig,
                          res["dist_dict"][pair],
                          res["prdf_dict"][pair],
                          f"{pair[0]}–{pair[1]}",
                          COLORS[idx % len(COLORS)])
            fig.update_layout(**make_layout(
                f"PRDF – {selected_struct}: all pairs",
                barmode="overlay" if plot_style == "Bars (Histogram)" else None,
            ))
            st.plotly_chart(fig, use_container_width=True)

        else:
            for idx, pair in enumerate(pairs):
                fig = go.Figure()
                add_trace(fig,
                          res["dist_dict"][pair],
                          res["prdf_dict"][pair],
                          f"{pair[0]}–{pair[1]}",
                          COLORS[idx % len(COLORS)])
                fig.update_layout(**make_layout(
                    f"PRDF: {pair[0]}–{pair[1]}  |  {selected_struct}"
                ))
                st.plotly_chart(fig, use_container_width=True)

    with tab_comp:
        st.markdown("### Compare PRDFs across structures")

        if len(results) < 2:
            st.info(
                "Upload and calculate at least **2 structures** to use the "
                "comparison view."
            )
        else:
            col_pair, col_structs = st.columns([1, 2])

            with col_pair:
                pair_labels   = [f"{p[0]}–{p[1]}" for p in all_pairs]
                chosen_label  = st.selectbox(
                    "Element pair:", pair_labels, key="comp_pair"
                )
                chosen_pair   = all_pairs[pair_labels.index(chosen_label)]

            with col_structs:
                all_names     = list(results.keys())
                chosen_structs = st.multiselect(
                    "Structures to overlay:",
                    all_names,
                    default=all_names,
                    key="comp_structs",
                )

            if not chosen_structs:
                st.warning("Select at least one structure.")
            else:
                dash_styles = [
                    "solid", "dash", "dot",
                    "dashdot", "longdash", "longdashdot",
                ]

                fig_comp = go.Figure()
                skipped  = []
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
                        + ", ".join(f"*{s}*" for s in skipped)
                        + " — skipped."
                    )

                fig_comp.update_layout(**make_layout(
                    f"PRDF comparison: {chosen_label}",
                    barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                ))
                st.plotly_chart(fig_comp, use_container_width=True)

                st.markdown("#### Total RDF comparison")
                fig_tot = go.Figure()
                for s_idx, sname in enumerate(chosen_structs):
                    r    = results[sname]
                    bins = sorted(r["global_rdf"].keys())
                    vals = [r["global_rdf"][b] for b in bins]
                    short = sname if len(sname) <= 35 else sname[:32] + "…"
                    add_trace(
                        fig_tot, bins, vals, short,
                        COLORS[s_idx % len(COLORS)],
                        dash=dash_styles[s_idx % len(dash_styles)],
                    )
                fig_tot.update_layout(**make_layout(
                    "Total RDF comparison",
                    barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                ))
                st.plotly_chart(fig_tot, use_container_width=True)

    with tab_total:
        st.markdown("### Total RDF – individual structures")
        for s_idx, (fname, r) in enumerate(results.items()):
            bins = sorted(r["global_rdf"].keys())
            vals = [r["global_rdf"][b] for b in bins]
            fig_g = go.Figure()
            add_trace(fig_g, bins, vals, fname, COLORS[s_idx % len(COLORS)])
            fig_g.update_layout(**make_layout(f"Total RDF – {fname}"))
            st.plotly_chart(fig_g, use_container_width=True)

    with tab_dl:
        st.markdown("### Download results as CSV")
        st.button("Prepare CSV files", on_click=prepare_downloads, type="secondary")

        if st.session_state.prdf_download_ready:
            for fname, r in results.items():
                safe = fname.rsplit(".", 1)[0].replace(" ", "_")
                st.markdown(f"#### {fname}")

                for pair, intensities in r["prdf_dict"].items():
                    pair_label = f"{pair[0]}_{pair[1]}"
                    df = pd.DataFrame({
                        "Distance_Ang":   r["dist_dict"][pair],
                        "PRDF_Intensity": intensities,
                    })
                    st.download_button(
                        label=f"⬇️  {pair[0]}–{pair[1]} PRDF",
                        data=df.to_csv(index=False).encode(),
                        file_name=f"{safe}_PRDF_{pair_label}.csv",
                        mime="text/csv",
                        key=f"dl_{safe}_{pair_label}",
                    )

                bins = sorted(r["global_rdf"].keys())
                df_g = pd.DataFrame({
                    "Distance_Ang": bins,
                    "Total_RDF":    [r["global_rdf"][b] for b in bins],
                })
                st.download_button(
                    label="⬇️  Total RDF",
                    data=df_g.to_csv(index=False).encode(),
                    file_name=f"{safe}_Total_RDF.csv",
                    mime="text/csv",
                    key=f"dl_total_{safe}",
                )
                st.markdown("---")


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """<hr style="border:none;height:5px;background-color:#8b0000;
                 border-radius:6px;margin:0 0 12px 0;">""",
    unsafe_allow_html=True,
)

st.sidebar.info(
    "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. "
    "Spot a bug or have a feature idea? Let us know at: **lebedmi2@cvut.cz**. "
    "To compile the full app locally, visit our "
    "**[GitHub page](https://github.com/bracerino/xrdlicious)**. "
    "If you use this tool, please cite the "
    "**[article in IUCr](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)**. "
    "❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**"
)


st.markdown("""
### Acknowledgments

This module uses several open-source tools. We gratefully acknowledge their authors:

- **[Matminer](https://github.com/hackingmaterials/matminer)**
- **[Pymatgen](https://github.com/materialsproject/pymatgen)**
- **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)**
- **[Plotly](https://plotly.com)**
- **[SciPy](https://scipy.org)**

**XRDlicious (P)RDF module** is open-source and released under the
[MIT License](https://github.com/bracerino/xrdlicious/blob/main/LICENSE).
""")
