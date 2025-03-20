import streamlit as st
st.set_page_config(page_title="RDF and XRD Calculator for Crystal Structures (CIF, POSCAR, XYZ, ...)")

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from matminer.featurizers.structure import PartialRadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from collections import defaultdict
from itertools import combinations
import streamlit.components.v1 as components

# Inject custom CSS for buttons.
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0099ff;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 0.5em 1em;
        border: none;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    /* Override active and focus states to keep the text white */
    div.stButton > button:active,
    div.stButton > button:focus {
        background-color: #0099ff !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

components.html(
    """
    <head>
        <meta name="description" content="Online calculator of Partial Radial Distribution Function (PRDF), Global RDF, and XRD Pattern for Crystal Structures (CIF, POSCAR, XYZ, ...)">
    </head>
    """,
    height=0,
)

st.title("Partial Radial Distribution Function (PRDF), Global RDF, and XRD Pattern Calculator for Crystal Structures (CIF, POSCAR, XYZ, ...)")
st.divider()

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload Structure Files (CIF, POSCAR, XYZ, etc.)",
    type=None,
    accept_multiple_files=True
)
if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")
else:
    st.warning("Please upload at least one structure file. For XRD Pattern calculation, upload only one structure.\n\n [📺 Quick tutorial here](https://youtu.be/-zjuqwXT2-k)")

st.info(
    "**Note:**  Upload structure files (e.g., CIF, POSCAR, XYZ), and this tool will calculate either the "
    "Partial Radial Distribution Function (PRDF) for each element combination, as well as the Global RDF, or the XRD diffraction pattern. "
    "If multiple files are uploaded, the PRDF will be averaged for corresponding element combinations across the structures. "
    "Below, you can change the settings for PRDF or XRD calculation."
)

# --- Detect Atomic Species ---
if uploaded_files:
    species_set = set()
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        structure = read(file.name)
        for atom in structure:
            species_set.add(atom.symbol)
    species_list = sorted(species_set)
    st.subheader("📊 Detected Atomic Species")
    st.write(", ".join(species_list))
else:
    species_list = []

# --- RDF (PRDF) Settings and Calculation ---
st.divider()
st.subheader("⚙️ RDF (PRDF) Settings")
cutoff = st.number_input("⚙️ Cutoff (Å)", min_value=1.0, max_value=50.0, value=10.0, step=1.0, format="%.1f")
bin_size = st.number_input("⚙️ Bin Size (Å)", min_value=0.05, max_value=5.0, value=0.1, step=0.05, format="%.2f")

# Use session state to control RDF calculation.
if "calc_rdf" not in st.session_state:
    st.session_state.calc_rdf = False
if st.button("Calculate RDF"):
    st.session_state.calc_rdf = True

if st.session_state.calc_rdf and uploaded_files:
    st.subheader("📊 OUTPUT → RDF (PRDF & Global RDF)")
    bins = np.arange(0, cutoff + bin_size, bin_size)
    species_combinations = list(combinations(species_list, 2)) + [(s, s) for s in species_list]
    all_prdf_dict = defaultdict(list)
    all_distance_dict = {}
    global_rdf_list = []
    for file in uploaded_files:
        structure = read(file.name)
        mg_structure = AseAtomsAdaptor.get_structure(structure)
        prdf_featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
        prdf_featurizer.fit([mg_structure])
        prdf_data = prdf_featurizer.featurize(mg_structure)
        feature_labels = prdf_featurizer.feature_labels()
        prdf_dict = defaultdict(list)
        distance_dict = {}
        global_dict = {}
        for i, label in enumerate(feature_labels):
            parts = label.split(" PRDF r=")
            element_pair = tuple(parts[0].split("-"))
            distance_range = parts[1].split("-")
            bin_center = (float(distance_range[0]) + float(distance_range[1])) / 2
            prdf_dict[element_pair].append(prdf_data[i])
            if element_pair not in distance_dict:
                distance_dict[element_pair] = []
            distance_dict[element_pair].append(bin_center)
            global_dict[bin_center] = global_dict.get(bin_center, 0) + prdf_data[i]
        for pair, values in prdf_dict.items():
            if pair not in all_distance_dict:
                all_distance_dict[pair] = distance_dict[pair]
            if isinstance(values, float):
                values = [values]
            all_prdf_dict[pair].append(values)
        global_rdf_list.append(global_dict)
    multi_structures = len(uploaded_files) > 1
    colors = plt.cm.tab10.colors
    st.divider()
    st.subheader("PRDF Plots:")
    for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):
        valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]
        if valid_prdf:
            prdf_array = np.vstack(valid_prdf)
            prdf_avg = np.mean(prdf_array, axis=0) if multi_structures else prdf_array[0]
        else:
            prdf_avg = np.zeros_like(all_distance_dict[comb])
        title_str = f"Averaged PRDF: {comb[0]}-{comb[1]}" if multi_structures else f"PRDF: {comb[0]}-{comb[1]}"
        fig, ax = plt.subplots()
        color = colors[idx % len(colors)]
        ax.plot(all_distance_dict[comb], prdf_avg, label=f"{comb[0]}-{comb[1]}", color=color)
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("PRDF Intensity")
        ax.set_title(title_str)
        ax.legend()
        ax.set_ylim(bottom=0)
        st.pyplot(fig)
        with st.expander(f"View Data for {comb[0]}-{comb[1]}"):
            table_str = "#Distance (Å)    PRDF\n"
            for x, y in zip(all_distance_dict[comb], prdf_avg):
                table_str += f"{x:<12.3f} {y:<12.3f}\n"
            st.code(table_str, language="text")
    st.subheader("Global RDF Plot:")
    global_bins_set = set()
    for gd in global_rdf_list:
        global_bins_set.update(gd.keys())
    global_bins = sorted(list(global_bins_set))
    global_rdf_avg = []
    for b in global_bins:
        vals = []
        for gd in global_rdf_list:
            vals.append(gd.get(b, 0))
        global_rdf_avg.append(np.mean(vals))
    fig_global, ax_global = plt.subplots()
    title_global = "Averaged Global RDF" if multi_structures else "Global RDF"
    global_color = colors[len(all_prdf_dict) % len(colors)]
    ax_global.plot(global_bins, global_rdf_avg, label="Global RDF", color=global_color)
    ax_global.set_xlabel("Distance (Å)")
    ax_global.set_ylabel("Global RDF Intensity")
    ax_global.set_title(title_global)
    ax_global.legend()
    ax_global.set_ylim(bottom=0)
    st.pyplot(fig_global)
    with st.expander("View Data for Global RDF"):
        table_str = "#Distance (Å)    Global RDF\n"
        for x, y in zip(global_bins, global_rdf_avg):
            table_str += f"{x:<12.3f} {y:<12.3f}\n"
        st.code(table_str, language="text")

# --- XRD Settings and Calculation ---
st.divider()
st.subheader("⚙️ XRD Settings")

def format_index(index):
    s = str(index)
    if len(s) == 2:
        return s + " "
    return s

# ----- Conversion Functions -----
def twotheta_to_metric(twotheta_deg, metric, wavelength_A, wavelength_nm):
    """
    Converts 2θ (in degrees) to the desired x-axis metric.
    Works for both scalar and array inputs.
    """
    # Ensure input is a numpy array
    twotheta_deg = np.asarray(twotheta_deg)
    theta = np.deg2rad(twotheta_deg / 2)
    if metric == "2θ (°)":
        result = twotheta_deg
    elif metric == "2θ (rad)":
        result = np.deg2rad(twotheta_deg)
    elif metric == "q (1/Å)":
        result = (4 * np.pi / wavelength_A) * np.sin(theta)
    elif metric == "q (1/nm)":
        result = (4 * np.pi / wavelength_nm) * np.sin(theta)
    elif metric == "d (Å)":
        # Vectorized: for each theta, if sin(theta)==0, return np.inf
        result = np.where(np.sin(theta)==0, np.inf, wavelength_A / (2 * np.sin(theta)))
    elif metric == "d (nm)":
        result = np.where(np.sin(theta)==0, np.inf, wavelength_nm / (2 * np.sin(theta)))
    elif metric == "energy (keV)":
        result = (24.796 * np.sin(theta)) / wavelength_A
    elif metric == "frequency (PHz)":
        f_Hz = (24.796 * np.sin(theta)) / wavelength_A * 2.418e17
        result = f_Hz / 1e15
    else:
        result = twotheta_deg
    # If the input was a scalar, return a scalar.
    if np.ndim(twotheta_deg) == 0:
        return float(result)
    return result

def metric_to_twotheta(metric_value, metric, wavelength_A, wavelength_nm):
    """
    Inverts the conversion: given an x-axis value in the desired metric,
    returns the corresponding 2θ (in degrees). Assumes scalar input.
    """
    if metric == "2θ (°)":
        return metric_value
    elif metric == "2θ (rad)":
        return np.rad2deg(metric_value)
    elif metric == "q (1/Å)":
        theta = np.arcsin(np.clip(metric_value * wavelength_A / (4 * np.pi), 0, 1))
        return np.rad2deg(2 * theta)
    elif metric == "q (1/nm)":
        theta = np.arcsin(np.clip(metric_value * wavelength_nm / (4 * np.pi), 0, 1))
        return np.rad2deg(2 * theta)
    elif metric == "d (Å)":
        sin_theta = np.clip(wavelength_A / (2 * metric_value), 0, 1)
        theta = np.arcsin(sin_theta)
        return np.rad2deg(2 * theta)
    elif metric == "d (nm)":
        sin_theta = np.clip(wavelength_nm / (2 * metric_value), 0, 1)
        theta = np.arcsin(sin_theta)
        return np.rad2deg(2 * theta)
    elif metric == "energy (keV)":
        theta = np.arcsin(np.clip(metric_value * wavelength_A / 24.796, 0, 1))
        return np.rad2deg(2 * theta)
    elif metric == "frequency (PHz)":
        f_Hz = metric_value * 1e15
        E_keV = f_Hz / 2.418e17
        theta = np.arcsin(np.clip(E_keV * wavelength_A / 24.796, 0, 1))
        return np.rad2deg(2 * theta)
    else:
        return metric_value

# ----- Conversion Equations Help Information -----
conversion_info = {
    "2θ (°)": "Identity: 2θ in degrees.",
    "2θ (rad)": "Conversion: radians = degrees * (π/180).",
    "q (1/Å)": "q = (4π/λ) * sin(θ), with λ in Å.",
    "q (1/nm)": "q = (4π/λ) * sin(θ), with λ in nm.",
    "d (Å)": "d = λ / (2 sin(θ)), with λ in Å.",
    "d (nm)": "d = λ / (2 sin(θ)), with λ in nm.",
    "energy (keV)": "E = (24.796 * sin(θ)) / λ, with λ in Å.",
    "frequency (PHz)": "f = [(24.796 * sin(θ))/λ * 2.418e17] / 1e15, with λ in Å."
}

# -------------------------------
# --- Wavelength Selection ---
preset_options = ["CuKa", "MoKa", "CoKa"]
preset_wavelengths = {"CuKa": 0.154, "MoKa": 0.071, "CoKa": 0.179}  # in nm
preset_choice = st.selectbox("Preset Wavelength", options=preset_options, index=0)
wavelength_value = st.number_input("Wavelength (nm)",
                                   value=preset_wavelengths[preset_choice],
                                   min_value=0.001,
                                   step=0.001, format="%.3f")
st.write(f"**Using wavelength = {wavelength_value} nm**")
# Convert wavelength from nm to Ångströms (1 nm = 10 Å)
wavelength_A = wavelength_value * 10
wavelength_nm = wavelength_value  # For clarity

# -------------------------------
# --- X-axis Metric Selection ---
x_axis_options = [
    "2θ (°)", "2θ (rad)",
    "q (1/Å)", "q (1/nm)",
    "d (Å)", "d (nm)",
    "energy (keV)", "frequency (PHz)"
]
if "x_axis_metric" not in st.session_state:
    st.session_state.x_axis_metric = x_axis_options[0]

x_axis_metric = st.selectbox(
    "⚙️ XRD x-axis Metric",
    x_axis_options,
    index=x_axis_options.index(st.session_state.x_axis_metric),
    key="x_axis_metric",
    help=conversion_info[st.session_state.x_axis_metric]
)

# -------------------------------
# Default underlying 2θ boundaries.
if x_axis_metric in ["energy (keV)", "frequency (PHz)"]:
    default_twotheta_min = 1.0
elif x_axis_metric in ["d (Å)", "d (nm)"]:
    default_twotheta_min = 20.0
else:
    default_twotheta_min = 0.0
default_twotheta_max = 165.0

# Compute default values in the chosen metric.
default_metric_min = twotheta_to_metric(default_twotheta_min, x_axis_metric, wavelength_A, wavelength_nm)
default_metric_max = twotheta_to_metric(default_twotheta_max, x_axis_metric, wavelength_A, wavelength_nm)

# Choose an appropriate step size.
if x_axis_metric == "2θ (°)":
    step_val = 1.0
elif x_axis_metric == "2θ (rad)":
    step_val = 0.0174533
else:
    step_val = 0.1

col1, col2 = st.columns(2)
min_val = col1.number_input(f"Minimum {x_axis_metric}", value=default_metric_min, step=step_val)
max_val = col2.number_input(f"Maximum {x_axis_metric}", value=default_metric_max, step=step_val)

# Convert the user-specified x-axis limits back to 2θ (in degrees)
two_theta_min = metric_to_twotheta(min_val, x_axis_metric, wavelength_A, wavelength_nm)
two_theta_max = metric_to_twotheta(max_val, x_axis_metric, wavelength_A, wavelength_nm)
two_theta_range = (two_theta_min, two_theta_max)

# -------------------------------
sigma = st.number_input("⚙️ Gaussian sigma (°) for peak sharpness (smaller = sharper peaks)",
                        min_value=0.01, max_value=1.0, value=0.1, step=0.01)
num_annotate = st.number_input("⚙️ Annotate top how many peaks (by intensity):",
                               min_value=0, max_value=30, value=5, step=1)

if "calc_xrd" not in st.session_state:
    st.session_state.calc_xrd = False
if st.button("Calculate XRD"):
    st.session_state.calc_xrd = True

# -------------------------------
# --- XRD Calculation ---
if st.session_state.calc_xrd and uploaded_files:
    st.subheader("📊 OUTPUT → XRD Pattern:")
    # Initialize XRDCalculator with the wavelength in Å (float)
    xrd_calc = XRDCalculator(wavelength=wavelength_A)
    for file in uploaded_files:
        structure = read(file.name)
        mg_structure = AseAtomsAdaptor.get_structure(structure)
        xrd_pattern = xrd_calc.get_pattern(mg_structure, two_theta_range=two_theta_range)
        x_dense = np.linspace(two_theta_min, two_theta_max, 2000)
        x_dense_plot = twotheta_to_metric(x_dense, x_axis_metric, wavelength_A, wavelength_nm)
        peak_vals = twotheta_to_metric(np.array(xrd_pattern.x), x_axis_metric, wavelength_A, wavelength_nm)
        y_dense = np.zeros_like(x_dense)
        for peak, intensity in zip(xrd_pattern.x, xrd_pattern.y):
            y_dense += intensity * np.exp(-((x_dense - peak)**2) / (2 * sigma**2))
        xrd_y_array = np.array(xrd_pattern.y)
        annotate_indices = set(np.argsort(xrd_y_array)[-num_annotate:])
        fig_xrd, ax_xrd = plt.subplots()
        ax_xrd.plot(x_dense_plot, y_dense, label=f"{file.name}")
        for i, (peak, intensity, hkl_group) in enumerate(zip(peak_vals, xrd_pattern.y, xrd_pattern.hkls)):
            if i in annotate_indices:
                hkl_str = ", ".join([f"({format_index(h['hkl'][0])}{format_index(h['hkl'][1])}{format_index(h['hkl'][2])})"
                                      for h in hkl_group])
                ax_xrd.annotate(hkl_str, xy=(peak, intensity), xytext=(0, 4),
                                textcoords='offset points', fontsize=8, rotation=90,
                                ha='center', va='bottom')
        ax_xrd.set_xlabel(x_axis_metric)
        ax_xrd.set_ylabel("Intensity (a.u.)")
        ax_xrd.set_title(f"XRD Pattern: {file.name}")
        ax_xrd.legend()
        st.pyplot(fig_xrd)

        with st.expander(f"View Data for XRD Pattern: {file.name}"):
            table_str = "#X-axis    Intensity    hkl\n"
            for theta, intensity, hkl_group in zip(peak_vals, xrd_pattern.y, xrd_pattern.hkls):
                hkl_str = ", ".join([f"({format_index(h['hkl'][0])}{format_index(h['hkl'][1])}{format_index(h['hkl'][2])})"
                                     for h in hkl_group])
                table_str += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
            st.code(table_str, language="text")

        with st.expander(f"View Data for Highest Intensity Peaks for XRD Pattern: {file.name}", expanded=True):
            table_str2 = "#X-axis    Intensity    hkl\n"
            for i, (theta, intensity, hkl_group) in enumerate(zip(peak_vals, xrd_pattern.y, xrd_pattern.hkls)):
                if i in annotate_indices:
                    hkl_str = ", ".join([f"({format_index(h['hkl'][0])}{format_index(h['hkl'][1])}{format_index(h['hkl'][2])})"
                                         for h in hkl_group])
                    table_str2 += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
            st.code(table_str2, language="text")
