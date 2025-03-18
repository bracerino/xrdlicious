import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from matminer.featurizers.structure import PartialRadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from collections import defaultdict
from itertools import combinations
import streamlit.components.v1 as components


st.set_page_config(page_title="Pair Radial Distribution Function (PRDF) for Crystal Structures")
components.html(
    """
    <head>
        <meta name="description" content="Online calculator of Pair Radial Distribution Function (PRDF) and Global RDF for crystal structures">
    </head>
    """,
    height=0,  # Adjust height as needed
)

st.title("Pair Radial Distribution Function (PRDF) and Global RDF Calculator for Crystal Structures")
st.divider()
# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload Structure Files (CIF, POSCAR, XYZ, etc.)",
    type=None,
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")

st.info(
    "Note: Upload structure files (e.g., CIF, POSCAR, XYZ), and the tool will automatically calculate the "
    "Pair Radial Distribution Function (PRDF) for each element combination, as well as the Global RDF. "
    "If multiple files are uploaded, the PRDF will be averaged for corresponding element combinations across the structures. "
    "Below, you can change the cut-off distance (Å) and the bin size to use for PRDF calculation."
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

# --- PRDF Parameters ---
st.divider()
st.subheader("⚙️ PRDF Parameters")
cutoff = st.number_input("⚙️ Cutoff (Å)", min_value=1.0, max_value=50.0, value=10.0, step=1.0, format="%.1f")
bin_size = st.number_input("⚙️ Bin Size (Å)", min_value=0.01, max_value=5.0, value=0.2, step=0.1, format="%.1f")

# --- Calculate and Plot PRDF Automatically ---
if uploaded_files:
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
    st.subheader("📊 OUTPUT → PRDF:")
    # Plot each pair (or element) PRDF
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
            table_str = "#Distance(A)    PRDF\n"
            for x, y in zip(all_distance_dict[comb], prdf_avg):
                table_str += f"{x:<12.3f} {y:<12.3f}\n"
            st.code(table_str, language="text")

    # --- Global RDF Calculation ---
    st.subheader("Global RDF")
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
        table_str = "#Distance(A)    Global RDF\n"
        for x, y in zip(global_bins, global_rdf_avg):
            table_str += f"{x:<12.3f} {y:<12.3f}\n"
        st.code(table_str, language="text")

