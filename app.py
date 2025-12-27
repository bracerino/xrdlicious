import streamlit as st

st.set_page_config(
    page_title="XRDlicious: Online Calculator for Powder XRD/ND patterns and (P)RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, ...), or XRD data conversion",
    layout="wide"
)
# Remove top padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
    }
    </style>
""", unsafe_allow_html=True)
from helpers import *
from xrd_convert import *
from equivalent_planes import *
from more_funct.reorient import *
from more_funct.citation_section import *

import gc
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from matminer.featurizers.structure import PartialRadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from collections import defaultdict
from itertools import combinations
import streamlit.components.v1 as components
from pymatgen.analysis.prototypes import AflowPrototypeMatcher

import py3Dmol
from io import StringIO
import pandas as pd
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
from pymatgen.core import Structure as PmgStructure
import matplotlib.colors as mcolors
import streamlit as st
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from math import cos, radians, sqrt
import io
import re
import spglib
from pymatgen.core import Structure
from aflow import search, K
from aflow import search  # ensure your file is not named aflow.py!
import aflow.keywords as AFLOW_K
import requests
from PIL import Image
import os
import psutil
import time
import warnings

# Suppersing pymatgen warning about rounding coordinates from CIF
warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")

# import aflow.keywords as K
from pymatgen.io.cif import CifWriter

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"

memory_use_limit = 1600

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.markdown(
#    f"#### **XRDlicious**: Online Calculator for Powder XRD/ND Patterns, (P)RDF, Peak Matching, Structure Modification and Point Defects Creation from Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)")
st.markdown(
    """
    <h4>
        <span style='color:#8b0000;'>
            <strong>XRDlicious</strong> – <em>powder diffraction and more</em>
        </span>
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <hr style="border: none; height: 6px; background-color: #8b0000; border-radius: 8px; margin: 0px 0;">
    """,
    unsafe_allow_html=True
)

# Get current memory usage
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
memory_usage = mem_info.rss / (1024 ** 2)  # in MB

col1, col2, col3 = st.columns([0.2, 0.4, 0.4])
with col3:
    st.markdown(
        """
        ###### 🔹 Separated Modules: 
        - Create point defects in a crystal structure: **[Open App 🌐](https://xrdlicious-point-defects.streamlit.app/)**  
        - Convert between `.xrdml`, `.ras` and `.xy` formats or X/Y-axis: **[Open App 🧩](https://xrd-convert.streamlit.app/)**  
        - Relations between austenite-martensite crystallographic planes for NiTiHf: **[Open App 🪄](https://austenite-martensite.streamlit.app/)**  
        """
    )


with col2:
    st.info(
        "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. Spot a bug or have a feature idea? Let us know at: "
        "**lebedmi2@cvut.cz**. To compile the app locally, visit our **[GitHub page](https://github.com/bracerino/xrdlicious)**. If you like the app, please cite **[article in IUCr](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)**. ❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**"
    )

# with col3:
#    st.link_button("", "https://github.com/bracerino/xrdlicious", type="primary")

with col1:
    about_app_show = st.checkbox(f"📖 About the app")
if about_app_show:
    about_app()
with col1:
    show_roadmap = st.checkbox(f"🧭 Roadmap", value=False)
if show_roadmap:
    with st.expander("Roadmap", icon="🧭", expanded=True):
        show_xrdlicious_roadmap()
with col1:
    citations = st.checkbox("📚 How to cite", value=False)
if citations:
    show_citation_section()

with col1:
    tutorials = st.checkbox("📺 Tutorials", value=False)
if tutorials:
    with st.expander("Tutorials", icon="📺", expanded=True):
        st.markdown(""" 

        - [Calculate powder diffraction patterns](https://youtu.be/jHdaNVB2UWE?si=5OPPsrt-8vr3c9aI)  
        - [Calculate partial and total radial distribution functions](https://youtu.be/aU7BfwlnqGM?si=Hlyl9_cnt9hTf9wD)  
        - [Convert XRD file formats (.ras, .xrdml ↔ .xy)](https://youtu.be/KwxVKadPZ6s?si=IvvZQtmlWl9gOGPw)  
        - [Plot online two-column data & convert XRD between wavelengths / slit types](https://youtu.be/YTzDSI4Jyh0?si=YJt-FS4nBgGA8YhT)  
        - [Create point defects (vacancies, interstitials, substitutions) in a crystal structure](https://youtu.be/cPp-NPxhAYQ?si=vETf52_IHnsps62f)  
        """)

pattern_details = None

st.markdown("""
    <style>
    /* Target tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* Sidebar background with subtle, more transparent gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(155, 89, 182, 0.15),   /* very soft purple */
                rgba(52, 152, 219, 0.15)    /* very soft blue */
            );
            backdrop-filter: blur(6px);  /* adds a glass effect */
        }

        /* Custom caption style */
        .sidebar-caption {
            font-size: 1.15rem;
            font-weight: 600;
            color: inherit;
            margin: 1rem 0 0.5rem 0;
            position: relative;
            display: inline-block;
        }

        .sidebar-caption::after {
            content: "";
            display: block;
            width: 100%;
            height: 3px;
            margin-top: 4px;
            border-radius: 2px;
            background: linear-gradient(to right, #6a11cb, #2575fc);  /* vivid purple → blue underline */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## 🍕 XRDlicious")
mode = "Advanced"
st.markdown(
    """
    <hr style="border: none; height: 6px; background-color: #8b0000; border-radius: 8px; margin: 0px 0;">
    """,
    unsafe_allow_html=True
)

calc_mode = st.sidebar.multiselect(
    "Choose Type(s) of Calculation/Analysis",
    options=[
        "🔬 Structure Modification",
        "💥 Powder Diffraction",
        "📊 (P)RDF",
        "🛠️ Online Search/Match** (UNDER TESTING, being regularly upgraded 😊)",
        "📈 Interactive Data Plot",
        "📉 PRDF from LAMMPS/XYZ trajectories",
        "➡️ .xrdml ↔️ .xy ↔️ .ras Converter",
        "↔️ Equivalent Planes",
    ],
    default=["🔬 Structure Modification", "💥 Powder Diffraction"]
)

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

if "➡️ .xrdml ↔️ .xy ↔️ .ras Converter" in calc_mode:
    run_data_converter()

if "↔️ Equivalent Planes" in calc_mode:
    run_equivalent_hkl_app()

if "📉 PRDF from LAMMPS/XYZ trajectories" in calc_mode:
    st.subheader(
        "This module calculates the Pair Radial Distribution Function (PRDF) across frames in LAMMPS or XYZ trajectories. Due to its high computational demands, it cannot be run on our free online server. Instead, it is provided as a standalone module that must be compiled and executed locally. Please visit to see how to compile and run the code:")
    st.markdown(
        '<p style="font-size:24px;">🔗 <a href="https://github.com/bracerino/PRDF-CP2K-LAMMPS" target="_blank">Download the PRDF calculator for LAMMPS/XYZ trajectories</a></p>',
        unsafe_allow_html=True
    )

if "🛠️ Online Search/Match** (UNDER TESTING, being regularly upgraded 😊)" in calc_mode:
    st.subheader("For the Online Peak Search/Match Subtool, Please visit (USE ONLY FOR TESTING PURPOSES): ")
    st.markdown(
        '<p style="font-size:24px;">🔗 <a href="https://xrdlicious-peak-match.streamlit.app/" target="_blank">Go to Peak Matching Tool</a></p>',
        unsafe_allow_html=True
    )

st.session_state.two_theta_min = 5


def update_element_indices(df):
    element_counts = {}
    for i, row in df.iterrows():
        element = row['Element']
        if element not in element_counts:
            element_counts[element] = 1
        else:
            element_counts[element] += 1
        df.at[i, 'Element_Index'] = f"{element}{element_counts[element]}"
    return df


if 'mp_options' not in st.session_state:
    st.session_state['mp_options'] = None
if 'selected_structure' not in st.session_state:
    st.session_state['selected_structure'] = None
if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []


def remove_fractional_occupancies_safely(structure):
    species = []
    coords = []

    for site in structure:
        if site.is_ordered:
            species.append(site.specie)
        else:
            dominant_sp = max(site.species.items(), key=lambda x: x[1])[0]
            species.append(dominant_sp)
        coords.append(site.frac_coords)
    ordered_structure = Structure(
        lattice=structure.lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=False
    )

    return ordered_structure


col3, col1, col2 = st.columns(3)

if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

st.sidebar.subheader("📁📤 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload structure files (CIF, POSCAR, LMP, XSF, PW, CFG, XYZ (with cell)):",
    type=["cif", "xyz", "vasp", "poscar", "lmp", "data", "xsf", "pw", "cfg"],
    accept_multiple_files=True,
    key="sidebar_uploader"
)

st.sidebar.subheader("📁🧫 Upload Your Experimental Data ")
user_pattern_file = st.sidebar.file_uploader(
    "Upload additional XRD pattern (2 columns: X-values and Intensity. The first line is skipped assuming a header.)",
    type=["csv", "txt", "xy", "data", "dat"],
    key="user_xrd", accept_multiple_files=True
)

if uploaded_files_user_sidebar:
    for file in uploaded_files_user_sidebar:
        if file.name not in st.session_state.full_structures:
            try:
                structure = load_structure(file)
                st.session_state.full_structures[file.name] = structure
                # check_structure_size_and_warn(structure, file.name)
            except Exception as e:
                # st.error(f"Failed to parse {file.name}: {e}")
                st.error(
                    f"This does not work. Are you sure you tried to upload here the structure files (CIF, POSCAR, LMP, XSF, PW)? For the **experimental XY data**, put them to the other uploader\n"
                    f"and please remove this wrongly placed file. 😊")

# Then in Streamlit main block
# display_structure_types()
show_database_search = st.checkbox("🗃️ Enable **database search** (MP, AFLOW, COD, MC3D)",
                                   value=False, )
# Define button colors
buttons_colors()

# Show helpful illustrative image when the app is first opened
if "first_run_note" not in st.session_state:
    st.session_state["first_run_note"] = True
first_run_note()


def get_space_group_info(number):
    symbol = SPACE_GROUP_SYMBOLS.get(number, f"SG#{number}")
    return symbol




if show_database_search:
    with st.expander("Search for Structures Online in Databases", icon="🔍", expanded=True):
        cols, cols2, cols3 = st.columns([1.5, 1.5, 3.5])
        with cols:
            db_choices = st.multiselect(
                "Select Database(s)",
                options=["Materials Project", "AFLOW", "COD", "MC3D"],
                default=["Materials Project", "COD", "MC3D"],
                help="Choose which databases to search for structures. You can select multiple databases."
            )

            if not db_choices:
                st.warning("Please select at least one database to search.")

            st.markdown("**Maximum number of structures to be found in each database (for improving performance):**")
            col_limits = st.columns(4)

            search_limits = {}
            if "Materials Project" in db_choices:
                with col_limits[0]:
                    search_limits["Materials Project"] = st.number_input(
                        "MP Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from Materials Project"
                    )
            if "AFLOW" in db_choices:
                with col_limits[1]:
                    search_limits["AFLOW"] = st.number_input(
                        "AFLOW Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from AFLOW"
                    )
            if "COD" in db_choices:
                with col_limits[2]:
                    search_limits["COD"] = st.number_input(
                        "COD Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from COD"
                    )
            if "MC3D" in db_choices:
                with col_limits[3]:
                    search_limits["MC3D"] = st.number_input(
                        "MC3D Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from MC3D"
                    )

        with cols2:
            search_mode = st.radio(
                "Search by:",
                options=["Elements", "Structure ID", "Space Group + Elements", "Formula", "Search Mineral"],
                help="Choose your search strategy"
            )

            if search_mode == "Elements":
                selected_elements = st.multiselect(
                    "Select elements for search:",
                    options=ELEMENTS,
                    default=["Na", "Cl"],
                    help="Choose one or more chemical elements"
                )
                search_query = " ".join(selected_elements) if selected_elements else ""

            elif search_mode == "Structure ID":
                structure_ids = st.text_area(
                    "Enter Structure IDs (one per line):",
                    value="mp-5229\ncod_1512124\naflow:010158cb2b41a1a5\nmc3d-82734",
                    help="Enter structure IDs. Examples:\n- Materials Project: mp-5229\n- COD: cod_1512124 (with cod_ prefix)\n- AFLOW: aflow:010158cb2b41a1a5 (AUID format)"
                )

            elif search_mode == "Space Group + Elements":
                selected_space_group = st.selectbox(
                    "Select Space Group:",
                    options=SPACE_GROUP_OPTIONS,
                    index=224,  #
                    help="Start typing to search by number or symbol",
                    key="db_search_space_group"
                )

                space_group_number = extract_space_group_number(selected_space_group)
                space_group_symbol = selected_space_group.split('(')[1][:-1] if selected_space_group else ""

                # st.info(f"Selected: **{space_group_number}** ({space_group_symbol})")

                selected_elements = st.multiselect(
                    "Select elements for search:",
                    options=ELEMENTS,
                    default=["Na", "Cl"],
                    help="Choose one or more chemical elements"
                )

            elif search_mode == "Formula":
                formula_input = st.text_input(
                    "Enter Chemical Formula:",
                    value="Sr Ti O3",
                    help="Enter chemical formula with spaces between elements. Examples:\n- Sr Ti O3 (strontium titanate)\n- Ca C O3 (calcium carbonate)\n- Al2 O3 (alumina)"
                )

            elif search_mode == "Search Mineral":
                mineral_options = []
                mineral_mapping = {}

                for space_group, minerals in MINERALS.items():
                    for mineral_name, formula in minerals.items():
                        option_text = f"{mineral_name} - SG #{space_group}"
                        mineral_options.append(option_text)
                        mineral_mapping[option_text] = {
                            'space_group': space_group,
                            'formula': formula,
                            'mineral_name': mineral_name
                        }

                # Sort mineral options alphabetically
                mineral_options.sort()

                selected_mineral = st.selectbox(
                    "Select Mineral Structure:",
                    options=mineral_options,
                    help="Choose a mineral structure type. The exact formula and space group will be automatically set.",
                    index=2
                )

                if selected_mineral:
                    mineral_info = mineral_mapping[selected_mineral]

                    # col_mineral1, col_mineral2 = st.columns(2)
                    # with col_mineral1:
                    sg_symbol = get_space_group_info(mineral_info['space_group'])
                    st.info(
                        f"**Structure:** {mineral_info['mineral_name']}, **Space Group:** {mineral_info['space_group']} ({sg_symbol}), "
                        f"**Formula:** {mineral_info['formula']}")

                    space_group_number = mineral_info['space_group']
                    formula_input = mineral_info['formula']

                    st.success(f"**Search will use:** Formula = {formula_input}, Space Group = {space_group_number}")

            show_element_info = st.checkbox("ℹ️ Show information about element groups")
            if show_element_info:
                st.markdown("""
                **Element groups note:**
                **Common Elements (14):** H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca  
                **Transition Metals (10):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn  
                **Alkali Metals (6):** Li, Na, K, Rb, Cs, Fr  
                **Alkaline Earth (6):** Be, Mg, Ca, Sr, Ba, Ra  
                **Noble Gases (6):** He, Ne, Ar, Kr, Xe, Rn  
                **Halogens (5):** F, Cl, Br, I, At  
                **Lanthanides (15):** La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu  
                **Actinides (15):** Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr  
                **Other Elements (51):** All remaining elements
                """)

        if st.button("Search Selected Databases"):
            if not db_choices:
                st.error("Please select at least one database to search.")
            else:
                for db_choice in db_choices:
                    if db_choice == "Materials Project":
                        mp_limit = search_limits.get("Materials Project", 50)
                        with st.spinner(f"Searching **the MP database** (limit: {mp_limit}), please wait. 😊"):
                            try:
                                with MPRester(MP_API_KEY) as mpr:
                                    docs = None

                                    if search_mode == "Elements":
                                        elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                        if not elements_list:
                                            st.error("Please enter at least one element for the search.")
                                            continue
                                        elements_list_sorted = sorted(set(elements_list))
                                        docs = mpr.materials.summary.search(
                                            elements=elements_list_sorted,
                                            num_elements=len(elements_list_sorted),
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    elif search_mode == "Structure ID":
                                        mp_ids = [id.strip() for id in structure_ids.split('\n')
                                                  if id.strip() and id.strip().startswith('mp-')]
                                        if not mp_ids:
                                            st.warning("No valid Materials Project IDs found (should start with 'mp-')")
                                            continue
                                        docs = mpr.materials.summary.search(
                                            material_ids=mp_ids,
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    elif search_mode == "Space Group + Elements":
                                        elements_list = sorted(set(selected_elements))
                                        if not elements_list:
                                            st.warning(
                                                "Please select elements for Materials Project space group search.")
                                            continue

                                        search_params = {
                                            "elements": elements_list,
                                            "num_elements": len(elements_list),
                                            "fields": ["material_id", "formula_pretty", "symmetry", "nsites", "volume"],
                                            "spacegroup_number": space_group_number
                                        }

                                        docs = mpr.materials.summary.search(**search_params)

                                    elif search_mode == "Formula":
                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for Materials Project search.")
                                            continue

                                        # Convert space-separated format to compact format (Sr Ti O3 -> SrTiO3)
                                        clean_formula = formula_input.strip()
                                        if ' ' in clean_formula:
                                            parts = clean_formula.split()
                                            compact_formula = ''.join(parts)
                                        else:
                                            compact_formula = clean_formula

                                        docs = mpr.materials.summary.search(
                                            formula=compact_formula,
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning(
                                                "Please select a mineral structure for Materials Project search.")
                                            continue
                                        clean_formula = formula_input.strip()
                                        if ' ' in clean_formula:
                                            parts = clean_formula.split()
                                            compact_formula = ''.join(parts)
                                        else:
                                            compact_formula = clean_formula

                                        # Search by formula and space group
                                        docs = mpr.materials.summary.search(
                                            formula=compact_formula,
                                            spacegroup_number=space_group_number,
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    if docs:
                                        status_placeholder = st.empty()
                                        st.session_state.mp_options = []
                                        st.session_state.full_structures_see = {}
                                        limited_docs = docs[:mp_limit]

                                        for doc in limited_docs:
                                            full_structure = mpr.get_structure_by_material_id(doc.material_id,
                                                                                              conventional_unit_cell=True)
                                            st.session_state.full_structures_see[doc.material_id] = full_structure
                                            lattice = full_structure.lattice
                                            leng = len(full_structure)
                                            lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                                           f"{lattice.alpha:.1f}, {lattice.beta:.1f}, {lattice.gamma:.1f} °")
                                            st.session_state.mp_options.append(
                                                f"{doc.formula_pretty} ({doc.symmetry.symbol} #{doc.symmetry.number}), {leng} atoms, [{lattice_str}], {float(doc.volume):.1f} Å³, {doc.material_id}:"
                                            )
                                            status_placeholder.markdown(
                                                f"- **Structure loaded:** `{full_structure.composition.reduced_formula}` ({doc.material_id})"
                                            )
                                        if len(limited_docs) < len(docs):
                                            st.info(
                                                f"Showing first {mp_limit} of {len(docs)} total Materials Project results. Increase limit to see more.")
                                        st.success(
                                            f"Found {len(st.session_state.mp_options)} structures in Materials Project.")
                                    else:
                                        st.session_state.mp_options = []
                                        st.warning("No matching structures found in Materials Project.")
                            except Exception as e:
                                st.error(f"An error occurred with Materials Project: {e}")

                    elif db_choice == "AFLOW":
                        aflow_limit = search_limits.get("AFLOW", 50)
                        with st.spinner(f"Searching **the AFLOW database** (limit: {aflow_limit}), please wait. 😊"):
                            try:
                                results = []

                                if search_mode == "Elements":
                                    elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                    if not elements_list:
                                        st.warning("Please enter elements for AFLOW search.")
                                        continue
                                    ordered_elements = sorted(elements_list)
                                    ordered_str = ",".join(ordered_elements)
                                    aflow_nspecies = len(ordered_elements)

                                    results = list(
                                        search(catalog="icsd")
                                        .filter((AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                                        .select(
                                            AFLOW_K.auid,
                                            AFLOW_K.compound,
                                            AFLOW_K.geometry,
                                            AFLOW_K.spacegroup_relax,
                                            AFLOW_K.aurl,
                                            AFLOW_K.files,
                                        )
                                    )

                                elif search_mode == "Structure ID":
                                    aflow_auids = []
                                    for id_line in structure_ids.split('\n'):
                                        id_line = id_line.strip()
                                        if id_line.startswith('aflow:'):
                                            auid = id_line.replace('aflow:', '').strip()
                                            aflow_auids.append(auid)

                                    if not aflow_auids:
                                        st.warning("No valid AFLOW AUIDs found (should start with 'aflow:')")
                                        continue

                                    results = []
                                    for auid in aflow_auids:
                                        try:
                                            result = list(search(catalog="icsd")
                                                          .filter(AFLOW_K.auid == f"aflow:{auid}")
                                                          .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                  AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                  AFLOW_K.files))
                                            results.extend(result)
                                        except Exception as e:
                                            st.warning(f"AFLOW search failed for AUID '{auid}': {e}")
                                            continue

                                elif search_mode == "Space Group + Elements":
                                    if not selected_elements:
                                        st.warning("Please select elements for AFLOW space group search.")
                                        continue
                                    ordered_elements = sorted(selected_elements)
                                    ordered_str = ",".join(ordered_elements)
                                    aflow_nspecies = len(ordered_elements)

                                    try:
                                        results = list(search(catalog="icsd")
                                                       .filter((AFLOW_K.species % ordered_str) &
                                                               (AFLOW_K.nspecies == aflow_nspecies) &
                                                               (AFLOW_K.spacegroup_relax == space_group_number))
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))
                                    except Exception as e:
                                        st.warning(f"AFLOW space group search failed: {e}")
                                        results = []


                                elif search_mode == "Formula":

                                    if not formula_input.strip():
                                        st.warning("Please enter a chemical formula for AFLOW search.")

                                        continue


                                    def convert_to_aflow_formula(formula_input):

                                        import re

                                        formula_parts = formula_input.strip().split()

                                        elements_dict = {}

                                        for part in formula_parts:

                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                            if match:
                                                element = match.group(1)

                                                count = match.group(2) if match.group(
                                                    2) else "1"  # Add "1" if no number

                                                elements_dict[element] = count

                                        aflow_parts = []

                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")

                                        return "".join(aflow_parts)


                                    # Generate 2x multiplied formula
                                    def multiply_formula_by_2(formula_input):

                                        import re

                                        formula_parts = formula_input.strip().split()

                                        elements_dict = {}

                                        for part in formula_parts:

                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                            if match:
                                                element = match.group(1)

                                                count = int(match.group(2)) if match.group(2) else 1

                                                elements_dict[element] = str(count * 2)  # Multiply by 2

                                        aflow_parts = []

                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")

                                        return "".join(aflow_parts)


                                    aflow_formula = convert_to_aflow_formula(formula_input)

                                    aflow_formula_2x = multiply_formula_by_2(formula_input)

                                    if aflow_formula_2x != aflow_formula:

                                        results = list(search(catalog="icsd")

                                                       .filter((AFLOW_K.compound == aflow_formula) |

                                                               (AFLOW_K.compound == aflow_formula_2x))

                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,

                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(
                                            f"Searching for both {aflow_formula} and {aflow_formula_2x} formulas simultaneously")

                                    else:
                                        results = list(search(catalog="icsd")
                                                       .filter(AFLOW_K.compound == aflow_formula)
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(f"Searching for formula {aflow_formula}")


                                elif search_mode == "Search Mineral":
                                    if not selected_mineral:
                                        st.warning("Please select a mineral structure for AFLOW search.")
                                        continue


                                    def convert_to_aflow_formula_mineral(formula_input):
                                        import re
                                        formula_parts = formula_input.strip().split()
                                        elements_dict = {}
                                        for part in formula_parts:

                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                            if match:
                                                element = match.group(1)

                                                count = match.group(2) if match.group(
                                                    2) else "1"  # Always add "1" for single atoms

                                                elements_dict[element] = count

                                        aflow_parts = []

                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")

                                        return "".join(aflow_parts)


                                    def multiply_mineral_formula_by_2(formula_input):

                                        import re

                                        formula_parts = formula_input.strip().split()

                                        elements_dict = {}

                                        for part in formula_parts:
                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                            if match:
                                                element = match.group(1)
                                                count = int(match.group(2)) if match.group(2) else 1
                                                elements_dict[element] = str(count * 2)  # Multiply by 2
                                        aflow_parts = []
                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")
                                        return "".join(aflow_parts)


                                    aflow_formula = convert_to_aflow_formula_mineral(formula_input)

                                    aflow_formula_2x = multiply_mineral_formula_by_2(formula_input)

                                    # Search for both formulas with space group constraint in a single query

                                    if aflow_formula_2x != aflow_formula:
                                        results = list(search(catalog="icsd")
                                                       .filter(((AFLOW_K.compound == aflow_formula) |
                                                                (AFLOW_K.compound == aflow_formula_2x)) &
                                                               (AFLOW_K.spacegroup_relax == space_group_number))
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(
                                            f"Searching {mineral_info['mineral_name']} for both {aflow_formula} and {aflow_formula_2x} with space group {space_group_number}")

                                    else:
                                        results = list(search(catalog="icsd")
                                                       .filter((AFLOW_K.compound == aflow_formula) &
                                                               (AFLOW_K.spacegroup_relax == space_group_number))
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(
                                            f"Searching {mineral_info['mineral_name']} for formula {aflow_formula} with space group {space_group_number}")

                                if results:
                                    status_placeholder = st.empty()
                                    st.session_state.aflow_options = []
                                    st.session_state.entrys = {}

                                    limited_results = results[:aflow_limit]

                                    for entry in limited_results:
                                        st.session_state.entrys[entry.auid] = entry
                                        st.session_state.aflow_options.append(
                                            f"{entry.compound} ({entry.spacegroup_relax}) {entry.geometry}, {entry.auid}"
                                        )
                                        status_placeholder.markdown(
                                            f"- **Structure loaded:** `{entry.compound}` (aflow_{entry.auid})"
                                        )
                                    if len(limited_results) < len(results):
                                        st.info(
                                            f"Showing first {aflow_limit} of {len(results)} total AFLOW results. Increase limit to see more.")
                                    st.success(f"Found {len(st.session_state.aflow_options)} structures in AFLOW.")
                                else:
                                    st.session_state.aflow_options = []
                                    st.warning("No matching structures found in AFLOW.")
                            except Exception as e:
                                st.warning(f"No matching structures found in AFLOW.")
                                st.session_state.aflow_options = []
                    elif db_choice == "MC3D":
                        mc3d_limit = search_limits.get("MC3D", 300)
                        with st.spinner(f"Searching **the MC3D database** (limit: {mc3d_limit}), please wait. 😊"):
                            results = []

                            try:
                                query_params = {}

                                if search_mode == "Elements":
                                    elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                    if not elements_list:
                                        st.warning("Please enter elements for MC3D search.")
                                    else:
                                        query_params['elements'] = sorted(set(elements_list))
                                        results = search_mc3d_optimade(query_params, limit=mc3d_limit)

                                elif search_mode == "Structure ID":
                                    mc3d_ids = []
                                    for id_line in structure_ids.split('\n'):
                                        id_line = id_line.strip()
                                        if id_line.startswith('mc3d-') or id_line.startswith('mcloud-'):
                                            mc3d_ids.append(id_line)

                                    if not mc3d_ids:
                                        st.warning("No valid MC3D IDs found (should start with 'mc3d-' or 'mcloud-')")
                                    else:
                                        for mc3d_id in mc3d_ids:
                                            structure = get_mc3d_structure_by_id(mc3d_id)
                                            if structure:
                                                results.append({
                                                    'id': mc3d_id,
                                                    'structure': structure,
                                                    'formula': structure.composition.reduced_formula
                                                })

                                elif search_mode == "Space Group + Elements":
                                    if not selected_elements:
                                        st.warning("Please select elements for MC3D space group search.")
                                    else:
                                        query_params['elements'] = sorted(selected_elements)
                                        results = search_mc3d_optimade(query_params, limit=mc3d_limit)

                                        filtered_results = []
                                        for result in results:
                                            structure = result['structure']
                                            analyzer = SpacegroupAnalyzer(structure)
                                            if analyzer.get_space_group_number() == space_group_number:
                                                filtered_results.append(result)
                                        results = filtered_results

                                elif search_mode == "Formula":
                                    if not formula_input.strip():
                                        st.warning("Please enter a chemical formula for MC3D search.")
                                    else:
                                        try:
                                            from pymatgen.core import Composition

                                            comp = Composition(formula_input.strip())
                                            normalized_formula = comp.reduced_formula
                                            query_params['formula'] = normalized_formula
                                            st.info(f"Searching for normalized formula: {normalized_formula}")
                                        except:
                                            query_params['formula'] = formula_input.strip()
                                        results = search_mc3d_optimade(query_params, limit=mc3d_limit)

                                elif search_mode == "Search Mineral":
                                    if not selected_mineral:
                                        st.warning("Please select a mineral structure for MC3D search.")
                                    else:
                                        try:
                                            from pymatgen.core import Composition

                                            comp = Composition(formula_input.strip())
                                            normalized_formula = comp.reduced_formula
                                            query_params['formula'] = normalized_formula
                                        except:
                                            query_params['formula'] = formula_input.strip()

                                        results = search_mc3d_optimade(query_params, limit=mc3d_limit)

                                        filtered_results = []
                                        for result in results:
                                            structure = result['structure']
                                            analyzer = SpacegroupAnalyzer(structure)
                                            if analyzer.get_space_group_number() == space_group_number:
                                                filtered_results.append(result)
                                        results = filtered_results

                                else:
                                    if query_params:
                                        results = search_mc3d_optimade(query_params, limit=mc3d_limit)

                                if results:
                                    st.session_state.mc3d_options = []
                                    st.session_state.mc3d_structures = {}

                                    for result in results:
                                        mc3d_id = result['id']
                                        structure = result['structure']
                                        formula = result['formula']

                                        st.session_state.mc3d_structures[mc3d_id] = structure

                                        analyzer = SpacegroupAnalyzer(structure)
                                        sg_number = analyzer.get_space_group_number()
                                        sg_symbol = SPACE_GROUP_SYMBOLS.get(sg_number, f"SG#{sg_number}")

                                        #n_elements = len(structure.composition.elements)
                                        n_atoms = len(structure)

                                        option_str = (
                                            f"{formula} ({sg_symbol} #{sg_number}), "
                                            f"{n_atoms} atoms, "
                                            f"{mc3d_id}"
                                        )
                                        st.session_state.mc3d_options.append(option_str)

                                    st.success(
                                        f"✅ Found {len(st.session_state.mc3d_options)} structures in MC3D via OPTIMADE.")
                                else:
                                    st.session_state.mc3d_options = []
                                    st.warning("No matching structures found in MC3D.")

                            except Exception as e:
                                st.error(f"MC3D search error: {str(e)}")
                                import traceback

                                st.write(traceback.format_exc())
                                st.session_state.mc3d_options = []
                    elif db_choice == "COD":
                        cod_limit = search_limits.get("COD", 50)
                        with st.spinner(f"Searching **the COD database** (limit: {cod_limit}), please wait. 😊"):
                            try:
                                cod_entries = []

                                if search_mode == "Elements":
                                    elements = [el.strip() for el in search_query.split() if el.strip()]
                                    if elements:
                                        params = {'format': 'json', 'detail': '1'}
                                        for i, el in enumerate(elements, start=1):
                                            params[f'el{i}'] = el
                                        params['strictmin'] = str(len(elements))
                                        params['strictmax'] = str(len(elements))
                                        cod_entries = get_cod_entries(params)
                                    else:
                                        st.warning("Please enter elements for COD search.")
                                        continue

                                elif search_mode == "Structure ID":
                                    cod_ids = []
                                    for id_line in structure_ids.split('\n'):
                                        id_line = id_line.strip()
                                        if id_line.startswith('cod_'):
                                            # Extract numeric ID from cod_XXXXX format
                                            numeric_id = id_line.replace('cod_', '').strip()
                                            if numeric_id.isdigit():
                                                cod_ids.append(numeric_id)

                                    if not cod_ids:
                                        st.warning(
                                            "No valid COD IDs found (should start with 'cod_' followed by numbers)")
                                        continue

                                    cod_entries = []
                                    for cod_id in cod_ids:
                                        try:
                                            params = {'format': 'json', 'detail': '1', 'id': cod_id}
                                            entry = get_cod_entries(params)
                                            if entry:
                                                if isinstance(entry, list):
                                                    cod_entries.extend(entry)
                                                else:
                                                    cod_entries.append(entry)
                                        except Exception as e:
                                            st.warning(f"COD search failed for ID {cod_id}: {e}")
                                            continue

                                elif search_mode == "Space Group + Elements":
                                    elements = selected_elements
                                    if elements:
                                        params = {'format': 'json', 'detail': '1'}
                                        for i, el in enumerate(elements, start=1):
                                            params[f'el{i}'] = el
                                        params['strictmin'] = str(len(elements))
                                        params['strictmax'] = str(len(elements))
                                        params['space_group_number'] = str(space_group_number)

                                        cod_entries = get_cod_entries(params)
                                    else:
                                        st.warning("Please select elements for COD space group search.")
                                        continue

                                elif search_mode == "Formula":
                                    if not formula_input.strip():
                                        st.warning("Please enter a chemical formula for COD search.")
                                        continue

                                    # alphabet sorting
                                    alphabet_form = sort_formula_alphabetically(formula_input)
                                    print(alphabet_form)
                                    params = {'format': 'json', 'detail': '1', 'formula': alphabet_form}
                                    cod_entries = get_cod_entries(params)

                                elif search_mode == "Search Mineral":
                                    if not selected_mineral:
                                        st.warning("Please select a mineral structure for COD search.")
                                        continue

                                    # Use both formula and space group for COD search
                                    alphabet_form = sort_formula_alphabetically(formula_input)
                                    params = {
                                        'format': 'json',
                                        'detail': '1',
                                        'formula': alphabet_form,
                                        'space_group_number': str(space_group_number)
                                    }
                                    cod_entries = get_cod_entries(params)

                                if cod_entries and isinstance(cod_entries, list):
                                    st.session_state.cod_options = []
                                    st.session_state.full_structures_see_cod = {}
                                    status_placeholder = st.empty()
                                    limited_entries = cod_entries[:cod_limit]
                                    errors = []

                                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                        future_to_entry = {executor.submit(fetch_and_parse_cod_cif, entry): entry for
                                                           entry in limited_entries}

                                        processed_count = 0
                                        for future in concurrent.futures.as_completed(future_to_entry):
                                            processed_count += 1
                                            status_placeholder.markdown(
                                                f"- **Processing:** {processed_count}/{len(limited_entries)} entries...")
                                            try:
                                                cod_id, structure, entry_data, error = future.result()
                                                if error:
                                                    original_entry = future_to_entry[future]
                                                    errors.append(
                                                        f"Entry `{original_entry.get('file', 'N/A')}` failed: {error}")
                                                    continue  # Skip to the next completed future
                                                if cod_id and structure and entry_data:
                                                    st.session_state.full_structures_see_cod[cod_id] = structure

                                                    spcs = entry_data.get("sg", "Unknown")
                                                    spcs_number = entry_data.get("sgNumber", "Unknown")
                                                    cell_volume = structure.lattice.volume
                                                    option_str = (
                                                        f"{structure.composition.reduced_formula} ({spcs} #{spcs_number}), {len(structure)} atoms, [{structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} Å, {structure.lattice.alpha:.2f}, "
                                                        f"{structure.lattice.beta:.2f}, {structure.lattice.gamma:.2f}°], {cell_volume:.1f} Å³, {cod_id}"
                                                    )
                                                    st.session_state.cod_options.append(option_str)

                                            except Exception as e:
                                                errors.append(
                                                    f"A critical error occurred while processing a result: {e}")
                                    status_placeholder.empty()
                                    if st.session_state.cod_options:
                                        if len(limited_entries) < len(cod_entries):
                                            st.info(
                                                f"Showing first {cod_limit} of {len(cod_entries)} total COD results. Increase limit to see more.")
                                        st.success(
                                            f"Found and processed {len(st.session_state.cod_options)} structures from COD.")
                                    else:
                                        st.warning("COD: No matching structures could be successfully processed.")
                                    if errors:
                                        st.error(f"Encountered {len(errors)} error(s) during the search.")
                                        with st.container(border=True):
                                            for e in errors:
                                                st.warning(e)
                                else:
                                    st.session_state.cod_options = []
                                    st.warning("COD: No matching structures found.")
                            except Exception as e:
                                st.warning(f"COD search error: {e}")
                                st.session_state.cod_options = []

        # with cols2:
        #     image = Image.open("images/Rabbit2.png")
        #     st.image(image, use_container_width=True)

        with cols3:
            if any(x in st.session_state for x in ['mp_options', 'aflow_options', 'cod_options', 'mc3d_options']):
                tabs = []
                if 'mp_options' in st.session_state and st.session_state.mp_options:
                    tabs.append("Materials Project")
                if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                    tabs.append("AFLOW")
                if 'cod_options' in st.session_state and st.session_state.cod_options:
                    tabs.append("COD")
                if 'mc3d_options' in st.session_state and st.session_state.mc3d_options:
                    tabs.append("MC3D")
                if tabs:
                    selected_tab = st.tabs(tabs)

                    tab_index = 0
                    if 'mp_options' in st.session_state and st.session_state.mp_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in Materials Project")
                            selected_structure = st.selectbox("Select a structure from MP:",
                                                              st.session_state.mp_options)
                            selected_id = selected_structure.split(",")[-1].replace(":", "").strip()
                            composition = selected_structure.split("(")[0].strip()
                            file_name = f"{selected_id}_{composition}.cif"
                            file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

                            if selected_id in st.session_state.full_structures_see:
                                selected_entry = st.session_state.full_structures_see[selected_id]

                                conv_lattice = selected_entry.lattice
                                cell_volume = selected_entry.lattice.volume
                                density = str(selected_entry.density).split()[0]
                                n_atoms = len(selected_entry)
                                atomic_den = n_atoms / cell_volume

                                structure_type = identify_structure_type(selected_entry)
                                st.write(f"**Structure type:** {structure_type}")
                                analyzer = SpacegroupAnalyzer(selected_entry)
                                st.write(
                                    f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                st.write(
                                    f"**Material ID:** {selected_id}, **Formula:** {composition}, **N. of Atoms:** {n_atoms}")

                                st.write(
                                    f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                mp_url = f"https://materialsproject.org/materials/{selected_id}"
                                st.write(f"**Link:** {mp_url}")

                                col_mpd, col_mpb = st.columns([2, 1])
                                with col_mpd:
                                    if st.button("Add Selected Structure (MP)", key="add_btn_mp"):
                                        pmg_structure = st.session_state.full_structures_see[selected_id]
                                        check_structure_size_and_warn(pmg_structure, f"MP structure {selected_id}")
                                        st.session_state.full_structures[file_name] = pmg_structure
                                        cif_writer = CifWriter(pmg_structure)
                                        cif_content = cif_writer.__str__()
                                        cif_file = io.BytesIO(cif_content.encode('utf-8'))
                                        cif_file.name = file_name
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        if all(f.name != file_name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)
                                        st.success("Structure added from Materials Project!")
                                with col_mpb:
                                    st.download_button(
                                        label="Download MP CIF",
                                        data=str(
                                            CifWriter(st.session_state.full_structures_see[selected_id], symprec=0.01)),
                                        file_name=file_name,
                                        type="primary",
                                        mime="chemical/x-cif"
                                    )
                                st.info(
                                    f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                        tab_index += 1

                    if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in AFLOW")
                            st.warning(
                                "The AFLOW does not provide atomic occupancies and includes only information about primitive cell in API. For better performance, volume and n. of atoms are purposely omitted from the expander.")
                            selected_structure = st.selectbox("Select a structure from AFLOW:",
                                                              st.session_state.aflow_options)
                            selected_auid = selected_structure.split(",")[-1].strip()
                            selected_entry = next(
                                (entry for entry in st.session_state.entrys.values() if entry.auid == selected_auid),
                                None)
                            if selected_entry:

                                cif_files = [f for f in selected_entry.files if
                                             f.endswith("_sprim.cif") or f.endswith(".cif")]

                                if cif_files:

                                    cif_filename = cif_files[0]

                                    # Correct the AURL: replace the first ':' with '/'

                                    host_part, path_part = selected_entry.aurl.split(":", 1)

                                    corrected_aurl = f"{host_part}/{path_part}"

                                    file_url = f"http://{corrected_aurl}/{cif_filename}"
                                    response = requests.get(file_url)
                                    cif_content = response.content

                                    structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'), fmt="cif")
                                    converted_structure = get_full_conventional_structure(structure_from_aflow,
                                                                                          symprec=0.1)

                                    conv_lattice = converted_structure.lattice
                                    cell_volume = converted_structure.lattice.volume
                                    density = str(converted_structure.density).split()[0]
                                    n_atoms = len(converted_structure)
                                    atomic_den = n_atoms / cell_volume

                                    structure_type = identify_structure_type(converted_structure)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(structure_from_aflow)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
                                    st.write(
                                        f"**AUID:** {selected_entry.auid}, **Formula:** {selected_entry.compound}, **N. of Atoms:** {n_atoms}")
                                    st.write(
                                        f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, "
                                        f"γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    linnk = f"https://aflowlib.duke.edu/search/ui/material/?id=" + selected_entry.auid
                                    st.write("**Link:**", linnk)

                                    if st.button("Add Selected Structure (AFLOW)", key="add_btn_aflow"):
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        cif_file = io.BytesIO(cif_content)
                                        cif_file.name = f"{selected_entry.compound}_{selected_entry.auid}.cif"

                                        st.session_state.full_structures[cif_file.name] = structure_from_aflow

                                        check_structure_size_and_warn(structure_from_aflow, cif_file.name)
                                        if all(f.name != cif_file.name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)
                                        st.success("Structure added from AFLOW!")

                                    st.download_button(
                                        label="Download AFLOW CIF",
                                        data=cif_content,
                                        file_name=f"{selected_entry.compound}_{selected_entry.auid}.cif",
                                        type="primary",
                                        mime="chemical/x-cif"
                                    )
                                    st.info(
                                        f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                                else:
                                    st.warning("No CIF file found for this AFLOW entry.")
                        tab_index += 1

                    # COD tab
                    if 'cod_options' in st.session_state and st.session_state.cod_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in COD")
                            selected_cod_structure = st.selectbox(
                                "Select a structure from COD:",
                                st.session_state.cod_options,
                                key='sidebar_select_cod'
                            )
                            cod_id = selected_cod_structure.split(",")[-1].strip()
                            if cod_id in st.session_state.full_structures_see_cod:
                                selected_entry = st.session_state.full_structures_see_cod[cod_id]
                                lattice = selected_entry.lattice
                                cell_volume = selected_entry.lattice.volume
                                density = str(selected_entry.density).split()[0]
                                n_atoms = len(selected_entry)
                                atomic_den = n_atoms / cell_volume

                                idcodd = cod_id.removeprefix("cod_")

                                structure_type = identify_structure_type(selected_entry)
                                st.write(f"**Structure type:** {structure_type}")
                                analyzer = SpacegroupAnalyzer(selected_entry)
                                st.write(
                                    f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                st.write(
                                    f"**COD ID:** {idcodd}, **Formula:** {selected_entry.composition.reduced_formula}, **N. of Atoms:** {n_atoms}")
                                st.write(
                                    f"**Conventional Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}° (Volume {cell_volume:.1f} Å³)")
                                st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                cod_url = f"https://www.crystallography.net/cod/{cod_id.split('_')[1]}.html"
                                st.write(f"**Link:** {cod_url}")

                                file_name = f"{selected_entry.composition.reduced_formula}_COD_{cod_id.split('_')[1]}.cif"

                                if st.button("Add Selected Structure (COD)", key="sid_add_btn_cod"):
                                    cif_writer = CifWriter(selected_entry, symprec=0.01)
                                    cif_data = str(cif_writer)
                                    st.session_state.full_structures[file_name] = selected_entry
                                    cif_file = io.BytesIO(cif_data.encode('utf-8'))
                                    cif_file.name = file_name
                                    if 'uploaded_files' not in st.session_state:
                                        st.session_state.uploaded_files = []
                                    if all(f.name != file_name for f in st.session_state.uploaded_files):
                                        st.session_state.uploaded_files.append(cif_file)

                                    check_structure_size_and_warn(selected_entry, file_name)
                                    st.success("Structure added from COD!")

                                st.download_button(
                                    label="Download COD CIF",
                                    data=str(CifWriter(selected_entry, symprec=0.01)),
                                    file_name=file_name,
                                    mime="chemical/x-cif", type="primary",
                                )
                                st.info(
                                    f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                        tab_index += 1
                    # MC3D tab
                    if 'mc3d_options' in st.session_state and st.session_state.mc3d_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in MC3D")
                            st.info("ℹ️ MC3D structures accessed via OPTIMADE API from Materials Cloud.")

                            selected_structure = st.selectbox("Select a structure from MC3D:",
                                                              st.session_state.mc3d_options,
                                                              key='sidebar_select_mc3d')
                            mc3d_id = selected_structure.split(",")[-1].strip()

                            if mc3d_id in st.session_state.mc3d_structures:
                                selected_entry = st.session_state.mc3d_structures[mc3d_id]

                                lattice = selected_entry.lattice
                                cell_volume = lattice.volume
                                density = str(selected_entry.density).split()[0]
                                n_atoms = len(selected_entry)
                                atomic_den = n_atoms / cell_volume

                                structure_type = identify_structure_type(selected_entry)
                                st.write(f"**Structure type:** {structure_type}")

                                analyzer = SpacegroupAnalyzer(selected_entry)
                                st.write(
                                    f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                composition = selected_entry.composition.reduced_formula
                                st.write(
                                    f"**MC3D ID:** {mc3d_id}, **Formula:** {composition}, **N. of Atoms:** {n_atoms}")
                                st.write(
                                    f"**Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, "
                                    f"α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}° "
                                    f"(Volume {cell_volume:.1f} Å³)")
                                st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                #mc3d_url = f"https://mc3d.materialscloud.org/#/details/{mc3d_id}/pbe-v1"
                                mc3d_url = f"https://mc3d.materialscloud.org/#/details/{mc3d_id}/pbesol-v2"
                                st.write(f"**Link:** [View on Materials Cloud]({mc3d_url})")

                                file_name = f"{mc3d_id}_{composition}.cif"
                                file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

                                col_mc3d1, col_mc3d2 = st.columns([1, 1])
                                with col_mc3d1:
                                    if st.button("Add Selected Structure (MC3D)", key="add_btn_mc3d"):
                                        cif_writer = CifWriter(selected_entry, symprec=0.01)
                                        cif_data = str(cif_writer)
                                        st.session_state.full_structures[file_name] = selected_entry

                                        cif_file = io.BytesIO(cif_data.encode('utf-8'))
                                        cif_file.name = file_name

                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        if all(f.name != file_name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)

                                        check_structure_size_and_warn(selected_entry, file_name)
                                        st.success("Structure added from MC3D!")

                                with col_mc3d2:
                                    st.download_button(
                                        label="💾 Download MC3D CIF",
                                        data=str(CifWriter(selected_entry, symprec=0.01)),
                                        file_name=file_name,
                                        mime="chemical/x-cif",
                                        type="primary"
                                    )

                        tab_index += 1


def validate_atom_dataframe(df):
    required_columns = ["Element", "Frac X", "Frac Y", "Frac Z", "Occupancy"]

    for col in required_columns:
        if col not in df.columns:
            return False, f"Missing required column: {col}"

    for col in required_columns:
        if df[col].isna().any() or (df[col] == "").any():
            rows_with_empty = df[df[col].isna() | (df[col] == "")].index.tolist()
            return False, f"Empty values in '{col}' column (rows {rows_with_empty}). Please complete all fields."

    for col in ["Frac X", "Frac Y", "Frac Z", "Occupancy"]:
        try:
            df[col].astype(float)
        except (ValueError, TypeError):
            return False, f"Invalid numeric values in '{col}' column. Please enter valid numbers."

    for idx, elem in enumerate(df["Element"]):
        if not isinstance(elem, str) or not elem.strip():
            return False, f"Row {idx + 1}: Element cannot be empty"

    return True, ""


if uploaded_files_user_sidebar:
    uploaded_files = st.session_state['uploaded_files'] + uploaded_files_user_sidebar
    if 'full_structures' not in st.session_state:
        st.session_state.full_structures = {}
    for file in uploaded_files_user_sidebar:
        try:
            structure = load_structure(file)
            st.session_state['full_structures'][file.name] = structure
            check_structure_size_and_warn(structure, file.name)
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")
else:
    uploaded_files = st.session_state['uploaded_files']

if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")

if uploaded_files:
    species_set = set()
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        structure = load_structure(file)
        for atom in structure:
            if atom.is_ordered:
                species_set.add(atom.specie.symbol)
            else:
                for sp in atom.species:
                    species_set.add(sp.symbol)
    species_list = sorted(species_set)
else:
    species_list = []

if "current_structure" not in st.session_state:
    st.session_state["current_structure"] = None

if "original_structures" not in st.session_state:
    st.session_state["original_structures"] = {}

if "base_modified_structure" not in st.session_state:
    st.session_state["base_modified_structure"] = None

if "new_symmetry" not in st.session_state:
    st.session_state["new_symmetry"] = None


def has_partial_occupancies(structure):
    for site in structure:
        if not site.is_ordered:
            return True
    return False


def recalc_computed_columns(df, lattice):
    import numpy as np

    df = df.copy()

    if "No" in df.columns:
        df = df.drop(columns=["No"])
    df.insert(0, "No", range(1, len(df) + 1))

    for col in ["Frac X", "Frac Y", "Frac Z", "X", "Y", "Z"]:
        if col not in df.columns:
            df[col] = 0.0
    try:
        frac_coords = df[["Frac X", "Frac Y", "Frac Z"]].astype(float).values
        cart_coords = lattice.get_cartesian_coords(frac_coords)
        df["X"] = np.round(cart_coords[:, 0], 3)
        df["Y"] = np.round(cart_coords[:, 1], 3)
        df["Z"] = np.round(cart_coords[:, 2], 3)
        df["Occupancy"] = df["Occupancy"].astype(float)
    except Exception as e:
        st.warning(f"Could not recalculate Cartesian coordinates: {e}")

    for col in ["Elements", "Atom", "Wyckoff", "Occupancy"]:
        if col not in df.columns:
            df[col] = ""

    return df


if "xrd_download_prepared" not in st.session_state:
    st.session_state.xrd_download_prepared = False


def auto_save_structure_function(auto_save_filename, visual_pmg_structure):
    try:
        grouped_data = st.session_state.modified_atom_df.copy()
        grouped_data[['Frac X', 'Frac Y', 'Frac Z']] = grouped_data[['Frac X', 'Frac Y', 'Frac Z']].round(5)
        position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])
        new_struct = Structure(visual_pmg_structure.lattice, [], [])

        for (x, y, z), group in position_groups:
            position = (float(x), float(y), float(z))
            species_dict = {}
            for _, row in group.iterrows():
                species_dict[row['Element']] = float(row['Occupancy'])

            props = {"wyckoff": group.iloc[0]["Wyckoff"]}
            new_struct.append(species=species_dict, coords=position,
                              coords_are_cartesian=False, properties=props)

        st.session_state.auto_saved_structure = new_struct.copy()
        cif_writer = CifWriter(new_struct, symprec=0.1, write_site_properties=True)
        cif_content = cif_writer.__str__()

        cif_file = io.BytesIO(cif_content.encode('utf-8'))
        cif_file.name = auto_save_filename

        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []

        st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if
                                           f.name != auto_save_filename]
        st.session_state.uploaded_files.append(cif_file)

        if "final_structures" not in st.session_state:
            st.session_state.final_structures = {}

        file_key = auto_save_filename.replace(".cif", "")
        st.session_state.final_structures[file_key] = new_struct
        st.session_state["original_structures"][file_key] = new_struct

        st.success(f"Structure automatically saved as '{auto_save_filename}'.")
        return True


    except Exception as e:
        st.error(f"Auto-saving failed: {e}")
        return False


if "removal_message" not in st.session_state:
    st.session_state.removal_message = ""
if not isinstance(st.session_state.removal_message, str):
    st.session_state.removal_message = str(st.session_state.removal_message)

if "modified_defects" not in st.session_state:
    st.session_state["modified_defects"] = {}

if "expander_atomic_sites" not in st.session_state:
    st.session_state["expander_atomic_sites"] = False
if "expander_lattice" not in st.session_state:
    st.session_state["expander_lattice"] = False

if "auto_saved_structure" not in st.session_state:
    st.session_state["auto_saved_structure"] = None
if "expander_supercell" not in st.session_state:
    st.session_state["expander_supercell"] = False
if "expander_defects" not in st.session_state:
    st.session_state["expander_defects"] = False


def generate_initial_df_with_occupancy_and_wyckoff(structure: Structure):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        wyckoffs = sga.get_symmetry_dataset().wyckoffs
    except Exception as e:
        wyckoffs = ["-"] * len(structure.sites)

    initial_data = []
    row_index = 1
    element_counts = {}

    for i, site in enumerate(structure.sites):
        frac = site.frac_coords
        cart = structure.lattice.get_cartesian_coords(frac)
        for sp, occ in site.species.items():
            element = sp.symbol
            if element not in element_counts:
                element_counts[element] = 1
            else:
                element_counts[element] += 1
            element_indexed = f"{element}{element_counts[element]}"

            row = {
                "No": row_index,
                "Element": element,
                "Element_Index": element_indexed,
                "Occupancy": round(occ, 3),
                "Frac X": round(frac[0], 3),
                "Frac Y": round(frac[1], 3),
                "Frac Z": round(frac[2], 3),
                "X": round(cart[0], 3),
                "Y": round(cart[1], 3),
                "Z": round(cart[2], 3),
                "Wyckoff": wyckoffs[i],
            }

            initial_data.append(row)
            row_index += 1
    df = pd.DataFrame(initial_data)

    return df


if "run_before" not in st.session_state:
    st.session_state["run_before"] = False

if "🔬 Structure Modification" in calc_mode:
    auto_save_structure = False
    auto_save_filename = False
    show_structure = True
    # st.info("First, upload your crystal structures or add them from online databases. ")
    if uploaded_files:
        if "helpful" not in st.session_state:
            st.session_state["helpful"] = False
        tab01, tab02, tab03 = st.tabs(["🔬 Structure visualization", "🖥️ Atomic sites", "🔧 Lattice parameters"])
        with tab01:
            if show_structure:

                col_viz, col_mod, col_download = st.columns(3)
                if "current_structure" not in st.session_state:
                    st.session_state["current_structure"] = None

                # FOR COMPARISON IF SELECTED FILE CHANGED
                if "selected_file" not in st.session_state:
                    st.session_state["selected_file"] = None
                prev_selected_file = st.session_state.get("selected_file")
                with col_viz:
                    file_options = [file.name for file in uploaded_files]
                    st.subheader("Select Structure for Interactive Visualization:")
                    if len(file_options) > 1:
                        selected_file = st.selectbox("Select file", file_options, label_visibility="collapsed")
                    else:
                        selected_file = st.radio("Select file", file_options, label_visibility="collapsed")
                with col_mod:
                    cell_convert_or = False
                    st.info(
                        "To convert between different cell representations, please use [this XRDlicious submodule](https://xrdlicious-point-defects.streamlit.app/)")

                if selected_file != st.session_state["selected_file"]:

                    # IF SELECTED FILE CHANGED, RESETTING ALL MODIFICATIONS
                    st.session_state["current_structure"] = None
                    st.session_state["selected_file"] = selected_file
                    try:
                        mp_struct = load_structure(selected_file)
                    except Exception as e:
                        structure = read(selected_file)
                        mp_struct = AseAtomsAdaptor.get_structure(structure)
                    st.session_state["current_structure"] = mp_struct
                    st.session_state["original_structures"][selected_file] = mp_struct.copy()
                    st.session_state["original_for_supercell"] = mp_struct

                    st.session_state["auto_saved_structure"] = mp_struct.copy()
                    st.session_state["supercell_n_a"] = 1
                    st.session_state["supercell_n_b"] = 1
                    st.session_state["supercell_n_c"] = 1
                else:
                    mp_struct = st.session_state["current_structure"]
                    visual_pmg_structure = mp_struct.copy()

                selected_file = st.session_state.get("selected_file")
                original_structures = st.session_state["original_structures"]

            if "selected_file" not in st.session_state or st.session_state.selected_file != selected_file:
                st.session_state.selected_file = selected_file

            color_map = jmol_colors

            visual_pmg_structure = mp_struct.copy()

            for i, site in enumerate(mp_struct.sites):
                frac = site.frac_coords
                cart = mp_struct.lattice.get_cartesian_coords(frac)

            structure_type = identify_structure_type(visual_pmg_structure)

            composition = visual_pmg_structure.composition
            formula = composition.reduced_formula
            full_formula = composition.formula
            element_counts = composition.get_el_amt_dict()

            composition_str = " ".join([f"{el}{count:.2f}" if count % 1 != 0 else f"{el}{int(count)}"
                                        for el, count in element_counts.items()])
            st.subheader(f"{composition_str}, {structure_type}    ⬅️ Selected structure")

            st.markdown(
                'To create **Supercell** and **Point Defects**, please visit [this site](https://xrdlicious-point-defects.streamlit.app/).'
            )

            st.session_state["current_structure"] = mp_struct
            visual_pmg_structure = mp_struct

            st.session_state.modified_atom_df = generate_initial_df_with_occupancy_and_wyckoff(mp_struct)
            col_g1, col_g2 = st.columns([1, 3.2])

            with col_g1:
                show_plot_str = st.checkbox(f"Show 3D structure plot", value=True)
                if show_plot_str:
                    viz_type = st.radio(
                        "Choose visualization type:",
                        options=["Plotly", "py3Dmol (Molecular viewer)"],
                        index=1,
                        horizontal=True,
                        help="Choose between Plotly's interactive 3D plotting or py3Dmol's molecular visualization"
                    )
                unique_wyckoff_only = False
                if show_plot_str and viz_type == "py3Dmol (Molecular viewer)":
                    show_lattice_vectors = st.checkbox(
                        "🔴🟢🔵 Show lattice vectors and unit cell",
                        value=True,
                        help="Show lattice vectors and unit cell box",
                        key="show_lattice_vectors_main"
                    )

                    use_orthographic = st.checkbox(
                        "📐 Use orthographic projection (remove perspective)",
                        value=False,
                        key="use_orthographic_main")
                else:
                    show_lattice_vectors = True
                    use_orthographic = False

                hkl_result = None

            full_df = st.session_state.modified_atom_df.copy()
            with col_g1:
                if show_plot_str and viz_type == "py3Dmol (Molecular viewer)":
                    hkl_result = add_hkl_plane_controls(key_suffix="main_viz")
            if unique_wyckoff_only:
                grouped = full_df.groupby(['Wyckoff', 'Element']).size().reset_index(name='count')

                unique_indices = []
                for _, row in grouped.iterrows():
                    wyckoff = row['Wyckoff']
                    element = row['Element']
                    count = row['count']

                    idx = full_df[(full_df['Wyckoff'] == wyckoff) & (full_df['Element'] == element)].index[0]
                    unique_indices.append(idx)

                display_df = full_df.loc[unique_indices].copy()
                for i, row in display_df.iterrows():
                    wyckoff = row['Wyckoff']
                    element = row['Element']
                    count = grouped[(grouped['Wyckoff'] == wyckoff) & (grouped['Element'] == element)]['count'].values[
                        0]
                    if count > 1:
                        display_df.at[i, 'Wyckoff'] = f"{count}{wyckoff}"
            else:
                display_df = full_df

            if unique_wyckoff_only:
                st.info(
                    "ℹ️ When editing atoms in asymmetric unit view, changes will be propagated to all symmetrically equivalent atoms with the same Wyckoff position.")
        with tab02:
            editor_key = "atom_editor_unique" if unique_wyckoff_only else "atom_editor_full"
            with st.expander("Modify atomic sites", icon='⚛️',
                             expanded=True):  # expanded=st.session_state["expander_atomic_sites"]
                st.session_state["expander_open"] = True
                edited_df = st.data_editor(
                    display_df,
                    num_rows="dynamic",
                    key=editor_key,
                    column_config={
                        "Occupancy": st.column_config.NumberColumn(
                            "Occupancy",
                            min_value=0.001,
                            max_value=1.000,
                            step=0.001,
                            format="%.3f",  # ensures decimals
                        ),
                        "Frac X": st.column_config.NumberColumn(format="%.5f"),
                        "Frac Y": st.column_config.NumberColumn(format="%.5f"),
                        "Frac Z": st.column_config.NumberColumn(format="%.5f"),
                    }
                )

                if 'previous_atom_df' not in st.session_state:
                    st.session_state.previous_atom_df = st.session_state.modified_atom_df.copy()
                if not unique_wyckoff_only:
                    st.session_state.df_last_before_wyck = edited_df

                if not edited_df.equals(st.session_state.previous_atom_df) and unique_wyckoff_only == False:
                    # st.session_state.modified_atom_df = edited_df.copy()
                    pass
                    if auto_save_structure:
                        auto_save_structure_function(auto_save_filename, visual_pmg_structure)
                    st.session_state.previous_atom_df = edited_df.copy()

                if 'modified_atom_df_help' not in st.session_state:
                    pass
                else:
                    display_df = st.session_state.modified_atom_df_help

                edited_df_reset = edited_df.reset_index(drop=True)
                display_df_reset = display_df.reset_index(drop=True)

                if not edited_df_reset.equals(display_df_reset):  # and allow_atomic_mod:
                    edited_df = edited_df.reset_index(drop=True)
                    display_df = display_df.reset_index(drop=True)
                    st.session_state.modified_atom_df_help = edited_df
                    st.session_state["run_before"] = True

                    if unique_wyckoff_only:
                        full_df_copy = full_df.copy()

                        for i, row in edited_df.iterrows():
                            original_row = st.session_state.df_last_before_wyck.iloc[i].copy()
                            display_wyckoff = row['Wyckoff']
                            match = re.match(r'\d*(\D+)', display_wyckoff)
                            if match:
                                wyckoff_letter = match.group(1)
                            else:
                                wyckoff_letter = display_wyckoff

                            element = row['Element']
                            original_element = original_row['Element']
                            changed_props = {}
                            for col in ['Element', 'Frac X', 'Frac Y', 'Frac Z', 'Occupancy']:
                                if col in row and col in original_row and row[col] != original_row[col]:
                                    changed_props[col] = row[col]
                            if not changed_props:
                                continue
                            wyckoff_mask = full_df_copy['Wyckoff'].str.endswith(wyckoff_letter)
                            if 'Element' in changed_props:
                                # Find all atoms with same Wyckoff letter and original element
                                element_mask = wyckoff_mask & (full_df_copy['Element'] == original_element)

                                if element_mask.sum() > 0:
                                    full_df_copy.loc[element_mask, 'Element'] = changed_props['Element']
                                    update_element_indices(full_df_copy)
                                    element = changed_props['Element']

                                    # st.info(
                                    #    f"Updated {element_mask.sum()} atoms at Wyckoff positions with '{wyckoff_letter}': Element changed from {original_element} to {element}")

                            if 'Occupancy' in changed_props:
                                occ_mask = wyckoff_mask & (full_df_copy['Element'] == element)

                                if occ_mask.sum() > 0:
                                    full_df_copy.loc[occ_mask, 'Occupancy'] = changed_props['Occupancy']

                                    # st.info(
                                    #    f"Updated {occ_mask.sum()} atoms at Wyckoff positions with '{wyckoff_letter}': Occupancy changed to {changed_props['Occupancy']}")
                            position_changed = any(col in changed_props for col in ['Frac X', 'Frac Y', 'Frac Z'])

                            if position_changed:

                                x_orig = original_row['Frac X']
                                y_orig = original_row['Frac Y']
                                z_orig = original_row['Frac Z']
                                coord_mask = (
                                        (abs(full_df_copy['Frac X'] - x_orig) < 1e-5) &
                                        (abs(full_df_copy['Frac Y'] - y_orig) < 1e-5) &
                                        (abs(full_df_copy['Frac Z'] - z_orig) < 1e-5)
                                )

                                exact_match = coord_mask & (full_df_copy['Element'] == original_element) & wyckoff_mask

                                if exact_match.sum() >= 1:
                                    match_idx = full_df_copy[exact_match].index[0]
                                    if 'Frac X' in changed_props:
                                        full_df_copy.at[match_idx, 'Frac X'] = changed_props['Frac X']
                                    if 'Frac Y' in changed_props:
                                        full_df_copy.at[match_idx, 'Frac Y'] = changed_props['Frac Y']
                                    if 'Frac Z' in changed_props:
                                        full_df_copy.at[match_idx, 'Frac Z'] = changed_props['Frac Z']

                                    # st.info(f"Position for {element} at Wyckoff position with '{wyckoff_letter}' was updated.")
                                    # st.warning("Note: Changing atomic positions may break the crystal symmetry.")
                                else:
                                    pass
                                    # st.error(
                                    #    f"Could not find exact matching atom to update position. Found {exact_match.sum()} matches.")

                        st.session_state.modified_atom_df = recalc_computed_columns(full_df_copy,
                                                                                    visual_pmg_structure.lattice)
                        df_plot = full_df_copy.copy()
                        try:
                            grouped_data = st.session_state.modified_atom_df.copy()
                            st.session_state.df_last_before_wyck = grouped_data
                            grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                            grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                            grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                            position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])

                            new_struct = Structure(visual_pmg_structure.lattice, [], [])

                            for (x, y, z), group in position_groups:
                                position = (float(x), float(y), float(z))
                                species_dict = {}

                                for _, row in group.iterrows():
                                    element = row['Element']
                                    occupancy = float(row['Occupancy'])

                                    if element in species_dict:
                                        species_dict[element] += occupancy
                                    else:
                                        species_dict[element] = occupancy

                                props = {}
                                if "Wyckoff" in group.columns:
                                    props["wyckoff"] = group.iloc[0]["Wyckoff"]

                                new_struct.append(
                                    species=species_dict,
                                    coords=position,
                                    coords_are_cartesian=False,
                                    properties=props
                                )

                            visual_pmg_structure = new_struct

                            mp_struct = new_struct

                            st.session_state["current_structure"] = mp_struct
                            # st.session_state["original_for_supercell"] = mp_struct
                            st.session_state["supercell_n_a"] = 1
                            st.session_state["supercell_n_b"] = 1
                            st.session_state["supercell_n_c"] = 1

                            st.success("Structure rebuilt from the modified atomic positions!")
                            st.session_state["run_before"] = True
                        except Exception as e:
                            st.error(f"Error rebuilding structure: {e}")
                    else:
                        st.session_state.modified_atom_df = recalc_computed_columns(edited_df.copy(),
                                                                                    visual_pmg_structure.lattice)
                        df_plot = edited_df.copy()

                        try:
                            grouped_data = st.session_state.modified_atom_df.copy()
                            st.session_state.df_last_before_wyck = grouped_data
                            grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                            grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                            grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                            position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])

                            new_struct = Structure(visual_pmg_structure.lattice, [], [])

                            for (x, y, z), group in position_groups:
                                position = (float(x), float(y), float(z))
                                species_dict = {}

                                for _, row in group.iterrows():
                                    element = row['Element']
                                    occupancy = float(row['Occupancy'])

                                    if element in species_dict:
                                        species_dict[element] += occupancy
                                    else:
                                        species_dict[element] = occupancy

                                props = {}
                                if "Wyckoff" in group.columns:
                                    props["wyckoff"] = group.iloc[0]["Wyckoff"]

                                new_struct.append(
                                    species=species_dict,
                                    coords=position,
                                    coords_are_cartesian=False,
                                    properties=props
                                )

                            visual_pmg_structure = new_struct

                            mp_struct = new_struct

                            st.session_state["current_structure"] = mp_struct
                            # st.session_state["original_for_supercell"] = mp_struct
                            st.session_state["supercell_n_a"] = 1
                            st.session_state["supercell_n_b"] = 1
                            st.session_state["supercell_n_c"] = 1

                            st.success("Structure rebuilt from the modified atomic positions!")
                            st.session_state["run_before"] = True
                        except Exception as e:
                            st.error(f"Error rebuilding structure: {e}")

                df_plot = edited_df

            if st.session_state["run_before"] == True:
                st.session_state["run_before"] = False
                st.rerun()
            with col_g1:
                show_atom_labels = st.checkbox(f"**Show** atom **labels** in 3D visualization", value=False,
                                               key='atom_labels')

        custom_filename = st.text_input("Enter a name for the modified structure file:", value="MODIFIED_STR")
        if not custom_filename.endswith(".cif"):
            custom_filename += ".cif"
        if st.button("Add Modified Structure to Calculator"):
            try:
                grouped_data = st.session_state.modified_atom_df.copy()
                grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])

                new_struct = Structure(visual_pmg_structure.lattice, [], [])

                for (x, y, z), group in position_groups:
                    position = (float(x), float(y), float(z))
                    species_dict = {}
                    for _, row in group.iterrows():
                        element = row['Element']
                        occupancy = float(row['Occupancy'])

                        if element in species_dict:
                            species_dict[element] += occupancy
                        else:
                            species_dict[element] = occupancy
                    props = {"wyckoff": group.iloc[0]["Wyckoff"]}

                    new_struct.append(
                        species=species_dict,
                        coords=position,
                        coords_are_cartesian=False,
                        properties=props
                    )

                cif_writer = CifWriter(new_struct, symprec=0.1, write_site_properties=True)
                cif_content = cif_writer.__str__()
                cif_file = io.BytesIO(cif_content.encode('utf-8'))
                cif_file.name = custom_filename

                if 'uploaded_files' not in st.session_state:
                    st.session_state.uploaded_files = []

                if any(f.name == custom_filename for f in st.session_state.uploaded_files):
                    st.warning(f"A file named '{custom_filename}' already exists. The new version will replace it.")

                    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if
                                                       f.name != custom_filename]

                st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if
                                                   f.name != custom_filename]
                if 'uploaded_files' in locals():
                    uploaded_files[:] = [f for f in uploaded_files if f.name != custom_filename]
                st.session_state.uploaded_files.append(cif_file)
                uploaded_files.append(cif_file)
                #               uploaded_files = st.session_state.uploaded_files
                if "final_structures" not in st.session_state:
                    st.session_state.final_structures = {}

                new_key = custom_filename.replace(".cif", "")
                st.session_state.final_structures[new_key] = new_struct

                st.success(f"Modified structure added as '{new_key}'!")
                # st.write("Final list of structures in calculator:")
                # st.write(list(st.session_state.final_structures.keys()))

                if "calc_xrd" not in st.session_state:
                    st.session_state.calc_xrd = False
                if "new_structure_added" not in st.session_state:
                    st.session_state.new_structure_added = False
                if "intensity_scale_option" not in st.session_state:
                    st.session_state.intensity_scale_option = "Normalized"

                st.session_state.new_structure_added = True
                st.session_state.calc_xrd = True

                # Store the new structure name for feedback
                st.session_state.new_structure_name = new_key
            except Exception as e:
                st.error(f"Error reconstructing structure: {e}")
                st.error(
                    f"You probably added some new atom which has the same fractional coordinates as already defined atom, but you did not modify their occupancies. If the atoms share the same atomic site, their total occupancy must be equal to 1.")

                # st.rerun()

        with tab03:
            modify_lattice = st.checkbox("Modify lattice parameters", value=False)
            if modify_lattice:

                if "lattice_a_input" not in st.session_state:
                    st.session_state["lattice_a_input"] = visual_pmg_structure.lattice.a
                if "lattice_b_input" not in st.session_state:
                    st.session_state["lattice_b_input"] = visual_pmg_structure.lattice.b
                if "lattice_c_input" not in st.session_state:
                    st.session_state["lattice_c_input"] = visual_pmg_structure.lattice.c
                if "lattice_alpha_input" not in st.session_state:
                    st.session_state["lattice_alpha_input"] = visual_pmg_structure.lattice.alpha
                if "lattice_beta_input" not in st.session_state:
                    st.session_state["lattice_beta_input"] = visual_pmg_structure.lattice.beta
                if "lattice_gamma_input" not in st.session_state:
                    st.session_state["lattice_gamma_input"] = visual_pmg_structure.lattice.gamma

                # Reset lattice parameters when file changes
                if selected_file != st.session_state.get("previous_selected_file_lattice"):
                    st.session_state["previous_selected_file_lattice"] = selected_file
                    st.session_state["lattice_a_input"] = visual_pmg_structure.lattice.a
                    st.session_state["lattice_b_input"] = visual_pmg_structure.lattice.b
                    st.session_state["lattice_c_input"] = visual_pmg_structure.lattice.c
                    st.session_state["lattice_alpha_input"] = visual_pmg_structure.lattice.alpha
                    st.session_state["lattice_beta_input"] = visual_pmg_structure.lattice.beta
                    st.session_state["lattice_gamma_input"] = visual_pmg_structure.lattice.gamma

                try:
                    sga = SpacegroupAnalyzer(visual_pmg_structure)
                    crystal_system = sga.get_crystal_system()
                    spg_symbol = sga.get_space_group_symbol()
                    spg_number = sga.get_space_group_number()
                    st.info(
                        f"Crystal system: **{crystal_system.upper()}** | Space group: **{spg_symbol} (#{spg_number})**")

                    override_symmetry = st.checkbox("Override symmetry constraints (allow editing all parameters)",
                                                    value=False)
                    if override_symmetry:
                        crystal_system = "triclinic"
                except Exception as e:
                    crystal_system = "unknown"
                    st.warning(f"Could not determine crystal system: {e}")
                    override_symmetry = st.checkbox("Override symmetry constraints (allow editing all parameters)",
                                                    value=False)
                    if override_symmetry:
                        crystal_system = "triclinic"

                params_info = {
                    "cubic": {
                        "modifiable": ["a"],
                        "info": "In cubic systems, only parameter 'a' can be modified (b=a, c=a, α=β=γ=90°)"
                    },
                    "tetragonal": {
                        "modifiable": ["a", "c"],
                        "info": "In tetragonal systems, only parameters 'a' and 'c' can be modified (b=a, α=β=γ=90°)"
                    },
                    "orthorhombic": {
                        "modifiable": ["a", "b", "c"],
                        "info": "In orthorhombic systems, you can modify 'a', 'b', and 'c' (α=β=γ=90°)"
                    },
                    "hexagonal": {
                        "modifiable": ["a", "c"],
                        "info": "In hexagonal systems, only parameters 'a' and 'c' can be modified (b=a, α=β=90°, γ=120°)"
                    },
                    "trigonal": {
                        "modifiable": ["a", "c", "alpha"],
                        "info": "In trigonal systems, parameters 'a', 'c', and 'α' can be modified (b=a, β=α, γ=120° or γ=α depending on the specific space group)"
                    },
                    "monoclinic": {
                        "modifiable": ["a", "b", "c", "beta"],
                        "info": "In monoclinic systems, parameters 'a', 'b', 'c', and 'β' can be modified (α=γ=90°)"
                    },
                    "triclinic": {
                        "modifiable": ["a", "b", "c", "alpha", "beta", "gamma"],
                        "info": "In triclinic systems, all parameters can be modified"
                    },
                    "unknown": {
                        "modifiable": ["a", "b", "c", "alpha", "beta", "gamma"],
                        "info": "All parameters can be modified (system unknown)"
                    }
                }

                st.markdown(params_info[crystal_system]["info"])
                modifiable = params_info[crystal_system]["modifiable"]

                col_a, col_b, col_c = st.columns(3)
                col_alpha, col_beta, col_gamma = st.columns(3)

                with col_a:
                    new_a = st.number_input("a (Å)",
                                            value=float(st.session_state["lattice_a_input"]),
                                            min_value=0.1,
                                            max_value=100.0,
                                            step=0.01,
                                            format="%.5f",
                                            key="lattice_a")

                with col_b:
                    if "b" in modifiable:
                        new_b = st.number_input("b (Å)",
                                                value=float(st.session_state["lattice_b_input"]),
                                                min_value=0.1,
                                                max_value=100.0,
                                                step=0.01,
                                                format="%.5f",
                                                key="lattice_b")
                    else:
                        if crystal_system in ["cubic", "tetragonal", "hexagonal", "trigonal"]:
                            st.text_input("b (Å) = a", value=f"{float(new_a):.5f}", disabled=True)
                            new_b = new_a
                        else:
                            st.text_input("b (Å)", value=f"{float(st.session_state['lattice_b_input']):.5f}",
                                          disabled=True)
                            new_b = st.session_state["lattice_b_input"]

                with col_c:
                    if "c" in modifiable:
                        new_c = st.number_input("c (Å)",
                                                value=float(st.session_state["lattice_c_input"]),
                                                min_value=0.1,
                                                max_value=100.0,
                                                step=0.01,
                                                format="%.5f",
                                                key="lattice_c")
                    else:
                        if crystal_system == "cubic":
                            st.text_input("c (Å) = a", value=f"{float(new_a):.5f}", disabled=True)
                            new_c = new_a
                        else:
                            st.text_input("c (Å)", value=f"{float(st.session_state['lattice_c_input']):.5f}",
                                          disabled=True)
                            new_c = st.session_state["lattice_c_input"]

                with col_alpha:
                    if "alpha" in modifiable:
                        new_alpha = st.number_input("α (°)",
                                                    value=float(st.session_state["lattice_alpha_input"]),
                                                    min_value=0.1,
                                                    max_value=179.9,
                                                    step=0.1,
                                                    format="%.5f",
                                                    key="lattice_alpha")
                    else:
                        if crystal_system in ["cubic", "tetragonal", "orthorhombic", "hexagonal", "monoclinic"]:
                            st.text_input("α (°)", value="90.00000", disabled=True)
                            new_alpha = 90.0
                        else:
                            st.text_input("α (°)", value=f"{float(st.session_state['lattice_alpha_input']):.5f}",
                                          disabled=True)
                            new_alpha = st.session_state["lattice_alpha_input"]

                with col_beta:
                    if "beta" in modifiable:
                        new_beta = st.number_input("β (°)",
                                                   value=float(st.session_state["lattice_beta_input"]),
                                                   min_value=0.1,
                                                   max_value=179.9,
                                                   step=0.1,
                                                   format="%.5f",
                                                   key="lattice_beta")
                    else:
                        if crystal_system in ["cubic", "tetragonal", "orthorhombic", "hexagonal"]:
                            st.text_input("β (°)", value="90.00000", disabled=True)
                            new_beta = 90.0
                        elif crystal_system == "trigonal" and "alpha" in modifiable:
                            st.text_input("β (°) = α", value=f"{float(new_alpha):.5f}", disabled=True)
                            new_beta = new_alpha
                        else:
                            st.text_input("β (°)", value=f"{float(st.session_state['lattice_beta_input']):.5f}",
                                          disabled=True)
                            new_beta = st.session_state["lattice_beta_input"]

                with col_gamma:
                    if "gamma" in modifiable:
                        new_gamma = st.number_input("γ (°)",
                                                    value=float(st.session_state["lattice_gamma_input"]),
                                                    min_value=0.1,
                                                    max_value=179.9,
                                                    step=0.1,
                                                    format="%.5f",
                                                    key="lattice_gamma")
                    else:
                        if crystal_system in ["cubic", "tetragonal", "orthorhombic", "monoclinic"]:
                            st.text_input("γ (°)", value="90.00000", disabled=True)
                            new_gamma = 90.0
                        elif crystal_system == "hexagonal":
                            st.text_input("γ (°)", value="120.00000", disabled=True)
                            new_gamma = 120.0
                        elif crystal_system == "trigonal" and spg_symbol.startswith("R"):
                            st.text_input("γ (°) = α", value=f"{float(new_alpha):.5f}", disabled=True)
                            new_gamma = new_alpha
                        else:
                            st.text_input("γ (°)", value=f"{float(st.session_state['lattice_gamma_input']):.5f}",
                                          disabled=True)
                            new_gamma = st.session_state["lattice_gamma_input"]

                if st.button("Apply Lattice Changes"):
                    try:
                        st.session_state["expander_lattice"] = True
                        current_selected_file = st.session_state.get("selected_file")

                        from pymatgen.core import Lattice

                        new_lattice = Lattice.from_parameters(
                            a=new_a,
                            b=new_b,
                            c=new_c,
                            alpha=new_alpha,
                            beta=new_beta,
                            gamma=new_gamma
                        )

                        frac_coords = [site.frac_coords for site in mp_struct.sites]
                        species = [site.species for site in mp_struct.sites]
                        props = [site.properties for site in mp_struct.sites]

                        from pymatgen.core import Structure

                        updated_structure = Structure(
                            lattice=new_lattice,
                            species=species,
                            coords=frac_coords,
                            coords_are_cartesian=False,
                            site_properties={k: [p.get(k, None) for p in props] for k in set().union(*props)}
                        )

                        mp_struct = updated_structure
                        visual_pmg_structure = updated_structure
                        st.session_state["current_structure"] = mp_struct

                        if current_selected_file and "original_structures" in st.session_state:
                            st.session_state["original_structures"][current_selected_file] = updated_structure

                        if "modified_atom_df" in st.session_state:
                            st.session_state.modified_atom_df = recalc_computed_columns(
                                st.session_state.modified_atom_df.copy(),
                                updated_structure.lattice
                            )

                        st.session_state["lattice_a_input"] = new_a
                        st.session_state["lattice_b_input"] = new_b
                        st.session_state["lattice_c_input"] = new_c
                        st.session_state["lattice_alpha_input"] = new_alpha
                        st.session_state["lattice_beta_input"] = new_beta
                        st.session_state["lattice_gamma_input"] = new_gamma

                        try:

                            lattice_modified_filename = custom_filename

                            cif_writer = CifWriter(updated_structure, symprec=0.1, write_site_properties=True)
                            cif_content = cif_writer.__str__()
                            cif_file = io.BytesIO(cif_content.encode('utf-8'))
                            cif_file.name = lattice_modified_filename

                            if 'uploaded_files' not in st.session_state:
                                st.session_state.uploaded_files = []

                            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if
                                                               f.name != lattice_modified_filename]

                            if 'uploaded_files' in locals():
                                uploaded_files[:] = [f for f in uploaded_files if f.name != lattice_modified_filename]
                                uploaded_files.append(cif_file)

                            st.session_state.uploaded_files.append(cif_file)

                            if "final_structures" not in st.session_state:
                                st.session_state.final_structures = {}

                            file_key = lattice_modified_filename.replace(".cif", "")
                            st.session_state.final_structures[file_key] = updated_structure

                            st.success(f"Lattice parameters updated! Structure saved as '{lattice_modified_filename}'")
                            st.info(
                                f"The modified structure is now available in the calculator with the name you specified.")

                        except Exception as e:
                            st.error(f"Error saving structure: {e}")
                            st.success("Lattice parameters updated successfully, but structure could not be saved.")

                    except Exception as e:
                        st.error(f"Error updating lattice parameters: {e}")

        df_plot = df_plot.copy()
        with tab01:
            with col_g1:
                base_atom_size = st.slider(
                    "Base atom size in visualization:",
                    min_value=1,
                    max_value=30,
                    value=10,
                    step=1,
                    help="Adjust the base size of atoms in the 3D visualization - size will adjust with zooming"
                )

            has_partial_occupancies = False
            for site in visual_pmg_structure:
                if not site.is_ordered:
                    has_partial_occupancies = True
                    break

            if has_partial_occupancies:
                st.info(
                    "This structure contains sites with partial occupancies. Combined labels will be shown for these sites.")

            if not show_atom_labels:

                atom_labels_dict = {}
            else:

                atom_labels_dict = {}

                processed_df = df_plot.copy()
                processed_df['X_round'] = processed_df['X'].round(3)
                processed_df['Y_round'] = processed_df['Y'].round(3)
                processed_df['Z_round'] = processed_df['Z'].round(3)

                coord_groups = processed_df.groupby(['X_round', 'Y_round', 'Z_round'])

                for (x, y, z), group in coord_groups:
                    position_key = (x, y, z)

                    if len(group) == 1 and abs(group['Occupancy'].values[0] - 1.0) < 0.01:
                        atom_labels_dict[position_key] = group['Element_Index'].values[0]
                        continue

                    total_occ = group['Occupancy'].sum()

                    vacancy = 1.0 - total_occ if total_occ < 0.99 else 0

                    label_parts = []

                    for _, row in group.iterrows():
                        element = row['Element']
                        occ = row['Occupancy']
                        if occ > 0.01:
                            label_parts.append(f"{element}{occ:.3f}")

                    if vacancy > 0.01:
                        label_parts.append(f"□{vacancy:.3f}")  # Square symbol for vacancy

                    atom_labels_dict[position_key] = "/".join(label_parts)

            atom_traces = []

            df_plot['X_round'] = df_plot['X'].round(3)
            df_plot['Y_round'] = df_plot['Y'].round(3)
            df_plot['Z_round'] = df_plot['Z'].round(3)

            # Group by coordinates to get dominant element at each position
            position_groups = df_plot.groupby(['X_round', 'Y_round', 'Z_round'])
            element_positions = {}
            element_labels = {}

            if show_plot_str:
                if viz_type == "Plotly":
                    df_for_viz = display_df if unique_wyckoff_only else df_plot

                    df_for_viz['X_round'] = df_for_viz['X'].round(3)
                    df_for_viz['Y_round'] = df_for_viz['Y'].round(3)
                    df_for_viz['Z_round'] = df_for_viz['Z'].round(3)

                    position_groups = df_for_viz.groupby(['X_round', 'Y_round', 'Z_round'])
                    element_positions = {}
                    element_labels = {}

                    for (x, y, z), group in position_groups:
                        position = (x, y, z)

                        if len(group) > 1:
                            max_row = group.loc[group['Occupancy'].idxmax()]
                            dominant_element = max_row['Element']
                        else:
                            dominant_element = group['Element'].iloc[0]

                        if dominant_element not in element_positions:
                            element_positions[dominant_element] = []
                            element_labels[dominant_element] = []

                        element_positions[dominant_element].append(position)

                        if show_atom_labels:
                            if 'Element_Index' in group.columns and len(group) == 1:
                                label = group['Element_Index'].iloc[0]
                            else:
                                label = f"{dominant_element}{len(element_positions[dominant_element])}"
                            element_labels[dominant_element].append(label)
                        else:
                            element_labels[dominant_element].append("")

                    for element, positions in element_positions.items():
                        if not positions:
                            continue

                        x_vals = [pos[0] for pos in positions]
                        y_vals = [pos[1] for pos in positions]
                        z_vals = [pos[2] for pos in positions]
                        labels = element_labels[element]

                        mode = 'markers+text' if show_atom_labels else 'markers'

                        trace = go.Scatter3d(
                            x=x_vals, y=y_vals, z=z_vals,
                            mode=mode,
                            marker=dict(
                                size=base_atom_size,
                                color=color_map.get(element, "gray"),
                                opacity=1,
                                sizemode='area',
                                sizeref=2.5,
                                sizemin=0.5,
                            ),
                            text=labels,
                            textposition="top center",
                            textfont=dict(
                                size=14,
                                color="black"
                            ),
                            name=element
                        )
                        atom_traces.append(trace)

                    cell = visual_pmg_structure.lattice.matrix
                    a, b, c = cell[0], cell[1], cell[2]
                    corners = []
                    for i in [0, 1]:
                        for j in [0, 1]:
                            for k in [0, 1]:
                                corner = i * a + j * b + k * c
                                corners.append(corner)

                    edges = []
                    for i in [0, 1]:
                        for j in [0, 1]:
                            for k in [0, 1]:
                                start_coord = np.array([i, j, k])
                                start_point = i * a + j * b + k * c
                                for axis in range(3):
                                    if start_coord[axis] == 0:
                                        neighbor = start_coord.copy()
                                        neighbor[axis] = 1
                                        end_point = neighbor[0] * a + neighbor[1] * b + neighbor[2] * c
                                        edges.append((start_point, end_point))

                    edge_x, edge_y, edge_z = [], [], []
                    for start, end in edges:
                        edge_x.extend([start[0], end[0], None])
                        edge_y.extend([start[1], end[1], None])
                        edge_z.extend([start[2], end[2], None])

                    edge_trace = go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z, opacity=0.8,
                        mode="lines",
                        line=dict(color="black", width=3),
                        name="Unit Cell"
                    )

                    arrow_trace = go.Cone(
                        x=[0, 0, 0],
                        y=[0, 0, 0],
                        z=[0, 0, 0],
                        u=[a[0], b[0], c[0]],
                        v=[a[1], b[1], c[1]],
                        w=[a[2], b[2], c[2]],
                        anchor="tail",
                        colorscale=[[0, "black"], [1, "black"]],
                        showscale=False,
                        sizemode="absolute",
                        sizeref=0.3,
                        name="Lattice Vectors"
                    )

                    labels_x, labels_y, labels_z, vec_texts = [], [], [], []
                    for vec, label in zip([a, b, c],
                                          [f"a = {np.linalg.norm(a):.3f} Å",
                                           f"b = {np.linalg.norm(b):.3f} Å",
                                           f"c = {np.linalg.norm(c):.3f} Å"]):
                        norm = np.linalg.norm(vec)
                        pos = vec + (0.1 * vec / (norm + 1e-6))
                        labels_x.append(pos[0])
                        labels_y.append(pos[1])
                        labels_z.append(pos[2])
                        vec_texts.append(label)

                    label_trace = go.Scatter3d(
                        x=labels_x, y=labels_y, z=labels_z,
                        mode="text",
                        text=vec_texts,
                        textposition="top center",
                        textfont=dict(
                            size=14,
                            color="black"
                        ),
                        showlegend=False
                    )

                    data = atom_traces + [edge_trace, label_trace]

                    layout = go.Layout(
                        scene=dict(
                            xaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showline=False,
                                visible=False,
                            ),
                            yaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showline=False,
                                visible=False,
                            ),
                            zaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showline=False,
                                visible=False,
                            ),
                            annotations=[],
                        ),
                        margin=dict(l=20, r=20, b=20, t=50),
                        legend=dict(
                            font=dict(
                                size=16
                            )
                        ),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                    )

                    fig = go.Figure(data=data, layout=layout)

                    fig.update_layout(
                        width=1000,
                        height=800,
                        shapes=[
                            dict(
                                type="rect",
                                xref="paper",
                                yref="paper",
                                x0=0,
                                y0=0,
                                x1=1,
                                y1=1,
                                line=dict(
                                    color="black",
                                    width=3,
                                ),
                                fillcolor="rgba(0,0,0,0)",
                            )
                        ]
                    )

                    fig.update_scenes(
                        aspectmode='data',
                        camera=dict(
                            eye=dict(x=1.5, y=1.2, z=1)
                        ),
                        dragmode='orbit'
                    )

                    with col_g2:
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    structure_for_viz = visual_pmg_structure
                    df_for_viz = display_df if unique_wyckoff_only else df_plot
                    supercell_x, supercell_y, supercell_z = 1, 1, 1

                    if hkl_result is not None:
                        # h, k, l, apply_orientation, supercell_x, supercell_y, supercell_z, show_lattice_vectors = hkl_result
                        h, k, l, apply_orientation, supercell_x, supercell_y, supercell_z = hkl_result

                    if supercell_x == 1 and supercell_y == 1 and supercell_z == 1:
                        xyz_lines = [str(len(df_for_viz))]
                        xyz_lines.append("py3Dmol visualization")
                        for _, row in df_for_viz.iterrows():
                            element = row['Element']
                            x, y, z = row['X'], row['Y'], row['Z']
                            xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
                        xyz_str = "\n".join(xyz_lines)

                    else:

                        xyz_str = create_supercell_xyz_for_visualization(
                            df_for_viz,
                            structure_for_viz.lattice.matrix,
                            supercell_x,
                            supercell_y,
                            supercell_z
                        )
                    with col_g2:
                        view = py3Dmol.view(width=1000, height=800)
                        view.addModel(xyz_str, "xyz")
                        view.setStyle({'model': 0}, {"sphere": {"radius": base_atom_size / 30, "colorscheme": "Jmol"}})

                        # if use_orthographic:
                        #    view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})
                        #    # Set orthographic camera
                        #    view.setView({
                        #        'fov': 0,  # Field of view = 0 means orthographic
                        #    })

                        cell_3dmol = structure_for_viz.lattice.matrix

                        if np.linalg.det(cell_3dmol) > 1e-6:
                            if show_lattice_vectors:
                                if supercell_x == 1 and supercell_y == 1 and supercell_z == 1:
                                    add_box(view, cell_3dmol, color='black', linewidth=2)
                                else:
                                    add_supercell_unit_cell_boxes(view, cell_3dmol, supercell_x, supercell_y,
                                                                  supercell_z)

                            if show_lattice_vectors:
                                a, b, c = cell_3dmol[0], cell_3dmol[1], cell_3dmol[2]

                                view.addArrow({
                                    'start': {'x': 0, 'y': 0, 'z': 0},
                                    'end': {'x': a[0], 'y': a[1], 'z': a[2]},
                                    'color': 'red',
                                    'radius': 0.1
                                })
                                view.addArrow({
                                    'start': {'x': 0, 'y': 0, 'z': 0},
                                    'end': {'x': b[0], 'y': b[1], 'z': b[2]},
                                    'color': 'green',
                                    'radius': 0.1
                                })
                                view.addArrow({
                                    'start': {'x': 0, 'y': 0, 'z': 0},
                                    'end': {'x': c[0], 'y': c[1], 'z': c[2]},
                                    'color': 'blue',
                                    'radius': 0.1
                                })
                                a_norm = np.linalg.norm(a)
                                b_norm = np.linalg.norm(b)
                                c_norm = np.linalg.norm(c)
                                view.addLabel(f"a = {a_norm:.3f} Å", {
                                    "position": {"x": a[0] * 1.1, "y": a[1] * 1.1, "z": a[2] * 1.1},
                                    "backgroundColor": "red",
                                    "fontColor": "white",
                                    "fontSize": 12
                                })
                                view.addLabel(f"b = {b_norm:.3f} Å", {
                                    "position": {"x": b[0] * 1.1, "y": b[1] * 1.1, "z": b[2] * 1.1},
                                    "backgroundColor": "green",
                                    "fontColor": "white",
                                    "fontSize": 12
                                })
                                view.addLabel(f"c = {c_norm:.3f} Å", {
                                    "position": {"x": c[0] * 1.1, "y": c[1] * 1.1, "z": c[2] * 1.1},
                                    "backgroundColor": "blue",
                                    "fontColor": "white",
                                    "fontSize": 12
                                })

                        total_atoms = len(df_for_viz) * supercell_x * supercell_y * supercell_z
                        if show_atom_labels and total_atoms <= 200:
                            if supercell_x == 1 and supercell_y == 1 and supercell_z == 1:
                                for i, row in df_for_viz.iterrows():
                                    element = row['Element']
                                    x, y, z = row['X'], row['Y'], row['Z']
                                    if 'Element_Index' in row:
                                        label = row['Element_Index']
                                    else:
                                        label = f"{element}{i + 1}"
                                    view.addLabel(label, {
                                        "position": {"x": x, "y": y, "z": z},
                                        "backgroundColor": "white",
                                        "fontColor": "black",
                                        "fontSize": 12,
                                        "borderThickness": 1,
                                        "borderColor": "grey"
                                    })
                            else:
                                atom_count = 0
                                for i in range(supercell_x):
                                    for j in range(supercell_y):
                                        for k in range(supercell_z):
                                            translation = i * cell_3dmol[0] + j * cell_3dmol[1] + k * cell_3dmol[2]
                                            for _, row in df_for_viz.iterrows():
                                                element = row['Element']
                                                original_pos = np.array([row['X'], row['Y'], row['Z']])
                                                new_pos = original_pos + translation
                                                if atom_count % 5 == 0:
                                                    view.addLabel(element, {
                                                        "position": {"x": new_pos[0], "y": new_pos[1], "z": new_pos[2]},
                                                        "backgroundColor": "white",
                                                        "fontColor": "black",
                                                        "fontSize": 10,
                                                        "borderThickness": 1,
                                                        "borderColor": "grey"
                                                    })
                                                atom_count += 1
                        elif show_atom_labels and total_atoms > 500:
                            st.warning("⚠️ Too many atoms for labeling. Labels disabled for performance.")
                        key_suffix = "main_viz"
                        stored_orientation_key = f"stored_orientation_{key_suffix}"
                        should_apply_orientation = False
                        orientation_h, orientation_k, orientation_l = 1, 0, 0
                        show_success_message = False
                        if hkl_result is not None:
                            h, k, l, apply_orientation, supercell_x, supercell_y, supercell_z = hkl_result
                            plane_info = get_hkl_plane_info(visual_pmg_structure.lattice.matrix, h, k, l)
                            if plane_info["success"]:
                                st.write(f"**d-spacing for ({h} {k} {l}) plane:** {plane_info['d_spacing']:.4f} Å")

                            if apply_orientation:
                                should_apply_orientation = True

                                orientation_h, orientation_k, orientation_l = h, k, l

                                show_success_message = True
                                st.session_state[stored_orientation_key] = {
                                    "active": True,
                                    "h": h,
                                    "k": k,
                                    "l": l
                                }

                        if stored_orientation_key in st.session_state and st.session_state[stored_orientation_key][
                            "active"]:
                            stored = st.session_state[stored_orientation_key]
                            should_apply_orientation = True
                            orientation_h = stored["h"]
                            orientation_k = stored["k"]
                            orientation_l = stored["l"]
                        if should_apply_orientation:
                            success, message = apply_hkl_orientation_to_py3dmol(
                                view,
                                visual_pmg_structure.lattice.matrix,
                                orientation_h, orientation_k, orientation_l,
                                supercell_x, supercell_y, supercell_z
                            )
                            if show_success_message:
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)

                        if use_orthographic:
                            view.setProjection('orthogonal')
                            view.setCameraParameters({'orthographic': True})
                            view.zoomTo()
                            # view.zoom(1.1)
                            # view.rotate(10, 'x')
                        else:
                            view.setProjection('perspective')
                            view.setCameraParameters({'orthographic': False})
                            view.zoomTo()
                            view.zoom(1.1)
                            view.rotate(10, 'x')

                        html_content = view._make_html()

                        st.components.v1.html(

                            f"<div style='display:flex;justify-content:center;border:2px solid #333;border-radius:10px;overflow:hidden;background-color:#f8f9fa;'>{html_content}</div>",
                            height=820
                        )
                        elements_in_viz = df_for_viz['Element'].unique()
                        elems_legend = sorted(list(elements_in_viz))
                        legend_items = [
                            f"<div style='margin-right:15px;display:flex;align-items:center;'>"
                            f"<div style='width:18px;height:18px;background-color:{color_map.get(e, '#CCCCCC')};margin-right:8px;border:2px solid black;border-radius:50%;'></div>"
                            f"<span style='font-weight:bold;font-size:14px;'>{e}</span></div>"
                            for e in elems_legend
                        ]
                        st.markdown(
                            f"<div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;margin-top:15px;padding:10px;background-color:#f0f2f6;border-radius:10px;'>{''.join(legend_items)}</div>",
                            unsafe_allow_html=True
                        )
                        st.info(
                            "🖱️ **py3Dmol Controls:** Left click + drag to rotate, scroll to zoom, middle click + drag to pan"
                        )

            lattice = visual_pmg_structure.lattice
            a_para = lattice.a
            b_para = lattice.b
            c_para = lattice.c
            alpha = lattice.alpha
            beta = lattice.beta
            gamma = lattice.gamma
            volume = lattice.volume

            density_g = str(visual_pmg_structure.density).split()[0]
            str_len = len(visual_pmg_structure)
            density_a = str_len / volume

            # Get lattice parameters

            lattice_str = (
                f"a = {a_para:.4f} Å<br>"
                f"b = {b_para:.4f} Å<br>"
                f"c = {c_para:.4f} Å<br>"
                f"α = {alpha:.2f}°<br>"
                f"β = {beta:.2f}°<br>"
                f"γ = {gamma:.2f}°<br>"
                f"Volume = {volume:.2f} Å³"
            )

            with col_g1:
                st.markdown(f"""
                 <div style='text-align: center; font-size: 18px;'>
                     <p><strong>Lattice Parameters:</strong><br>{lattice_str}</p>
                     <p><strong>Number of Atoms:</strong> {str_len}</p>
                     <p><strong>Density:</strong> {float(density_g):.2f} g/cm³ ({float(density_a):.4f} 1/Å³) </p>
                     <p><strong>Structure Type:</strong> {structure_type}</p>
                 </div>
                 """, unsafe_allow_html=True)

            with col_download:
                file_format = st.radio(
                    f"Select file **format**",
                    ("CIF", "VASP", "LAMMPS", "XYZ",),
                    horizontal=True
                )

                file_content = None
                download_file_name = None
                mime = "text/plain"

                try:
                    if file_format == "CIF":
                        from pymatgen.io.cif import CifWriter

                        download_file_name = selected_file.split('.')[
                                                 0] + f'.cif'

                        mime = "chemical/x-cif"
                        grouped_data = st.session_state.modified_atom_df.copy()
                        grouped_data = df_plot.copy()
                        grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                        grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                        grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                        # Group by position
                        position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])

                        new_struct = Structure(visual_pmg_structure.lattice, [], [])

                        for (x, y, z), group in position_groups:
                            position = (float(x), float(y), float(z))

                            species_dict = {}
                            for _, row in group.iterrows():
                                element = row['Element']
                                occupancy = float(row['Occupancy'])

                                if element in species_dict:
                                    species_dict[element] += occupancy
                                else:
                                    species_dict[element] = occupancy

                            props = {"wyckoff": group.iloc[0]["Wyckoff"]}

                            new_struct.append(
                                species=species_dict,
                                coords=position,
                                coords_are_cartesian=False,
                                properties=props
                            )

                        file_content = CifWriter(new_struct, symprec=0.1, write_site_properties=True).__str__()
                    elif file_format == "VASP":
                        from pymatgen.io.cif import CifWriter

                        mime = "chemical/x-cif"

                        grouped_data = st.session_state.modified_atom_df.copy()
                        grouped_data = df_plot.copy()

                        grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                        grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                        grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                        position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])

                        # Create a new structure with properly defined partial occupancies
                        new_struct = Structure(visual_pmg_structure.lattice, [], [])

                        for (x, y, z), group in position_groups:
                            position = (float(x), float(y), float(z))

                            species_dict = {}
                            for _, row in group.iterrows():
                                element = row['Element']
                            new_struct.append(
                                species=element,
                                coords=position,
                                coords_are_cartesian=False,
                            )

                        out = StringIO()
                        current_ase_structure = AseAtomsAdaptor.get_atoms(new_struct)

                        colsss, colyyy = st.columns([1, 1])
                        with colsss:
                            use_fractional = st.checkbox("Output POSCAR with fractional coordinates",
                                                         value=True,
                                                         key="poscar_fractional")

                        with colyyy:
                            from ase.constraints import FixAtoms

                            use_selective_dynamics = st.checkbox("Include Selective dynamics (all atoms free)",
                                                                 value=False, key="poscar_sd")
                            if use_selective_dynamics:
                                constraint = FixAtoms(indices=[])  # No atoms are fixed, so all will be T T T
                                current_ase_structure.set_constraint(constraint)
                        write(out, current_ase_structure, format="vasp", direct=use_fractional, sort=True)
                        file_content = out.getvalue()
                        download_file_name = selected_file.split('.')[
                                                 0] + f'.poscar'

                    elif file_format == "LAMMPS":
                        from pymatgen.io.cif import CifWriter

                        mime = "chemical/x-cif"

                        grouped_data = st.session_state.modified_atom_df.copy()
                        grouped_data = df_plot.copy()

                        grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                        grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                        grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                        position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])

                        # Create a new structure with properly defined partial occupancies
                        new_struct = Structure(visual_pmg_structure.lattice, [], [])

                        for (x, y, z), group in position_groups:
                            position = (float(x), float(y), float(z))
                            species_dict = {}
                            for _, row in group.iterrows():
                                element = row['Element']

                            new_struct.append(
                                species=element,
                                coords=position,
                                coords_are_cartesian=False,
                            )

                        st.markdown("**LAMMPS Export Options**")

                        atom_style = st.selectbox("Select atom_style", ["atomic", "charge", "full"], index=0)
                        units = st.selectbox("Select units", ["metal", "real", "si"], index=0)
                        include_masses = st.checkbox("Include atomic masses", value=True)
                        force_skew = st.checkbox("Force triclinic cell (skew)", value=False)
                        current_ase_structure = AseAtomsAdaptor.get_atoms(new_struct)
                        out = StringIO()
                        write(
                            out,
                            current_ase_structure,
                            format="lammps-data",
                            atom_style=atom_style,
                            units=units,
                            masses=include_masses,
                            force_skew=force_skew
                        )
                        file_content = out.getvalue()

                        download_file_name = selected_file.split('.')[
                                                 0] + f'_.lmp'


                    elif file_format == "XYZ":
                        mime = "chemical/x-xyz"
                        grouped_data = st.session_state.modified_atom_df.copy()
                        grouped_data = df_plot.copy()
                        grouped_data['Frac X'] = grouped_data['Frac X'].round(5)
                        grouped_data['Frac Y'] = grouped_data['Frac Y'].round(5)
                        grouped_data['Frac Z'] = grouped_data['Frac Z'].round(5)

                        position_groups = grouped_data.groupby(['Frac X', 'Frac Y', 'Frac Z'])
                        new_struct = Structure(visual_pmg_structure.lattice, [], [])
                        for (x, y, z), group in position_groups:
                            position = (float(x), float(y), float(z))
                            species_dict = {}
                            for _, row in group.iterrows():
                                element = row['Element']
                            new_struct.append(
                                species=element,
                                coords=position,
                                coords_are_cartesian=False,
                            )
                        lattice_vectors = new_struct.lattice.matrix
                        cart_coords = []
                        elements = []
                        for site in new_struct:
                            cart_coords.append(new_struct.lattice.get_cartesian_coords(site.frac_coords))
                            elements.append(site.specie.symbol)
                        xyz_lines = []
                        xyz_lines.append(str(len(new_struct)))

                        lattice_string = " ".join([f"{x:.6f}" for row in lattice_vectors for x in row])
                        properties = "Properties=species:S:1:pos:R:3"
                        comment_line = f'Lattice="{lattice_string}" {properties}'

                        xyz_lines.append(comment_line)

                        for element, coord in zip(elements, cart_coords):
                            line = f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}"
                            xyz_lines.append(line)

                        file_content = "\n".join(xyz_lines)
                        download_file_name = selected_file.split('.')[0] + f'_.xyz'

                except Exception as e:
                    st.error(f"Error generating {file_format} file: {e}")
                    st.error(
                        f"You probably added some new atom which has the same fractional coordinates as already defined atom, but you did not modify their occupancies. If the atoms share the same atomic site, their total occupancy must be equal to 1.")

                if file_content is not None:
                    st.download_button(
                        label=f"Download {file_format} file",
                        data=file_content,
                        file_name=download_file_name,
                        type="primary",
                        mime=mime
                    )

# --- Diffraction Settings and Calculation ---


if mode == "Basic":
    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("""
            <hr style="height:3px;border:none;color:#333;background-color:#333;" />
            """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 24px;'>
        🪧 <strong>Step 3 / 4:</strong> Configure Settings for the Calculation of Diffraction Patterns or (P)RDF and Press 'Calculate XRD / ND'  or 'Calculate RDF' Button: 
        <br><span style="font-size: 28px;">⬇️</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
            <hr style="height:3px;border:none;color:#333;background-color:#333;" />
            """, unsafe_allow_html=True)

current_user_file_names = set()
if uploaded_files_user_sidebar:
    current_user_file_names = {f.name for f in uploaded_files_user_sidebar}

if 'files_marked_for_removal' not in st.session_state:
    st.session_state.files_marked_for_removal = set()

removable_files = [f for f in uploaded_files
                   if f.name not in current_user_file_names
                   and f.name not in st.session_state.files_marked_for_removal]

with st.sidebar.expander("🗑️ Remove database/modified structure(s)", expanded=False):
    if removable_files:
        st.write(f"**{len(removable_files)} removable file(s):**")

        for i, file in enumerate(removable_files):
            col1, col2 = st.columns([4, 1])
            col1.write(file.name)

            if col2.button("❌", key=f"remove_db_{i}"):
                st.session_state.files_marked_for_removal.add(file.name)

                st.session_state['uploaded_files'] = [f for f in st.session_state['uploaded_files']
                                                      if f.name != file.name]

                uploaded_files[:] = [f for f in uploaded_files if f.name != file.name]

                st.success(f"✅ Removed: {file.name}")

        remaining_count = len([f for f in uploaded_files
                               if f.name not in current_user_file_names
                               and f.name not in st.session_state.files_marked_for_removal])

        if remaining_count != len(removable_files):
            st.info(f"📊 {remaining_count} files remaining to remove")

    else:
        removed_count = len(st.session_state.files_marked_for_removal)
        if removed_count > 0:
            st.success(f"✅ All database/modified structures removed ({removed_count} total)")
        else:
            st.info("No database or modified structures to remove")
    if st.session_state.files_marked_for_removal:
        if st.button("🔄 Update list of files", help="Update the list of files to be removed"):
            pass

unique_files = {f.name: f for f in uploaded_files
                if f.name not in st.session_state.files_marked_for_removal}.values()
uploaded_files[:] = list(unique_files)

with st.sidebar.expander("📁 Final list of structure files", expanded=True):
    if uploaded_files:
        st.write(f"**Total: {len(uploaded_files)} file(s)**")

        for i, file in enumerate(uploaded_files, 1):
            source_icon = "👤" if file.name in current_user_file_names else "🌐"
            st.write(f"{i}. {source_icon} {file.name}")

        user_count = len([f for f in uploaded_files if f.name in current_user_file_names])
        db_count = len(uploaded_files) - user_count

        st.caption(f"👤 User: {user_count} | 🌐 Database/Modified: {db_count}")

    else:
        st.info("No files uploaded yet")

    st.session_state.files_marked_for_removal.clear()

if "expander_diff_settings" not in st.session_state:
    st.session_state["expander_diff_settings"] = True

if "parsed_exp_data" not in st.session_state:
    st.session_state.parsed_exp_data = {}
#


if "💥 Powder Diffraction" in calc_mode:
    colmain_1, colmain_2 = st.columns([0.5, 1])
    with colmain_1:
        with st.expander("Diffraction Settings", icon="⚙️", expanded=st.session_state["expander_diff_settings"]):
            st.subheader(
                "⚙️ Diffraction Settings",
                help=(
                    "The powder XRD pattern is calculated using Bragg-Brentano geometry. First, the reciprocal lattice is computed "
                    "and all points within a sphere of radius 2/λ are identified. For each (hkl) plane, the Bragg condition "
                    "(sinθ = λ/(2dₕₖₗ)) is applied. The structure factor, Fₕₖₗ, is computed as the sum of the atomic scattering "
                    "factors. The atomic scattering factor is given by:\n\n"
                    "  f(s) = Z − 41.78214·s²·Σᵢ aᵢ exp(−bᵢ s²)  with s = sinθ/λ\n\n"
                    "Here:\n"
                    " • f(s) is the atomic scattering factor.\n"
                    " • Z is the atomic number.\n"
                    " • aᵢ and bᵢ are tabulated fitted parameters that describe the decay of f(s) with increasing s.\n\n"
                    "The intensity is then computed as Iₕₖₗ = |Fₕₖₗ|², and a Lorentz-polarization correction P(θ) = "
                    "(1+cos²(2θ))/(sin²θ cosθ) is applied."
                )
            )
            st.session_state["expander_diff_settings"] = True


            # --- Save all diffraction parameters to session state ---
            def save_params_to_session_state():
                # This will run every time the diffraction section is processed
                # Getting values from widgets and storing them in session state
                if "peak_representation" in st.session_state:
                    peak_representation = st.session_state.peak_representation
                if "intensity_scale_option" in st.session_state:
                    # intensity_scale_option = st.session_state.intensity_scale_option
                    pass
                if "diffraction_choice" in st.session_state:
                    diffraction_choice = st.session_state.diffraction_choice
                if "line_thickness" in st.session_state:
                    line_thickness = st.session_state.line_thickness
                if "use_debye_waller" in st.session_state:
                    use_debye_waller = st.session_state.use_debye_waller
                if "wavelength_value" in st.session_state:
                    wavelength_value = st.session_state.wavelength_value
                if "sigma" in st.session_state and peak_representation == "Gaussian":
                    sigma = st.session_state.sigma
                if "x_axis_metric" in st.session_state:
                    x_axis_metric = st.session_state.x_axis_metric
                if "y_axis_scale" in st.session_state:
                    y_axis_scale = st.session_state.y_axis_scale
                if "intensity_filter" in st.session_state:
                    intensity_filter = st.session_state.intensity_filter
                if "num_annotate" in st.session_state:
                    num_annotate = st.session_state.num_annotate


            # Initialize parameters if not already in session state
            if "peak_representation" not in st.session_state:
                st.session_state.peak_representation = "Delta"
            #   if "intensity_scale_option" not in st.session_state:
            #       st.session_state.intensity_scale_option = "Normalized"
            if "diffraction_choice" not in st.session_state:
                st.session_state.diffraction_choice = "XRD (X-ray)"
            if "line_thickness" not in st.session_state:
                st.session_state.line_thickness = 2.0
            if "use_debye_waller" not in st.session_state:
                st.session_state.use_debye_waller = False
            if "wavelength_value" not in st.session_state:
                st.session_state.wavelength_value = 0.15406  # Default to CuKa1
            if "sigma" not in st.session_state:
                st.session_state.sigma = 0.5
            if "x_axis_metric" not in st.session_state:
                st.session_state.x_axis_metric = "2θ (°)"
            if "y_axis_scale" not in st.session_state:
                st.session_state.y_axis_scale = "Linear"
            if "intensity_filter" not in st.session_state:
                st.session_state.intensity_filter = 0.0
            if "num_annotate" not in st.session_state:
                st.session_state.num_annotate = 5

            # --- Diffraction Calculator Selection ---
            col2, col4 = st.columns(2)
            with col4:
                diffraction_choice = st.radio(
                    "Diffraction Calculator",
                    ["XRD (X-ray)", "ND (Neutron)"],
                    help="🔬 The X-ray diffraction (XRD) patterns are for **powder samples**, assuming **randomly oriented crystallites**. "
                         "The calculator applies the **Lorentz-polarization correction**: `LP(θ) = (1 + cos²(2θ)) / (sin²θ cosθ)`. It does not account for other corrections, such as preferred orientation, absorption, "
                         "instrumental broadening, or temperature effects (Debye-Waller factors). 🔬 The neutron diffraction (ND) patterns are for **powder samples**, assuming **randomly oriented crystallites**. "
                         "The calculator applies the **Lorentz correction**: `L(θ) = 1  / sin²θ cosθ`. It does not account for other corrections, such as preferred orientation, absorption, "
                         "instrumental broadening, or temperature effects (Debye-Waller factors). The main differences in the calculation from the XRD pattern are: "
                         " (1) Atomic scattering lengths are constant, and (2) Polarization correction is not necessary.",
                    key="diffraction_choice"
                )
                # st.session_state.diffraction_choice = diffraction_choice

            with col2:
                peak_representation = st.radio(
                    "Peak Representation",
                    ["Delta", "Gaussian"],
                    key="peak_representation",
                    help=("Choose whether to represent each diffraction peak as a delta function "
                          "or as a Gaussian. When using Gaussian, the area under each peak equals "
                          "the calculated intensity, and overlapping Gaussians are summed.")
                )
                # st.session_state.peak_representation = peak_representation
            col3, colhhh = st.columns(2)
            with col3:
                intensity_scale_option = st.radio(
                    "Intensity scale",
                    options=["Normalized", "Absolute"],
                    key="intensity_scale_option",
                    help="Normalized sets maximum peak to 100; Absolute shows raw calculated intensities."
                )
                # st.session_state.intensity_scale_option = intensity_scale_option

            with colhhh:
                line_thickness = st.slider(
                    "⚙️ Line thickness for peaks:",
                    min_value=0.5,
                    max_value=6.0,
                    step=0.1,
                    key="line_thickness",
                    help="Adjust the thickness of diffraction peak lines."
                )
                # st.session_state.line_thickness = line_thickness
            use_debye_waller = st.checkbox(
                "✓ Apply Debye-Waller temperature factors",
                key="use_debye_waller",
                help="Apply temperature-dependent intensity correction using Debye-Waller factors (B-factors) for each element. "
                     "This accounts for thermal motion of atoms, which reduces diffraction peak intensities. "
                     "Enter B-factor values for each element in Å² for each structure file. Typical values range from 0.5 to 3.0 Å² "
                     "Higher values (2-3 Å²) represent more thermal motion or disorder. Lower values (0.5-1 Å²) represent less thermal motion (e.g., at low temperatures). "
                     "The intensity correction is applied as: exp(-B·sin²θ/λ²)."
            )
            # st.session_state.use_debye_waller = use_debye_waller

            if use_debye_waller:
                st.markdown(f"### 🔥 Debye-Waller B-factors")
                if "debye_waller_factors_per_file" not in st.session_state:
                    st.session_state.debye_waller_factors_per_file = {}
                preset_col1, preset_col2 = st.columns([1, 3])
                with preset_col1:
                    apply_preset = st.selectbox(
                        "Apply a preset to all files",
                        ["Custom (No Preset)", "Room Temperature (300K)", "Low Temperature (100K)",
                         "High Temperature (500K)"],
                        key="dw_preset"
                    )

                with preset_col2:
                    if apply_preset != "Custom (No Preset)":
                        preset_values = {
                            "Room Temperature (300K)": {
                                "H": 1.1, "C": 0.8, "N": 0.9, "O": 0.7, "F": 0.8, "Na": 1.1, "Mg": 0.5,
                                "Al": 0.6, "Si": 0.5, "P": 0.7, "S": 0.6, "Cl": 0.8, "K": 1.2, "Ca": 0.6,
                                "Ti": 0.4, "V": 0.4, "Cr": 0.4, "Mn": 0.5, "Fe": 0.4, "Co": 0.4, "Ni": 0.4,
                                "Cu": 0.5, "Zn": 0.6, "Ga": 0.7, "Ge": 0.6, "As": 0.5, "Se": 0.7, "Br": 0.9,
                                "Rb": 1.3, "Sr": 0.7, "Y": 0.5, "Zr": 0.4, "Nb": 0.4, "Mo": 0.4, "Tc": 0.4,
                                "Ru": 0.4, "Rh": 0.4, "Pd": 0.5, "Ag": 0.6, "Cd": 0.7, "In": 0.8, "Sn": 0.7,
                                "Sb": 0.6, "Te": 0.7, "I": 1.0, "Cs": 1.4, "Ba": 0.7, "La": 0.5, "Ce": 0.5
                            },
                            "Low Temperature (100K)": {
                                "H": 0.6, "C": 0.4, "N": 0.5, "O": 0.3, "F": 0.4, "Na": 0.6, "Mg": 0.3,
                                "Al": 0.3, "Si": 0.2, "P": 0.3, "S": 0.3, "Cl": 0.4, "K": 0.7, "Ca": 0.3,
                                "Ti": 0.2, "V": 0.2, "Cr": 0.2, "Mn": 0.3, "Fe": 0.2, "Co": 0.2, "Ni": 0.2,
                                "Cu": 0.3, "Zn": 0.3, "Ga": 0.4, "Ge": 0.3, "As": 0.3, "Se": 0.4, "Br": 0.5,
                                "Rb": 0.7, "Sr": 0.4, "Y": 0.3, "Zr": 0.2, "Nb": 0.2, "Mo": 0.2, "Tc": 0.2,
                                "Ru": 0.2, "Rh": 0.2, "Pd": 0.3, "Ag": 0.3, "Cd": 0.4, "In": 0.4, "Sn": 0.4,
                                "Sb": 0.3, "Te": 0.4, "I": 0.5, "Cs": 0.8, "Ba": 0.4, "La": 0.3, "Ce": 0.3
                            },
                            "High Temperature (500K)": {
                                "H": 1.8, "C": 1.3, "N": 1.5, "O": 1.2, "F": 1.3, "Na": 1.8, "Mg": 0.9,
                                "Al": 1.0, "Si": 0.8, "P": 1.2, "S": 1.0, "Cl": 1.4, "K": 2.0, "Ca": 1.0,
                                "Ti": 0.7, "V": 0.7, "Cr": 0.7, "Mn": 0.8, "Fe": 0.7, "Co": 0.7, "Ni": 0.7,
                                "Cu": 0.8, "Zn": 1.0, "Ga": 1.2, "Ge": 1.0, "As": 0.9, "Se": 1.2, "Br": 1.5,
                                "Rb": 2.2, "Sr": 1.2, "Y": 0.9, "Zr": 0.7, "Nb": 0.7, "Mo": 0.7, "Tc": 0.7,
                                "Ru": 0.7, "Rh": 0.7, "Pd": 0.8, "Ag": 1.0, "Cd": 1.2, "In": 1.3, "Sn": 1.2,
                                "Sb": 1.0, "Te": 1.2, "I": 1.7, "Cs": 2.3, "Ba": 1.2, "La": 0.9, "Ce": 0.9
                            }
                        }

                        selected_preset = preset_values[apply_preset]
                        st.write(f"{apply_preset} preset with default B-factors for common elements")

                        sample_elements = ["Si", "O", "Fe", "Ca", "Al"]
                        sample_values = {el: selected_preset.get(el, "N/A") for el in sample_elements if
                                         el in selected_preset}
                        if sample_values:
                            st.write("Example values: " + ", ".join(
                                [f"{el}: {val} Å²" for el, val in sample_values.items()]))

                        apply_preset_button = st.button("Apply preset to all files")
                        if apply_preset_button:
                            st.success(f"{apply_preset} preset to all files")
                if not uploaded_files:
                    st.info(
                        "No structure files uploaded. Please upload structure files to set Debye-Waller factors.")
                else:
                    file_tabs = st.tabs([file.name for file in uploaded_files])

                    for i, (file, tab) in enumerate(zip(uploaded_files, file_tabs)):
                        with tab:
                            file_key = file.name
                            if file_key not in st.session_state.debye_waller_factors_per_file:
                                st.session_state.debye_waller_factors_per_file[file_key] = {}

                            structure_elements = set()
                            try:
                                structure = load_structure(file.name)
                                for site in structure:
                                    if hasattr(site, 'specie') and hasattr(site.specie, 'symbol'):
                                        structure_elements.add(site.specie.symbol)
                                    elif hasattr(site, 'species'):
                                        for sp, _ in site.species.items():
                                            if hasattr(sp, 'symbol'):
                                                structure_elements.add(sp.symbol)
                            except Exception as e:
                                st.warning(f"Could not extract elements from {file.name}: {e}")
                                continue
                            if not structure_elements:
                                st.warning(f"No elements found in {file.name}")
                                continue

                            if apply_preset != "Custom (No Preset)" and apply_preset_button:
                                for element in structure_elements:
                                    if element in selected_preset:
                                        st.session_state.debye_waller_factors_per_file[file_key][element] = \
                                            selected_preset[element]

                            st.write(f"**Elements in {file.name}:** {', '.join(sorted(structure_elements))}")

                            num_cols = min(4, len(structure_elements))
                            cols = st.columns(num_cols)

                            #  input fields for each element
                            for j, element in enumerate(sorted(structure_elements)):
                                col_idx = j % num_cols
                                with cols[col_idx]:
                                    # Use stored value, preset value, or default
                                    if apply_preset != "Custom (No Preset)" and apply_preset_button and element in selected_preset:
                                        default_value = selected_preset[element]
                                    else:
                                        default_value = st.session_state.debye_waller_factors_per_file[
                                            file_key].get(element, 1.0)

                                    b_factor = st.number_input(
                                        f"B-factor for {element} (Å²)",
                                        min_value=0.0,
                                        max_value=10.0,
                                        value=default_value,
                                        step=0.1,
                                        format="%.2f",
                                        key=f"b_factor_{file_key}_{element}"
                                    )
                                    st.session_state.debye_waller_factors_per_file[file_key][element] = b_factor


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


            fig_interactive = go.Figure()


            def twotheta_to_metric(twotheta_deg, metric, wavelength_A, wavelength_nm, diffraction_choice):
                twotheta_deg = np.asarray(twotheta_deg)
                theta = np.deg2rad(twotheta_deg / 2)
                if metric == "2θ (°)":
                    result = twotheta_deg
                elif metric == "2θ (rad)":
                    result = np.deg2rad(twotheta_deg)
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
                    result = np.where(np.sin(theta) == 0, np.inf, wavelength_A / (2 * np.sin(theta)))
                elif metric == "d (nm)":
                    result = np.where(np.sin(theta) == 0, np.inf, wavelength_nm / (2 * np.sin(theta)))
                elif metric == "energy (keV)":
                    if diffraction_choice == "ND (Neutron)":
                        return 0.003956 / (wavelength_nm ** 2)
                    else:
                        return (24.796 * np.sin(theta)) / wavelength_A
                elif metric == "frequency (PHz)":
                    f_Hz = (24.796 * np.sin(theta)) / wavelength_A * 2.418e17
                    result = f_Hz / 1e15
                else:
                    result = twotheta_deg
                if np.ndim(twotheta_deg) == 0:
                    return float(result)
                return result


            def metric_to_twotheta(metric_value, metric, wavelength_A, wavelength_nm, diffraction_choice):
                if metric == "2θ (°)":
                    return metric_value
                elif metric == "2θ (rad)":
                    return np.rad2deg(metric_value)
                elif metric == "q (1/Å)":
                    theta = np.arcsin(np.clip(metric_value * wavelength_A / (4 * np.pi), 0, 1))
                    return np.rad2deg(2 * theta)
                elif metric == "θ (°)":
                    return 2 * metric_value
                elif metric == "θ (rad)":
                    return 2 * np.rad2deg(metric_value)
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
                    if diffraction_choice == "ND (Neutron)":
                        λ_nm = np.sqrt(0.003956 / metric_value)
                        sin_theta = λ_nm / (2 * wavelength_nm)
                        theta = np.arcsin(np.clip(sin_theta, 0, 1))
                    else:
                        sin_theta = np.clip(metric_value * wavelength_A / 24.796, 0, 1)
                        theta = np.arcsin(np.clip(sin_theta, 0, 1))
                    return np.rad2deg(2 * theta)
                elif metric == "frequency (PHz)":
                    f_Hz = metric_value * 1e15
                    E_keV = f_Hz / 2.418e17
                    theta = np.arcsin(np.clip(E_keV * wavelength_A / 24.796, 0, 1))
                    return np.rad2deg(2 * theta)
                else:
                    return metric_value


            conversion_info = {
                "2θ (°)": "Identity: 2θ in degrees.",
                "2θ (rad)": "Conversion: radians = degrees * (π/180).",
                "θ (°)": "Identity: 2θ in degrees.",
                "θ (rad)": "Conversion: radians = degrees * (π/180).",
                "q (1/Å)": "q = (4π/λ) * sin(θ), with λ in Å.",
                "q (1/nm)": "q = (4π/λ) * sin(θ), with λ in nm.",
                "d (Å)": "d = λ / (2 sin(θ)), with λ in Å.",
                "d (nm)": "d = λ / (2 sin(θ)), with λ in nm.",
                "energy (keV)": "E = (24.796 * sin(θ)) / λ, with λ in Å.",
                "frequency (PHz)": "f = [(24.796 * sin(θ))/λ * 2.418e17] / 1e15, with λ in Å."
            }

            # --- Wavelength Selection ---
            # preset_options = [
            #    'CoKa1', 'CoKa2', 'Co(Ka1+Ka2)', 'Co(Ka1+Ka2+Kb1)', 'CoKb1',
            #    'MoKa1', 'MoKa2', 'Mo(Ka1+Ka2)', 'Mo(Ka1+Ka2+Kb1)', 'MoKb1',
            #    'CuKa1', 'CuKa2', 'Cu(Ka1+Ka2)', 'Cu(Ka1+Ka2+Kb1)', 'CuKb1',
            #    'CrKa1', 'CrKa2', 'Cr(Ka1+Ka2)', 'Cr(Ka1+Ka2+Kb1)', 'CrKb1',
            #    'FeKa1', 'FeKa2', 'Fe(Ka1+Ka2)', 'Fe(Ka1+Ka2+Kb1)', 'FeKb1',
            #    'AgKa1', 'AgKa2', 'Ag(Ka1+Ka2)', 'Ag(Ka1+Ka2+Kb1)', 'AgKb1'
            # ]
            preset_options = [
                'Cobalt (CoKa1)', 'Copper (CuKa1)', 'Molybdenum (MoKa1)', 'Chromium (CrKa1)', 'Iron (FeKa1)',
                'Silver (AgKa1)',
                'Co(Ka1+Ka2)', 'Co(Ka1+Ka2+Kb1)',
                'Mo(Ka1+Ka2)', 'Mo(Ka1+Ka2+Kb1)',
                'Cu(Ka1+Ka2)', 'Cu(Ka1+Ka2+Kb1)',
                'Cr(Ka1+Ka2)', 'Cr(Ka1+Ka2+Kb1)',
                'Fe(Ka1+Ka2)', 'Fe(Ka1+Ka2+Kb1)',
                'Ag(Ka1+Ka2)', 'Ag(Ka1+Ka2+Kb1)',
            ]
            preset_wavelengths = {
                'Custom': 0.17889,
                'Cu(Ka1+Ka2)': 0.154,
                'CuKa2': 0.15444,
                'Copper (CuKa1)': 0.15406,
                'Cu(Ka1+Ka2+Kb1)': 0.153339,
                'CuKb1': 0.13922,
                'Mo(Ka1+Ka2)': 0.071,
                'MoKa2': 0.0711,
                'Molybdenum (MoKa1)': 0.07093,
                'Mo(Ka1+Ka2+Kb1)': 0.07059119,
                'MoKb1': 0.064,
                'Cr(Ka1+Ka2)': 0.229,
                'CrKa2': 0.22888,
                'Chromium (CrKa1)': 0.22897,
                'Cr(Ka1+Ka2+Kb1)': 0.22775471,
                'CrKb1': 0.208,
                'Fe(Ka1+Ka2)': 0.194,
                'FeKa2': 0.194,
                'Iron (FeKa1)': 0.19360,
                'Fe(Ka1+Ka2+Kb1)': 0.1927295,
                'FeKb1': 0.176,
                'Co(Ka1+Ka2)': 0.179,
                'CoKa2': 0.17927,
                'Cobalt (CoKa1)': 0.17889,
                'Co(Ka1+Ka2+Kb1)': 0.1781100,
                'CoKb1': 0.163,
                'Silver (AgKa1)': 0.0561,
                'AgKa2': 0.05634,
                'Ag(Ka1+Ka2)': 0.0561,
                'AgKb1': 0.0496,
                'Ag(Ka1+Ka2+Kb1)': 0.0557006
            }


            def energy_to_wavelength(energy_kev):
                return 1.2398 / energy_kev


            def wavelength_to_energy(wavelength_nm):
                return 1.2398 / wavelength_nm


            preset_options_neutron = ['Thermal Neutrons', 'Cold Neutrons', 'Hot Neutrons']
            preset_wavelengths_neutrons = {
                'Custom': 0.154,
                'Thermal Neutrons': 0.154,
                'Cold Neutrons': 0.475,
                'Hot Neutrons': 0.087
            }

            if "input_mode" not in st.session_state:
                st.session_state.input_mode = "Preset Wavelength"
            if "energy_kev" not in st.session_state:
                st.session_state.energy_kev = 8.048
            if "preset_choice" not in st.session_state:
                st.session_state.preset_choice = "Cobalt (CoKa1)"

            if "wavelength_value" not in st.session_state:
                if diffraction_choice == "XRD (X-ray)":
                    st.session_state.wavelength_value = 0.17889
                else:
                    st.session_state.wavelength_value = 0.154

            if "last_diffraction_choice" not in st.session_state:
                st.session_state.last_diffraction_choice = diffraction_choice
            elif st.session_state.last_diffraction_choice != diffraction_choice:
                if diffraction_choice == "XRD (X-ray)":
                    st.session_state.wavelength_value = 0.17889
                else:
                    st.session_state.wavelength_value = 0.154
                st.session_state.last_diffraction_choice = diffraction_choice

            if diffraction_choice == "XRD (X-ray)":
                # with col1:
                input_mode = st.radio(
                    "Input Mode",
                    ["Preset Wavelength", "Custom Wavelength", "X-ray Energy (keV)"],
                    key="input_mode",
                    help="Choose how to specify the X-ray source"
                )
                col2, col3h = st.columns(2)
                if "last_input_mode" not in st.session_state:
                    st.session_state.last_input_mode = input_mode
                elif st.session_state.last_input_mode != input_mode and input_mode == "Preset Wavelength":
                    st.session_state.wavelength_value = preset_wavelengths["Cobalt (CoKa1)"]
                    st.session_state.preset_choice = "Cobalt (CoKa1)"
                st.session_state.last_input_mode = input_mode

                if input_mode == "Preset Wavelength":
                    with col2:
                        preset_choice = st.selectbox(
                            "🌊 Preset Wavelength",
                            options=preset_options,
                            key="preset_choice",
                            index=0,
                            help="I_Kalpha2 = 1/2 I_Kalpha1, I_Kbeta = 1/9 I_Kalpha1"
                        )

                    with col2:
                        if "preset_choice" in st.session_state and st.session_state.preset_choice != st.session_state.get(
                                "previous_preset", ""):
                            st.session_state.wavelength_value = preset_wavelengths[st.session_state.preset_choice]
                            st.session_state.previous_preset = st.session_state.preset_choice
                            selected_preset_name = st.session_state.preset_choice
                            if selected_preset_name in DEFAULT_TWO_THETA_MAX_FOR_PRESET:
                                new_max_2theta = DEFAULT_TWO_THETA_MAX_FOR_PRESET[selected_preset_name]
                                st.session_state.two_theta_max = new_max_2theta
                                if st.session_state.two_theta_min >= new_max_2theta:
                                    st.session_state.two_theta_min = 5.0
                                    if new_max_2theta <= 10.0:
                                        st.session_state.two_theta_min = 1.0
                                    if st.session_state.two_theta_min >= new_max_2theta:
                                        st.session_state.two_theta_min = max(0.1, new_max_2theta * 0.5)

                        hide_input_for = ['Cu(Ka1+Ka2+Kb1)', 'Cu(Ka1+Ka2)']
                        if preset_choice not in hide_input_for:
                            wavelength_value = st.number_input(
                                "🌊 Wavelength (nm)",
                                min_value=0.001,
                                step=0.001,
                                format="%.5f",
                                key="wavelength_value"
                            )
                        else:
                            wavelength_value = preset_wavelengths[preset_choice]
                            st.session_state.wavelength_value = wavelength_value
                elif input_mode == "Custom Wavelength":
                    if st.session_state.last_input_mode != input_mode:
                        st.session_state.wavelength_value = 0.17889
                    with col2:
                        wavelength_value = st.number_input(
                            "🌊 Wavelength (nm)",
                            min_value=0.01,
                            max_value=5.0,
                            value=st.session_state.wavelength_value,
                            step=0.001,
                            format="%.5f",
                            key="wavelength_value",
                            help="Typical X-ray wavelengths: 0.05-2.5 nm"
                        )
                    with col3h:
                        calculated_energy = wavelength_to_energy(wavelength_value)
                        st.metric("Calculated Energy", f"{calculated_energy:.3f} keV")

                elif input_mode == "X-ray Energy (keV)":
                    if st.session_state.last_input_mode != input_mode:
                        st.session_state.energy_kev = 6
                    if st.session_state.get("last_input_mode") != input_mode:
                        if ("energy_kev" not in st.session_state
                                or st.session_state.energy_kev is None
                                or st.session_state.energy_kev < 1.0):
                            st.session_state.energy_kev = 6  # default within [1.0, 100.0]
                        st.session_state.wavelength_value = energy_to_wavelength(st.session_state.energy_kev)
                    with col2:
                        energy_kev = st.number_input(
                            "⚡ X-ray Energy (keV)",
                            min_value=6.0,
                            max_value=65.0,
                            step=0.1,
                            format="%.3f",
                            key="energy_kev",
                            help="Typical synchrotron energies: 8-30 keV"
                        )

                    with col3h:
                        wavelength_value = energy_to_wavelength(energy_kev)
                        st.session_state.wavelength_value = wavelength_value
                        st.metric("Calculated Wavelength", f"{wavelength_value:.5f} nm")




            elif diffraction_choice == "ND (Neutron)":
                col1, col2 = st.columns(2)
                with col1:
                    preset_choice = st.selectbox(
                        "Preset Wavelength",
                        options=preset_options_neutron,
                        index=0,
                        key="preset_choice_neutron",
                        help="Factors for weighted average of wavelengths are: I1 = 2 (ka1), I2 = 1 (ka2), I3 = 0.18 (kb1)"
                    )
                with col2:
                    if "preset_choice_neutron" in st.session_state and st.session_state.preset_choice_neutron != st.session_state.get(
                            "previous_preset_neutron", ""):
                        st.session_state.wavelength_value = preset_wavelengths_neutrons[
                            st.session_state.preset_choice_neutron]
                        st.session_state.previous_preset_neutron = st.session_state.preset_choice_neutron
                    selected_preset_name_neutron = st.session_state.preset_choice_neutron
                    st.session_state.previous_preset_neutron = selected_preset_name_neutron

                    if selected_preset_name_neutron in DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET:
                        new_max_2theta_neutron = DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET[selected_preset_name_neutron]
                        st.session_state.two_theta_max = new_max_2theta_neutron
                        if st.session_state.two_theta_min >= new_max_2theta_neutron:
                            st.session_state.two_theta_min = 5.0
                            if new_max_2theta_neutron <= 10.0:
                                st.session_state.two_theta_min = 1.0
                            if st.session_state.two_theta_min >= new_max_2theta_neutron:
                                st.session_state.two_theta_min = max(0.1, new_max_2theta_neutron * 0.5)

                    wavelength_value = st.number_input(
                        "Wavelength (nm)",
                        min_value=0.001,
                        step=0.001,
                        format="%.5f",
                        key="wavelength_value"
                    )
            _, col3h = st.columns(2)
            wavelength_A = wavelength_value * 10
            wavelength_nm = wavelength_value

            preset_choice = st.session_state.get("preset_choice", "Cobalt (CoKa1)")

            x_axis_options = [
                "2θ (°)", "2θ (rad)", "θ (°)", "θ (rad)",
                "q (1/Å)", "q (1/nm)",
                "d (Å)", "d (nm)",
                "energy (keV)", "frequency (PHz)"
            ]
            x_axis_options_neutron = [
                "2θ (°)", "2θ (rad)", "θ (°)", "θ (rad)",
                "q (1/Å)", "q (1/nm)",
                "d (Å)", "d (nm)",
            ]
            # --- X-axis Metric Selection ---
            colx, colx1, = st.columns(2)
            with colx:
                if diffraction_choice == "ND (Neutron)":
                    x_axis_metric = st.selectbox(
                        "⚙️ ND x-axis Metric",
                        x_axis_options_neutron,
                        key="x_axis_metric",
                        # help=conversion_info.get(x_axis_metric, "X-axis metric selection")
                    )
                else:
                    x_axis_metric = st.selectbox(
                        "⚙️ XRD x-axis Metric",
                        x_axis_options,
                        key="x_axis_metric",
                        # help=conversion_info.get(x_axis_metric, "X-axis metric selection")
                    )
            with colx1:
                y_axis_scale = st.selectbox(
                    "⚙️ Y-axis Scale",
                    ["Linear", "Square Root", "Logarithmic"],
                    key="y_axis_scale",
                    help="Choose how to display intensity values. Linear shows original values. Square Root..."
                )
            if y_axis_scale == "Linear":
                y_axis_title = "Intensity (a.u.)"
            elif y_axis_scale == "Square Root":
                y_axis_title = "√Intensity (a.u.)"
            elif y_axis_scale == "Logarithmic":
                y_axis_title = "log₁₀(Intensity) (a.u.)"
            # --- Initialize canonical two_theta_range in session_state (always in degrees) ---
            if "two_theta_min" not in st.session_state:
                if x_axis_metric in ["energy (keV)", "frequency (PHz)"]:
                    st.session_state.two_theta_min = 5.0
                elif x_axis_metric in ["d (Å)", "d (nm)"]:
                    st.session_state.two_theta_min = 20.0
                else:
                    st.session_state.two_theta_min = 5.0
            if "two_theta_max" not in st.session_state:
                st.session_state.two_theta_max = 165.0

            # --- Compute display values by converting canonical two_theta values to current unit ---
            display_metric_min = twotheta_to_metric(st.session_state.two_theta_min, x_axis_metric, wavelength_A,
                                                    wavelength_nm,
                                                    diffraction_choice)
            display_metric_max = twotheta_to_metric(st.session_state.two_theta_max, x_axis_metric, wavelength_A,
                                                    wavelength_nm,
                                                    diffraction_choice)

            if x_axis_metric == "2θ (°)":
                step_val = 1.0
            elif x_axis_metric == "2θ (rad)":
                step_val = 0.0174533
            else:
                step_val = 0.1

            # col1, col2 = st.columns(2)
            colx2, colx3 = st.columns(2)
            if x_axis_metric == "d (Å)" or x_axis_metric == "d (nm)":

                min_val = colx3.number_input(f"⚙️ Maximum {x_axis_metric}", value=display_metric_min, step=step_val,
                                             key=f"min_val_{x_axis_metric}")
                max_val = colx2.number_input(f"⚙️ Minimum {x_axis_metric}", value=display_metric_max, step=step_val,
                                             key=f"max_val_{x_axis_metric}")
            else:
                min_val = colx2.number_input(f"⚙️ Minimum {x_axis_metric}", value=display_metric_min, step=step_val,
                                             key=f"min_val_{x_axis_metric}")
                max_val = colx3.number_input(f"⚙️ Maximum {x_axis_metric}", value=display_metric_max, step=step_val,
                                             key=f"max_val_{x_axis_metric}")

            # --- Update the canonical two_theta values based on current inputs ---
            st.session_state.two_theta_min = metric_to_twotheta(min_val, x_axis_metric, wavelength_A, wavelength_nm,
                                                                diffraction_choice)
            st.session_state.two_theta_max = metric_to_twotheta(max_val, x_axis_metric, wavelength_A, wavelength_nm,
                                                                diffraction_choice)
            two_theta_display_range = (st.session_state.two_theta_min, st.session_state.two_theta_max)
            user_calculation_range = (st.session_state.two_theta_min, st.session_state.two_theta_max)

            if st.session_state.peak_representation != "Delta":
                sigma = st.number_input(
                    "⚙️ Gaussian sigma (°) for peak sharpness (smaller = sharper peaks)",
                    min_value=0.1,
                    max_value=1.5,
                    step=0.01,
                    key="sigma"
                )
                # st.session_state.sigma = sigma
            else:
                sigma = st.session_state.sigma = 0.5
            with col3h:
                num_annotate = st.number_input(
                    "⚙️ How many highest peaks to annotate in table:",
                    min_value=0,
                    max_value=30,
                    value=st.session_state.num_annotate,
                    step=1,
                    key="num_annotate_widget"
                )
                st.session_state.num_annotate = num_annotate

            # with col4h:
            intensity_filter = st.slider(
                "⚙️ Filter peaks (% of max intensity):",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.intensity_filter,
                step=0.1,
                key="intensity_filter_widget",
                help="Filter out peaks with intensity below this percentage of the maximum peak intensity. Set to 0 to show all peaks."
            )
            st.session_state.intensity_filter = intensity_filter

            if "calc_xrd" not in st.session_state:
                st.session_state.calc_xrd = False
            if "prev_energy_kev" not in st.session_state:
                st.session_state.prev_energy_kev = 8.048

            if diffraction_choice == "ND (Neutron)":
                if st.button("Calculate ND"):
                    st.session_state.calc_xrd = True
            else:
                if st.button("Calculate XRD"):
                    st.session_state.calc_xrd = True

    st.sidebar.subheader(
        "📉 Experimental Data Background Subtraction",
        help=(
            "This section allows you to subtract background from your uploaded experimental diffraction data. "
            "You can choose from different background estimation methods and adjust parameters."
        )
    )
    if user_pattern_file:
        bg_subtraction_container = st.container()

        with bg_subtraction_container:
            if isinstance(user_pattern_file, list) and len(user_pattern_file) > 1:
                selected_exp_file = st.sidebar.selectbox(
                    "Select experimental data file for background subtraction",
                    options=[file.name for file in user_pattern_file],
                    index=0
                )
                selected_file_obj = next((file for file in user_pattern_file if file.name == selected_exp_file),
                                         None)
            else:
                selected_file_obj = user_pattern_file[0] if isinstance(user_pattern_file,
                                                                       list) else user_pattern_file
                selected_exp_file = selected_file_obj.name

            try:
                df = pd.read_csv(selected_file_obj, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                x_exp = df.iloc[:, 0].values
                y_exp = df.iloc[:, 1].values

                if "original_exp_data" not in st.session_state:
                    st.session_state.original_exp_data = {}

                if selected_exp_file not in st.session_state.original_exp_data:
                    st.session_state.original_exp_data[selected_exp_file] = {
                        "x": x_exp.copy(),
                        "y": y_exp.copy()
                    }

                # Background subtraction method selection
                bg_method = st.sidebar.radio(
                    "Background Estimation Method",
                    ["None", "Polynomial Fit", "SNIP Algorithm", "Rolling Ball Algorithm"],
                    index=0,
                    key="bg_method",
                    help=(
                        "Choose a method to estimate the background:\n"
                        "- Polynomial Fit: Fits a polynomial of specified degree to the data\n"
                        "- SNIP Algorithm: Statistics-sensitive Non-linear Iterative Peak-clipping\n"
                        "- Rolling Ball Algorithm: Simulates rolling a ball under the spectrum"
                    )
                )

                # If any method other than "None" is selected
                if bg_method != "None":
                    param_col1, param_col2 = st.columns(2)

                    if "bg_subtracted_data" not in st.session_state:
                        st.session_state.bg_subtracted_data = {}

                    if bg_method == "Polynomial Fit":
                        with param_col1:
                            poly_degree = st.slider("Polynomial Degree", 1, 10, 3, 1,
                                                    help="Higher degree allows more complex background shapes")

                        with param_col2:
                            smoothing_factor = st.slider("Smoothing Factor", 0.0, 1.0, 0.0, 0.01,
                                                         help="Higher values create smoother fits (0=exact fit)")


                        def poly_bg(x, y, degree, smoothing):
                            sort_idx = np.argsort(x)
                            x_sorted = x[sort_idx]
                            y_sorted = y[sort_idx]

                            if smoothing > 0:
                                from scipy.signal import savgol_filter
                                window = max(3, int(len(x) * smoothing / 5) * 2 + 1)  # Must be odd
                                window = min(window, len(x) - 1)
                                if window >= 3:
                                    y_smoothed = savgol_filter(y_sorted, window, min(degree, window - 1))
                                else:
                                    y_smoothed = y_sorted
                            else:
                                y_smoothed = y_sorted

                            coeffs = np.polyfit(x_sorted, y_smoothed, degree)
                            background = np.polyval(coeffs, x)

                            return background


                        background = poly_bg(x_exp, y_exp, poly_degree, smoothing_factor)

                    elif bg_method == "SNIP Algorithm":
                        with param_col1:
                            snip_iterations = st.slider("SNIP Iterations", 1, 100, 20, 1,
                                                        help="Number of iterations for the SNIP algorithm")

                        with param_col2:
                            snip_window = st.slider("Window Size", 3, 101, 21, 2,
                                                    help="Window size for the SNIP algorithm (must be odd)")
                            if snip_window % 2 == 0:
                                snip_window += 1  # Ensure window is odd


                        def snip_bg(y, iterations, window_size):
                            if window_size % 2 == 0:
                                window_size += 1

                            half_window = window_size // 2
                            y_bg = y.copy()

                            # SNIP algorithm
                            for _ in range(iterations):
                                for i in range(half_window, len(y) - half_window):
                                    left_val = y_bg[i - half_window]
                                    right_val = y_bg[i + half_window]
                                    min_val = min(left_val, right_val)
                                    y_bg[i] = min(y_bg[i], min_val)

                            return y_bg


                        background = snip_bg(y_exp, snip_iterations, snip_window)

                    elif bg_method == "Rolling Ball Algorithm":
                        with param_col1:
                            ball_radius = st.slider("Ball Radius", 1, 100, 30, 1,
                                                    help="Radius of the rolling ball (larger = smoother background)")

                        with param_col2:
                            ball_smoothing = st.slider("Smoothing Passes", 0, 10, 3, 1,
                                                       help="Number of smoothing passes before background estimation")


                        # Apply the Rolling Ball background estimation
                        def rolling_ball_bg(y, radius, smoothing_passes):

                            y_smoothed = y.copy()
                            if smoothing_passes > 0:
                                window_size = min(len(y) // 10, 20)
                                if window_size % 2 == 0:
                                    window_size += 1  # Ensure window is odd

                                if window_size >= 3:
                                    from scipy.signal import savgol_filter
                                    for _ in range(smoothing_passes):
                                        y_smoothed = savgol_filter(y_smoothed, window_size, 2)

                            y_bg = np.zeros_like(y_smoothed)
                            for i in range(len(y_smoothed)):
                                start = max(0, i - radius)
                                end = min(len(y_smoothed), i + radius + 1)

                                window_min = np.min(y_smoothed[start:end])

                                y_bg[i] = window_min

                            return y_bg


                        background = rolling_ball_bg(y_exp, ball_radius, ball_smoothing)

                    y_bg_subtracted = np.maximum(0, y_exp - background)

                    # Store the background-subtracted data
                    st.session_state.bg_subtracted_data[selected_exp_file] = {
                        "x": x_exp,
                        "y": y_bg_subtracted,
                        "background": background,
                        "method": bg_method,
                        "params": {
                            "poly_degree": poly_degree if bg_method == "Polynomial Fit" else None,
                            "smoothing_factor": smoothing_factor if bg_method == "Polynomial Fit" else None,
                            "snip_iterations": snip_iterations if bg_method == "SNIP Algorithm" else None,
                            "snip_window": snip_window if bg_method == "SNIP Algorithm" else None,
                            "ball_radius": ball_radius if bg_method == "Rolling Ball Algorithm" else None,
                            "ball_smoothing": ball_smoothing if bg_method == "Rolling Ball Algorithm" else None
                        }
                    }

                    apply_permanent = st.button("Permanently Apply This Background Subtraction", type='primary',
                                                help="This will permanently modify the data for this file. The background-subtracted data will be used for all future operations, even if you change parameters or work with other files.")

                    if apply_permanent:
                        if "permanent_exp_data" not in st.session_state:
                            st.session_state.permanent_exp_data = {}
                        st.session_state.permanent_exp_data[selected_exp_file] = {
                            "x": x_exp.copy(),
                            "y": y_bg_subtracted.copy(),
                            "original_x": x_exp.copy(),
                            "original_y": y_exp.copy(),
                            "background": background.copy(),
                            "method": bg_method,
                            "params": {
                                "poly_degree": poly_degree if bg_method == "Polynomial Fit" else None,
                                "smoothing_factor": smoothing_factor if bg_method == "Polynomial Fit" else None,
                                "snip_iterations": snip_iterations if bg_method == "SNIP Algorithm" else None,
                                "snip_window": snip_window if bg_method == "SNIP Algorithm" else None,
                                "ball_radius": ball_radius if bg_method == "Rolling Ball Algorithm" else None,
                                "ball_smoothing": ball_smoothing if bg_method == "Rolling Ball Algorithm" else None
                            }
                        }
                        st.success(f"Background subtraction has been permanently applied to {selected_exp_file}!")
                        
                    # Download button for background-subtracted data
                        col_download1, col_download2 = st.columns(2)
                        
                        with col_download1:
                            download_data = np.column_stack((x_exp, y_bg_subtracted))
                            download_str = "# X-axis  Intensity (Background Subtracted)\n"
                            download_str += "\n".join([f"{row[0]:.6f}  {row[1]:.6f}" for row in download_data])
                            
                            st.download_button(
                                label="💾 Download Background-Subtracted Data (.xy)",
                                data=download_str,
                                file_name=f"{selected_exp_file.rsplit('.', 1)[0]}_bg_subtracted.xy",
                                mime="text/plain",
                                type="primary",
                            )
                        
                        with col_download2:
                            background_data = np.column_stack((x_exp, background))
                            background_str = "# X-axis  Background\n"
                            background_str += "\n".join([f"{row[0]:.6f}  {row[1]:.6f}" for row in background_data])
                            
                            st.download_button(
                                label="💾 Download Background Curve (.xy)",
                                data=background_str,
                                file_name=f"{selected_exp_file.rsplit('.', 1)[0]}_background.xy",
                                mime="text/plain",
                                type="primary",
                            )
                    col1, col2, col3 = st.columns([1, 2, 1])

                    # with col2:
                    fig_bg = go.Figure()

                    fig_bg.add_trace(go.Scatter(
                        x=x_exp,
                        y=y_exp,
                        mode='lines',
                        name='Original Data',
                        line=dict(color='black', width=3),
                        hovertemplate='Original Data<br>%{x:.3f}<br>%{y:.2f}<extra></extra>'
                    ))

                    fig_bg.add_trace(go.Scatter(
                        x=x_exp,
                        y=background,
                        mode='lines',
                        name='Estimated Background',
                        line=dict(color='red', width=3),
                        hovertemplate='Background<br>%{x:.3f}<br>%{y:.2f}<extra></extra>'
                    ))

                    fig_bg.add_trace(go.Scatter(
                        x=x_exp,
                        y=y_bg_subtracted,
                        mode='lines',
                        name='After Subtraction',
                        line=dict(color='blue', width=3),
                        hovertemplate='Subtracted<br>%{x:.3f}<br>%{y:.2f}<extra></extra>'
                    ))

                    fig_bg.update_layout(
                        title=dict(
                            text='Background Subtraction',
                            font=dict(size=32)
                        ),
                        xaxis=dict(
                            title=dict(text=x_axis_metric, font=dict(size=28)),
                            tickfont=dict(size=24)
                        ),
                        yaxis=dict(
                            title=dict(text='Intensity (a.u.)', font=dict(size=28)),
                            tickfont=dict(size=24)
                        ),
                        legend=dict(
                            font=dict(size=24),
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        height=600,
                        width=1200,
                        margin=dict(l=100, r=100, t=120, b=100),
                        hovermode='x unified',
                        showlegend=True,
                        hoverlabel=dict(font=dict(size=20))
                    )

                    st.plotly_chart(fig_bg, use_container_width=True)

                    use_bg_subtracted = st.checkbox("Use background-subtracted data for visualization", value=True,
                                                    help="When checked, the background-subtracted data will be used in the main plot")

                    if use_bg_subtracted:

                        st.session_state.use_bg_subtracted = True
                        st.session_state.active_bg_subtracted_file = selected_exp_file
                    else:
                        st.session_state.use_bg_subtracted = False
            except Exception as e:
                st.error(f"Error processing experimental file {selected_exp_file}: {e}")

            # --- XRD Calculation ---
            colors = ["black", "brown", "grey", "purple"]
            if not st.session_state.calc_xrd:
                st.info(f"Hint: You can **remove background** from the **experimental files** using sidebar.")
                st.warning("Provide first crystal structure files.")

            if user_pattern_file and (not st.session_state.calc_xrd or not uploaded_files):
                if "parsed_exp_data" not in st.session_state:
                    st.session_state.parsed_exp_data = {}

                if isinstance(user_pattern_file, list):
                    for i, file in enumerate(user_pattern_file):
                        file_name = file.name

                        if "permanent_exp_data" in st.session_state and file_name in st.session_state.permanent_exp_data:
                            x_user = st.session_state.permanent_exp_data[file_name]["x"]
                            y_user = st.session_state.permanent_exp_data[file_name]["y"]
                            display_name = file_name + " (BG removed)"

                        elif ("use_bg_subtracted" in st.session_state and st.session_state.use_bg_subtracted and
                              "active_bg_subtracted_file" in st.session_state and
                              st.session_state.active_bg_subtracted_file == file_name and
                              "bg_subtracted_data" in st.session_state and
                              file_name in st.session_state.bg_subtracted_data):

                            x_user = st.session_state.bg_subtracted_data[file_name]["x"]
                            y_user = st.session_state.bg_subtracted_data[file_name]["y"]
                            display_name = file_name + " (temp BG removed)"
                        else:
                            file.seek(0)

                            file_contents = file.read()

                            if isinstance(file_contents, bytes):
                                file_contents = file_contents.decode('utf-8')

                            from io import StringIO

                            data_io = StringIO(file_contents)

                            df = pd.read_csv(data_io, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                            x_user = df.iloc[:, 0].values
                            y_user = df.iloc[:, 1].values
                            display_name = file_name

                            file.seek(0)
                        if intensity_scale_option == "Normalized" and np.max(y_user) > 0:
                            y_user = (y_user / np.max(y_user)) * 100

                        mask_user = (x_user >= st.session_state.two_theta_min) & (
                                    x_user <= st.session_state.two_theta_max)
                        x_user_filtered = x_user[mask_user]
                        y_user_filtered = y_user[mask_user]

                        color = colors[i % len(colors)]
                        fig_interactive.add_trace(go.Scatter(
                            x=x_user_filtered,
                            y=y_user_filtered,
                            mode="lines",
                            name=file.name + (" (BG subtracted)" if ("use_bg_subtracted" in st.session_state and
                                                                     st.session_state.use_bg_subtracted and
                                                                     "active_bg_subtracted_file" in st.session_state and
                                                                     st.session_state.active_bg_subtracted_file == file_name) else ""),
                            line=dict(dash='solid', width=1, color=color),
                            marker=dict(color=color, size=3),
                            hovertemplate=(
                                f"<span style='color:{color};'><b>{file.name}:</b><br>"
                                "2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
                            )
                        ))
                        fig_interactive.update_layout(
                            height=800,
                            margin=dict(t=80, b=80, l=60, r=30),
                            hovermode="x",
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=24)
                            ),
                            xaxis=dict(
                                title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
                                tickfont=dict(size=36, color='black')
                            ),
                            yaxis=dict(
                                title=dict(text=y_axis_title, font=dict(size=36, color='black')),
                                tickfont=dict(size=36, color='black')
                            ),
                            hoverlabel=dict(font=dict(size=24)),
                            font=dict(size=18),
                            autosize=True
                        )
                else:
                    file_name = user_pattern_file.name
                    try:
                        if "permanent_exp_data" in st.session_state and file_name in st.session_state.permanent_exp_data:
                            # Use the permanently background-subtracted data
                            x_user = st.session_state.permanent_exp_data[file_name]["x"]
                            y_user = st.session_state.permanent_exp_data[file_name]["y"]
                            display_name = file_name + " (BG removed)"

                        elif ("use_bg_subtracted" in st.session_state and st.session_state.use_bg_subtracted and
                              "active_bg_subtracted_file" in st.session_state and
                              st.session_state.active_bg_subtracted_file == file_name and
                              "bg_subtracted_data" in st.session_state and
                              file_name in st.session_state.bg_subtracted_data):

                            x_user = st.session_state.bg_subtracted_data[file_name]["x"]
                            y_user = st.session_state.bg_subtracted_data[file_name]["y"]
                            display_name = file_name + " (temp BG removed)"
                        else:
                            user_pattern_file.seek(0)

                            file_contents = user_pattern_file.read()

                            if isinstance(file_contents, bytes):
                                file_contents = file_contents.decode('utf-8')

                            from io import StringIO

                            data_io = StringIO(file_contents)

                            df = pd.read_csv(data_io, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                            x_user = df.iloc[:, 0].values
                            y_user = df.iloc[:, 1].values
                            display_name = file_name

                            user_pattern_file.seek(0)

                        if x_user is not None and y_user is not None:
                            if intensity_scale_option == "Normalized" and np.max(y_user) > 0:
                                y_user = (y_user / np.max(y_user)) * 100

                            mask_user = (x_user >= st.session_state.two_theta_min) & (
                                    x_user <= st.session_state.two_theta_max)
                            x_user_filtered = x_user[mask_user]
                            y_user_filtered = y_user[mask_user]

                            color = "black"
                            fig_interactive.add_trace(go.Scatter(
                                x=x_user_filtered,
                                y=y_user_filtered,
                                mode="lines",
                                name=display_name,
                                line=dict(dash='solid', width=1, color=color),
                                marker=dict(color=color, size=3),
                                hovertemplate=(
                                    f"<span style='color:{color};'><b>{display_name}:</b><br>"
                                    "2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
                                )
                            ))
                    except Exception as e:
                        st.error(f"Error processing file {file_name}: {e}")
                        x_user, y_user = None, None
    with colmain_2:
        tab1, tab2 = st.tabs(["➡️ Patterns chart", "🖥️ Quantitative peak data"])
        if st.session_state.calc_xrd and uploaded_files:
            with st.spinner("Calculating the powder diffraction pattern(s), please wait. 😊"):
                with tab1:
                    if "new_structure_added" in st.session_state and st.session_state.new_structure_added:
                        # Show a success message at the top of the XRD section
                        if "new_structure_name" in st.session_state:
                            st.success(
                                f"Modified structure '{st.session_state.new_structure_name}' has been added to the calculation.")

                        # Reset the flag so the message doesn't show again
                        st.session_state.new_structure_added = False

                    save_params_to_session_state()
                    peak_representation = st.session_state.peak_representation
                    intensity_scale_option = st.session_state.intensity_scale_option
                    diffraction_choice = st.session_state.diffraction_choice
                    line_thickness = st.session_state.line_thickness
                    use_debye_waller = st.session_state.use_debye_waller
                    wavelength_value = st.session_state.wavelength_value
                    wavelength_A = wavelength_value * 10  # Convert nm to Å
                    wavelength_nm = wavelength_value
                    sigma = st.session_state.sigma
                    x_axis_metric = st.session_state.x_axis_metric
                    y_axis_scale = st.session_state.y_axis_scale
                    intensity_filter = st.session_state.intensity_filter
                    num_annotate = st.session_state.num_annotate

                    multi_component_presets = {
                        "Cu(Ka1+Ka2)": {
                            "wavelengths": [0.15406, 0.15444],
                            "factors": [1.0, 1 / 2.0]
                        },
                        "Cu(Ka1+Ka2+Kb1)": {
                            "wavelengths": [0.15406, 0.15444, 0.13922],
                            "factors": [1.0, 1 / 2.0, 1 / 9.0]
                        },
                        "Mo(Ka1+Ka2)": {
                            "wavelengths": [0.07093, 0.0711],
                            "factors": [1.0, 1 / 2.0]
                        },
                        "Mo(Ka1+Ka2+Kb1)": {
                            "wavelengths": [0.07093, 0.0711, 0.064],
                            "factors": [1.0, 1 / 2.0, 1 / 9.0]
                        },
                        "Cr(Ka1+Ka2)": {
                            "wavelengths": [0.22897, 0.22888],
                            "factors": [1.0, 1 / 2.0]
                        },
                        "Cr(Ka1+Ka2+Kb1)": {
                            "wavelengths": [0.22897, 0.22888, 0.208],
                            "factors": [1.0, 1 / 2.0, 1 / 9.0]
                        },
                        "Fe(Ka1+Ka2)": {
                            "wavelengths": [0.19360, 0.194],
                            "factors": [1.0, 1 / 2.0]
                        },
                        "Fe(Ka1+Ka2+Kb1)": {
                            "wavelengths": [0.19360, 0.194, 0.176],
                            "factors": [1.0, 1 / 2.0, 1 / 9.0]
                        },
                        "Co(Ka1+Ka2)": {
                            "wavelengths": [0.17889, 0.17927],
                            "factors": [1.0, 1 / 2.0]
                        },
                        "Co(Ka1+Ka2+Kb1)": {
                            "wavelengths": [0.17889, 0.17927, 0.163],
                            "factors": [1.0, 1 / 2.0, 1 / 9.0]
                        },
                        "Ag(Ka1+Ka2)": {
                            "wavelengths": [0.0561, 0.05634],
                            "factors": [1.0, 1 / 2.0]
                        },
                        "Ag(Ka1+Ka2+Kb1)": {
                            "wavelengths": [0.0561, 0.05634, 0.0496],
                            "factors": [1.0, 1 / 2.0, 1 / 9.0]
                        }
                    }

                    is_multi_component = preset_choice in multi_component_presets
                    if is_multi_component:
                        comp_info = multi_component_presets[preset_choice]
                        if "labels" not in comp_info:
                            n = len(comp_info["wavelengths"])
                            if n == 2:
                                comp_info["labels"] = ["Kα1", "Kα2"]
                            elif n == 3:
                                comp_info["labels"] = ["Kα1", "Kα2", "Kβ"]
                            else:
                                comp_info["labels"] = ["Kα1"] * n

                    colors = plt.cm.tab10.colors
                    pattern_details = {}
                    full_range = (0.01, 179.9)


                    #  Converting .xyz, .lmp, .poscar to .cif file otherwise I am getting wrong (h k l) indecis for hexagonal structures
                    def convert_to_cif_then_load(file):
                        from pymatgen.io.cif import CifWriter
                        import tempfile
                        import os
                        structure = load_structure(file)

                        cif_writer = CifWriter(structure, symprec=0.01)

                        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as temp_cif:
                            cif_writer.write_file(temp_cif.name)
                            temp_cif_path = temp_cif.name

                        try:
                            cif_structure = PmgStructure.from_file(temp_cif_path)
                            return cif_structure
                        finally:
                            if os.path.exists(temp_cif_path):
                                os.remove(temp_cif_path)


                    for idx, file in enumerate(uploaded_files):
                        if file.name.lower().endswith('.cif'):
                            mg_structure = load_structure(file)
                        else:
                            mg_structure = convert_to_cif_then_load(file)
                        mg_structure = get_full_conventional_structure_diffra(mg_structure)

                        debye_waller_dict = None
                        if use_debye_waller and "debye_waller_factors_per_file" in st.session_state:
                            file_key = file.name
                            if file_key in st.session_state.debye_waller_factors_per_file:
                                debye_waller_dict = st.session_state.debye_waller_factors_per_file[file_key]

                        if is_multi_component:
                            num_points = 2000  # 20000
                            x_dense_full = np.linspace(full_range[0], full_range[1], num_points)
                            dx = x_dense_full[1] - x_dense_full[0]
                            y_dense_total = np.zeros_like(x_dense_full)
                            all_filtered_x = []
                            all_filtered_y = []
                            all_filtered_hkls = []
                            all_peak_types = []
                            comp_info = multi_component_presets[preset_choice]
                            for comp_index, (wl, factor) in enumerate(
                                    zip(comp_info["wavelengths"], comp_info["factors"])):
                                wavelength_A_comp = wl * 10  # convert nm to Å
                                if diffraction_choice == "ND (Neutron)":
                                    diff_calc = NDCalculator(wavelength=wavelength_A_comp,
                                                             debye_waller_factors=debye_waller_dict)
                                else:
                                    diff_calc = XRDCalculator(wavelength=wavelength_A_comp,
                                                              debye_waller_factors=debye_waller_dict)
                                diff_pattern = diff_calc.get_pattern(mg_structure,
                                                                     two_theta_range=user_calculation_range,
                                                                     scaled=False)

                                filtered_x = []
                                filtered_y = []
                                filtered_hkls = []
                                max_intensity = np.max(diff_pattern.y) if len(diff_pattern.y) > 0 else 1.0
                                intensity_threshold = (
                                                              intensity_filter / 100.0) * max_intensity if intensity_filter > 0 else 0

                                for x_val, y_val, hkl_group in zip(diff_pattern.x, diff_pattern.y, diff_pattern.hkls):
                                    if any(len(h['hkl']) == 3 and tuple(h['hkl'][:3]) == (0, 0, 0) for h in hkl_group):
                                        continue
                                    if any(len(h['hkl']) == 4 and tuple(h['hkl'][:4]) == (0, 0, 0, 0) for h in
                                           hkl_group):
                                        continue

                                    if intensity_filter > 0 and y_val < intensity_threshold:
                                        continue
                                    filtered_x.append(x_val)
                                    filtered_y.append(y_val * factor)  # scale intensity
                                    filtered_hkls.append(hkl_group)
                                    all_peak_types.append(comp_info["labels"][comp_index])
                                y_dense_comp = np.zeros_like(x_dense_full)
                                if peak_representation == "Gaussian":
                                    for peak, intensity in zip(filtered_x, filtered_y):
                                        gauss = np.exp(-((x_dense_full - peak) ** 2) / (2 * sigma ** 2))
                                        area = np.sum(gauss) * dx
                                        y_dense_comp += (intensity / area) * gauss
                                else:
                                    for peak, intensity in zip(filtered_x, filtered_y):
                                        idx_closest = np.argmin(np.abs(x_dense_full - peak))
                                        y_dense_comp[idx_closest] += intensity
                                if y_axis_scale != "Linear":
                                    y_dense_comp = convert_intensity_scale(y_dense_comp, y_axis_scale)
                                if y_axis_scale != "Linear":
                                    filtered_y = convert_intensity_scale(filtered_y, y_axis_scale)

                                y_dense_total += y_dense_comp
                                # if y_axis_scale != "Linear":
                                #    y_dense_total = convert_intensity_scale(y_dense_total, y_axis_scale)
                                all_filtered_x.extend(filtered_x)
                                all_filtered_y.extend(filtered_y)
                                # if y_axis_scale != "Linear":
                                #    for i in range(len(all_filtered_y)):
                                #        all_filtered_y[i] = convert_intensity_scale(np.array([all_filtered_y[i]]), y_axis_scale)[0]

                                all_filtered_hkls.extend(filtered_hkls)
                        else:
                            if diffraction_choice == "ND (Neutron)":
                                diff_calc = NDCalculator(wavelength=wavelength_A,
                                                         debye_waller_factors=debye_waller_dict)
                            else:
                                diff_calc = XRDCalculator(wavelength=wavelength_A,
                                                          debye_waller_factors=debye_waller_dict)
                            try:
                                diff_pattern = diff_calc.get_pattern(mg_structure,
                                                                     two_theta_range=user_calculation_range,
                                                                     scaled=False)
                            except Exception as e:
                                st.write("No peaks available for this wavelenght.")
                                st.stop()
                            filtered_x = []
                            filtered_y = []
                            filtered_hkls = []

                            max_intensity = np.max(diff_pattern.y) if len(diff_pattern.y) > 0 else 1.0
                            intensity_threshold = (
                                                              intensity_filter / 100.0) * max_intensity if intensity_filter > 0 else 0
                            for x_val, y_val, hkl_group in zip(diff_pattern.x, diff_pattern.y, diff_pattern.hkls):
                                if any(len(h['hkl']) == 3 and tuple(h['hkl'][:3]) == (0, 0, 0) for h in hkl_group):
                                    continue
                                if any(len(h['hkl']) == 4 and tuple(h['hkl'][:4]) == (0, 0, 0, 0) for h in hkl_group):
                                    continue
                                if intensity_filter > 0 and y_val < intensity_threshold:
                                    continue
                                filtered_x.append(x_val)
                                filtered_y.append(y_val)
                                filtered_hkls.append(hkl_group)
                            num_points = 2000  # 20000
                            x_dense_full = np.linspace(full_range[0], full_range[1], num_points)
                            dx = x_dense_full[1] - x_dense_full[0]
                            y_dense_total = np.zeros_like(x_dense_full)
                            if peak_representation == "Gaussian":
                                for peak, intensity in zip(filtered_x, filtered_y):
                                    gauss = np.exp(-((x_dense_full - peak) ** 2) / (2 * sigma ** 2))
                                    area = np.sum(gauss) * dx
                                    y_dense_total += (intensity / area) * gauss
                            else:
                                for peak, intensity in zip(filtered_x, filtered_y):
                                    idx_closest = np.argmin(np.abs(x_dense_full - peak))
                                    y_dense_total[idx_closest] += intensity
                            if y_axis_scale != "Linear":
                                # Convert the dense y values (continuous curve)
                                y_dense_total = convert_intensity_scale(y_dense_total, y_axis_scale)
                            all_filtered_x = filtered_x
                            all_filtered_y = filtered_y
                            if y_axis_scale != "Linear":
                                for i in range(len(all_filtered_y)):
                                    all_filtered_y[i] = \
                                        convert_intensity_scale(np.array([all_filtered_y[i]]), y_axis_scale)[0]

                            all_filtered_hkls = filtered_hkls
                            all_peak_types = ["Kα1"] * len(filtered_x)

                        if intensity_scale_option == "Normalized":
                            norm_factor = np.max(all_filtered_y) if np.max(all_filtered_y) > 0 else 1.0
                            y_dense_total = (y_dense_total / np.max(y_dense_total)) * 100
                            displayed_intensity_array = (np.array(all_filtered_y) / norm_factor) * 100
                        else:
                            displayed_intensity_array = np.array(all_filtered_y)

                        peak_vals = twotheta_to_metric(np.array(all_filtered_x), x_axis_metric, wavelength_A,
                                                       wavelength_nm,
                                                       diffraction_choice)
                        ka1_indices = [i for i, pt in enumerate(all_peak_types) if pt == "Kα1"]
                        ka1_intensities = [displayed_intensity_array[i] for i in ka1_indices]
                        if ka1_intensities:
                            sorted_ka1 = sorted(zip(ka1_indices, ka1_intensities), key=lambda x: x[1], reverse=True)
                            annotate_indices = set(i for i, _ in sorted_ka1[:num_annotate])
                        else:
                            annotate_indices = set()
                        pattern_details[file.name] = {
                            "peak_vals": peak_vals,
                            "intensities": displayed_intensity_array,
                            "hkls": all_filtered_hkls,
                            "peak_types": all_peak_types,
                            "annotate_indices": annotate_indices,
                            "x_dense_full": x_dense_full,
                            "y_dense": y_dense_total
                        }

                    if peak_representation != "Delta":
                        if preset_choice in multi_component_presets:
                            st.sidebar.subheader("Include Kα1 or Kα2/Kβ for hovering:")
                            num_components = len(multi_component_presets[preset_choice]["wavelengths"])
                            if num_components > 1:
                                show_Kalpha1_hover = st.sidebar.checkbox("Include Kα1 hover", value=True)
                            if num_components >= 2:
                                show_Kalpha2_hover = st.sidebar.checkbox("Include Kα2 hover", value=False)
                            if num_components >= 3:
                                show_Kbeta_hover = st.sidebar.checkbox("Include Kβ hover", value=False)
                        else:
                            st.sidebar.subheader("Include Kα1 for hovering:")
                            show_Kalpha1_hover = st.sidebar.checkbox("Include Kα1 hover", value=True)

                    for idx, (file_name, details) in enumerate(pattern_details.items()):

                        base_color = rgb_color(colors[idx % len(colors)], opacity=0.8)
                        mask = (details["x_dense_full"] >= st.session_state.two_theta_min) & (
                                details["x_dense_full"] <= st.session_state.two_theta_max)
                        x_dense_range = twotheta_to_metric(details["x_dense_full"][mask],
                                                           x_axis_metric, wavelength_A, wavelength_nm,
                                                           diffraction_choice)
                        y_dense_range = details["y_dense"][mask]

                        if peak_representation == "Delta":
                            if "peak_types" in details:
                                groups = {}
                                for i, peak in enumerate(details["peak_vals"]):
                                    canonical = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm,
                                                                   diffraction_choice)
                                    if st.session_state.two_theta_min <= canonical <= st.session_state.two_theta_max:
                                        pt = details["peak_types"][i]
                                        groups.setdefault(pt, {"x": [], "y": [], "hover": []})
                                        groups[pt]["x"].append(details["peak_vals"][i])
                                        groups[pt]["y"].append(details["intensities"][i])
                                        hkl_group = details["hkls"][i]
                                        if len(hkl_group[0]['hkl']) == 3:
                                            hkl_str = ", ".join([
                                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                                for h in hkl_group])
                                        else:
                                            hkl_str = ", ".join([
                                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                                for h in hkl_group])
                                        groups[pt]["hover"].append(f"(hkl): {hkl_str}")
                                for pt, data in groups.items():
                                    if pt == "Kα1":
                                        pt_color = base_color
                                        dash_type = "solid"
                                        hover_info = "text"
                                        hover_template = f"<br>{file_name} - {pt}<br><b>{x_axis_metric}: %{{x:.2f}}</b><br>Intensity: %{{y:.2f}}<br><b>%{{text}}</b><extra></extra>"
                                    elif pt == "Kα2":
                                        pt_color = rgb_color(colors[idx % len(colors)], opacity=0.6)
                                        dash_type = "dot"
                                        hover_info = "skip"
                                        hover_template = None
                                    elif pt == "Kβ":
                                        pt_color = rgb_color(colors[idx % len(colors)], opacity=0.4)
                                        dash_type = "dash"
                                        hover_info = "skip"
                                        hover_template = None
                                    else:
                                        pt_color = base_color
                                        dash_type = "solid"
                                        hover_info = "text"
                                        hover_template = f"<br>{file_name} - {pt}<br><b>{x_axis_metric}: %{{x:.2f}}</b><br>Intensity: %{{y:.2f}}<br><b>%{{text}}</b><extra></extra>"

                                    vertical_x = []
                                    vertical_y = []
                                    vertical_hover = []
                                    for j in range(len(data["x"])):
                                        vertical_x.extend([data["x"][j], data["x"][j], None])
                                        vertical_y.extend([0, data["y"][j], None])
                                        vertical_hover.extend([data["hover"][j], data["hover"][j], None])
                                    fig_interactive.add_trace(go.Scatter(
                                        x=vertical_x,
                                        y=vertical_y,
                                        mode='lines',
                                        name=f"{file_name} - {pt}",
                                        showlegend=True,
                                        line=dict(color=pt_color, width=line_thickness, dash=dash_type),
                                        hoverinfo=hover_info,
                                        text=vertical_hover,
                                        hovertemplate=hover_template,
                                        hoverlabel=dict(bgcolor=pt_color, font=dict(color="white", size=24))
                                    ))
                            else:
                                vertical_x = []
                                vertical_y = []
                                vertical_hover = []
                                for i, peak in enumerate(details["peak_vals"]):
                                    canonical = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm,
                                                                   diffraction_choice)
                                    if st.session_state.two_theta_min <= canonical <= st.session_state.two_theta_max:
                                        vertical_x.extend([peak, peak, None])
                                        vertical_y.extend([0, details["intensities"][i], None])
                                        hkl_group = details["hkls"][i]
                                        if len(hkl_group[0]['hkl']) == 3:
                                            hkl_str = ", ".join([
                                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                                for h in hkl_group])
                                        else:
                                            hkl_str = ", ".join([
                                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                                for h in hkl_group])
                                        vertical_hover.extend([f"(hkl): {hkl_str}", f"(hkl): {hkl_str}", None])
                                fig_interactive.add_trace(go.Scatter(
                                    x=vertical_x,
                                    y=vertical_y,
                                    mode='lines',
                                    name=file_name,
                                    showlegend=True,
                                    line=dict(color=base_color, width=line_thickness, dash="solid"),
                                    hoverinfo="text",
                                    text=vertical_hover,
                                    hovertemplate=f"<br>{file_name}<br><b>{x_axis_metric}: %{{x:.2f}}</b><br>Intensity: %{{y:.2f}}<br><b>%{{text}}</b><extra></extra>",
                                    hoverlabel=dict(bgcolor=base_color, font=dict(color="white", size=24))
                                ))
                        else:
                            fig_interactive.add_trace(go.Scatter(
                                x=x_dense_range,
                                y=y_dense_range,
                                mode='lines',
                                name=file_name,
                                line=dict(color=base_color, width=line_thickness),
                                hoverinfo='skip'
                            ))
                            peak_vals_in_range = []
                            intensities_in_range = []
                            peak_hover_texts = []
                            gaussian_max_intensities = []
                            for i, peak in enumerate(details["peak_vals"]):
                                peak_type = details["peak_types"][i]
                                if (peak_type == "Kα1" and not show_Kalpha1_hover) or (
                                        peak_type == "Kα2" and not show_Kalpha2_hover) or (
                                        peak_type == "Kβ" and not show_Kbeta_hover):
                                    continue
                                canonical = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm,
                                                               diffraction_choice)
                                if st.session_state.two_theta_min <= canonical <= st.session_state.two_theta_max:
                                    peak_vals_in_range.append(peak)
                                    gauss = np.exp(-((details["x_dense_full"] - peak) ** 2) / (2 * sigma ** 2))
                                    area = np.sum(gauss) * dx
                                    scaled_gauss = (details["intensities"][i] / area) * gauss
                                    max_gauss = np.max(scaled_gauss)
                                    gaussian_max_intensities.append(max_gauss)
                                    hkl_group = details["hkls"][i]
                                    if len(hkl_group[0]['hkl']) == 3:
                                        hkl_str = ", ".join(
                                            [
                                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                                for h in hkl_group])
                                    else:
                                        hkl_str = ", ".join(
                                            [
                                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                                for h in hkl_group])
                                    if peak_type == "Kα1":
                                        hover_text = f"Kα1 (hkl): {hkl_str}"
                                    elif peak_type == "Kα2":
                                        hover_text = f"Kα2 (hkl): {hkl_str}"
                                    elif peak_type == "Kβ":
                                        hover_text = f"Kβ (hkl): {hkl_str}"
                                    else:
                                        hover_text = f"Kα1 (hkl): {hkl_str}"
                                    peak_hover_texts.append(hover_text)
                            if intensity_scale_option == "Normalized" and gaussian_max_intensities:
                                norm_marker = max(gaussian_max_intensities)
                                gaussian_max_intensities = [val / norm_marker * 100 for val in gaussian_max_intensities]
                            fig_interactive.add_trace(go.Scatter(
                                x=peak_vals_in_range,
                                y=gaussian_max_intensities,
                                mode='markers',
                                name=file_name,
                                showlegend=True,
                                marker=dict(color=base_color, size=8, opacity=0.5),
                                text=peak_hover_texts,
                                hovertemplate=f"<br>{file_name}<br><b>{x_axis_metric}: %{{x:.2f}}</b><br>Intensity: %{{y:.2f}}<br><b>%{{text}}</b><extra></extra>",
                                hoverlabel=dict(bgcolor=base_color, font=dict(color="white", size=20))
                            ))

                    display_metric_min = twotheta_to_metric(st.session_state.two_theta_min, x_axis_metric, wavelength_A,
                                                            wavelength_nm, diffraction_choice)
                    display_metric_max = twotheta_to_metric(st.session_state.two_theta_max, x_axis_metric, wavelength_A,
                                                            wavelength_nm, diffraction_choice)
                    colors = ["black", "brown", "grey", "purple"]
                    if user_pattern_file:
                        # Initialize parsed data cache if not already done
                        if "parsed_exp_data" not in st.session_state:
                            st.session_state.parsed_exp_data = {}

                        if isinstance(user_pattern_file, list):
                            for i, file in enumerate(user_pattern_file):
                                file_name = file.name

                                if "permanent_exp_data" in st.session_state and file_name in st.session_state.permanent_exp_data:
                                    x_user = st.session_state.permanent_exp_data[file_name]["x"]
                                    y_user = st.session_state.permanent_exp_data[file_name]["y"]
                                    display_name = file_name + " (BG removed)"

                                elif ("use_bg_subtracted" in st.session_state and st.session_state.use_bg_subtracted and
                                      "active_bg_subtracted_file" in st.session_state and
                                      st.session_state.active_bg_subtracted_file == file_name and
                                      "bg_subtracted_data" in st.session_state and
                                      file_name in st.session_state.bg_subtracted_data):

                                    x_user = st.session_state.bg_subtracted_data[file_name]["x"]
                                    y_user = st.session_state.bg_subtracted_data[file_name]["y"]
                                    display_name = file_name + " (temp BG removed)"
                                else:

                                    file.seek(0)
                                    file_contents = file.read()

                                    if isinstance(file_contents, bytes):
                                        file_contents = file_contents.decode('utf-8')

                                    from io import StringIO

                                    data_io = StringIO(file_contents)

                                    df = pd.read_csv(data_io, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                                    x_user = df.iloc[:, 0].values
                                    y_user = df.iloc[:, 1].values
                                    display_name = file_name

                                    file.seek(0)

                                if y_axis_scale != "Linear":
                                    y_user = convert_intensity_scale(y_user, y_axis_scale)
                                if intensity_scale_option == "Normalized" and np.max(y_user) > 0:
                                    y_user = (y_user / np.max(y_user)) * 100

                                mask_user = (x_user >= st.session_state.two_theta_min) & (
                                        x_user <= st.session_state.two_theta_max)
                                x_user_filtered = x_user[mask_user]
                                y_user_filtered = y_user[mask_user]

                                color = colors[i % len(colors)]
                                fig_interactive.add_trace(go.Scatter(
                                    x=x_user_filtered,
                                    y=y_user_filtered,
                                    mode="lines",
                                    name=file.name + (" (BG subtracted)" if ("use_bg_subtracted" in st.session_state and
                                                                             st.session_state.use_bg_subtracted and
                                                                             "active_bg_subtracted_file" in st.session_state and
                                                                             st.session_state.active_bg_subtracted_file == file_name) else ""),
                                    line=dict(dash='solid', width=1, color=color),
                                    marker=dict(color=color, size=3),
                                    hovertemplate=(
                                        f"<span style='color:{color};'><b>{file.name}:</b><br>"
                                        "2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
                                    )
                                ))
                                fig_interactive.update_layout(
                                    height=800,
                                    margin=dict(t=80, b=80, l=60, r=30),
                                    hovermode="x",
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.2,
                                        xanchor="center",
                                        x=0.5,
                                        font=dict(size=24)
                                    ),
                                    xaxis=dict(
                                        title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
                                        tickfont=dict(size=36, color='black')
                                    ),
                                    yaxis=dict(
                                        title=dict(text=y_axis_title, font=dict(size=36, color='black')),
                                        tickfont=dict(size=36, color='black')
                                    ),
                                    hoverlabel=dict(font=dict(size=24)),
                                    font=dict(size=18),
                                    autosize=True
                                )
                        else:
                            file_name = user_pattern_file.name
                            try:

                                if "permanent_exp_data" in st.session_state and file_name in st.session_state.permanent_exp_data:
                                    x_user = st.session_state.permanent_exp_data[file_name]["x"]
                                    y_user = st.session_state.permanent_exp_data[file_name]["y"]
                                    display_name = file_name + " (BG removed)"

                                elif ("use_bg_subtracted" in st.session_state and st.session_state.use_bg_subtracted and
                                      "active_bg_subtracted_file" in st.session_state and
                                      st.session_state.active_bg_subtracted_file == file_name and
                                      "bg_subtracted_data" in st.session_state and
                                      file_name in st.session_state.bg_subtracted_data):

                                    x_user = st.session_state.bg_subtracted_data[file_name]["x"]
                                    y_user = st.session_state.bg_subtracted_data[file_name]["y"]
                                    display_name = file_name + " (temp BG removed)"
                                else:
                                    user_pattern_file.seek(0)
                                    file_contents = user_pattern_file.read()

                                    if isinstance(file_contents, bytes):
                                        file_contents = file_contents.decode('utf-8')

                                    from io import StringIO

                                    data_io = StringIO(file_contents)

                                    df = pd.read_csv(data_io, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                                    x_user = df.iloc[:, 0].values
                                    y_user = df.iloc[:, 1].values
                                    display_name = file_name

                                    user_pattern_file.seek(0)

                                if x_user is not None and y_user is not None:
                                    if y_axis_scale != "Linear":
                                        y_user = convert_intensity_scale(y_user, y_axis_scale)
                                    if intensity_scale_option == "Normalized" and np.max(y_user) > 0:
                                        y_user = (y_user / np.max(y_user)) * 100

                                    mask_user = (x_user >= st.session_state.two_theta_min) & (
                                            x_user <= st.session_state.two_theta_max)
                                    x_user_filtered = x_user[mask_user]
                                    y_user_filtered = y_user[mask_user]

                                    color = "black"  # Default color for single file
                                    fig_interactive.add_trace(go.Scatter(
                                        x=x_user_filtered,
                                        y=y_user_filtered,
                                        mode="lines",
                                        name=display_name,
                                        line=dict(dash='solid', width=1, color=color),
                                        marker=dict(color=color, size=3),
                                        hovertemplate=(
                                            f"<span style='color:{color};'><b>{display_name}:</b><br>"
                                            "2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
                                        )
                                    ))
                            except Exception as e:
                                st.error(f"Error processing file {file_name}: {e}")
                                x_user, y_user = None, None

                        fig_interactive.update_layout(
                            xaxis_title="2θ (°)",
                            yaxis_title="Intensity",
                            autosize=True,
                            height=500
                        )
                    if x_axis_metric in ["d (Å)", "d (nm)"]:
                        fig_interactive.update_layout(xaxis=dict(range=[display_metric_max, display_metric_min]))
                    else:
                        fig_interactive.update_layout(xaxis=dict(range=[display_metric_min, display_metric_max]))

                    if peak_representation == "Delta" and intensity_scale_option != "Absolute":
                        fig_interactive.update_layout(
                            height=800,
                            margin=dict(t=80, b=80, l=60, r=30),
                            hovermode="x",
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=24)
                            ),
                            xaxis=dict(
                                title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
                                tickfont=dict(size=36, color='black')
                            ),
                            yaxis=dict(
                                title=dict(text=y_axis_title, font=dict(size=36, color='black')),
                                tickfont=dict(size=36, color='black'), range=[0, 125]
                            ),
                            hoverlabel=dict(font=dict(size=24)),
                            font=dict(size=18),
                            autosize=True
                        )
                    else:
                        fig_interactive.update_layout(
                            height=1000,
                            margin=dict(t=80, b=80, l=60, r=30),
                            hovermode="x",
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=24)
                            ),
                            xaxis=dict(
                                title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
                                tickfont=dict(size=36, color='black')
                            ),
                            yaxis=dict(
                                title=dict(text=y_axis_title, font=dict(size=36, color='black')),
                                tickfont=dict(size=36, color='black')
                            ),
                            hoverlabel=dict(font=dict(size=24)),
                            font=dict(size=18),
                            autosize=True
                        )
        # st.rerun()
        with tab1:
            st.session_state.placeholder_interactive = st.empty()
            st.session_state.fig_interactive = fig_interactive
            st.session_state.placeholder_interactive.plotly_chart(st.session_state.fig_interactive,
                                                                  use_container_width=True)
            st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)


            def find_equivalent_hkls(hkl_base, space_group, crystal_system):
                import numpy as np

                equivalent_hkls = set()
                h, k, l = hkl_base

                if space_group:
                    try:
                        for symm_op in space_group:
                            rotation_matrix = symm_op.rotation_matrix

                            if crystal_system in ['hexagonal', 'trigonal']:
                                i = -(h + k)
                                hkl_3d_for_rotation = np.array([h, k, l])
                                equivalent_hkl_3d = rotation_matrix.dot(hkl_3d_for_rotation).astype(int)
                                h_eq, k_eq, l_eq = equivalent_hkl_3d
                                i_eq = -(h_eq + k_eq)
                                equivalent_hkl = (h_eq, k_eq, i_eq, l_eq)
                            else:
                                hkl_3d = np.array([h, k, l])
                                equivalent_hkl_3d = rotation_matrix.dot(hkl_3d).astype(int)
                                equivalent_hkl = tuple(equivalent_hkl_3d)

                            equivalent_hkls.add(equivalent_hkl)
                            equivalent_hkls.add(tuple(-x for x in equivalent_hkl))

                    except Exception as e:
                        st.warning(f"Error in symmetry operations: {e}")

                if crystal_system == 'tetragonal':
                    equivalent_hkls.add((h, k, l))
                    equivalent_hkls.add((k, h, l))
                    equivalent_hkls.add((-h, -k, l))
                    equivalent_hkls.add((-k, -h, l))
                    equivalent_hkls.add((h, -k, -l))
                    equivalent_hkls.add((k, -h, -l))
                    equivalent_hkls.add((-h, k, -l))
                    equivalent_hkls.add((-k, h, -l))
                elif crystal_system == 'cubic':
                    for perm_h, perm_k, perm_l in [(h, k, l), (h, l, k), (k, h, l), (k, l, h), (l, h, k), (l, k, h)]:
                        for sign_h, sign_k, sign_l in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, -1, -1),
                                                       (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
                            equivalent_hkls.add((sign_h * perm_h, sign_k * perm_k, sign_l * perm_l))
                elif crystal_system == 'orthorhombic':
                    for sign_h, sign_k, sign_l in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, -1, -1),
                                                   (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
                        equivalent_hkls.add((sign_h * h, sign_k * k, sign_l * l))
                elif crystal_system in ['hexagonal', 'trigonal']:
                    pass
                else:
                    equivalent_hkls.add((h, k, l))
                    equivalent_hkls.add((-h, -k, -l))

                return list(equivalent_hkls)


            def find_plane_multiples_with_symmetry(hkl_base, max_multiple, space_group, crystal_system):
                equivalent_base = find_equivalent_hkls(hkl_base, space_group, crystal_system)
                all_multiples = set()

                for equiv_hkl in equivalent_base:
                    if len(equiv_hkl) == 4:
                        h, k, i, l = equiv_hkl
                        for n in range(1, max_multiple + 1):
                            all_multiples.add((h * n, k * n, i * n, l * n))
                            all_multiples.add((-h * n, -k * n, -i * n, -l * n))
                    else:
                        h, k, l = equiv_hkl[:3]
                        for n in range(1, max_multiple + 1):
                            all_multiples.add((h * n, k * n, l * n))
                            all_multiples.add((-h * n, -k * n, -l * n))

                return list(all_multiples)


            def get_structure_space_group(structure):
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                try:
                    sga = SpacegroupAnalyzer(structure)
                    crystal_system = sga.get_crystal_system()
                    return sga.get_space_group_symbol(), sga.get_space_group_operations(), crystal_system
                except:
                    return "Unknown", None, "unknown"


            if st.session_state.calc_xrd and uploaded_files and 'pattern_details' in locals():
                with st.expander("🏷️ Annotate Specific Planes", expanded=True):

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        h_index = st.number_input("h index", min_value=-10, max_value=10, value=0, step=1,
                                                  key="h_plane")
                    with col2:
                        k_index = st.number_input("k index", min_value=-10, max_value=10, value=0, step=1,
                                                  key="k_plane")
                    with col3:
                        l_index = st.number_input("l index", min_value=-10, max_value=10, value=1, step=1,
                                                  key="l_plane")
                    with col4:
                        max_multiple = st.number_input("Max multiple", min_value=1, max_value=20, value=5, step=1,
                                                       key="max_mult")

                    plane_hkl = (h_index, k_index, l_index)

                    if st.button("Generate Annotated Plot", type="primary"):
                        with st.spinner("Analyzing space groups and generating annotations..."):

                            structure_space_groups = {}
                            for file in uploaded_files:
                                try:
                                    structure = load_structure(file)
                                    sg_symbol, sg_ops, crystal_system = get_structure_space_group(structure)
                                    structure_space_groups[file.name] = {
                                        "symbol": sg_symbol,
                                        "operations": sg_ops,
                                        "crystal_system": crystal_system,
                                        "structure": structure
                                    }
                                except Exception as e:
                                    st.warning(f"Could not determine space group for {file.name}: {e}")
                                    structure_space_groups[file.name] = {
                                        "symbol": "Unknown",
                                        "operations": None,
                                        "crystal_system": "unknown",
                                        "structure": None
                                    }

                            st.write("**Space Groups of loaded structures:**")
                            for file_name, sg_info in structure_space_groups.items():
                                crystal_sys = sg_info.get('crystal_system', 'unknown')
                                st.write(f"• {file_name}: {sg_info['symbol']} ({crystal_sys})")

                            fig_annotated = go.Figure()

                            for trace in st.session_state.fig_interactive.data:
                                fig_annotated.add_trace(trace)

                            total_annotations = 0

                            for file_name, details in pattern_details.items():
                                peak_vals = details["peak_vals"]
                                intensities = details["intensities"]
                                hkls = details["hkls"]

                                sg_info = structure_space_groups.get(file_name, {})
                                space_group = sg_info.get("operations")
                                crystal_system = sg_info.get("crystal_system", "unknown")

                                multiples = find_plane_multiples_with_symmetry(plane_hkl, max_multiple, space_group,
                                                                               crystal_system)

                                annotated_x = []
                                annotated_y = []
                                annotated_text = []

                                for i, (peak_val, intensity, hkl_group) in enumerate(zip(peak_vals, intensities, hkls)):
                                    for hkl_dict in hkl_group:
                                        hkl = hkl_dict['hkl']
                                        if len(hkl) >= 3:
                                            if crystal_system in ['hexagonal', 'trigonal'] and len(hkl) == 4:
                                                hkl_tuple = tuple(hkl[:4])
                                            else:
                                                hkl_tuple = tuple(hkl[:3])

                                            if hkl_tuple in multiples:
                                                canonical_2theta = metric_to_twotheta(peak_val, x_axis_metric,
                                                                                      wavelength_A, wavelength_nm,
                                                                                      diffraction_choice)
                                                if st.session_state.two_theta_min <= canonical_2theta <= st.session_state.two_theta_max:
                                                    annotated_x.append(peak_val)
                                                    annotated_y.append(intensity)
                                                    if len(hkl) == 4:
                                                        annotated_text.append(f"({hkl[0]} {hkl[1]} {hkl[2]} {hkl[3]})")
                                                    else:
                                                        annotated_text.append(f"({hkl[0]} {hkl[1]} {hkl[2]})")
                                                    total_annotations += 1

                                if annotated_x:
                                    fig_annotated.add_trace(go.Scatter(
                                        x=annotated_x,
                                        y=annotated_y,
                                        mode='markers+text',
                                        text=annotated_text,
                                        textposition="top center",
                                        textfont=dict(size=20, color="red", family="Arial Black"),
                                        marker=dict(size=12, color="red", symbol="diamond",
                                                    line=dict(width=2, color="darkred")),
                                        name=f"{file_name} - {plane_hkl} family ({sg_info.get('symbol', 'Unknown')})",
                                        showlegend=True
                                    ))

                            fig_annotated.update_layout(st.session_state.fig_interactive.layout)
                            fig_annotated.update_layout(
                                title=f"Diffraction Pattern with {plane_hkl} Family Annotations (Symmetry-Aware)",
                                plot_bgcolor='white'
                            )

                            annotation_traces = []
                            other_traces = []
                            for trace in fig_annotated.data:
                                if 'family' in trace.name:
                                    annotation_traces.append(trace)
                                else:
                                    other_traces.append(trace)

                            fig_annotated.data = other_traces + annotation_traces

                            st.plotly_chart(fig_annotated, use_container_width=True, key="annotated_plot")
                            st.success(
                                f"Generated annotated plot for plane family {plane_hkl} with {total_annotations} annotations considering space group symmetry")

        if pattern_details is not None:
            with tab2:
                st.subheader("Quantitative Data for Calculated Diffraction Patterns")
                for file in uploaded_files:
                    details = pattern_details[file.name]
                    peak_vals = details["peak_vals"]
                    intensities = details["intensities"]
                    hkls = details["hkls"]
                    annotate_indices = details["annotate_indices"]
                    x_dense_full = details["x_dense_full"]
                    y_dense = details["y_dense"]
                    with st.expander(f"View Peak Data for Diffraction Pattern: **{file.name}**"):
                        table_str = "#X-axis    Intensity    hkl\n"
                        for theta, intensity, hkl_group in zip(peak_vals, intensities, hkls):
                            if len(hkl_group[0]['hkl']) == 3:
                                hkl_str = ", ".join(
                                    [
                                        f"({h['hkl'][0]} {h['hkl'][1]} {h['hkl'][2]})"
                                        for h in hkl_group
                                    ])
                            else:
                                hkl_str = ", ".join(
                                    [
                                        f"({h['hkl'][0]} {h['hkl'][1]} {h['hkl'][3]})"
                                        for h in hkl_group
                                    ])
                            table_str += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
                        st.code(table_str, language="text")
                    with st.expander(f"View Highest Intensity Peaks for Diffraction Pattern: **{file.name}**",
                                     expanded=True):
                        table_str2 = "#X-axis    Intensity    hkl\n"
                        for i, (theta, intensity, hkl_group) in enumerate(zip(peak_vals, intensities, hkls)):
                            if i in annotate_indices:
                                if len(hkl_group[0]['hkl']) == 3:
                                    hkl_str = ", ".join(
                                        [
                                            f"({h['hkl'][0]} {h['hkl'][1]} {h['hkl'][2]})"
                                            for h in hkl_group
                                        ])
                                else:
                                    hkl_str = ", ".join(
                                        [
                                            f"({h['hkl'][0]} {h['hkl'][1]} {h['hkl'][3]})"
                                            for h in hkl_group
                                        ])
                                table_str2 += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
                        st.code(table_str2, language="text")

                    button_key = f"prepare_download_{file.name}"
                    if button_key not in st.session_state:
                        st.session_state[button_key] = False


                    def prepare_xrd_download(file_key):
                        st.session_state[file_key] = True


                    st.button(f"Download Continuous Curve Data for {file.name}",
                              key=f"button_{file.name}",
                              on_click=prepare_xrd_download,
                              args=(button_key,))
                    if st.session_state[button_key]:
                        import base64

                        # Prepare the data for download
                        df = pd.DataFrame({
                            "X-axis": x_dense_full,
                            "Y-value": y_dense
                        })
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        filename = f"continuous_curve_data_{file.name.replace('.', '_')}.csv"
                        download_link = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Continuous Curve Data for {file.name}</a>'
                        st.markdown(download_link, unsafe_allow_html=True)

                combined_data = {}
                for file in uploaded_files:
                    file_name = file.name
                    details = pattern_details[file_name]
                    combined_data[file_name] = {
                        "Peak Vals": details["peak_vals"],
                        "Intensities": details["intensities"],
                        "HKLs": details["hkls"]
                    }
                selected_metric = st.session_state.x_axis_metric
                st.markdown(
                    """
                    <style>
                    div[data-testid="stDataFrameContainer"] table td {
                         font-size: 22px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                view_combined = st.checkbox("📈 View peak data across all structures in an interactive table",
                                            )
                if view_combined:
                    with st.expander("📊 View Combined Peak Data Across All Structures", expanded=True):
                        combined_df = pd.DataFrame()
                        data_list = []
                        for file in uploaded_files:
                            file_name = file.name
                            if file_name in combined_data:
                                peak_vals = combined_data[file_name]["Peak Vals"]
                                intensities = combined_data[file_name]["Intensities"]
                                hkls = combined_data[file_name]["HKLs"]
                                for i in range(len(peak_vals)):
                                    for group in hkls:
                                        for item in group:
                                            hkl = item['hkl']
                                            if len(hkl) == 3 and tuple(hkl[:3]) == (0, 0, 0):
                                                continue
                                            if len(hkl) == 4 and tuple(hkl[:4]) == (0, 0, 0, 0):
                                                continue
                                    if len(hkl) == 3:
                                        hkl_str = ", ".join([
                                            f"({h['hkl'][0]} {h['hkl'][1]} {h['hkl'][2]})"
                                            for h in hkls[i]])
                                    else:
                                        hkl_str = ", ".join([
                                            f"({h['hkl'][0]} {h['hkl'][1]} {h['hkl'][3]})"
                                            for h in hkls[i]])
                                    data_list.append([peak_vals[i], intensities[i], hkl_str, file_name])
                        combined_df = pd.DataFrame(data_list,
                                                   columns=["{}".format(selected_metric), "Intensity", "(hkl)",
                                                            "Phase"])
                        st.dataframe(combined_df)

if "calc_rdf" not in st.session_state:
    st.session_state.calc_rdf = False
if "display_mode" not in st.session_state:
    st.session_state.display_mode = "Average PRDF across frames"
if "selected_frame_idx" not in st.session_state:
    st.session_state.selected_frame_idx = 0
if "frame_indices" not in st.session_state:
    st.session_state.frame_indices = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {
        "all_prdf_dict": {},
        "all_distance_dict": {},
        "global_rdf_list": [],
        "multi_structures": False
    }
if "animate" not in st.session_state:
    st.session_state.animate = False
if "do_calculation" not in st.session_state:
    st.session_state.do_calculation = False


def update_selected_frame():
    st.session_state.selected_frame_idx = st.session_state.frame_slider


def update_display_mode():
    st.session_state.display_mode = st.session_state.display_mode_radio
    st.session_state.animate = False


def trigger_calculation():
    st.session_state.calc_rdf = True
    st.session_state.do_calculation = True
    for key in ["all_prdf_dict", "all_distance_dict", "global_rdf_list"]:
        if key in st.session_state.processed_data:
            del st.session_state.processed_data[key]
    st.session_state.frame_indices = []
    st.session_state.processed_data = {
        "all_prdf_dict": {},
        "all_distance_dict": {},
        "global_rdf_list": [],
        "multi_structures": False
    }
    import gc
    gc.collect()


def toggle_animation():
    st.session_state.animate = not st.session_state.animate

# Add these function definitions after your imports
def smooth_gaussian(y_data, sigma=1.5):
    """Smooth using Gaussian filter"""
    return gaussian_filter1d(y_data, sigma=sigma)

def smooth_savgol(y_data, window_length=11, polyorder=3):
    """Smooth using Savitzky-Golay filter"""
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(y_data):
        window_length = len(y_data) if len(y_data) % 2 == 1 else len(y_data) - 1
    polyorder = min(polyorder, window_length - 1)
    return savgol_filter(y_data, window_length, polyorder)

def smooth_spline(x_data, y_data, smoothing_factor=300):
    """Smooth using cubic spline"""
    x_smooth = np.linspace(x_data[0], x_data[-1], smoothing_factor)
    spl = make_interp_spline(x_data, y_data, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, np.maximum(0, y_smooth)

# Main PRDF section
if "📊 (P)RDF" in calc_mode:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    from scipy.interpolate import make_interp_spline
    if 'uploaded_files_user_sidebar' in locals() and uploaded_files_user_sidebar:
        uploaded_files = st.session_state['uploaded_files'] + uploaded_files_user_sidebar
    else:
        uploaded_files = st.session_state['uploaded_files']
    # uploaded_files = st.session_state.uploaded_files
    # --- RDF (PRDF) Settings and Calculation ---
    st.subheader("⚙️ (P)RDF Settings",
                 help="🔬 **PRDF** describes the atomic element pair distances distribution within a structure, "
                      "providing insight into **local environments** and **structural disorder**. "
                      "It is commonly used in **diffusion studies** to track atomic movement and ion transport, "
                      "as well as in **phase transition analysis**, revealing changes in atomic ordering during melting or crystallization. "
                      "Additionally, PRDF/RDF can be employed as one of the **structural descriptors in machine learning**. "
                      "Here, the (P)RDF values are **unitless** (relative PRDF intensity). Peaks = preferred bonding distances. "
                      "Peak width = disorder. Height = relative likelihood.")

    use_lammps_traj = st.checkbox("📈 Use LAMMPS Trajectory File",
                                  help="Enable this for a LAMMPS dump trajectory file with multiple frames",
                                  disabled=True)

    st.warning(
        "⚠️ Due to the large computational demand connected with the trajectories processing, this option was put as a separated module that runs "
        "only locally. Please visit **[this site](https://github.com/bracerino/PRDF-CP2K-LAMMPS)** to see how to compile it.")

    plot_style = st.radio(
        "Plot Style",
        ["Bars (Histogram)", "Smooth Curve", "Raw Data Points"],
        index=1,
        key="plot_style",
        horizontal=True,
        help="Choose visualization style: bars show discrete bins, smooth curve applies interpolation"
    )
    if plot_style == "Bars (Histogram)":
        bar_width_factor = st.slider(
            "Bar Width Factor",
            0.1, 2.0, 0.8, 0.1,
            help="Adjust bar thickness relative to bin size (default: 0.8)"
        )
    if plot_style == "Smooth Curve":
        smoothing_method = st.radio(
            "Smoothing Method",
            ["Gaussian", "Savitzky-Golay", "Cubic Spline"],
            index=0,
            key="smoothing_method",
            horizontal=True,
            help="Gaussian: best for general RDF | Savitzky-Golay: preserves peaks | Spline: smoothest visual"
        )

        if smoothing_method == "Gaussian":
            sigma = st.slider("Gaussian Sigma", 0.5, 5.0, 1.5, 0.1,
                              help="Higher values = more smoothing")
        elif smoothing_method == "Savitzky-Golay":
            window = st.slider("Window Length", 5, 21, 11, 2,
                               help="Must be odd. Larger = more smoothing")
            polyorder = st.slider("Polynomial Order", 2, 5, 3, 1,
                                  help="Must be less than window length")
        else:  # Cubic Spline
            spline_points = st.slider("Interpolation Points", 100, 500, 300, 50,
                                      help="More points = smoother curve")
    plot_display_mode = st.radio(
        "Plot Display Mode",
        ["Separate plots for each pair", "Combined plot with all pairs"],
        index=0,
        key="plot_display_mode",
        horizontal=True,
        help="Choose whether to show separate plots for each element pair or combine all pairs in one plot"
    )

    line_style = st.radio(
        "Line Style",
        ["Lines + Markers", "Lines Only"],
        index=1,
        key="line_style",
        horizontal=True,
        help="Select how to display PRDF lines - with or without point markers"
    )

    if use_lammps_traj:
        lammps_file = st.file_uploader("Upload LAMMPS Trajectory File (.dump, .lammpstrj)",
                                       type=["dump", "lammpstrj"],
                                       help="Upload a LAMMPS trajectory file to analyze PRDF for each frame")
        frame_sampling = st.slider("Frame Sampling Rate",
                                   min_value=1,
                                   max_value=500,
                                   value=1,
                                   help="Select every Nth frame from the trajectory (1 = use all frames)")

        st.radio(
            "Display Mode",
            ["Average PRDF across frames", "Individual frame PRDFs"],
            index=0 if st.session_state.display_mode == "Average PRDF across frames" else 1,
            key="display_mode_radio",
            on_change=update_display_mode
        )

    cutoff = st.number_input("⚙️ Cutoff (Å)", min_value=1.0, max_value=50.0, value=10.0, step=1.0, format="%.1f")
    bin_size = st.number_input("⚙️ Bin Size (Å)", min_value=0.001, max_value=5.000, value=0.100, step=0.005,
                               format="%.3f")

    st.button("Calculate RDF", on_click=trigger_calculation)

    if st.session_state.calc_rdf and (uploaded_files or (use_lammps_traj and lammps_file)):
        with st.spinner(f"Calculating (P)RDF, please wait. 😊"):
            if st.session_state.do_calculation:
                species_combinations = list(combinations(species_list, 2)) + [(s, s) for s in species_list]
                all_prdf_dict = defaultdict(list)
                all_distance_dict = {}
                global_rdf_list = []

                if use_lammps_traj and lammps_file:
                    st.info(f"Processing LAMMPS trajectory file: {lammps_file.name}")
                    progress_bar = st.progress(0)
                    # with st.expander("Log from reading LAMMPS trajectory file"):
                    with st.status("Reading LAMMPS trajectory file..."):
                        file_content_sample = lammps_file.read(2048)
                        lammps_file.seek(0)
                        try:
                            sample_text = file_content_sample.decode('utf-8')
                        except UnicodeDecodeError:
                            sample_text = file_content_sample.decode('latin-1')

                        if "ITEM: TIMESTEP" in sample_text:
                            st.success("Detected standard LAMMPS dump format")
                        elif "ITEM: NUMBER OF ATOMS" in sample_text:
                            st.success("Detected LAMMPS dump format with atom counts")
                        else:
                            st.warning("Could not detect standard LAMMPS format markers. Will attempt to read anyway.")
                        try:
                            import tempfile
                            import io

                            bytes_data = io.BytesIO(lammps_file.getbuffer())

                            st.info("Attempting to read LAMMPS trajectory directly from memory...")

                            from ase.io import read as ase_read

                            frames = []

                            read_methods = [
                                {'format': 'lammps-dump', 'description': 'Standard LAMMPS dump format'},
                                {'format': 'lammps-dump-text', 'description': 'LAMMPS dump text format'},
                                {'format': None, 'description': 'Automatic format detection'}
                            ]

                            success = False


                            def parse_lammps_dump_from_string(content):
                                from ase import Atoms
                                import numpy as np

                                frames = []
                                lines = content.splitlines()

                                i = 0
                                while i < len(lines):
                                    if 'ITEM: TIMESTEP' in lines[i]:
                                        i += 2
                                        if i >= len(lines) or 'ITEM: NUMBER OF ATOMS' not in lines[i]:
                                            continue

                                        i += 1
                                        try:
                                            num_atoms = int(lines[i].strip())
                                            i += 1
                                            while i < len(lines) and 'ITEM: ATOMS' not in lines[i]:
                                                i += 1
                                            if i >= len(lines):
                                                break
                                            header = lines[i].replace('ITEM: ATOMS', '').strip().split()
                                            i += 1
                                            positions = np.zeros((num_atoms, 3))
                                            symbols = []
                                            for j in range(num_atoms):
                                                if i + j >= len(lines):
                                                    break
                                                values = lines[i + j].strip().split()
                                                if len(values) < len(header):
                                                    continue
                                                x_idx = header.index('x') if 'x' in header else -1
                                                y_idx = header.index('y') if 'y' in header else -1
                                                z_idx = header.index('z') if 'z' in header else -1

                                                # Find element column
                                                type_idx = header.index('type') if 'type' in header else -1
                                                element_idx = header.index('element') if 'element' in header else -1

                                                if x_idx >= 0 and y_idx >= 0 and z_idx >= 0:
                                                    positions[j] = [float(values[x_idx]), float(values[y_idx]),
                                                                    float(values[z_idx])]

                                                if element_idx >= 0:
                                                    symbols.append(values[element_idx])
                                                elif type_idx >= 0:
                                                    type_num = int(values[type_idx])
                                                    element_map = {1: 'Si', 2: 'O', 3: 'Al', 4: 'Na'}  # Example mapping
                                                    symbols.append(element_map.get(type_num, f'X{type_num}'))
                                            i += num_atoms
                                            if len(symbols) == num_atoms:
                                                atoms = Atoms(symbols=symbols, positions=positions)
                                                frames.append(atoms)
                                        except Exception as inner_e:
                                            st.error(f"Error parsing frame: {str(inner_e)}")
                                            i += 1
                                    else:
                                        i += 1

                                return frames


                            for method in read_methods:
                                if success:
                                    break

                                try:
                                    bytes_data.seek(0)
                                    st.info(f"Trying to read using {method['description']} directly from memory...")
                                    if method['format'] == 'lammps-dump' or method['format'] == 'lammps-dump-text':
                                        # Get bytes data and convert to string
                                        raw_bytes = bytes_data.getvalue()
                                        try:
                                            text_content = raw_bytes.decode('utf-8')
                                        except UnicodeDecodeError:
                                            text_content = raw_bytes.decode('latin-1')

                                        temp_bytes = io.BytesIO()
                                        temp_bytes.write(text_content.encode('utf-8'))
                                        temp_bytes.seek(0)

                                        frames = ase_read(temp_bytes, index=':', format=method['format'])
                                    elif method['format'] is None:
                                        bytes_data.seek(0)
                                        raw_data = bytes_data.getvalue()
                                        temp_bytes = io.BytesIO(raw_data)
                                        frames = ase_read(temp_bytes, index=':')

                                    if frames and len(frames) > 0:
                                        success = True
                                        st.success(f"Successfully read using {method['description']} from memory")
                                except Exception as e:
                                    st.warning(f"Failed with {method['description']} from memory: {str(e)}")
                            if not success:
                                st.warning("Direct memory reading failed. Trying with temporary file...")
                                import os

                                temp_dir = os.path.join(os.getcwd(), ".streamlit/temp") if os.path.exists(
                                    os.path.join(os.getcwd(), ".streamlit")) else tempfile.gettempdir()
                                os.makedirs(temp_dir, exist_ok=True)
                                import uuid

                                temp_file_path = os.path.join(temp_dir, f"temp_lammps_{uuid.uuid4().hex}.dump")
                                bytes_data.seek(0)
                                with open(temp_file_path, "wb") as f:
                                    f.write(bytes_data.getbuffer())
                                st.info(f"Saved temporary file for processing at: {temp_file_path}")
                                for method in read_methods:
                                    if success:
                                        break

                                    try:
                                        st.info(f"Trying to read using {method['description']} from temp file...")

                                        if method['format'] is None:
                                            frames = ase_read(temp_file_path, index=':')
                                        else:
                                            frames = ase_read(temp_file_path, index=':', format=method['format'])

                                        if frames and len(frames) > 0:
                                            success = True
                                            st.success(
                                                f"Successfully read using {method['description']} from temp file")
                                    except Exception as e:
                                        st.warning(f"Failed with {method['description']} from temp file: {str(e)}")

                                try:
                                    os.remove(temp_file_path)
                                    st.info("Temporary file removed")
                                except Exception as clean_err:
                                    st.warning(f"Could not remove temporary file: {str(clean_err)}")
                            if not success:
                                st.warning("All standard methods failed. Attempting custom parsing...")
                                bytes_data.seek(0)
                                try:
                                    text_content = bytes_data.getvalue().decode('utf-8')
                                except UnicodeDecodeError:
                                    text_content = bytes_data.getvalue().decode('latin-1')
                                frames = parse_lammps_dump_from_string(text_content)

                                if frames and len(frames) > 0:
                                    success = True
                                    st.success(f"Successfully read using custom parser")

                            if not frames or len(frames) == 0:
                                raise Exception("Could not extract any frames from the trajectory file")
                        except Exception as e:
                            st.error(f"Error reading LAMMPS trajectory file: {str(e)}")

                    lammps_file = None
                    bytes_data = None

                    total_frames = len(frames)
                    st.write(f"Found {total_frames} frames in the trajectory")

                    selected_frames = frames[::frame_sampling]
                    st.write(f"Analyzing {len(selected_frames)} frames with sampling rate of {frame_sampling}")

                    frame_indices = [i * frame_sampling for i in range(len(selected_frames))]
                    st.session_state.frame_indices = frame_indices
                    st.session_state.animate = False

                    for i, frame in enumerate(selected_frames):
                        progress_bar.progress((i + 1) / len(selected_frames))

                        try:
                            mg_structure = AseAtomsAdaptor.get_structure(frame)

                            prdf_featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
                            prdf_featurizer.fit([mg_structure])
                            prdf_data = prdf_featurizer.featurize(mg_structure)
                            feature_labels = prdf_featurizer.feature_labels()

                            prdf_dict = defaultdict(list)
                            distance_dict = {}
                            global_dict = {}

                            for j, label in enumerate(feature_labels):
                                parts = label.split(" PRDF r=")
                                element_pair = tuple(parts[0].split("-"))
                                distance_range = parts[1].split("-")
                                bin_center = (float(distance_range[0]) + float(distance_range[1])) / 2
                                prdf_dict[element_pair].append(prdf_data[j])

                                if element_pair not in distance_dict:
                                    distance_dict[element_pair] = []
                                distance_dict[element_pair].append(bin_center)
                                global_dict[bin_center] = global_dict.get(bin_center, 0) + prdf_data[j]

                            for pair, values in prdf_dict.items():
                                if pair not in all_distance_dict:
                                    all_distance_dict[pair] = distance_dict[pair]
                                if isinstance(values, float):
                                    values = [values]
                                all_prdf_dict[pair].append(values)

                            global_rdf_list.append(global_dict)

                        except Exception as e:
                            st.error(f"Error processing frame {i}: {str(e)}")

                    progress_bar.progress(1.0)
                    multi_structures = True

                else:
                    for file in uploaded_files:
                        try:
                            structure = read(file.name)
                            mg_structure = AseAtomsAdaptor.get_structure(structure)
                        except Exception as e:
                            mg_structure = load_structure(file)

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
                    frame_indices = [0]
                    st.session_state.frame_indices = frame_indices

                st.session_state.processed_data = {
                    "all_prdf_dict": all_prdf_dict,
                    "all_distance_dict": all_distance_dict,
                    "global_rdf_list": global_rdf_list,
                    "multi_structures": multi_structures
                }

                st.session_state.do_calculation = False

        all_prdf_dict = st.session_state.processed_data["all_prdf_dict"]
        all_distance_dict = st.session_state.processed_data["all_distance_dict"]
        global_rdf_list = st.session_state.processed_data["global_rdf_list"]
        multi_structures = st.session_state.processed_data["multi_structures"]
        frame_indices = st.session_state.frame_indices

        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        import numpy as np

        colors = plt.cm.tab10.colors


        def rgb_to_hex(color):
            return '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))


        font_dict = dict(size=24, color="black")

        st.divider()
        st.subheader("PRDF Plots:")
        hex_color_global = rgb_to_hex(colors[len(all_prdf_dict) % len(colors)])

        if use_lammps_traj and lammps_file and st.session_state.display_mode == "Individual frame PRDFs":
            if "animation_speed" not in st.session_state:
                st.session_state.animation_speed = 0.5


            def update_speed():
                st.session_state.animation_speed = st.session_state.speed_slider


            st.slider("Animation Speed",
                      min_value=0.05,
                      max_value=2.00,
                      value=st.session_state.animation_speed,
                      step=0.05,
                      key="speed_slider",
                      on_change=update_speed,
                      help="Seconds per frame")

        if not use_lammps_traj and 'plot_display_mode' in locals() and plot_display_mode == "Combined plot with all pairs":
            fig_combined = go.Figure()

            for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):
                hex_color = rgb_to_hex(colors[idx % len(colors)])
                valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                if not valid_prdf:
                    prdf_data = np.zeros_like(all_distance_dict[comb])
                else:
                    prdf_array = np.vstack(valid_prdf) if valid_prdf else np.zeros((1, len(all_distance_dict[comb])))
                    prdf_data = np.mean(prdf_array, axis=0) if multi_structures else prdf_array[0]


                if plot_style == "Bars (Histogram)":
                    fig_combined.add_trace(go.Bar(
                        x=all_distance_dict[comb],
                        y=prdf_data,
                        name=f"{comb[0]}-{comb[1]}",
                        marker=dict(color=hex_color, line=dict(width=0)),
                        width=bin_size * bar_width_factor,
                        opacity=0.7  # Add transparency for overlapping bars
                    ))

                elif plot_style == "Smooth Curve":
                    if smoothing_method == "Gaussian":
                        y_smooth = smooth_gaussian(prdf_data, sigma=sigma)
                        x_plot = all_distance_dict[comb]

                    elif smoothing_method == "Savitzky-Golay":
                        y_smooth = smooth_savgol(prdf_data, window, polyorder)
                        x_plot = all_distance_dict[comb]

                    else:  # Cubic Spline
                        x_plot, y_smooth = smooth_spline(
                            np.array(all_distance_dict[comb]),
                            prdf_data,
                            spline_points
                        )

                    fig_combined.add_trace(go.Scatter(
                        x=x_plot,
                        y=y_smooth,
                        mode='lines',
                        name=f"{comb[0]}-{comb[1]}",
                        line=dict(color=hex_color, width=2)
                    ))

                else:  # Raw Data Points
                    fig_combined.add_trace(go.Scatter(
                        x=all_distance_dict[comb],
                        y=prdf_data,
                        mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                        name=f"{comb[0]}-{comb[1]}",
                        line=dict(color=hex_color, width=2),
                        marker=dict(size=8) if st.session_state.line_style == "Lines + Markers" else dict()
                    ))

            title_str = "Combined Averaged PRDF: All Pairs" if multi_structures else "Combined PRDF: All Pairs"

            fig_combined.update_layout(
                title={'text': title_str, 'font': font_dict},
                xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                hovermode='x',
                font=font_dict,
                xaxis=dict(tickfont=font_dict),
                yaxis=dict(tickfont=font_dict, range=[0, None]),
                hoverlabel=dict(font=font_dict),
                barmode='overlay' if plot_style == "Bars (Histogram)" else None,  # Important for bar overlap
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=24)
                )
            )

            st.plotly_chart(fig_combined, use_container_width=True)

        else:
            for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):

                hex_color = rgb_to_hex(colors[idx % len(colors)])

                valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                if not valid_prdf:

                    prdf_data = np.zeros_like(all_distance_dict[comb])
                    title_str = f"PRDF: {comb[0]}-{comb[1]}"

                    fig = go.Figure()
                    if plot_style == "Bars (Histogram)":
                        fig.add_trace(go.Bar(
                            x=all_distance_dict[comb],
                            y=prdf_data,
                            name=f"{comb[0]}-{comb[1]}",
                            marker=dict(color=hex_color, line=dict(width=0)),
                            width=bin_size * 0.8
                        ))

                    elif plot_style == "Smooth Curve":
                        if smoothing_method == "Gaussian":
                            y_smooth = smooth_gaussian(prdf_data, sigma=sigma)
                            x_plot = all_distance_dict[comb]

                        elif smoothing_method == "Savitzky-Golay":
                            y_smooth = smooth_savgol(prdf_data, window, polyorder)
                            x_plot = all_distance_dict[comb]

                        else:  # Cubic Spline
                            x_plot, y_smooth = smooth_spline(
                                np.array(all_distance_dict[comb]),
                                prdf_data,
                                spline_points
                            )

                        fig.add_trace(go.Scatter(
                            x=x_plot,
                            y=y_smooth,
                            mode='lines',
                            name=f"{comb[0]}-{comb[1]}",
                            line=dict(color=hex_color, width=2)
                        ))

                    else:  # Raw Data Points (original behavior)
                        fig.add_trace(go.Scatter(
                            x=all_distance_dict[comb],
                            y=prdf_data,
                            mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                            name=f"{comb[0]}-{comb[1]}",
                            line=dict(color=hex_color, width=2),
                            marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
                        ))

                elif use_lammps_traj and lammps_file and st.session_state.display_mode == "Individual frame PRDFs":

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=all_distance_dict[comb],
                        y=valid_prdf[0],  # First frame data
                        mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                        name=f"{comb[0]}-{comb[1]}",
                        line=dict(color=hex_color, width=2),
                        marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
                    ))

                    frames = []
                    for i, frame_data in enumerate(valid_prdf):
                        frame = go.Frame(
                            data=[go.Scatter(
                                x=all_distance_dict[comb],
                                y=frame_data,
                                mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                                line=dict(color=hex_color, width=2),
                                marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
                            )],
                            name=f"frame_{i}"
                        )
                        frames.append(frame)

                    fig.frames = frames

                    updatemenus = [
                        dict(
                            type="buttons",
                            direction="right",
                            x=0.1,
                            y=-0.1,
                            showactive=False,
                            buttons=[
                                dict(
                                    label="▶️ Play",
                                    method="animate",
                                    args=[None, {
                                        "frame": {"duration": int(st.session_state.animation_speed * 1000),
                                                  "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                                ),
                                dict(
                                    label="⏹️ Pause",
                                    method="animate",
                                    args=[[None], {"frame": {"duration": 0, "redraw": True},
                                                   "mode": "immediate", "transition": {"duration": 0}}],
                                ),
                            ],
                        )
                    ]

                    sliders = [
                        dict(
                            active=0,
                            yanchor="top",
                            xanchor="left",
                            currentvalue=dict(
                                font=dict(size=16),
                                prefix="Frame: ",
                                visible=True,
                                xanchor="right"
                            ),
                            pad=dict(b=10, t=50),
                            len=0.9,
                            x=0.1,
                            y=0,
                            steps=[
                                dict(
                                    method="animate",
                                    args=[
                                        [f"frame_{k}"],
                                        {"frame": {"duration": 100, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}
                                    ],
                                    label=f"{frame_indices[k]}"
                                )
                                for k in range(len(valid_prdf))
                            ]
                        )
                    ]

                    all_y_values = [y for data in valid_prdf for y in data]
                    max_y = max(all_y_values) * 1.1 if all_y_values else 1.0

                    title_str = f"PRDF: {comb[0]}-{comb[1]} Animation"

                    fig.update_layout(
                        title={'text': title_str, 'font': font_dict},
                        xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                        yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                        hovermode='x',
                        updatemenus=updatemenus,
                        sliders=sliders,
                        font=font_dict,
                        xaxis=dict(tickfont=font_dict),
                        yaxis=dict(tickfont=font_dict, range=[0, max_y]),
                        hoverlabel=dict(font=font_dict)
                    )

                else:
                    prdf_array = np.vstack(valid_prdf) if valid_prdf else np.zeros((1, len(all_distance_dict[comb])))
                    prdf_data = np.mean(prdf_array, axis=0) if multi_structures else prdf_array[0]

                    if use_lammps_traj and lammps_file:
                        title_str = f"Trajectory-Averaged PRDF: {comb[0]}-{comb[1]}"
                    else:
                        title_str = f"Averaged PRDF: {comb[0]}-{comb[1]}" if multi_structures else f"PRDF: {comb[0]}-{comb[1]}"

                    fig = go.Figure()
                    if plot_style == "Bars (Histogram)":
                        fig.add_trace(go.Bar(
                            x=all_distance_dict[comb],
                            y=prdf_data,
                            name=f"{comb[0]}-{comb[1]}",
                            marker=dict(color=hex_color, line=dict(width=0)),
                            width=bin_size * 0.8
                        ))

                    elif plot_style == "Smooth Curve":
                        if smoothing_method == "Gaussian":
                            y_smooth = smooth_gaussian(prdf_data, sigma=sigma)
                            x_plot = all_distance_dict[comb]

                        elif smoothing_method == "Savitzky-Golay":
                            y_smooth = smooth_savgol(prdf_data, window, polyorder)
                            x_plot = all_distance_dict[comb]

                        else:  # Cubic Spline
                            x_plot, y_smooth = smooth_spline(
                                np.array(all_distance_dict[comb]),
                                prdf_data,
                                spline_points
                            )

                        fig.add_trace(go.Scatter(
                            x=x_plot,
                            y=y_smooth,
                            mode='lines',
                            name=f"{comb[0]}-{comb[1]}",
                            line=dict(color=hex_color, width=2)
                        ))

                    else:  # Raw Data Points
                        fig.add_trace(go.Scatter(
                            x=all_distance_dict[comb],
                            y=prdf_data,
                            mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                            name=f"{comb[0]}-{comb[1]}",
                            line=dict(color=hex_color, width=2),
                            marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
                        ))

                    if use_lammps_traj and lammps_file and multi_structures and len(valid_prdf) > 1:
                        prdf_std = np.std(prdf_array, axis=0)
                        fig.add_trace(go.Scatter(
                            x=all_distance_dict[comb],
                            y=prdf_data + prdf_std,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=all_distance_dict[comb],
                            y=np.maximum(0, prdf_data - prdf_std),
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(100,100,100,0.2)',
                            fill='tonexty',
                            showlegend=False
                        ))

                    fig.update_layout(
                        title={'text': title_str, 'font': font_dict},
                        xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                        yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                        hovermode='x',
                        font=font_dict,
                        xaxis=dict(tickfont=font_dict),
                        yaxis=dict(tickfont=font_dict, range=[0, None]),
                        hoverlabel=dict(font=font_dict)
                    )

                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Total RDF Plot:")
        global_bins_set = set()
        for gd in global_rdf_list:
            global_bins_set.update(gd.keys())
        global_bins = sorted(list(global_bins_set))

        if use_lammps_traj and lammps_file and st.session_state.display_mode == "Individual frame PRDFs":

            fig_global = go.Figure()

            initial_frame = global_rdf_list[0]
            initial_values = [initial_frame.get(b, 0) for b in global_bins]

            fig_global.add_trace(go.Scatter(
                x=global_bins,
                y=initial_values,
                mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                name=f"Global RDF",
                line=dict(color=hex_color_global, width=2),
                marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
            ))

            frames = []
            for i, global_dict in enumerate(global_rdf_list):
                frame_values = [global_dict.get(b, 0) for b in global_bins]

                frame = go.Frame(
                    data=[go.Scatter(
                        x=global_bins,
                        y=frame_values,
                        mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                        line=dict(color=hex_color_global, width=2),
                        marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
                    )],
                    name=f"frame_{i}"
                )
                frames.append(frame)

            fig_global.frames = frames

            updatemenus = [
                dict(
                    type="buttons",
                    direction="right",
                    x=0.1,
                    y=-0.1,
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶️ Play",
                            method="animate",
                            args=[None,
                                  {"frame": {"duration": int(st.session_state.animation_speed * 1000), "redraw": True},
                                   "fromcurrent": True, "mode": "immediate"}],
                        ),
                        dict(
                            label="⏹️ Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ]

            # Add slider
            sliders = [
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=16),
                        prefix="Frame: ",
                        visible=True,
                        xanchor="right"
                    ),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f"frame_{k}"],
                                {"frame": {"duration": 100, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 0}}
                            ],
                            label=f"{frame_indices[k]}"
                        )
                        for k in range(len(global_rdf_list))
                    ]
                )
            ]

            all_values = []
            for gdict in global_rdf_list:
                values = [gdict.get(b, 0) for b in global_bins]
                all_values.extend(values)
            max_y = max(all_values) * 1.1 if all_values else 1.0

            fig_global.update_layout(
                title={'text': "Global RDF Animation", 'font': font_dict},
                xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                yaxis_title={'text': "Total RDF Intensity", 'font': font_dict},
                hovermode='x',
                updatemenus=updatemenus,
                sliders=sliders,
                font=font_dict,
                xaxis=dict(tickfont=font_dict),
                yaxis=dict(tickfont=font_dict, range=[0, max_y]),
                hoverlabel=dict(font=font_dict)
            )

            title_global = "Global RDF Animation"

        else:

            global_rdf_avg = []
            global_rdf_std = []

            for b in global_bins:
                vals = []
                for gd in global_rdf_list:
                    vals.append(gd.get(b, 0))
                global_rdf_avg.append(np.mean(vals))
                global_rdf_std.append(np.std(vals))

            fig_global = go.Figure()
            if plot_style == "Bars (Histogram)":
                fig_global.add_trace(go.Bar(
                    x=global_bins,
                    y=global_rdf_avg,
                    name="Global RDF",
                    marker=dict(color=hex_color_global, line=dict(width=0)),
                    width=bin_size * bar_width_factor
                ))

            elif plot_style == "Smooth Curve":
                if smoothing_method == "Gaussian":
                    y_smooth = smooth_gaussian(np.array(global_rdf_avg), sigma=sigma)
                    x_plot = global_bins

                elif smoothing_method == "Savitzky-Golay":
                    y_smooth = smooth_savgol(np.array(global_rdf_avg), window, polyorder)
                    x_plot = global_bins

                else:  # Cubic Spline
                    x_plot, y_smooth = smooth_spline(
                        np.array(global_bins),
                        np.array(global_rdf_avg),
                        spline_points
                    )

                fig_global.add_trace(go.Scatter(
                    x=x_plot,
                    y=y_smooth,
                    mode='lines',
                    name="Global RDF",
                    line=dict(color=hex_color_global, width=2)
                ))

            else:  # Raw Data Points
                fig_global.add_trace(go.Scatter(
                    x=global_bins,
                    y=global_rdf_avg,
                    mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                    name="Global RDF",
                    line=dict(color=hex_color_global, width=2),
                    marker=dict(size=10) if st.session_state.line_style == "Lines + Markers" else dict()
                ))

            if use_lammps_traj and lammps_file:
                fig_global.add_trace(go.Scatter(
                    x=global_bins,
                    y=[a + s for a, s in zip(global_rdf_avg, global_rdf_std)],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig_global.add_trace(go.Scatter(
                    x=global_bins,
                    y=[max(0, a - s) for a, s in zip(global_rdf_avg, global_rdf_std)],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(100,100,100,0.2)',
                    fill='tonexty',
                    showlegend=False
                ))
                title_global = "Trajectory-Averaged Global RDF"
            else:
                title_global = "Averaged Global RDF" if multi_structures else "Global RDF"

            fig_global.update_layout(
                title={'text': title_global, 'font': font_dict},
                xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                yaxis_title={'text': "Total RDF Intensity", 'font': font_dict},
                hovermode='x',
                font=font_dict,
                xaxis=dict(tickfont=font_dict),
                yaxis=dict(tickfont=font_dict, range=[0, None]),
                hoverlabel=dict(font=font_dict)
            )

        st.plotly_chart(fig_global, use_container_width=True)


        def toggle_animation():
            st.session_state.animate = not st.session_state.animate
            if st.session_state.animate:
                st.rerun()


        st.subheader("Download Options")

        if "download_prepared" not in st.session_state:
            st.session_state.download_prepared = False


        def prepare_downloads():
            st.session_state.download_prepared = True


        st.button("Prepare Data for Download", on_click=prepare_downloads)

        if st.session_state.download_prepared:
            import io
            import pandas as pd
            import base64

            is_individual_mode = (use_lammps_traj and lammps_file and
                                  st.session_state.display_mode == "Individual frame PRDFs")

            for comb, prdf_list in all_prdf_dict.items():
                valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                if valid_prdf:
                    if is_individual_mode:

                        df = pd.DataFrame()
                        df["Distance (Å)"] = all_distance_dict[comb]

                        for i, frame_data in enumerate(valid_prdf):
                            df[f"Frame_{frame_indices[i]}"] = frame_data

                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{comb[0]}_{comb[1]}_prdf_frames.csv">Download {comb[0]}-{comb[1]} PRDF data for all frames</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:

                        df = pd.DataFrame()
                        df["Distance (Å)"] = all_distance_dict[comb]

                        if multi_structures:

                            prdf_array = np.vstack(valid_prdf)
                            prdf_data = np.mean(prdf_array, axis=0)
                            df["Average"] = prdf_data

                            if len(valid_prdf) > 1:
                                prdf_std = np.std(prdf_array, axis=0)
                                df["StdDev"] = prdf_std

                            filename = f"{comb[0]}_{comb[1]}_prdf_average.csv"
                        else:

                            prdf_data = valid_prdf[0]
                            df["PRDF"] = prdf_data
                            filename = f"{comb[0]}_{comb[1]}_prdf.csv"

                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {comb[0]}-{comb[1]} PRDF data</a>'
                        st.markdown(href, unsafe_allow_html=True)

            global_bins_set = set()
            for gd in global_rdf_list:
                global_bins_set.update(gd.keys())
            global_bins = sorted(list(global_bins_set))

            if is_individual_mode:

                global_df = pd.DataFrame()
                global_df["Distance (Å)"] = global_bins

                for i, gd in enumerate(global_rdf_list):
                    global_df[f"Frame_{frame_indices[i]}"] = [gd.get(b, 0) for b in global_bins]

                global_filename = "global_rdf_frames.csv"
                download_text = "Download Total RDF data for all frames"
            elif multi_structures:

                global_df = pd.DataFrame()
                global_df["Distance (Å)"] = global_bins

                global_avgs = []
                global_stds = []
                for b in global_bins:
                    vals = [gd.get(b, 0) for gd in global_rdf_list]
                    global_avgs.append(np.mean(vals))
                    if len(global_rdf_list) > 1:
                        global_stds.append(np.std(vals))

                global_df["Average"] = global_avgs

                if len(global_rdf_list) > 1:
                    global_df["StdDev"] = global_stds

                global_filename = "global_rdf_average.csv"
                download_text = "Download Average Total RDF data"
            else:

                global_df = pd.DataFrame()
                global_df["Distance (Å)"] = global_bins
                global_df["RDF"] = [global_rdf_list[0].get(b, 0) for b in global_bins]

                global_filename = "global_rdf.csv"
                download_text = "Download Total RDF data"

            global_csv = global_df.to_csv(index=False)
            global_b64 = base64.b64encode(global_csv.encode()).decode()
            global_href = f'<a href="data:file/csv;base64,{global_b64}" download="{global_filename}">{download_text}</a>'
            st.markdown(global_href, unsafe_allow_html=True)

if "📈 Interactive Data Plot" in calc_mode:

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'grey']

    st.markdown(
        "#### 📂 Upload your two-column data files in the sidebar to see them in an interactive plot. Multiple files are supported, and your columns can be separated by spaces, tabs, commas, or semicolons."
    )

    colss, colzz, colx, colc, cold = st.columns([1, 1, 1, 1, 1])
    has_header = colss.checkbox("Files contain a header row", value=False)
    skip_header = colzz.checkbox("Skip header row", value=True)
    normalized_intensity = colx.checkbox("Normalized intensity", value=False)
    x_axis_log = colc.checkbox("Logarithmic X-axis", value=False)
    y_axis_log = cold.checkbox("Logarithmic Y-axis", value=False)

    col_thick, col_size, col_fox, col_xmin, col_xmax, = st.columns([2, 1, 1, 1, 1])
    with col_thick:
        st.info(
            f"ℹ️ You can modify the **graph layout** from the sidebar.️ ℹ️ You can **convert** your **XRD** data below the plot. Enable 'Normalized intensity' to **automatically shift data files in vertical direction**.")
    fix_x_axis = col_fox.checkbox("Fix x-axis range?", value=False)
    if fix_x_axis == True:
        x_axis_min = col_xmin.number_input("X-axis Minimum", value=0.0)
        x_axis_max = col_xmax.number_input("X-axis Maximum", value=10.0)
    x_axis_metric = "X-data"
    y_axis_metric = "Y-data"
    if user_pattern_file:
        files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

        if has_header:
            try:
                sample_file = files[0]
                sample_file.seek(0)
                df_sample = pd.read_csv(
                    sample_file,
                    sep=r'\s+|,|;',
                    engine='python',
                    header=0
                )
                x_axis_metric = df_sample.columns[0]
                y_axis_metric = df_sample.columns[1]
            except Exception as e:
                st.error(f"Error reading header from file {sample_file.name}: {e}")
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"
        else:
            x_axis_metric = "X-data"
            y_axis_metric = "Y-data"

    if normalized_intensity:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("✨ Stack Plots"):
                offset_gap_value = st.session_state.get('stack_offset_gap', 10.0)
                auto_normalize_and_stack_plots(files, skip_header, has_header, offset_gap_value)
                # st.rerun()
        with col2:
            if st.button("🔄 Reset Layout"):
                reset_layout(files)
                # st.rerun()
        with col3:
            st.number_input(
                "Stacking Gap",
                min_value=0.0,
                value=10.0,
                step=5.0,
                key='stack_offset_gap',
                help="The vertical space to add between stacked, normalized plots."
            )

    plot_placeholder = st.empty()
    st.sidebar.markdown("### Interactive Data Plot layout")
    customize_layout = st.sidebar.checkbox(f"Modify the **graph layout**", value=False)
    if customize_layout:
        # st.sidebar.markdown("### Graph Appearance Settings")

        col_line, col_marker = st.sidebar.columns(2)
        show_lines = col_line.checkbox("Show Lines", value=True, key="show_lines")
        show_markers = col_marker.checkbox("Show Markers", value=False, key="show_markers")

        col_thick, col_size = st.sidebar.columns(2)
        line_thickness = col_thick.number_input("Line Thickness", min_value=0.1, max_value=15.0, value=1.0,
                                                step=0.3,
                                                key="line_thickness2")
        marker_size = col_size.number_input("Marker Size", min_value=0.5, max_value=50.0, value=3.0,
                                            step=1.0,
                                            key="marker_size")

        col_title_font, col_axis_font, col_tick_font = st.sidebar.columns(3)
        title_font_size = col_title_font.number_input("Title Font Size", min_value=10, max_value=50,
                                                      value=36,
                                                      step=2,
                                                      key="title_font_size")
        axis_label_font_size = col_axis_font.number_input("Axis Label Font Size", min_value=10,
                                                          max_value=50,
                                                          value=36,
                                                          step=2, key="axis_font_size")
        tick_font_size = col_tick_font.number_input("Tick Label Font Size", min_value=8, max_value=40,
                                                    value=24,
                                                    step=2,
                                                    key="tick_font_size")

        col_leg_font, col_leg_pos = st.sidebar.columns(2)
        legend_font_size = col_leg_font.number_input("Legend Font Size", min_value=8, max_value=40,
                                                     value=28,
                                                     step=2,
                                                     key="legend_font_size")
        legend_position = col_leg_pos.selectbox(
            "Legend Position",
            options=["Top", "Bottom", "Left", "Right"],
            index=0,
            key="legend_position"
        )

        col_width, col_height = st.sidebar.columns(2)
        graph_width = col_width.number_input("Graph Width (pixels)", min_value=400, max_value=2000,
                                             value=1000,
                                             step=50,
                                             key="graph_width")
        graph_height = col_height.number_input("Graph Height (pixels)", min_value=300, max_value=1500,
                                               value=900,
                                               step=50, key="graph_height")

        st.sidebar.markdown("#### Custom Axis Labels")
        col_x_label, col_y_label = st.sidebar.columns(2)
        custom_x_label = col_x_label.text_input("X-axis Label", value=x_axis_metric, key="custom_x_label")
        custom_y_label = col_y_label.text_input("Y-axis FLabel", value=y_axis_metric, key="custom_y_label")

        if user_pattern_file:
            st.sidebar.markdown("#### Custom Series Names")
            series_names = {}

            if isinstance(user_pattern_file, list):
                for i, file in enumerate(user_pattern_file):
                    series_names[i] = st.sidebar.text_input(f"Label for {file.name}", value=file.name,
                                                            key=f"series_name_{i}")
            else:
                series_names[0] = st.sidebar.text_input(f"Label for {user_pattern_file.name}",
                                                        value=user_pattern_file.name,
                                                        key="series_name_0")
        if user_pattern_file:
            st.sidebar.markdown("#### Custom Series Colors")
            series_colors = {}
            files_for_color = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

            colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#000000', '#7f7f7f']
            for i, file in enumerate(files_for_color):
                default_color = colors[i % len(colors)]
                series_colors[i] = st.sidebar.color_picker(
                    f"Color for {file.name}",
                    value=default_color,
                    key=f"series_color_{i}"
                )
    else:
        series_colors = {}
        show_lines = True
        show_markers = False
        line_thickness = 1.0
        marker_size = 3.0
        title_font_size = 36
        axis_label_font_size = 36
        tick_font_size = 24
        legend_font_size = 28
        legend_position = "Top"
        graph_width = 1000
        graph_height = 900
        custom_x_label = x_axis_metric
        custom_y_label = y_axis_metric
        series_names = {}

    enable_conversion = st.checkbox(f"Enable powder **XRD data conversion**", value=False)

    if user_pattern_file:
        files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

        if has_header:
            try:
                sample_file = files[0]
                sample_file.seek(0)
                df_sample = pd.read_csv(
                    sample_file,
                    sep=r'\s+|,|;',
                    engine='python',
                    header=0
                )
                x_axis_metric = df_sample.columns[0]
                y_axis_metric = df_sample.columns[1]
            except Exception as e:
                st.error(f"Error reading header from file {sample_file.name}: {e}")
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"
        else:
            x_axis_metric = "X-data"
            y_axis_metric = "Y-data"

        file_conversion_settings = []

        if enable_conversion:
            with st.expander("❓ How does Divergence Slit Conversion work?", expanded=False):
                st.markdown("""
                ### Divergence Slit Conversion Explained

                #### Auto Slit
                - The slit **automatically adjusts** with angle (2θ) to **keep the irradiated area constant**.
                - Produces intensity that remains relatively **consistent** across angles.

                #### Fixed Slit
                - The slit has a **fixed opening angle**.
                - As 2θ increases, the **irradiated area is smaller**.
                - Results in **reduced intensity at higher angles**.

                #### Conversion Types

                - **Fixed Slit → Auto Slit**  
                  Adjusts for loss of intensity at higher angles by simulating constant irradiated area:

                  $$
                  \\text{Intensity}_{\\text{auto}} = \\text{Intensity}_{\\text{fixed}} \\times \\frac{\\text{Irradiated Length} \\times \\sin(\\theta)}{\\text{Fixed Slit Size}}
                  $$

                - **Auto Slit → Fixed Slit**  
                  Simulates reduced illuminated area at higher angles:

                  $$
                  \\text{Intensity}_{\\text{fixed}} = \\text{Intensity}_{\\text{auto}} \\times \\frac{\\text{Fixed Slit Size}}{\\text{Irradiated Length} \\times \\sin(\\theta)}
                  $$

                #### Parameters

                - **Fixed slit size (degrees)**: The opening angle of the slit in degrees.
                - **Irradiated sample length (mm)**: Physical length of sample that is illuminated.
                  - Reflection geometry: *10–20 mm*  
                  - Transmission geometry: *1–2 mm*

                """)
            for i, file in enumerate(files):
                with st.expander(f"🔄 Conversion settings for **{file.name}**", expanded=(i == 0)):
                    wave_col, slit_col = st.columns(2)

                    with wave_col:
                        st.markdown("**Diffraction data conversion:**")
                        input_format = st.selectbox(
                            "Convert from:",
                            [
                                "No conversion",
                                "d-spacing (Å)",
                                "2theta (Copper CuKa1)",
                                "2theta (Cobalt CoKa1)",
                                "2theta (Custom)",
                                "q-vector (Å⁻¹)"
                            ],
                            key=f"input_format_{i}",
                            help=f"Copper (CuKa1): 1.5406 Å\n\n"
                                 " Molybdenum (MoKa1): 0.7093 Å\n\n"
                                 " Chromium (CrKa1): 2.2897 Å\n\n"
                                 " Iron (FeKa1): 1.9360 Å\n\n"
                                 " Cobalt (CoKa1): 1.7889 Å\n\n"
                                 " Silver (AgKa1): 0.5594 Å\n\n"
                                 " q-vector = 4π·sin(θ)/λ\n"
                        )

                        all_output_options = [
                            "No conversion",
                            "d-spacing (Å)",
                            "2theta (Copper CuKa1)",
                            "2theta (Cobalt CoKa1)",
                            "2theta (Custom)",
                            "q-vector (Å⁻¹)"
                        ]

                        if input_format != "No conversion":
                            filtered_options = all_output_options.copy()
                            if input_format in filtered_options and input_format != "2theta (Custom)":
                                filtered_options.remove(input_format)
                        else:
                            filtered_options = all_output_options

                        output_format = st.selectbox(
                            "Convert to:",
                            filtered_options,
                            index=0,
                            key=f"output_format_{i}"
                        )

                        input_custom_wavelength = None
                        output_custom_wavelength = None

                        if input_format == "2theta (Custom)":
                            input_custom_wavelength = st.number_input(
                                "Input custom wavelength (Å)",
                                min_value=0.1,
                                max_value=10.0,
                                value=1.54056,
                                step=0.01,
                                format="%.5f",
                                key=f"input_custom_wl_{i}"
                            )

                        if output_format == "2theta (Custom)":
                            output_custom_wavelength = st.number_input(
                                "Output custom wavelength (Å)",
                                min_value=0.1,
                                max_value=10.0,
                                value=1.54056 if input_custom_wavelength is None else input_custom_wavelength * 0.9,
                                step=0.01,
                                format="%.5f",
                                key=f"output_custom_wl_{i}"
                            )

                        if input_format == "q-vector (Å⁻¹)" and "2theta" in output_format and output_custom_wavelength is None:
                            if "Custom" in output_format:
                                output_custom_wavelength = st.number_input(
                                    "Output wavelength for q-vector to 2theta conversion (Å)",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=1.54056,
                                    step=0.01,
                                    format="%.5f",
                                    key=f"q_to_2theta_wl_{i}"
                                )

                        if "2theta" in input_format and output_format == "q-vector (Å⁻¹)" and input_custom_wavelength is None:
                            if "Custom" in input_format:
                                input_custom_wavelength = st.number_input(
                                    "Input wavelength for 2theta to q-vector conversion (Å)",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=1.54056,
                                    step=0.01,
                                    format="%.5f",
                                    key=f"2theta_to_q_wl_{i}"
                                )

                    with slit_col:

                        st.markdown("**Divergence slit conversion:**")
                        slit_conversion_type = st.selectbox(
                            "Convert slit type:",
                            [
                                "No conversion",
                                "Auto slit to fixed slit",
                                "Fixed slit to auto slit"
                            ],
                            key=f"slit_conv_{i}"
                        )
                        fixed_slit_size = None
                        irradiated_length = None
                        if slit_conversion_type != "No conversion":
                            fixed_slit_size = st.number_input(
                                "Fixed slit size (degrees)",
                                min_value=0.1,
                                max_value=2.0,
                                value=1.0,
                                step=0.1,
                                format="%.2f",
                                key=f"fixed_slit_{i}"
                            )
                            irradiated_length = st.number_input(
                                "Irradiated sample length (mm)",
                                min_value=1.0,
                                max_value=50.0,
                                value=10.0,
                                step=1.0,
                                key=f"irradiated_{i}"
                            )

                            st.info("Typical values: 10-20 mm for reflection, 1-2 mm for transmission geometry.")

                    file_conversion_settings.append({
                        "file_index": i,
                        "conversion_type": "No conversion" if input_format == "No conversion" or not output_format else f"{input_format} to {output_format}",
                        "input_format": input_format,
                        "output_format": output_format,
                        "input_custom_wavelength": input_custom_wavelength,
                        "output_custom_wavelength": output_custom_wavelength,
                        "slit_conversion_type": slit_conversion_type,
                        "fixed_slit_size": fixed_slit_size,
                        "irradiated_length": irradiated_length
                    })
        offset_cols = st.columns(len(files))
        y_offsets = []
        for i, file in enumerate(files):
            offset_val = offset_cols[i].number_input(
                f"Y offset for {file.name}",
                value=0.0,
                key=f"y_offset_{i}"
            )
            y_offsets.append(offset_val)
        scale_cols = st.columns(len(files))
        y_scales = []
        for i, file in enumerate(files):
            scale_val = scale_cols[i].number_input(
                f"Scale factor for {file.name}",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key=f"y_scale_{i}"
            )
            y_scales.append(scale_val)

        fig_interactive = go.Figure()
        for i, file in enumerate(files):
            try:
                file.seek(0)
                if has_header:
                    df = pd.read_csv(
                        file,
                        sep=r'\s+|,|;',
                        engine='python',
                        header=0
                    )
                else:
                    if skip_header:
                        file.seek(0)
                        try:
                            file_content = file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            file_content = file.read().decode('latin-1')

                        lines = file_content.splitlines()
                        comment_line_indices = [i for i, line in enumerate(lines) if line.strip().startswith('#')]
                        lines_to_skip = [0] + comment_line_indices
                        lines_to_skip = sorted(set(lines_to_skip))
                        file.seek(0)

                        df = pd.read_csv(
                            file,
                            sep=r'\s+|,|;',
                            engine='python',
                            header=None,
                            skiprows=lines_to_skip
                        )
                    else:
                        df = pd.read_csv(
                            file,
                            sep=r'\s+|,|;',
                            engine='python',
                            header=None
                        )
                    df.columns = [f"Column {j + 1}" for j in range(len(df.columns))]

                x_data = df.iloc[:, 0].values
                y_data = df.iloc[:, 1].values

                if st.session_state.get('auto_stack_enabled', False):
                    min_adjustments = st.session_state.get('min_adjustments', [])
                    if i < len(min_adjustments):
                        y_data = y_data - min_adjustments[i]

                if enable_conversion:
                    settings = file_conversion_settings[i]  # this must match the same index i as files[i]
                    conversion_type = settings.get("conversion_type", "No conversion")
                    if conversion_type == "No conversion":
                        pass
                    else:
                        def convert_data(x_values, conversion_type, input_custom_wavelength=None,
                                         output_custom_wavelength=None):

                            wavelength_map = {
                                "Copper": 1.54056,
                                "CuKa1": 1.54056,
                                "Cobalt": 1.78897,
                                "CoKa1": 1.78897
                            }

                            parts = conversion_type.split(" to ")
                            if len(parts) != 2:
                                print(f"Invalid conversion type format: {conversion_type}")
                                return x_values

                            input_format = parts[0].strip()
                            output_format = parts[1].strip()

                            lambda_in = None
                            if "Copper" in input_format or "CuKa1" in input_format:
                                lambda_in = wavelength_map["Copper"]
                            elif "Cobalt" in input_format or "CoKa1" in input_format:
                                lambda_in = wavelength_map["Cobalt"]
                            elif "Custom" in input_format and input_custom_wavelength is not None:
                                lambda_in = input_custom_wavelength

                            lambda_out = None
                            if "Copper" in output_format or "CuKa1" in output_format:
                                lambda_out = wavelength_map["Copper"]
                            elif "Cobalt" in output_format or "CoKa1" in output_format:
                                lambda_out = wavelength_map["Cobalt"]
                            elif "Custom" in output_format and output_custom_wavelength is not None:
                                lambda_out = output_custom_wavelength

                            if "q-vector" in input_format and "d-spacing" in output_format:
                                valid = x_values > 0
                                d_values = np.zeros_like(x_values)
                                d_values[valid] = 2 * np.pi / x_values[valid]
                                d_values[~valid] = np.nan
                                return d_values

                            elif "d-spacing" in input_format and "q-vector" in output_format:
                                valid = x_values > 0
                                q_values = np.zeros_like(x_values)
                                q_values[valid] = 2 * np.pi / x_values[valid]
                                q_values[~valid] = np.nan
                                return q_values

                            elif "q-vector" in input_format and "2theta" in output_format:
                                if lambda_out is None:
                                    print(f"Missing output wavelength for q-vector to 2theta conversion")
                                    return x_values

                                valid = x_values >= 0
                                sin_arg = (x_values[valid] * lambda_out) / (4 * np.pi)

                                mask = (sin_arg >= -1) & (sin_arg <= 1)
                                sin_arg = sin_arg[mask]

                                theta = np.arcsin(sin_arg)
                                twotheta = 2 * np.degrees(theta)

                                result = np.zeros_like(x_values)
                                result_indices = np.where(valid)[0][mask]
                                result[result_indices] = twotheta
                                result[~valid] = np.nan

                                return result

                            elif "2theta" in input_format and "q-vector" in output_format:
                                if lambda_in is None:
                                    print(f"Missing input wavelength for 2theta to q-vector conversion")
                                    return x_values

                                theta_rad = np.radians(x_values) / 2
                                q_values = (4 * np.pi * np.sin(theta_rad)) / lambda_in

                                return q_values

                            elif ("2theta" in input_format) and ("d-spacing" in output_format):
                                if lambda_in is None:
                                    print(f"Missing input wavelength for conversion: {input_format}")
                                    return x_values

                                theta_rad = np.radians(x_values / 2)
                                valid = np.abs(np.sin(theta_rad)) > 1e-6

                                d = np.zeros_like(x_values)
                                d[valid] = lambda_in / (2 * np.sin(theta_rad[valid]))
                                d[~valid] = np.nan

                                return d

                            elif ("d-spacing" in input_format) and ("2theta" in output_format):
                                if lambda_out is None:
                                    print(f"Missing output wavelength for conversion: {output_format}")
                                    return x_values
                                valid = x_values > 0
                                sin_arg = lambda_out / (2 * x_values[valid])
                                sin_arg = np.clip(sin_arg, 0, 1)

                                theta = np.degrees(np.arcsin(sin_arg))
                                result = np.zeros_like(x_values)
                                result[valid] = 2 * theta
                                result[~valid] = np.nan

                                return result

                            elif ("2theta" in input_format) and ("2theta" in output_format):
                                if lambda_in is None or lambda_out is None:
                                    print(
                                        f"Missing wavelength for 2θ to 2θ conversion. Input λ: {lambda_in}, Output λ: {lambda_out}")
                                    return x_values

                                if abs(lambda_in - lambda_out) < 1e-6:
                                    return x_values

                                theta_rad = np.radians(x_values / 2)
                                valid = np.abs(np.sin(theta_rad)) > 1e-6
                                d = np.zeros_like(x_values)
                                d[valid] = lambda_in / (2 * np.sin(theta_rad[valid]))

                                sin_arg = lambda_out / (2 * d[valid])
                                sin_arg = np.clip(sin_arg, 0, 1)
                                theta_new = np.degrees(np.arcsin(sin_arg))

                                result = np.zeros_like(x_values)
                                result[valid] = 2 * theta_new
                                result[~valid] = np.nan

                                return result

                            else:
                                print(f"No matching conversion logic for: {conversion_type}")
                                return x_values


                        try:
                            x_data = convert_data(
                                x_values=x_data,
                                conversion_type=conversion_type,
                                input_custom_wavelength=settings["input_custom_wavelength"],
                                output_custom_wavelength=settings["output_custom_wavelength"]
                            )

                            if "to q-vector" in conversion_type:
                                x_axis_metric = "q (Å⁻¹)"
                            elif "to d-spacing" in conversion_type:
                                x_axis_metric = "d-spacing (Å)"
                            elif "to 2theta" in conversion_type:
                                if "Copper" in conversion_type:
                                    x_axis_metric = "2θ (Cu Kα, λ=1.54056Å)"
                                elif "Cobalt" in conversion_type:
                                    x_axis_metric = "2θ (Co Kα, λ=1.78897Å)"
                                elif "custom" in conversion_type or "Custom" in conversion_type:
                                    wavelength = settings['output_custom_wavelength']
                                    if wavelength:
                                        x_axis_metric = f"2θ (λ={wavelength}Å)"
                                    else:
                                        x_axis_metric = "2θ (°)"

                            st.success(f"Converted {file.name}: {conversion_type}")

                            valid_mask = ~np.isnan(x_data)
                            if not np.all(valid_mask):
                                x_data = x_data[valid_mask]
                                y_data = y_data[valid_mask]
                                st.warning(f"Some points in {file.name} were invalid for conversion and were removed")

                        except Exception as e:
                            st.error(f"Error in data conversion for {file.name}: {e}")
                            st.warning("Using original data for this file.")


            except Exception as e:
                st.error(
                    f"Error occurred in file processing for {file.name}: {str(e)}")

            if enable_conversion and i < len(file_conversion_settings):
                slit_settings = file_conversion_settings[i]

                slit_conversion_type = slit_settings.get("slit_conversion_type", "No conversion")
                fixed_slit_size = slit_settings.get("fixed_slit_size")
                irradiated_length = slit_settings.get("irradiated_length")

                if slit_conversion_type != "No conversion":
                    theta_rad = np.radians(x_data / 2)  # convert 2θ to θ in radians for calculations

                    # Avoid division by zero by creating a valid mask
                    valid_mask = np.abs(np.sin(theta_rad)) > 1e-6

                    if slit_conversion_type == "Auto slit to fixed slit":
                        adjustment_factor = np.zeros_like(y_data)
                        adjustment_factor[valid_mask] = fixed_slit_size / (
                                irradiated_length * np.sin(theta_rad[valid_mask])
                        )

                        y_data = y_data * adjustment_factor
                        st.success(f"Applied Auto slit → Fixed slit conversion for {file.name}")

                    elif slit_conversion_type == "Fixed slit to auto slit":
                        adjustment_factor = np.zeros_like(y_data)
                        adjustment_factor[valid_mask] = (
                                irradiated_length * np.sin(theta_rad[valid_mask]) / fixed_slit_size
                        )

                        y_data = y_data * adjustment_factor
                        st.success(f"Applied Fixed slit → Auto slit conversion for {file.name}")
                    if not np.all(valid_mask):
                        st.warning(
                            f"Some points in {file.name} were omitted during slit conversion due to invalid angles."
                        )
                        x_data = x_data[valid_mask]
                        y_data = y_data[valid_mask]

            if normalized_intensity and np.max(y_data) > 0:
                y_data = (y_data / np.max(y_data)) * 100

            try:

                if i < len(y_scales):
                    y_data = y_data * y_scales[i]

                y_data = y_data + y_offsets[i]

                mask = np.ones_like(x_data, dtype=bool)
                if x_axis_log:
                    mask &= (x_data > 0)
                if y_axis_log:
                    mask &= (y_data > 0)
                if not np.all(mask):
                    st.warning(
                        f"In file '{file.name}', some data points were omitted because they are not positive and are required for logarithmic scaling.")
                x_data = x_data[mask]
                y_data = y_data[mask]

                if x_axis_log:
                    x_data = np.log10(x_data)
                if y_axis_log:
                    y_data = np.log10(y_data)

                if customize_layout and i in series_colors:
                    color = series_colors[i]
                else:
                    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#000000', '#7f7f7f']
                    color = colors[i % len(colors)]
                mode_str = ""
                if show_lines:
                    mode_str += "lines"
                if show_markers:
                    if mode_str:
                        mode_str += "+markers"
                    else:
                        mode_str = "markers"
                if not mode_str:
                    mode_str = "markers"

                trace_name = series_names.get(i, file.name) if customize_layout else file.name

                fig_interactive.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode_str,
                    name=trace_name,
                    line=dict(dash='solid', width=line_thickness, color=color),
                    marker=dict(color=color, size=marker_size),
                    hovertemplate=(
                        f"<span style='color:{color};'><b>{trace_name}</b><br>"
                        "x = %{x:.2f}<br>y = %{y:.2f}</span><extra></extra>"
                    )
                ))
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

                # Set axis scale
            fig_interactive.update_xaxes(type="linear")
            fig_interactive.update_yaxes(type="linear")

            # Configure legend position based on selection
            legend_config = {
                "font": dict(size=legend_font_size),
                # "title": "Legend Title"
            }

            if legend_position == "Top":
                legend_config.update({
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5
                })
            elif legend_position == "Bottom":
                legend_config.update({
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.2,
                    "xanchor": "center",
                    "x": 0.5
                })
            elif legend_position == "Left":
                legend_config.update({
                    "orientation": "v",
                    "yanchor": "middle",
                    "y": 0.5,
                    "xanchor": "right",
                    "x": -0.1
                })
            elif legend_position == "Right":
                legend_config.update({
                    "orientation": "v",
                    "yanchor": "middle",
                    "y": 0.5,
                    "xanchor": "left",
                    "x": 1.05
                })

            # Update layout with all customized settings
            fig_interactive.update_layout(
                height=graph_height,
                width=graph_width,
                margin=dict(t=80, b=80, l=60, r=30),
                hovermode="closest",
                showlegend=True,
                legend=legend_config,
                xaxis=dict(
                    title=dict(text=custom_x_label, font=dict(size=axis_label_font_size, color='black'), standoff=20),
                    tickfont=dict(size=tick_font_size, color='black'),
                    fixedrange=fix_x_axis
                ),
                yaxis=dict(
                    title=dict(text=custom_y_label, font=dict(size=axis_label_font_size, color='black')),
                    tickfont=dict(size=tick_font_size, color='black')
                ),
                # title=dict(
                #    text="Interactive Data Plot",
                #    font=dict(size=title_font_size, color='black')
                # ),
                hoverlabel=dict(font=dict(size=tick_font_size)),
                font=dict(size=18),
                autosize=False
            )

        if user_pattern_file:
            files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

            if has_header:
                try:
                    sample_file = files[0]
                    sample_file.seek(0)
                    df_sample = pd.read_csv(
                        sample_file,
                        sep=r'\s+|,|;',
                        engine='python',
                        header=0
                    )
                    x_axis_metric = df_sample.columns[0]
                    y_axis_metric = df_sample.columns[1]
                except Exception as e:
                    st.error(f"Error reading header from file {sample_file.name}: {e}")
                    x_axis_metric = "X-data"
                    y_axis_metric = "Y-data"
            else:
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"

        if fix_x_axis == True:
            fig_interactive.update_xaxes(range=[x_axis_min, x_axis_max])

        with plot_placeholder.container():
            st.plotly_chart(fig_interactive)

        import io

        st.markdown("### 💾 Download Processed Data")
        delimiter_label_to_value = {
            "Comma (`,`)": ",",
            "Space (` `)": " ",
            "Tab (`\\t`)": "\t",
            "Semicolon (`;`)": ";"
        }

        delimiter_label = st.selectbox("Choose delimiter for download:", list(delimiter_label_to_value.keys()))
        delimiter_option = delimiter_label_to_value[delimiter_label]

        for i, file in enumerate(files):
            x_data = fig_interactive.data[i].x
            y_data = fig_interactive.data[i].y

            if fix_x_axis:
                if x_axis_log:
                    x_values = 10 ** x_data
                else:
                    x_values = x_data

                if x_axis_log:
                    mask = (x_values >= x_axis_min) & (x_values <= x_axis_max)
                else:
                    mask = (x_data >= x_axis_min) & (x_data <= x_axis_max)

                filtered_x = x_data[mask]
                filtered_y = y_data[mask]
            else:
                filtered_x = x_data
                filtered_y = y_data

            df_out = pd.DataFrame({
                x_axis_metric: filtered_x,
                y_axis_metric: filtered_y
            })

            buffer = io.StringIO()
            df_out.to_csv(buffer, sep=delimiter_option, index=False)

            base_name = file.name.rsplit(".", 1)[0]
            download_name = f"{base_name}_processed.xy"

            download_info = ""
            if fix_x_axis:
                download_info = f" (filtered to x-range: {x_axis_min}-{x_axis_max})"

            st.download_button(
                label=f"⬇️ Download processed data for {file.name}{download_info}",
                data=buffer.getvalue(),
                file_name=download_name,
                mime="text/plain",
                key=f"download_btn_{i}_{base_name}"
            )
    else:
        st.info(f"Upload your data file first to see all options.")
st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
import sys

components.html(
    """
    <head>
        <meta name="description" content="XRDlicious, Online Calculator for Powder XRD/ND Patterns (Diffractograms), Partial Radial Distribution Function (PRDF), and Total RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, XYZ ...), or XRD data conversion">
    </head>
    """,
    height=0,
)


def get_session_memory_usage():
    total_size = 0
    for key in st.session_state:
        try:
            total_size += sys.getsizeof(st.session_state[key])
        except Exception:
            pass
    return total_size / 1024  # in KB


st.markdown("""
**The XRDlicious application is open-source and released under the [MIT License](https://github.com/bracerino/xrdlicious/blob/main/LICENSE).**
""")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # in MB


memory_usage = get_memory_usage()
st.write(
    f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are now using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. For more extensive computations, please compile the application locally from the [GitHub](https://github.com/bracerino/xrdlicious).")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://pypi.org/project/py3Dmol/)** Licensed under the [BSD-3-Clause License](https://github.com/3dmol/3Dmol.js/blob/master/LICENSE). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html)
 **[Crystallographic Open Database (COD)](https://www.crystallography.net/cod/)** under the CC0 license.
""")
