import streamlit as st

st.set_page_config(
    page_title="XRDlicious: Online Calculator for Powder XRD/ND patterns and (P)RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, ...), or XRD data conversion",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Remove top padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)
from helpers import *
from xrd_convert import *
from equivalent_planes import *
from more_funct.reorient import *
from more_funct.citation_section import *
from more_funct.db_results_display import show_database_results
from more_funct.interactive_data_plot import render_interactive_data_plot

import gc
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
try:
    from xrd_rust_calculator import XRDCalculatorRust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
from collections import defaultdict
from itertools import combinations
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
    /* hamburger menu + footer + top decoration bar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    /* the Share / Star / Fork / GitHub / Edit / Deploy buttons (Community Cloud) */
    [data-testid="stToolbarActions"] {display: none;}

    /* the "Hosted with Streamlit" / fork badge, if present */
    .viewerBadge_link__qRIco {display: none;}
    [data-testid="stStatusWidget"] {display: none;}

    /* SAFETY NET: make sure the sidebar open/close control stays visible */
    [data-testid="stSidebarCollapsedControl"] {visibility: visible !important; display: block !important;}
    [data-testid="collapsedControl"] {visibility: visible !important; display: block !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.markdown(
#    f"#### **XRDlicious**: Online Calculator for Powder XRD/ND Patterns, (P)RDF, Peak Matching, Structure Modification and Point Defects Creation from Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)")
def is_running_locally():
    try:
        host = st.context.headers.get("host", "")
        return "localhost" in host or "127.0.0.1" in host
    except:
        return False

IS_LOCAL = is_running_locally()

# Get current memory usage
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
memory_usage = mem_info.rss / (1024 ** 2)  # in MB

# On first open the intro is shown inline; once the user interacts (the same
# moment the welcome note disappears) it collapses into an expander so the
# working area stays clean. The toggled content (About / Roadmap / How to cite /
# Tutorials) is rendered AFTER the container to avoid nesting expanders.
_intro_collapsed = not st.session_state.get("first_run_note", True)
intro_ctx = (
    st.expander("ℹ️ About XRDlicious — info, separated modules, roadmap, "
                "how to cite, tutorials", expanded=False)
    if _intro_collapsed else st.container()
)

with intro_ctx:
    st.markdown(
        """
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:16px; margin-bottom:8px;">
            <h4 style="margin:0;">
                <span style='color:#8b0000;'>
                    <strong>XRDlicious</strong> – <em>powder diffraction and more</em>
                </span>
            </h4>
            <div style="
                display: inline-block;
                background-color: #ffffff;
                border-left: 5px solid #8b0000;
                border-radius: 10px;
                padding: 10px 16px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.10);
                color: #111827;
                font-size: 0.95rem;
                font-weight: 600;
            ">
                <span style="color:#8b0000; font-weight:800;">Release:</span>
                v0.7.0 &nbsp; | &nbsp;
                <span style="color:#8b0000; font-weight:800;">Updated:</span>
                June 9, 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #8b0000; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([0.2, 0.4, 0.4])
    with col3:
        st.markdown(
            """
            ###### 🔹 Separated Modules:
            - Create point defects in a crystal structure: **[Open App 🌐](https://xrdlicious-point-defects.streamlit.app/)**
            - Convert between `.xrdml`, `.ras` and `.xy` formats or X/Y-axis: **[Open App 🧩](https://xrd-convert.streamlit.app/)**
            - Austenite-martensite crystallographic planes for NiTiHf: **[Open App 🌐](https://austenite-martensite.streamlit.app/)**
            - Calculate (P)RDF: **[Open App 🧩](https://prdf-xrdlicious.streamlit.app/)**
            """
        )
    with col2:
        st.info(
            "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. Spot a bug or have a feature idea? Let us know at: "
            "**lebedmi2@cvut.cz**. To compile the app locally, visit our **[GitHub page](https://github.com/bracerino/xrdlicious)**. If you like the app, please cite **[article in IUCr](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)**. 🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**"
        )
    with col1:
        about_app_show = st.checkbox(f"📖 About the app")
        show_roadmap = st.checkbox(f"🧭 Roadmap", value=False)
        citations = st.checkbox("📚 How to cite", value=False)
        tutorials = st.checkbox("📺 Tutorials", value=False)

# Toggled content rendered outside the intro container (top level) so the inner
# expanders are not nested inside the collapsed intro expander.
if about_app_show:
    about_app()
if show_roadmap:
    with st.expander("Roadmap", icon="🧭", expanded=True):
        show_xrdlicious_roadmap()
if citations:
    show_citation_section()
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

st.sidebar.markdown("## XRDlicious")
mode = "Advanced"
st.markdown(
    """
    <hr style="border: none; height: 6px; background-color: #8b0000; border-radius: 8px; margin: 0px 0;">
    """,
    unsafe_allow_html=True
)

calc_mode = st.sidebar.multiselect(
    "Choose Type(s) of Calculation",
    options=[
        "🔬 Structure Modification",
        "💥 Powder Diffraction",
        "🎯 Fitting Lattice Parameters",
        "📊 (P)RDF",
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


def section_divider():
    st.markdown(
        '<hr style="border: none; height: 6px; background-color: #1e3a8a; '
        'border-radius: 8px; margin: 20px 0;">',
        unsafe_allow_html=True,
    )


if "➡️ .xrdml ↔️ .xy ↔️ .ras Converter" in calc_mode:
    section_divider()
    run_data_converter()

if "↔️ Equivalent Planes" in calc_mode:
    section_divider()
    run_equivalent_hkl_app()

if "📉 PRDF from LAMMPS/XYZ trajectories" in calc_mode:
    section_divider()
    st.subheader(
        "This module calculates the Pair Radial Distribution Function (PRDF) across frames in LAMMPS or XYZ trajectories. Due to its high computational demands, it cannot be run on our free online server. Instead, it is provided as a standalone module that must be compiled and executed locally. Please visit to see how to compile and run the code:")
    st.markdown(
        '<p style="font-size:24px;">🔗 <a href="https://github.com/bracerino/PRDF-CP2K-LAMMPS" target="_blank">Download the PRDF calculator for LAMMPS/XYZ trajectories</a></p>',
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



allowed_types = ["cif", "xyz", "vasp", "poscar", "xsf", "pw", "cfg", "lmp"]

st.sidebar.subheader("📤 Upload Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "(CIF, POSCAR, XSF, PW, CFG, XYZ (with cell), LMP):",
    type=allowed_types,
    accept_multiple_files=True,
    key="sidebar_uploader"
)

st.sidebar.subheader("📁 Upload Experimental Data ")
user_pattern_file = st.sidebar.file_uploader(
    "(.xy, .xrdml, .ras — or any two-column text file).",
    type=["csv", "txt", "xy", "data", "dat", "xrdml", "xml", "ras"],
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
        cols, cols2 = st.columns([1.5, 1.5])
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
                                    #print(alphabet_form)
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

        show_database_results()
        st.write("")
        st.write("")
        st.write("")


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
    _size_items = []
    for file in uploaded_files_user_sidebar:
        try:
            structure = load_structure(file)
            st.session_state['full_structures'][file.name] = structure
            _size_items.append((file.name, len(structure)))
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")
    report_large_structures(_size_items, is_local=IS_LOCAL)
else:
    uploaded_files = st.session_state['uploaded_files']

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
    from more_funct.structure_editor import run_structure_editor
    uploaded_files = run_structure_editor(uploaded_files)

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

if uploaded_files:
    st.sidebar.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")

if "expander_diff_settings" not in st.session_state:
    st.session_state["expander_diff_settings"] = True

if "parsed_exp_data" not in st.session_state:
    st.session_state.parsed_exp_data = {}
#


if "💥 Powder Diffraction" in calc_mode:
    section_divider()
    from more_funct.xrd_nd_section import run_diffraction_section
    run_diffraction_section(uploaded_files, user_pattern_file, is_local=IS_LOCAL)

if "🎯 Fitting Lattice Parameters" in calc_mode:
    from more_funct.lattice_fitting import run_lattice_fitting_section
    run_lattice_fitting_section(uploaded_files, user_pattern_file, is_local=IS_LOCAL)

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
    section_divider()
    PRDF_APP_URL = "https://prdf-xrdlicious.streamlit.app/"
    st.markdown(
        """
        <hr style="border:none;height:6px;background-color:#8b0000;
                   border-radius:8px;margin:4px 0 12px 0;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #fff8e1, #fff3cd);
            border-left: 6px solid #8b0000;
            border-radius: 10px;
            padding: 24px 28px;
            margin: 8px 0 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        ">
            <h2 style="color:#8b0000;margin-top:0;">
                📊 (P)RDF Module Has Moved
            </h2>
            <p style="font-size:1.05rem;margin-bottom:10px;">
                To keep memory usage lower for powder diffraction,
                the <strong>Partial Radial Distribution Function (PRDF)</strong>
                calculator has been moved to its own application.
            </p>
            <p style="font-size:1.05rem;margin-bottom:18px;">
                All features are available there.
            </p>
            <a href="{PRDF_APP_URL}" target="_blank"
               style="
                   display:inline-block;
                   background-color:#8b0000;
                   color:white;
                   font-size:1.1rem;
                   font-weight:600;
                   padding:12px 28px;
                   border-radius:8px;
                   text-decoration:none;
                   box-shadow:0 2px 6px rgba(0,0,0,0.2);
               ">
                🔗 Open the (P)RDF Calculator →
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "📈 Interactive Data Plot" in calc_mode:
    section_divider()
    render_interactive_data_plot(user_pattern_file)
st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
import sys

st.html(
    """
    <head>
        <meta name="description" content="XRDlicious, Online Calculator for Powder XRD/ND Patterns (Diffractograms), Partial Radial Distribution Function (PRDF), and Total RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, XYZ ...), or XRD data conversion">
    </head>
    """
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
    f"🔍 Current memory usage: **{memory_usage:.2f} MB**.")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://pypi.org/project/py3Dmol/)** Licensed under the [BSD-3-Clause License](https://github.com/3dmol/3Dmol.js/blob/master/LICENSE). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html)
 **[Crystallography Open Database (COD)](https://www.crystallography.net/cod/)** under the CC0 license.
""")
