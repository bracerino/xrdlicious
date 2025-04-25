import streamlit as st

st.set_page_config(
    page_title="XRDlicious: Online Calculator for Powder XRD/ND patterns and (P)RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, ...)",
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

# import aflow.keywords as K
from pymatgen.io.cif import CifWriter

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"

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
    div.stButton > button:active,
    div.stButton > button:focus {
        background-color: #0099ff !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stDataFrameContainer"] table td {
         font-size: 22px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

components.html(
    """
    <head>
        <meta name="description" content="XRDlicious, Online Calculator for Powder XRD/ND Patterns (Diffractograms), Partial Radial Distribution Function (PRDF), and Total RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, ...)">
    </head>
    """,
    height=0,
)


st.markdown("#### 🍕 XRDlicious: Online Calculator for Powder XRD/ND Patterns, (P)RDF, Peak Matching, Structure Modification and Point Defects Creation from Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)")

st.markdown("""
<style>
@keyframes wave {
  0% { transform: rotate(0.0deg); }
  10% { transform: rotate(14.0deg); }
  20% { transform: rotate(-8.0deg); }
  30% { transform: rotate(14.0deg); }
  40% { transform: rotate(-4.0deg); }
  50% { transform: rotate(10.0deg); }
  60% { transform: rotate(0.0deg); }
  100% { transform: rotate(0.0deg); }
}

@keyframes fadeOut {
  0%   { opacity: 1; }
  80%  { opacity: 1; }
  100% { opacity: 0; display: none; }
}

.hello-container {
  position: fixed;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 9999;
  animation: fadeOut 7s ease-out forwards;
}

.hello-message {
  font-size: 2.5em;
  font-weight: bold;
  text-align: center;
  background-color: #f0f2f6;
  padding: 20px 30px;
  border-radius: 20px;
  box-shadow: 0 0 20px rgba(0,0,0,0.1);
}

.wave-emoji {
  display: inline-block;
  animation: wave 2s infinite;
  transform-origin: 70% 70%;
}
</style>

<div class="hello-container">
  <div class="hello-message">
      <span class="wave-emoji">👋</span> Hello there! Welcome to <span style="color:#0066cc;">XRDlicious</span>.<br>
  </div>
</div>
""", unsafe_allow_html=True)



col1, col2, col3 = st.columns([1.2, 0.5, 0.3])

with col2:
    st.info(
        "🌀 Developed by [IMPLANT team](https://implant.fs.cvut.cz/). 📺 [Quick tutorial HERE.](https://youtu.be/ZiRbcgS_cd0) The app will be continously updated. Spot a bug or have a feature idea? Let us know at: "
        "lebedmi2@cvut.cz"
    )
with col3:
    if st.button("💡 Need Help?"):
        st.markdown("""
        <style>
        @keyframes fadeInOut {
          0%   { opacity: 0; transform: translateY(-10px); }
          5%   { opacity: 1; transform: translateY(0); }
          95%  { opacity: 1; transform: translateY(0); }
          100% { opacity: 0; transform: translateY(-10px); }
        }

        .onboarding-tip {
          background-color: #ffffff;
          border-left: 5px solid #3399ff;
          padding: 18px 22px;
          border-radius: 14px;
          box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
          font-size: 1.1em;
          font-weight: 500;
          position: fixed;
          top: 100px;
          right: 40px;
          z-index: 9999;
          max-width: 400px;
          width: 100%;
          opacity: 0;
        }

        .tip1 { animation: fadeInOut 7s ease-in-out 0s forwards; }
        .tip2 { animation: fadeInOut 7s ease-in-out 7s forwards; }
        .tip3 { animation: fadeInOut 7s ease-in-out 14s forwards; }

        .tip-label {
          font-size: 0.9em;
          font-weight: 600;
          color: #0066cc;
          margin-bottom: 6px;
          display: block;
        }
        </style>

        <!-- Tip 1 -->
        <div class="onboarding-tip tip1">
          <span class="tip-label">Tip 1/3</span>
          🧭 From the <b>sidebar</b>, choose a tool like <b>Structure Modification</b>, <b>Powder Diffraction</b>, or <b>(P)RDF Calculator</b>.
        </div>

        <!-- Tip 2 -->
        <div class="onboarding-tip tip2">
          <span class="tip-label">Tip 2/3</span>
          📂 Upload your <b>structure files</b> (CIF, POSCAR, LMP, XSF) or <b>two-column data</b> using the sidebar.
        </div>

        <!-- Tip 3 -->
        <div class="onboarding-tip tip3">
          <span class="tip-label">Tip 3/3</span>
          🐣 No files? Use the <b>search interface</b> to fetch structures from online databases.
        </div>
        """, unsafe_allow_html=True)

with col1:
    with st.expander("Read here about this appllication.", icon="📖"):
        st.info(
            "Upload **structure files** (e.g., **CIF, LMP, POSCAR, XSF** format) and this tool will calculate either the "
            "**powder X-ray** or **neutron diffraction** (**XRD** or **ND**) patterns or **partial radial distribution function** (**PRDF**) for each **element combination**. Additionally, you can convert "
            "between primitive and conventional crystal structure representations, modify the structure, and introduce automatically interstitials, vacancies, or substitutes, downloading their outputs in CIF, POSCAR, LMP, or XYZ format. "
            "If **multiple files** are uploaded, the **PRDF** will be **averaged** for corresponding **element combinations** across the structures. For **XRD/ND patterns**, diffraction data from multiple structures are combined into a **single figure**."
            "Additionally, there is option to interactively plot and modify your two-columns data. "
        )
        st.warning(
            "🪧 **Step 1**: 📁 Choose which tool to use from the sidebar.\n\n"
            "**Structure Visualization** lets you view, convert (primitive ⇄ conventional), create **supercell and point defects**, and download structures (**CIF, POSCAR, LMP, XYZ**).\n\n "
            "**Diffraction** computes patterns on uploaded structures or shows **experimental data**.\n\n "
            "**PRDF** calculates **partial and total RDF** for all element pairs on the uploaded structures.\n\n"
            "**Peak Matching** allows users to upload their experimental powder XRD pattern and match peaks with structures from MP/AFLOW/COD databases. \n\n"
            f"🪧 **Step 2**:  📁 From the Sidebar, Upload Your Structure Files or Experimental Patterns, or Search Here in Online Databases."
            "💡 Tip: Make sure the file format is supported (e.g., CIF, POSCAR, LMP, xy)."
        )

        from PIL import Image

        image = Image.open("images/ts4.png")
        st.image(image)

pattern_details = None

st.sidebar.markdown("## 🍕 XRDlicious")
mode = "Advanced"

calc_mode = st.sidebar.multiselect(
    "Choose Type(s) of Calculation/Analysis",
    options=[
        "🔬 Structure Modification",
        "💥 Powder Diffraction",
        "📊 (P)RDF",
        "🛠️ Online Search/Match** (UNDER TESTING, being regularly upgraded 😊)",
        "📈 Interactive Data Plot"
    ],
    default=["🔬 Structure Modification","💥 Powder Diffraction" ]
)


if "🛠️ Online Search/Match** (UNDER TESTING, being regularly upgraded 😊)" in calc_mode:
    st.subheader("For the Online Peak Search/Match Subtool, Please visit (USE ONLY FOR TESTING PURPOSES): ")
    st.markdown(
        '<p style="font-size:24px;">🔗 <a href="https://xrdlicious-peak-match.streamlit.app/" target="_blank">Go to Peak Matching Tool</a></p>',
        unsafe_allow_html=True
    )


def update_element_indices(df):
    """Update the Element_Index column based on Element values"""
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


st.markdown(
    """
    <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
    """,
    unsafe_allow_html=True
)

col3, col1, col2 = st.columns(3)

if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

st.sidebar.subheader("📁📤 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, XSF, PW, CFG, ...):",
    type=None,
    accept_multiple_files=True,
    key="sidebar_uploader"
)

st.sidebar.subheader("📁🧫 Upload Your Experimental Data ")
user_pattern_file = st.sidebar.file_uploader(
    "Upload additional XRD pattern (2 columns: X-values and Intensity. The first line is skipped assuming a header.)",
    type=["csv", "txt", "xy"],
    key="user_xrd", accept_multiple_files=True
)

if uploaded_files_user_sidebar:
    for file in uploaded_files_user_sidebar:
        if file.name not in st.session_state.full_structures:
            try:
                structure = load_structure(file)
                st.session_state.full_structures[file.name] = structure
            except Exception as e:
                #st.error(f"Failed to parse {file.name}: {e}")
                st.error(f"This does not work. Are you sure you tried to upload here the structure files (CIF, POSCAR, LMP, XSF, PW)? For the **experimental XY data**, put them to the other uploader\n"
                         f"and please remove this wrongly placed file. 😊")



#if "📈 Interactive Data Plot" not in calc_mode:
with st.expander("Search for Structures Online in Databases", icon="🔍", expanded=False):
    cols, cols2, cols3 = st.columns([1.5, 1.5, 3.5])
    with cols:

        db_choices = st.multiselect(
            "Select Database(s)",
            options=["Materials Project", "AFLOW", "COD"],
            default=["Materials Project", "AFLOW", "COD"],
            help="Choose which databases to search for structures. You can select multiple databases."
        )

        if not db_choices:
            st.warning("Please select at least one database to search.")

    with cols2:
        search_query = st.text_input(
            "Enter elements separated by spaces (e.g., Sr Ti O):",
            value="Sr Ti O"
        )

    if st.button("Search Selected Databases"):
        if not db_choices:
            st.error("Please select at least one database to search.")
        else:
            elements_list = [el.strip() for el in search_query.split() if el.strip()]
            if not elements_list:
                st.error("Please enter at least one element for the search.")
            else:
                for db_choice in db_choices:
                    if db_choice == "Materials Project":
                        with st.spinner(f"Searching **the MP database**, please wait. 😊"):
                            elements_list_sorted = sorted(set(elements_list))
                            try:
                                with MPRester(MP_API_KEY) as mpr:
                                    docs = mpr.materials.summary.search(
                                        elements=elements_list_sorted,
                                        num_elements=len(elements_list_sorted),
                                        fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                    )
                                    if docs:
                                        status_placeholder = st.empty()
                                        st.session_state.mp_options = []
                                        st.session_state.full_structures_see = {}

                                        for doc in docs:
                                            full_structure = mpr.get_structure_by_material_id(doc.material_id,
                                                                                              conventional_unit_cell=True)
                                            structure_to_use = full_structure
                                            st.session_state.full_structures_see[doc.material_id] = full_structure
                                            lattice = structure_to_use.lattice
                                            leng = len(structure_to_use)
                                            lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                                           f"{lattice.alpha:.1f}, {lattice.beta:.1f}, {lattice.gamma:.1f} °")
                                            st.session_state.mp_options.append(
                                                f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol} #{doc.symmetry.number}) [{lattice_str}], {float(doc.volume):.1f} Å³, {leng} atoms"
                                            )
                                            status_placeholder.markdown(
                                                f"- **Structure loaded:** `{structure_to_use.composition.reduced_formula}` ({doc.material_id})"
                                            )
                                        st.success(
                                            f"Found {len(st.session_state.mp_options)} structures in Materials Project.")
                                    else:
                                        st.session_state.mp_options = []
                                        st.warning("No matching structures found in Materials Project.")
                            except Exception as e:
                                st.error(
                                    f"An error occurred with Materials Project: {e}.\nThis is likely due to an error within The Materials Project API. Please try again later.")

                    elif db_choice == "AFLOW":
                        with st.spinner(f"Searching **the AFLOW database**, please wait. 😊"):
                            try:
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
                                st.session_state.entrys = {}

                                if results:
                                    status_placeholder = st.empty()
                                    st.session_state.aflow_options = []
                                    st.session_state.entrys = {}  # store full AFLOW entry objects
                                    for entry in results:
                                        st.session_state.entrys[entry.auid] = entry
                                        st.session_state.aflow_options.append(
                                            f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax}) {entry.geometry}"
                                        )
                                        status_placeholder.markdown(
                                            f"- **Structure loaded:** `{entry.compound}` (aflow_{entry.auid})"
                                        )
                                    st.success(f"Found {len(st.session_state.aflow_options)} structures.")
                                else:
                                    st.session_state.aflow_options = []
                                    st.warning("No matching structures found in AFLOW.")
                            except Exception as e:
                                st.warning("No matching structures found in AFLOW.")

                    elif db_choice == "COD":
                        with st.spinner(f"Searching **the COD database**, please wait. 😊"):
                            elements = elements_list
                            if elements:
                                params = {'format': 'json', 'detail': '1'}
                                for i, el in enumerate(elements, start=1):
                                    params[f'el{i}'] = el
                                params['strictmin'] = str(len(elements))
                                params['strictmax'] = str(len(elements))
                                cod_entries = get_cod_entries(params)
                                if cod_entries:
                                    status_placeholder = st.empty()
                                    st.session_state.cod_options = []
                                    st.session_state.full_structures_see_cod = {}
                                    for entry in cod_entries:
                                        cif_content = get_cif_from_cod(entry)
                                        if cif_content:
                                            try:
                                                # structure = get_full_conventional_structure(
                                                #    get_cod_str(cif_content))
                                                structure = get_cod_str(cif_content)
                                                cod_id = f"cod_{entry.get('file')}"
                                                st.session_state.full_structures_see_cod[cod_id] = structure
                                                spcs = entry.get("sg")
                                                spcs_number = entry.get("sgNumber")
                                                # Listing all keywords in the entry
                                                # all_keys = list(entry.keys())
                                                # st.write(all_keys)

                                                cell_volume = structure.lattice.volume
                                                st.session_state.cod_options.append(
                                                    f"{cod_id}: {structure.composition.reduced_formula} ({spcs} #{spcs_number}) [{structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} Å, {structure.lattice.alpha:.2f} "
                                                    f"{structure.lattice.beta:.2f} {structure.lattice.gamma:.2f}] °, {cell_volume:.1f} Å³, {len(structure)} atoms "
                                                )
                                                status_placeholder.markdown(
                                                    f"- **Structure loaded:** `{structure.composition.reduced_formula}` (cod_{entry.get('file')})")
                                            except Exception as e:
                                                st.error(f"Error processing COD entry {entry.get('file')}: {e}")

                                    if st.session_state.cod_options:
                                        st.success(f"Found {len(st.session_state.cod_options)} structures in COD.")
                                    else:
                                        st.warning("COD: No matching structures found.")
                                else:
                                    st.session_state.cod_options = []
                                    st.warning("COD: No matching structures found.")
                            else:
                                st.error("Please enter at least one element for the COD search.")
        with cols2:
            image = Image.open("images/Rabbit2.png")
            st.image(image, use_container_width =True)



    with cols3:
        if any(x in st.session_state for x in ['mp_options', 'aflow_options', 'cod_options']):
            tabs = []
            if 'mp_options' in st.session_state and st.session_state.mp_options:
                tabs.append("Materials Project")
            if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                tabs.append("AFLOW")
            if 'cod_options' in st.session_state and st.session_state.cod_options:
                tabs.append("COD")

            if tabs:
                selected_tab = st.tabs(tabs)

                tab_index = 0
                if 'mp_options' in st.session_state and st.session_state.mp_options:
                    with selected_tab[tab_index]:
                        st.subheader("🧬 Structures Found in Materials Project")
                        selected_structure = st.selectbox("Select a structure from MP:",
                                                          st.session_state.mp_options)
                        selected_id = selected_structure.split(":")[0].strip()
                        composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
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
                                f"**Material ID:** {selected_id}, **Formula:** {composition}, N. of Atoms {n_atoms}")

                            st.write(
                                f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                            st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                            mp_url = f"https://materialsproject.org/materials/{selected_id}"
                            st.write(f"**Link:** {mp_url}")

                            col_mpd, col_mpb = st.columns([2, 1])
                            with col_mpd:
                                if st.button("Add Selected Structure (MP)", key="add_btn_mp"):
                                    pmg_structure = st.session_state.full_structures_see[selected_id]
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
                    tab_index += 1

                if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                    with selected_tab[tab_index]:
                        st.subheader("🧬 Structures Found in AFLOW")
                        st.warning(
                            "The AFLOW does not provide atomic occupancies and includes only information about primitive cell in API. For better performance, volume and n. of atoms are purposely omitted from the expander.")
                        selected_structure = st.selectbox("Select a structure from AFLOW:",
                                                          st.session_state.aflow_options)
                        selected_auid = selected_structure.split(": ")[0].strip()
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
                        cod_id = selected_cod_structure.split(":")[0].strip()
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
                                st.success("Structure added from COD!")

                            st.download_button(
                                label="Download COD CIF",
                                data=str(CifWriter(selected_entry, symprec=0.01)),
                                file_name=file_name,
                                mime="chemical/x-cif", type="primary",
                            )

if "first_run_note" not in st.session_state:
    st.session_state["first_run_note"] = True

if st.session_state["first_run_note"] == True:
    st.info("""
    From the **sidebar**, choose the calculation tool you'd like to use — **Structure Modification**, **Powder Diffraction Calculator**, **(P)RDF Calculator**, or **Interactive Data Plot**. Next, use the **sidebar** to **upload your crystal structure files** (**CIF**, **POSCAR**, **LMP**, or **XSF** formats) or your **two-column experimental data**.  
    If you don’t have crystal structure files, you can directly **add them using the search interface** for the **online databases** above 🐣.
    """)
    st.session_state["first_run_note"] = False


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
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")
else:
    uploaded_files = st.session_state['uploaded_files']

if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")

st.sidebar.markdown("### Final List of Structure Files:")
st.sidebar.write([f.name for f in uploaded_files])

st.sidebar.markdown("### 🗑️ Remove structure(s) added from online databases")

files_to_remove = []
for i, file in enumerate(st.session_state['uploaded_files']):
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(file.name)
    if col2.button("❌", key=f"remove_{i}"):
        files_to_remove.append(file)

if files_to_remove:
    for f in files_to_remove:
        st.session_state['uploaded_files'].remove(f)
    st.rerun()

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


def wrap_coordinates(frac_coords):
    coords = np.array(frac_coords)
    return coords % 1


def compute_periodic_distance_matrix(frac_coords):
    n = len(frac_coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            delta = frac_coords[i] - frac_coords[j]
            delta = delta - np.round(delta)
            dist = np.linalg.norm(delta)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


def select_spaced_points(frac_coords, n_points, mode, target_value=0.5):
    coords_wrapped = wrap_coordinates(frac_coords)
    dist_matrix = compute_periodic_distance_matrix(coords_wrapped)
    import random
    selected_indices = [random.randrange(len(coords_wrapped))]
    for _ in range(1, n_points):
        remaining = [i for i in range(len(coords_wrapped)) if i not in selected_indices]
        if mode == "farthest":
            next_index = max(remaining,
                             key=lambda i: min(dist_matrix[i, j] for j in selected_indices))
        elif mode == "nearest":
            next_index = min(remaining,
                             key=lambda i: min(dist_matrix[i, j] for j in selected_indices))
        elif mode == "moderate":
            next_index = min(remaining, key=lambda i: abs(
                sum(dist_matrix[i, j] for j in selected_indices) / len(
                    selected_indices) - target_value))
        else:
            raise ValueError(
                "Invalid selection mode. Use 'farthest', 'nearest', or 'moderate'.")
        selected_indices.append(next_index)

    selected_coords = np.array(coords_wrapped)[selected_indices].tolist()
    return selected_coords, selected_indices


# ---------- Interstitial Functions ----------

def classify_interstitial_site(structure, frac_coords, dummy_element="H"):
    from pymatgen.analysis.local_env import CrystalNN
    temp_struct = structure.copy()
    temp_struct.append(dummy_element, frac_coords, coords_are_cartesian=False)
    cnn = CrystalNN()
    try:
        nn_info = cnn.get_nn_info(temp_struct, len(temp_struct) - 1)
    except Exception as e:
        st.write("CrystalNN error:", e)
        nn_info = []
    cn = len(nn_info)

    if cn == 4:
        return f"CN = {cn} **(Tetrahedral)**"
    elif cn == 6:
        return f"CN = {cn} **(Octahedral)**"
    elif cn == 3:
        return f"CN = {cn} (Trigonal Planar)"
    elif cn == 5:
        return f"CN = {cn} (Trigonal Bipyramidal)"
    else:
        return f"CN = {cn}"


def insert_interstitials_into_structure(structure, interstitial_element, n_interstitials,
                                        which_interstitial=0, mode="farthest",
                                        clustering_tol=0.75,
                                        min_dist=0.5):
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
    with colb3:
        st.session_state.removal_message = ""
        with st.spinner(f"Calculating available interstitials positions, please wait. 😊"):
            generator = VoronoiInterstitialGenerator(clustering_tol=clustering_tol,
                                                     min_dist=min_dist)

            frac_coords = []
            frac_coords_dict = {}
            unique_int = []
            idx = 0
            # Collect candidate sites from the generator.
            for interstitial in generator.generate(structure, "H"):
                frac_coords_dict[idx] = []
                unique_int.append(interstitial.site.frac_coords)
                label = classify_interstitial_site(structure, interstitial.site.frac_coords)
                rounded_coords = [round(float(x), 3) for x in interstitial.site.frac_coords]
                #st.write(
                #    f"🧠 Unique interstitial site (**Type {idx + 1}**)  at {rounded_coords}, {label} (#{len(interstitial.equivalent_sites)} sites)")

                new_msg = f"🧠 Unique interstitial site (**Type {idx + 1}**)  at {rounded_coords}, {label} (#{len(interstitial.equivalent_sites)} sites)\n"
                st.session_state.removal_message += new_msg + "\n"
                for site in interstitial.equivalent_sites:
                    frac_coords.append(site.frac_coords)
                    frac_coords_dict[idx].append(site.frac_coords)
                idx += 1

            #st.write(f"**Total number of available interstitial positions:**", len(frac_coords))
            new_msg = f"**Total number of available interstitial positions:** {len(frac_coords)}\n"
            st.session_state.removal_message += new_msg + "\n"


            if st.session_state.removal_message:
                st.write(st.session_state.removal_message)


            if which_interstitial == 0:
                frac_coords_use = frac_coords
            else:
                frac_coords_use = frac_coords_dict.get(which_interstitial - 1, [])

            selected_points, _ = select_spaced_points(frac_coords_use, n_points=n_interstitials,
                                                      mode=mode)
            new_structure = structure.copy()
            for point in selected_points:
                new_structure.append(
                    species=Element(interstitial_element),
                    coords=point,
                    coords_are_cartesian=False
                )
        return new_structure


def remove_vacancies_from_structure(structure, vacancy_percentages, selection_mode="farthest",
                                    target_value=0.5):
    with colb3:
        with st.spinner(f"Creating substitutes, please wait. 😊"):
            new_structure = structure.copy()
            indices_to_remove = []
            for el, perc in vacancy_percentages.items():
                el_indices = [i for i, site in enumerate(new_structure.sites) if
                              site.specie.symbol == el]
                n_sites = len(el_indices)
                n_remove = int(round(n_sites * perc / 100.0))
                st.write(f"🧠 Removed {n_remove} atoms of {el}.")
                if n_remove < 1:
                    continue
                el_coords = [new_structure.sites[i].frac_coords for i in el_indices]
                if n_remove < len(el_coords):
                    _, selected_local_indices = select_spaced_points(el_coords,
                                                                     n_points=n_remove,
                                                                     mode=selection_mode,
                                                                     target_value=target_value)
                    selected_global_indices = [el_indices[i] for i in selected_local_indices]
                else:
                    selected_global_indices = el_indices
                indices_to_remove.extend(selected_global_indices)
            for i in sorted(indices_to_remove, reverse=True):
                new_structure.remove_sites([i])
    return new_structure


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
        wyckoffs = sga.get_symmetry_dataset()["wyckoffs"]
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


def substitute_atoms_in_structure(structure, substitution_dict, selection_mode="farthest",
                                  target_value=0.5):
    with colb3:
        with st.spinner(f"Creating substitutes, please wait. 😊"):
            new_species = [site.species_string for site in structure.sites]
            new_coords = [site.frac_coords for site in structure.sites]
            for orig_el, settings in substitution_dict.items():
                perc = settings.get("percentage", 0)
                sub_el = settings.get("substitute", "").strip()
                if perc <= 0 or not sub_el:
                    continue
                indices = [i for i, site in enumerate(structure.sites) if
                           site.specie.symbol == orig_el]
                n_sites = len(indices)
                n_substitute = int(round(n_sites * perc / 100.0))
                st.write(f"🧠 Replaced {n_substitute} atoms of {orig_el} with {sub_el}.")

                if n_substitute < 1:
                    continue
                el_coords = [new_coords[i] for i in indices]
                if n_substitute < len(el_coords):
                    _, selected_local_indices = select_spaced_points(el_coords,
                                                                     n_points=n_substitute,
                                                                     mode=selection_mode,
                                                                     target_value=target_value)
                    selected_global_indices = [indices[i] for i in selected_local_indices]
                else:
                    selected_global_indices = indices
                for i in selected_global_indices:
                    new_species[i] = sub_el
            new_structure = Structure(structure.lattice, new_species, new_coords,
                                      coords_are_cartesian=False)
    return new_structure

if "run_before" not in st.session_state:
    st.session_state["run_before"] = False

if "🔬 Structure Modification" in calc_mode:
    auto_save_structure = False
    auto_save_filename = False
    show_structure = True
    #st.info("First, upload your crystal structures or add them from online databases. ")
    if uploaded_files:
        if "helpful" not in st.session_state:
            st.session_state["helpful"] = False

        if show_structure:
            col_viz, col_mod, col_download = st.columns(3)
            if "current_structure" not in st.session_state:
                st.session_state["current_structure"] = None

            #FOR COMPARISON IF SELECTED FILE CHANGED
            if "selected_file" not in st.session_state:
                st.session_state["selected_file"] = None
            prev_selected_file = st.session_state.get("selected_file")
            with col_viz:
                file_options = [file.name for file in uploaded_files]
                st.subheader("Select Structure for Interactive Visualization:")
                if len(file_options) > 5:
                    selected_file = st.selectbox("", file_options)
                else:
                    selected_file = st.radio("", file_options)
            with col_mod:
                #apply_cell_conversion = st.checkbox(f"🧱 Find a **new symmetry**", value=False)
                cell_convert_or = st.checkbox(
                    f"🧱 Allow **conversion** between **cell representations** (will lead to lost occupancies)",
                    value=False)
                if cell_convert_or:
                    structure_cell_choice = st.radio(
                        "Structure Cell Type:",
                        options=["Conventional Cell", "Primitive Cell (Niggli)", "Primitive Cell (LLL)",
                                 "Primitive Cell (no reduction)"],
                        index=0,  # default to Conventional
                        help="Choose whether to use the crystallographic Primitive Cell or the Conventional Unit Cell for the structures. For Primitive Cell, you can select whether to use Niggli or LLL (Lenstra–Lenstra–Lovász) "
                             "lattice basis reduction algorithm to produce less skewed representation of the lattice. The MP database is using Niggli-reduced Primitive Cells."
                    )
                    convert_to_conventional = structure_cell_choice == "Conventional Cell"
                    pymatgen_prim_cell_niggli = structure_cell_choice == "Primitive Cell (Niggli)"
                    pymatgen_prim_cell_lll = structure_cell_choice == "Primitive Cell (LLL)"
                    pymatgen_prim_cell_no_reduce = structure_cell_choice == "Primitive Cell (no reduction)"
            if selected_file != st.session_state["selected_file"]:

                #IF SELECTED FILE CHANGED, RESETTING ALL MODIFICATIONS
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

        if cell_convert_or:
            if convert_to_conventional:
                converted_structure = get_full_conventional_structure(mp_struct, symprec=0.1)
            elif pymatgen_prim_cell_niggli:
                analyzer = SpacegroupAnalyzer(mp_struct)
                converted_structure = analyzer.get_primitive_standard_structure()
                converted_structure = converted_structure.get_reduced_structure(reduction_algo="niggli")
            elif pymatgen_prim_cell_lll:
                analyzer = SpacegroupAnalyzer(mp_struct)
                converted_structure = analyzer.get_primitive_standard_structure()
                converted_structure = converted_structure.get_reduced_structure(reduction_algo="LLL")
            else:
                analyzer = SpacegroupAnalyzer(mp_struct)
                converted_structure = analyzer.get_primitive_standard_structure()
            mp_struct = converted_structure
            st.session_state.modified_atom_df = generate_initial_df_with_occupancy_and_wyckoff(mp_struct)
            st.session_state["current_structure"] = mp_struct
            st.session_state["original_for_supercell"] = mp_struct

        for i, site in enumerate(mp_struct.sites):
            frac = site.frac_coords
            cart = mp_struct.lattice.get_cartesian_coords(frac)

        from pymatgen.transformations.standard_transformations import SupercellTransformation

        if "supercell_n_a" not in st.session_state:
            st.session_state["supercell_n_a"] = 1
        if "supercell_n_b" not in st.session_state:
            st.session_state["supercell_n_b"] = 1
        if "supercell_n_c" not in st.session_state:
            st.session_state["supercell_n_c"] = 1

        old_a = st.session_state.get("supercell_n_a", 1)
        old_b = st.session_state.get("supercell_n_b", 1)
        old_c = st.session_state.get("supercell_n_c", 1)
        st.subheader("Edit Structure if Needed")

        create_defects = st.checkbox(
            f"Create **Supercell** and **Point Defects**",
            value=False)
        #with col_mod:
            # apply_cell_conversion = st.checkbox(f"🧱 Find a **new symmetry**", value=False)
           #if st.button(f"🧱 **New symmetry** (conventional cell, will lead to lost occupancies"):
            #    mp_struct = st.session_state["current_structure"]
                #converted_structure = get_full_conventional_structure(mp_struct, symprec=0.1)
                #st.session_state["current_structure"] = converted_structure
            #    analyzer = SpacegroupAnalyzer(mp_struct, symprec=0.1)

                # Get the conventional standard structure
            #    converted_structure = analyzer.get_conventional_standard_structure()

                # Print symmetry information
            #    spacegroup = analyzer.get_space_group_symbol()
            #    st.write(f"Structure converted to conventional cell with spacegroup: {spacegroup}")

                # Update the session state with the new structure
            #    st.session_state["current_structure"] = converted_structure

                # Display information about the new structure
            #    st.write(f"Lattice parameters: {converted_structure.lattice.abc}")
            #    st.write(f"Lattice angles: {converted_structure.lattice.angles}")
            #    st.write(f"Number of sites: {len(converted_structure)}")
            #    st.session_state["current_structure"] = converted_structure
        if create_defects:
            from pymatgen.core import Structure, Element

            with st.expander("Create Supercell (Uncheck the conversion between cells) and Point Defects", icon='🧿', expanded=st.session_state["expander_defects"]):

                colb1, colb2, colb3 = st.columns(3)

                with colb1:
                    col1, col2, col3 = st.columns(3)
                    st.session_state["expander_supercell"] = True
                    n_a = col1.number_input("Repeat along a-axis", min_value=1, max_value=50,
                                            value=st.session_state["supercell_n_a"], step=1)
                    n_b = col2.number_input("Repeat along b-axis", min_value=1, max_value=50,
                                            value=st.session_state["supercell_n_b"], step=1)
                    n_c = col3.number_input("Repeat along c-axis", min_value=1, max_value=50,
                                            value=st.session_state["supercell_n_c"], step=1)

                st.session_state["supercell_n_a"] = n_a
                st.session_state["supercell_n_b"] = n_b
                st.session_state["supercell_n_c"] = n_c

                supercell_matrix = [[n_a, 0, 0], [0, n_b, 0], [0, 0, n_c]]

                if (n_a, n_b, n_c) != (old_a, old_b, old_c):
                    transformer = SupercellTransformation(supercell_matrix)

                    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

                    mp_struct = remove_fractional_occupancies_safely(st.session_state["original_for_supercell"])
                    mp_struct = transformer.apply_transformation(mp_struct)

                    st.session_state["current_structure"] = mp_struct
                    st.session_state["auto_saved_structure"] = mp_struct

                    converted_structure = mp_struct
                    # st.rerun()




                else:
                    print("DIDNT APPLY ANYTHING")
                    converted_structure = mp_struct

                st.session_state.modified_atom_df = generate_initial_df_with_occupancy_and_wyckoff(converted_structure)

                st.write("Cell representation conversion is now applied!")
                mp_struct = converted_structure
                visual_pmg_structure = mp_struct
                st.session_state["current_structure"] = mp_struct


                with colb2:
                    st.session_state["expander_defects"] = True
                    # ==================== Substitute Functions ====================
                    with colb3:
                        st.markdown(f"### Log output:")

                    operation_mode = st.selectbox("Choose Operation Mode",
                                                  ["Insert Interstitials (Voronoi method)", "Create Vacancies",
                                                   "Substitute Atoms"], help="""
                    #Interstitials settings
                    - **Element**: The chemical symbol of the interstitial atom you want to insert (e.g., `N` for nitrogen).
                    - **# to Insert**: The number of interstitial atoms to insert into the structure.
                    - **Type (0=all, 1=first...)**: Selects a specific interstitial site type.  
                      - `0` uses all detected interstitial sites.  
                      - `1` uses only the first unique type, `2` for second, etc.

                    - **Selection Mode**: How to choose which interstitial sites to use:  
                      - `farthest`: picks sites farthest apart from each other.  
                      - `nearest`: picks sites closest together.  
                      - `moderate`: balances distances around a target value.

                    - **Clustering Tol**: Tolerance for clustering nearby interstitial candidates together (higher = more merging).
                    - **Min Dist**: Minimum allowed distance between interstitials and other atoms when generating candidate sites. Do not consider any candidate site that is closer than this distance to an existing atom.

                    #Vacancy settings
                    - **Vacancy Selection Mode**: Strategy for choosing which atoms to remove:
                      - `farthest`: removes atoms that are farthest apart, to maximize spacing.
                      - `nearest`: removes atoms closest together, forming local vacancy clusters.
                      - `moderate`: selects atoms to remove so that the average spacing between them is close to a target value.

                    - **Target (moderate mode)**: Only used when `moderate` mode is selected.  
                      This value defines the average spacing (in fractional coordinates) between vacancies.

                    - **Vacancy % for [Element]**: Percentage of atoms to remove for each element.  
                      For example, if there are 20 O atoms and you set 10%, two O atoms will be randomly removed based on the selection mode.

                    #Substitution settings
                    - **Substitution Selection Mode**: Strategy to determine *which* atoms of a given element are substituted:
                      - `farthest`: substitutes atoms spaced far apart from each other.
                      - `nearest`: substitutes atoms that are close together.
                      - `moderate`: substitutes atoms spaced at an average distance close to the specified target.

                    - **Target (moderate mode)**: Only used when `moderate` mode is selected.  
                      It defines the preferred average spacing (in fractional coordinates) between substituted atoms.

                    - **Substitution % for [Element]**: How many atoms (as a percentage) of a given element should be substituted.

                    - **Substitute [Element] with**: The element symbol you want to use as a replacement.  
                      Leave blank or set substitution % to 0 to skip substitution for that element.
                            """)

                    if operation_mode == "Insert Interstitials (Voronoi method)":
                        st.markdown("""
                            **Insert Interstitials Settings**
                            """)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            interstitial_element_to_place = st.text_input("Element", value="N")
                        with col2:
                            number_of_interstitials_to_insert = st.number_input("# to Insert", value=2, min_value=1)
                        with col3:
                            which_interstitial_to_use = st.number_input("Type (0=all, 1=first...)", value=0,
                                                                        min_value=0)

                        col4, col5, col6 = st.columns(3)
                        with col4:
                            selection_mode = st.selectbox("Selection Mode",
                                                          options=["farthest", "nearest", "moderate"],
                                                          index=0)
                        with col5:
                            clustering_tol = st.number_input("Clustering Tol", value=0.75, step=0.05, format="%.2f")
                        with col6:
                            min_dist = st.number_input("Min Dist", value=0.5, step=0.05, format="%.2f")

                    elif operation_mode == "Create Vacancies":
                        st.markdown("""

                            """)

                        col1, col2 = st.columns(2)
                        vacancy_selection_mode = col1.selectbox("Vacancy Selection Mode",
                                                                ["farthest", "nearest", "moderate"], index=0)
                        if vacancy_selection_mode == "moderate":
                            vacancy_target_value = col2.number_input("Target (moderate mode)", value=0.5, step=0.05,
                                                                     format="%.2f")
                        else:
                            vacancy_target_value = 0.5

                        elements = sorted({site.specie.symbol for site in mp_struct.sites})
                        cols = st.columns(len(elements))
                        vacancy_percentages = {
                            el: cols[i].number_input(f"Vacancy % for {el}", value=0.0, min_value=0.0,
                                                     max_value=100.0,
                                                     step=1.0, format="%.1f")
                            for i, el in enumerate(elements)}

                    elif operation_mode == "Substitute Atoms":
                        st.markdown("""
                            **Substitution Settings**
                            """)

                        col1, col2 = st.columns(2)
                        substitution_selection_mode = col1.selectbox("Substitution Selection Mode",
                                                                     ["farthest", "nearest", "moderate"], index=0)
                        if substitution_selection_mode == "moderate":
                            substitution_target_value = col2.number_input("Target (moderate mode)", value=0.5,
                                                                          step=0.05,
                                                                          format="%.2f")
                        else:
                            substitution_target_value = 0.5

                        elements = sorted({site.specie.symbol for site in mp_struct.sites})
                        cols = st.columns(len(elements))
                        substitution_settings = {}
                        for i, el in enumerate(elements):
                            with cols[i]:
                                sub_perc = st.number_input(f"Substitution % for {el}", value=0.0, min_value=0.0,
                                                           max_value=100.0, step=1.0, format="%.1f",
                                                           key=f"sub_perc_{el}")
                                sub_target = st.text_input(f"Substitute {el} with", value="",
                                                           key=f"sub_target_{el}")
                            substitution_settings[el] = {"percentage": sub_perc, "substitute": sub_target.strip()}

                    # ==================== Execute Operation ====================
                    if operation_mode == "Insert Interstitials (Voronoi method)":

                        if st.button("Insert Interstitials"):
                            updated_structure = insert_interstitials_into_structure(mp_struct,
                                                                                    interstitial_element_to_place,
                                                                                    number_of_interstitials_to_insert,
                                                                                    which_interstitial_to_use,
                                                                                    mode=selection_mode,
                                                                                    clustering_tol=clustering_tol,
                                                                                    min_dist=min_dist)

                            mp_struct = updated_structure
                            st.session_state["current_structure"] = updated_structure
                            st.session_state["original_for_supercell"] = updated_structure
                            st.session_state["supercell_n_a"] = 1
                            st.session_state["supercell_n_b"] = 1
                            st.session_state["supercell_n_c"] = 1
                            visual_pmg_structure = mp_struct

                            with colb3:
                                st.success("Interstitials inserted and structure updated!")
                            st.session_state["helpful"] = True
                    elif operation_mode == "Create Vacancies":
                        if not st.session_state["modified_defects"]:
                            mp_struct = mp_struct.copy()
                        else:
                            mp_struct = st.session_state["modified_defects"]
                        if st.button("Create Vacancies"):
                            updated_structure = remove_vacancies_from_structure(mp_struct,
                                                                                vacancy_percentages,
                                                                                selection_mode=vacancy_selection_mode,
                                                                                target_value=vacancy_target_value)

                            mp_struct = updated_structure
                            st.session_state["current_structure"] = updated_structure
                            st.session_state["last_multiplier"] = (1, 1, 1)
                            visual_pmg_structure = mp_struct
                            st.session_state["original_for_supercell"] = updated_structure
                            st.session_state["supercell_n_a"] = 1
                            st.session_state["supercell_n_b"] = 1
                            st.session_state["supercell_n_c"] = 1
                            with colb3:
                                st.success("Vacancies created and structure updated!")
                            st.session_state["helpful"] = True
                    elif operation_mode == "Substitute Atoms":
                        if not st.session_state["modified_defects"]:
                            mp_struct = mp_struct.copy()
                        else:
                            mp_struct = st.session_state["modified_defects"]
                        if st.button("Substitute Atoms"):
                            updated_structure = substitute_atoms_in_structure(mp_struct,
                                                                              substitution_settings,
                                                                              selection_mode=substitution_selection_mode,
                                                                              target_value=substitution_target_value)

                            mp_struct = updated_structure
                            st.session_state["current_structure"] = updated_structure
                            st.session_state["original_for_supercell"] = updated_structure
                            st.session_state["supercell_n_a"] = 1
                            st.session_state["supercell_n_b"] = 1
                            st.session_state["supercell_n_c"] = 1
                            visual_pmg_structure = mp_struct
                            with colb3:
                                st.success("Substitutions applied and structure updated!")
                            st.session_state["helpful"] = True
        else:
            st.session_state["current_structure"] = mp_struct
            #st.session_state["original_for_supercell"] = mp_struct
            visual_pmg_structure = mp_struct


        st.session_state.modified_atom_df = generate_initial_df_with_occupancy_and_wyckoff(mp_struct)

        col_g1, col_g2 = st.columns([1, 4])

        with col_g1:
            show_plot_str = st.checkbox(f"Show 3D structure plot", value = True)
            #allow_atomic_mod = st.checkbox(f"Allow **atomic site modifications**", value=False)
            unique_wyckoff_only = st.checkbox(
                "Visualize only atoms in **asymmetric unit**",
                value=False)
        full_df = st.session_state.modified_atom_df.copy()

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
                count = grouped[(grouped['Wyckoff'] == wyckoff) & (grouped['Element'] == element)]['count'].values[0]
                if count > 1:
                    display_df.at[i, 'Wyckoff'] = f"{count}{wyckoff}"
        else:
            display_df = full_df


        if unique_wyckoff_only:
            st.info(
                "ℹ️ When editing atoms in asymmetric unit view, changes will be propagated to all symmetrically equivalent atoms with the same Wyckoff position.")

        editor_key = "atom_editor_unique" if unique_wyckoff_only else "atom_editor_full"
        with st.expander("Modify atomic sites", icon='⚛️', expanded = True): #expanded=st.session_state["expander_atomic_sites"]
            st.session_state["expander_open"] = True
            edited_df = st.data_editor(
                display_df,
                num_rows="dynamic",
                key=editor_key,
                column_config={
                    "Occupancy": st.column_config.NumberColumn(
                        "Occupancy",
                        min_value=0.01,
                        max_value=1.00,
                        step=0.01,
                        format="%.2f",  # ensures decimals
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
                #st.session_state.modified_atom_df = edited_df.copy()
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

            if not edited_df_reset.equals(display_df_reset): #and allow_atomic_mod:
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

                                #st.info(
                                #    f"Updated {element_mask.sum()} atoms at Wyckoff positions with '{wyckoff_letter}': Element changed from {original_element} to {element}")

                        if 'Occupancy' in changed_props:
                            occ_mask = wyckoff_mask & (full_df_copy['Element'] == element)

                            if occ_mask.sum() > 0:
                                full_df_copy.loc[occ_mask, 'Occupancy'] = changed_props['Occupancy']

                                #st.info(
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

                                #st.info(f"Position for {element} at Wyckoff position with '{wyckoff_letter}' was updated.")
                                #st.warning("Note: Changing atomic positions may break the crystal symmetry.")
                            else:
                                pass
                                #st.error(
                                #    f"Could not find exact matching atom to update position. Found {exact_match.sum()} matches.")

                    st.session_state.modified_atom_df = recalc_computed_columns(full_df_copy, visual_pmg_structure.lattice)
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
                        #st.session_state["original_for_supercell"] = mp_struct
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
                        #st.session_state["original_for_supercell"] = mp_struct
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
            show_atom_labels = st.checkbox(f"**Show** atom **labels** in 3D visualization", value=True)


        if create_defects == False:
            with st.expander("Modify Lattice Parameters", icon='📐', expanded=st.session_state["expander_lattice"]):

                if "lattice_a" not in st.session_state:
                    st.session_state["lattice_a"] = visual_pmg_structure.lattice.a
                if "lattice_b" not in st.session_state:
                    st.session_state["lattice_b"] = visual_pmg_structure.lattice.b
                if "lattice_c" not in st.session_state:
                    st.session_state["lattice_c"] = visual_pmg_structure.lattice.c
                if "lattice_alpha" not in st.session_state:
                    st.session_state["lattice_alpha"] = visual_pmg_structure.lattice.alpha
                if "lattice_beta" not in st.session_state:
                    st.session_state["lattice_beta"] = visual_pmg_structure.lattice.beta
                if "lattice_gamma" not in st.session_state:
                    st.session_state["lattice_gamma"] = visual_pmg_structure.lattice.gamma

                old_a = visual_pmg_structure.lattice.a
                old_b = visual_pmg_structure.lattice.b
                old_c = visual_pmg_structure.lattice.c
                old_alpha = visual_pmg_structure.lattice.alpha
                old_beta = visual_pmg_structure.lattice.beta
                old_gamma = visual_pmg_structure.lattice.gamma

                try:
                    sga = SpacegroupAnalyzer(visual_pmg_structure)
                    crystal_system = sga.get_crystal_system()
                    spg_symbol = sga.get_space_group_symbol()
                    spg_number = sga.get_space_group_number()
                    st.info(f"Crystal system: **{crystal_system.upper()}** | Space group: **{spg_symbol} (#{spg_number})**")

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
                                            value=float(old_a),
                                            min_value=0.1,
                                            max_value=100.0,
                                            step=0.01,
                                            format="%.5f")

                with col_b:
                    if "b" in modifiable:
                        new_b = st.number_input("b (Å)",
                                                value=float(old_b),
                                                min_value=0.1,
                                                max_value=100.0,
                                                step=0.01,
                                                format="%.5f")
                    else:

                        if crystal_system in ["cubic", "tetragonal", "hexagonal", "trigonal"]:
                            st.text_input("b (Å) = a", value=f"{float(new_a):.5f}", disabled=True)
                            new_b = new_a
                        else:
                            st.text_input("b (Å)", value=f"{float(old_b):.5f}", disabled=True)
                            new_b = old_b

                with col_c:
                    if "c" in modifiable:
                        new_c = st.number_input("c (Å)",
                                                value=float(old_c),
                                                min_value=0.1,
                                                max_value=100.0,
                                                step=0.01,
                                                format="%.5f")
                    else:
                        if crystal_system == "cubic":
                            st.text_input("c (Å) = a", value=f"{float(new_a):.5f}", disabled=True)
                            new_c = new_a
                        else:
                            st.text_input("c (Å)", value=f"{float(old_c):.5f}", disabled=True)
                            new_c = old_c

                with col_alpha:
                    if "alpha" in modifiable:
                        new_alpha = st.number_input("α (°)",
                                                    value=float(old_alpha),
                                                    min_value=0.1,
                                                    max_value=179.9,
                                                    step=0.1,
                                                    format="%.5f")
                    else:
                        if crystal_system in ["cubic", "tetragonal", "orthorhombic", "hexagonal", "monoclinic"]:
                            st.text_input("α (°)", value="90.00000", disabled=True)
                            new_alpha = 90.0
                        else:
                            st.text_input("α (°)", value=f"{float(old_alpha):.5f}", disabled=True)
                            new_alpha = old_alpha

                with col_beta:
                    if "beta" in modifiable:
                        new_beta = st.number_input("β (°)",
                                                   value=float(old_beta),
                                                   min_value=0.1,
                                                   max_value=179.9,
                                                   step=0.1,
                                                   format="%.5f")
                    else:
                        if crystal_system in ["cubic", "tetragonal", "orthorhombic", "hexagonal"]:
                            st.text_input("β (°)", value="90.00000", disabled=True)
                            new_beta = 90.0
                        elif crystal_system == "trigonal" and "alpha" in modifiable:
                            st.text_input("β (°) = α", value=f"{float(new_alpha):.5f}", disabled=True)
                            new_beta = new_alpha
                        else:
                            st.text_input("β (°)", value=f"{float(old_beta):.5f}", disabled=True)
                            new_beta = old_beta

                with col_gamma:
                    if "gamma" in modifiable:
                        new_gamma = st.number_input("γ (°)",
                                                    value=float(old_gamma),
                                                    min_value=0.1,
                                                    max_value=179.9,
                                                    step=0.1,
                                                    format="%.5f")
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
                            st.text_input("γ (°)", value=f"{float(old_gamma):.5f}", disabled=True)
                            new_gamma = old_gamma

                st.session_state["lattice_a"] = new_a
                st.session_state["lattice_b"] = new_b
                st.session_state["lattice_c"] = new_c
                st.session_state["lattice_alpha"] = new_alpha
                st.session_state["lattice_beta"] = new_beta
                st.session_state["lattice_gamma"] = new_gamma

                if st.button("Apply Lattice Changes"):
                    try:
                        st.session_state["expander_lattice"] = True

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
                        #st.session_state["original_for_supercell"] = mp_struct

                        if "modified_atom_df" in st.session_state:
                            st.session_state.modified_atom_df = recalc_computed_columns(
                                st.session_state.modified_atom_df.copy(),
                                updated_structure.lattice
                            )

                        st.success("Lattice parameters updated successfully!")


                    except Exception as e:
                        st.error(f"Error updating lattice parameters: {e}")
                    st.rerun()
        else:
            st.info(f'If you wish to directly modify lattice parameters, uncheck first the Create Supercell and Point Defects')

        df_plot = df_plot.copy()

        with col_g1:
            base_atom_size = st.slider(
                "Base atom size in visualization:",
                min_value=1,
                max_value=30,
                value=20,
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
                        label_parts.append(f"{element}{occ:.1f}")


                if vacancy > 0.01:
                    label_parts.append(f"□{vacancy:.1f}")  # Square symbol for vacancy

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
                    pos_key = (x, y, z)

                    if pos_key in atom_labels_dict:
                        label = atom_labels_dict[pos_key]
                    else:

                        label = dominant_element
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

            cell = visual_pmg_structure.lattice.matrix  # 3x3 array; each row is a lattice vector.
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

                st.session_state.uploaded_files.append(cif_file)

                if "final_structures" not in st.session_state:
                    st.session_state.final_structures = {}

                new_key = custom_filename.replace(".cif", "")
                st.session_state.final_structures[new_key] = new_struct

                st.success(f"Modified structure added as '{new_key}'!")
                st.write("Final list of structures in calculator:")
                st.write(list(st.session_state.final_structures.keys()))
            except Exception as e:
                st.error(f"Error reconstructing structure: {e}")
                st.error(
                    f"You probably added some new atom which has the same fractional coordinates as already defined atom, but you did not modify their occupancies. If the atoms share the same atomic site, their total occupancy must be equal to 1.")
            st.rerun()
        lattice = visual_pmg_structure.lattice
        a_para = lattice.a
        b_para = lattice.b
        c_para = lattice.c
        alpha = lattice.alpha
        beta = lattice.beta
        gamma = lattice.gamma
        volume = lattice.volume

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
            try:
                sg_analyzer = SpacegroupAnalyzer(visual_pmg_structure)
                spg_symbol = sg_analyzer.get_space_group_symbol()
                spg_number = sg_analyzer.get_space_group_number()
                space_group_str = f"{spg_symbol} ({spg_number})"

                structure_type = identify_structure_type(visual_pmg_structure)
                str_type = f"{structure_type}"

                same_lattice = lattice_same_conventional_vs_primitive(visual_pmg_structure)
                if same_lattice is None:
                    cell_note = "⚠️ Could not determine if cells are identical."
                    cell_note_color = "gray"
                elif same_lattice:
                    cell_note = "✅ Note: Conventional and Primitive Cells have the SAME cell volume."
                    cell_note_color = "green"
                else:
                    cell_note = "Note: Conventional and Primitive Cells have DIFFERENT cell volume."
                    cell_note_color = "gray"
            except Exception:
                space_group_str = "Not available"
                cell_note = "⚠️ Could not determine space group or cell similarity."
                cell_note_color = "gray"

            st.markdown(f"""
            <div style='text-align: center; font-size: 18px; color: {"green" if same_lattice else "gray"}'>
                <strong>{cell_note}</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='text-align: center; font-size: 18px;'>
                <p><strong>Lattice Parameters:</strong><br>{lattice_str}</p>
                <p><strong>Number of Atoms:</strong> {len(visual_pmg_structure)}</p>
                <p><strong>Space Group:</strong> {space_group_str}</p>
                <p><strong>Structure Type:</strong> {str_type}</p>
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
                                             0] + '_' + str(spg_number) + f'.cif'

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
                                             0] + '_' + str(spg_number) + f'.poscar'

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
                                             0] + '_' + str(spg_number) + f'_.lmp'

                elif file_format == "XYZ":
                    from pymatgen.io.cif import CifWriter

                    mime = "chemical/x-cif"

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
                    current_ase_structure = AseAtomsAdaptor.get_atoms(new_struct)
                    out = StringIO()
                    write(out, current_ase_structure, format="xyz")
                    file_content = out.getvalue()
                    download_file_name = selected_file.split('.')[
                                             0] + '_' + str(spg_number) + f'_.xyz'

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

# with col_settings:

if "expander_diff_settings" not in st.session_state:
    st.session_state["expander_diff_settings"] = True

#
if "💥 Powder Diffraction" in calc_mode:
    with st.expander("Diffraction Settings", icon="⚙️", expanded = st.session_state["expander_diff_settings"]):
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
        st.session_state["expander_diff_settings"] = False

        # --- Diffraction Calculator Selection ---
        col2, col3, col4 = st.columns(3)

        with col2:
            peak_representation = st.radio(
                "Peak Representation",
                ["Delta", "Gaussian"],
                index=0,
                key="peak_representation",
                help=("Choose whether to represent each diffraction peak as a delta function "
                      "or as a Gaussian. When using Gaussian, the area under each peak equals "
                      "the calculated intensity, and overlapping Gaussians are summed.")
            )
        with col3:
            intensity_scale_option = st.radio(
                "Select intensity scale",
                options=["Normalized", "Absolute"],
                index=0,
                help="Normalized sets maximum peak to 100; Absolute shows raw calculated intensities."
            )
        with col4:
            diffraction_choice = st.radio(
                "Select Diffraction Calculator",
                ["XRD (X-ray)", "ND (Neutron)"],
                index=0,
                help="🔬 The X-ray diffraction (XRD) patterns are for **powder samples**, assuming **randomly oriented crystallites**. "
                     "The calculator applies the **Lorentz-polarization correction**: `LP(θ) = (1 + cos²(2θ)) / (sin²θ cosθ)`. It does not account for other corrections, such as preferred orientation, absorption, "
                     "instrumental broadening, or temperature effects (Debye-Waller factors). 🔬 The neutron diffraction (ND) patterns are for **powder samples**, assuming **randomly oriented crystallites**. "
                     "The calculator applies the **Lorentz correction**: `L(θ) = 1  / sin²θ cosθ`. It does not account for other corrections, such as preferred orientation, absorption, "
                     "instrumental broadening, or temperature effects (Debye-Waller factors). The main differences in the calculation from the XRD pattern are: "
                     " (1) Atomic scattering lengths are constant, and (2) Polarization correction is not necessary."
            )
        use_debye_waller = st.checkbox(
            "✓ Apply Debye-Waller temperature factors",
            value=False,
            help="Apply temperature-dependent intensity correction using Debye-Waller factors (B-factors) for each element. "
                 "This accounts for thermal motion of atoms, which reduces diffraction peak intensities. "
                 "Enter B-factor values for each element in Å² for each structure file. Typical values range from 0.5 to 3.0 Å² "
                 "Higher values (2-3 Å²) represent more thermal motion or disorder. Lower values (0.5-1 Å²) represent less thermal motion (e.g., at low temperatures). "
                 "The intensity correction is applied as: exp(-B·sin²θ/λ²)."
        )

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
            'MoKa1', 'Mo(Ka1+Ka2)', 'Mo(Ka1+Ka2+Kb1)',
            'CuKa1', 'Cu(Ka1+Ka2)', 'Cu(Ka1+Ka2+Kb1)',
            'CrKa1', 'Cr(Ka1+Ka2)', 'Cr(Ka1+Ka2+Kb1)',
            'FeKa1', 'Fe(Ka1+Ka2)', 'Fe(Ka1+Ka2+Kb1)',
            'AgKa1', 'Ag(Ka1+Ka2)', 'Ag(Ka1+Ka2+Kb1)',
        ]
        preset_wavelengths = {
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
        col1, col2, col3h = st.columns(3)
        preset_options_neutron = ['Thermal Neutrons', 'Cold Neutrons', 'Hot Neutrons']
        preset_wavelengths_neutrons = {
            'Thermal Neutrons': 0.154,
            'Cold Neutrons': 0.475,
            'Hot Neutrons': 0.087
        }

        if diffraction_choice == "XRD (X-ray)":
            with col1:
                preset_choice = st.selectbox(
                    "🌊 Preset Wavelength",
                    options=preset_options,
                    index=0,
                    help="I_Kalpha2 = 1/2 I_Kalpha1, I_Kbeta = 1/9 I_Kalpha1"
                )

            hide_input_for = [
                'Cu(Ka1+Ka2+Kb1)', 'Cu(Ka1+Ka2)',
                'Mo(Ka1+Ka2+Kb1)', 'Mo(Ka1+Ka2)',
                'Cr(Ka1+Ka2+Kb1)', 'Cr(Ka1+Ka2)',
                'Fe(Ka1+Ka2+Kb1)', 'Fe(Ka1+Ka2)',
                'Co(Ka1+Ka2+Kb1)', 'Co(Ka1+Ka2)',
                'Ag(Ka1+Ka2+Kb1)', 'Ag(Ka1+Ka2)'
            ]

            with col2:
                if preset_choice not in hide_input_for:
                    wavelength_value = st.number_input(
                        "🌊 Wavelength (nm)",
                        value=preset_wavelengths[preset_choice],
                        min_value=0.001,
                        step=0.001,
                        format="%.5f"
                    )
                else:
                    wavelength_value = preset_wavelengths[preset_choice]


        elif diffraction_choice == "ND (Neutron)":
            with col1:
                preset_choice = st.selectbox(
                    "Preset Wavelength",
                    options=preset_options_neutron,
                    index=0,
                    help="Factors for weighted average of wavelengths are: I1 = 2 (ka1), I2 = 1 (ka2), I3 = 0.18 (kb1)"
                )
            with col2:
                wavelength_value = st.number_input(
                    "Wavelength (nm)",
                    value=preset_wavelengths_neutrons[preset_choice],
                    min_value=0.001,
                    step=0.001,
                    format="%.5f"
                )

        wavelength_A = wavelength_value * 10  # Convert nm to Å
        wavelength_nm = wavelength_value

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
        colx, colx2, colx3 = st.columns([1, 1, 1])
        with colx:
            if diffraction_choice == "ND (Neutron)":
                if "x_axis_metric" not in st.session_state:
                    st.session_state.x_axis_metric = x_axis_options_neutron[0]
                x_axis_metric = st.selectbox(
                    "⚙️ ND x-axis Metric",
                    x_axis_options_neutron,
                    index=x_axis_options_neutron.index(st.session_state.x_axis_metric)
                    if st.session_state.x_axis_metric in x_axis_options_neutron else 0,
                    key="x_axis_metric",
                    help=conversion_info[st.session_state.x_axis_metric]
                )
            else:
                if "x_axis_metric" not in st.session_state:
                    st.session_state.x_axis_metric = x_axis_options[0]
                x_axis_metric = st.selectbox(
                    "⚙️ XRD x-axis Metric",
                    x_axis_options,
                    index=x_axis_options.index(st.session_state.x_axis_metric),
                    key="x_axis_metric",
                    help=conversion_info[st.session_state.x_axis_metric]
                )

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

        if peak_representation != "Delta":
            sigma = st.number_input("⚙️ Gaussian sigma (°) for peak sharpness (smaller = sharper peaks)",
                                    min_value=0.01,
                                    max_value=1.5, value=0.5, step=0.01)
        else:
            sigma = 0.5
        with col3h:
            num_annotate = st.number_input("⚙️ How many highest peaks to annotate in table (by intensity):",
                                           min_value=0,
                                           max_value=30,
                                           value=5,
                                           step=1)

        if "calc_xrd" not in st.session_state:
            st.session_state.calc_xrd = False

        if diffraction_choice == "ND (Neutron)":
            if st.button("Calculate ND"):
                st.session_state.calc_xrd = True
        else:
            if st.button("Calculate XRD"):
                st.session_state.calc_xrd = True

        # --- XRD Calculation ---
        colors = ["black", "brown", "grey", "purple"]
        if not st.session_state.calc_xrd:
            st.subheader("📊 OUTPUT → Click first on the 'Calculate XRD / ND' button.")
        if user_pattern_file:
            if isinstance(user_pattern_file, list):
                for i, file in enumerate(user_pattern_file):
                    try:
                        df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)

                        x_user = df.iloc[:, 0].values
                        y_user = df.iloc[:, 1].values
                    except Exception as e:
                        #  st.error(f"Error processing experimental file {file.name}: {e}")
                        continue

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
                        mode="lines+markers",
                        name=file.name,
                        line=dict(dash='solid', width=1, color=color),
                        marker=dict(color=color, size=3),
                        hovertemplate=(
                            f"<span style='color::{color};'><b>{file.name}:</b><br>"
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
                            title=dict(text="Intensity (a.u.)", font=dict(size=36, color='black')),
                            tickfont=dict(size=36, color='black')
                        ),
                        hoverlabel=dict(font=dict(size=24)),
                        font=dict(size=18),
                        autosize=True
                    )
            else:
                try:
                    df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                    x_user = df.iloc[:, 0].values
                    y_user = df.iloc[:, 1].values
                except Exception as e:
                    # st.error(f"Error processing experimental file {user_pattern_file.name}: {e}")
                    x_user, y_user = None, None

    if st.session_state.calc_xrd and uploaded_files:

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

        for idx, file in enumerate(uploaded_files):
            mg_structure = load_structure(file)
            mg_structure = get_full_conventional_structure_diffra(mg_structure)
            debye_waller_dict = None
            if use_debye_waller and "debye_waller_factors_per_file" in st.session_state:
                file_key = file.name
                if file_key in st.session_state.debye_waller_factors_per_file:
                    debye_waller_dict = st.session_state.debye_waller_factors_per_file[file_key]

            if is_multi_component:
                num_points = 20000
                x_dense_full = np.linspace(full_range[0], full_range[1], num_points)
                dx = x_dense_full[1] - x_dense_full[0]
                y_dense_total = np.zeros_like(x_dense_full)
                all_filtered_x = []
                all_filtered_y = []
                all_filtered_hkls = []
                all_peak_types = []
                comp_info = multi_component_presets[preset_choice]
                for comp_index, (wl, factor) in enumerate(zip(comp_info["wavelengths"], comp_info["factors"])):
                    wavelength_A_comp = wl * 10  # convert nm to Å
                    if diffraction_choice == "ND (Neutron)":
                        diff_calc = NDCalculator(wavelength=wavelength_A_comp, debye_waller_factors=debye_waller_dict)
                    else:
                        diff_calc = XRDCalculator(wavelength=wavelength_A_comp, debye_waller_factors=debye_waller_dict)
                    diff_pattern = diff_calc.get_pattern(mg_structure, two_theta_range=full_range, scaled=False)

                    filtered_x = []
                    filtered_y = []
                    filtered_hkls = []
                    for x_val, y_val, hkl_group in zip(diff_pattern.x, diff_pattern.y, diff_pattern.hkls):
                        if any(len(h['hkl']) == 3 and tuple(h['hkl'][:3]) == (0, 0, 0) for h in hkl_group):
                            continue
                        if any(len(h['hkl']) == 4 and tuple(h['hkl'][:4]) == (0, 0, 0, 0) for h in hkl_group):
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
                    y_dense_total += y_dense_comp
                    all_filtered_x.extend(filtered_x)
                    all_filtered_y.extend(filtered_y)
                    all_filtered_hkls.extend(filtered_hkls)
            else:
                if diffraction_choice == "ND (Neutron)":
                    diff_calc = NDCalculator(wavelength=wavelength_A, debye_waller_factors=debye_waller_dict)
                else:
                    diff_calc = XRDCalculator(wavelength=wavelength_A, debye_waller_factors=debye_waller_dict)
                diff_pattern = diff_calc.get_pattern(mg_structure, two_theta_range=full_range, scaled=False)
                filtered_x = []
                filtered_y = []
                filtered_hkls = []
                for x_val, y_val, hkl_group in zip(diff_pattern.x, diff_pattern.y, diff_pattern.hkls):
                    if any(len(h['hkl']) == 3 and tuple(h['hkl'][:3]) == (0, 0, 0) for h in hkl_group):
                        continue
                    if any(len(h['hkl']) == 4 and tuple(h['hkl'][:4]) == (0, 0, 0, 0) for h in hkl_group):
                        continue
                    filtered_x.append(x_val)
                    filtered_y.append(y_val)
                    filtered_hkls.append(hkl_group)
                num_points = 20000
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
                all_filtered_x = filtered_x
                all_filtered_y = filtered_y
                all_filtered_hkls = filtered_hkls
                all_peak_types = ["Kα1"] * len(filtered_x)

            if intensity_scale_option == "Normalized":
                norm_factor = np.max(all_filtered_y) if np.max(all_filtered_y) > 0 else 1.0
                y_dense_total = (y_dense_total / np.max(y_dense_total)) * 100
                displayed_intensity_array = (np.array(all_filtered_y) / norm_factor) * 100
            else:
                displayed_intensity_array = np.array(all_filtered_y)

            peak_vals = twotheta_to_metric(np.array(all_filtered_x), x_axis_metric, wavelength_A, wavelength_nm,
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

        show_user_pattern = st.sidebar.checkbox("Show uploaded XRD pattern", value=True, key="show_user_pattern")
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
                                               x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
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
                            line=dict(color=pt_color, width=4, dash=dash_type),
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
                        line=dict(color=base_color, width=3, dash="solid"),
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
                    line=dict(color=base_color, width=2),
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
            if isinstance(user_pattern_file, list):
                for i, file in enumerate(user_pattern_file):
                    try:

                        df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)

                        x_user = df.iloc[:, 0].values
                        y_user = df.iloc[:, 1].values
                    except Exception as e:
                        continue

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
                        mode="lines+markers",
                        name=file.name,
                        line=dict(dash='solid', width=1, color=color),
                        marker=dict(color=color, size=3),
                        hovertemplate=(
                            f"<span style='color:{color};'><b>{file.name}:</b><br>"
                            "2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
                        )
                    ))
            else:
                try:
                    df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None, skiprows=1)
                    x_user = df.iloc[:, 0].values
                    y_user = df.iloc[:, 1].values
                except Exception as e:
                    # st.error(f"Error processing experimental file {user_pattern_file.name}: {e}")
                    x_user, y_user = None, None

                if x_user is not None and y_user is not None:
                    if intensity_scale_option == "Normalized" and np.max(y_user) > 0:
                        y_user = (y_user / np.max(y_user)) * 100

                    mask_user = (x_user >= st.session_state.two_theta_min) & (
                            x_user <= st.session_state.two_theta_max)
                    x_user_filtered = x_user[mask_user]
                    y_user_filtered = y_user[mask_user]

                    fig_interactive.add_trace(go.Scatter(
                        x=x_user_filtered,
                        y=y_user_filtered,
                        mode="lines+markers",
                        name=user_pattern_file.name,
                        line=dict(dash='solid', width=1, color=color),
                        marker=dict(color=color, size=3),
                        hovertemplate=(
                            f"<span style='color:{color};'><b>User XRD Data:</b><br>"
                            "2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
                        )
                    ))

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
                    title=dict(text="Intensity (a.u.)", font=dict(size=36, color='black')),
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
                    title=dict(text="Intensity (a.u.)", font=dict(size=36, color='black')),
                    tickfont=dict(size=36, color='black')
                ),
                hoverlabel=dict(font=dict(size=24)),
                font=dict(size=18),
                autosize=True
            )

    st.session_state.placeholder_interactive = st.empty()
    st.session_state.fig_interactive = fig_interactive
    st.session_state.placeholder_interactive.plotly_chart(st.session_state.fig_interactive,
                                                          use_container_width=True)

    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    if pattern_details is not None:
        st.subheader("Quantitative Data for Calculated Diffraction Patterns")
        for file in uploaded_files:
            details = pattern_details[file.name]
            peak_vals = details["peak_vals"]
            intensities = details["intensities"]
            hkls = details["hkls"]
            annotate_indices = details["annotate_indices"]
            x_dense_full = details["x_dense_full"]
            y_dense = details["y_dense"]
            with st.expander(f"View Peak Data for XRD Pattern: {file.name}"):
                table_str = "#X-axis    Intensity    hkl\n"
                for theta, intensity, hkl_group in zip(peak_vals, intensities, hkls):
                    if len(hkl_group[0]['hkl']) == 3:
                        hkl_str = ", ".join(
                            [
                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                for h in
                                hkl_group])
                    else:
                        hkl_str = ", ".join(
                            [
                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                for h in
                                hkl_group])
                    table_str += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
                st.code(table_str, language="text")
            with st.expander(f"View Highest Intensity Peaks for Diffraction Pattern: {file.name}", expanded=True):
                table_str2 = "#X-axis    Intensity    hkl\n"
                for i, (theta, intensity, hkl_group) in enumerate(zip(peak_vals, intensities, hkls)):
                    if i in annotate_indices:
                        if len(hkl_group[0]['hkl']) == 3:
                            hkl_str = ", ".join(
                                [
                                    f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                    for
                                    h in hkl_group])
                        else:
                            hkl_str = ", ".join(
                                [
                                    f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                    for
                                    h in hkl_group])
                        table_str2 += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
                st.code(table_str2, language="text")
            with st.expander(f"View Continuous Curve Data for Diffraction Pattern: {file.name}"):
                table_str3 = "#X-axis    Y-value\n"
                for x_val, y_val in zip(x_dense_full, y_dense):
                    table_str3 += f"{x_val:<12.5f} {y_val:<12.5f}\n"
                st.code(table_str3, language="text")


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
                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                for h in hkls[i]])
                        else:
                            hkl_str = ", ".join([
                                f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                for h in hkls[i]])
                        data_list.append([peak_vals[i], intensities[i], hkl_str, file_name])
            combined_df = pd.DataFrame(data_list, columns=["{}".format(selected_metric), "Intensity", "(hkl)", "Phase"])
            st.dataframe(combined_df)


# Add these session state initializations at the beginning of your script


# Add these session state initializations at the beginning of your script
# Make sure these are executed before any other Streamlit code

# Initialize session state variables
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


# Callback functions
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


# Main PRDF section
if "📊 (P)RDF" in calc_mode:
    uploaded_files = st.session_state.uploaded_files
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
                                  help="Enable this for a LAMMPS dump trajectory file with multiple frames")

    line_style = st.radio(
        "Line Style",
        ["Lines + Markers", "Lines Only"],
        index=0,
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

        if st.session_state.do_calculation:
            species_combinations = list(combinations(species_list, 2)) + [(s, s) for s in species_list]
            all_prdf_dict = defaultdict(list)
            all_distance_dict = {}
            global_rdf_list = []

            if use_lammps_traj and lammps_file:
                st.info(f"Processing LAMMPS trajectory file: {lammps_file.name}")
                progress_bar = st.progress(0)
                #with st.expander("Log from reading LAMMPS trajectory file"):
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
                                        st.success(f"Successfully read using {method['description']} from temp file")
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
                        mg_structure = load_structure(file.name)

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

        for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):

            hex_color = rgb_to_hex(colors[idx % len(colors)])

            valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

            if not valid_prdf:

                prdf_data = np.zeros_like(all_distance_dict[comb])
                title_str = f"PRDF: {comb[0]}-{comb[1]}"

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=all_distance_dict[comb],
                    y=prdf_data,
                    mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                    name=f"{comb[0]}-{comb[1]}",
                    line=dict(color=hex_color),
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
                                    "frame": {"duration": int(st.session_state.animation_speed * 1000), "redraw": True},
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
        "#### 📂 Upload your two-column data files in the sidebar to see them in an interactive plot. Multiple files are supported, and your columns can be separated by spaces, tabs, commas, or semicolons. 👍"
    )

    colss, colzz, colx, colc, cold = st.columns([1, 1, 1, 1, 1])
    has_header = colss.checkbox("Files contain a header row", value=False)
    skip_header = colzz.checkbox("Skip header row", value=True)
    normalized_intensity = colx.checkbox("Normalized intensity", value=False)
    x_axis_log = colc.checkbox("Logarithmic X-axis", value=False)
    y_axis_log = cold.checkbox("Logarithmic Y-axis", value=False)

    col_line, col_marker = st.columns([1, 1])
    show_lines = col_line.checkbox("Show Lines", value=True)
    show_markers = col_marker.checkbox("Show Markers", value=False)

    col_thick, col_size, col_fox, col_xmin, col_xmax, = st.columns([1, 1, 1, 1, 1])
    fix_x_axis = col_fox.checkbox("Fix x-axis range?", value=False)
    if fix_x_axis == True:
        x_axis_min = col_xmin.number_input("X-axis Minimum", value=0.0)
        x_axis_max = col_xmax.number_input("X-axis Maximum", value=10.0)
    line_thickness = col_thick.number_input("Line Thickness", min_value=0.1, max_value=15.0, value=1.0, step=0.3)
    marker_size = col_size.number_input("Marker Size", min_value=0.5, max_value=50.0, value=3.0, step=1.0)

    enable_conversion = st.checkbox("Enable powder XRD data conversion", value=False)

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
                with st.expander(f"🔄 Conversion settings for {file.name}", expanded=(i == 0)):
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

            try:

                if normalized_intensity and np.max(y_data) > 0:
                    y_data = (y_data / np.max(y_data)) * 100

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

                fig_interactive.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode_str,
                    name=file.name,
                    line=dict(dash='solid', width=line_thickness, color=color),
                    marker=dict(color=color, size=marker_size),
                    hovertemplate=(
                        f"<span style='color:{color};'><b>{file.name}</b><br>"
                        "x = %{x:.2f}<br>y = %{y:.2f}</span><extra></extra>"
                    )
                ))

            except Exception as e:
                st.error(
                    f"Error occurred in file processing. Please check whether your uploaded files are consistent: {e}")

        fig_interactive.update_xaxes(type="linear")
        fig_interactive.update_yaxes(type="linear")
        fig_interactive.update_layout(
            height=900,
            margin=dict(t=80, b=80, l=60, r=30),
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=28),
                title="Legend Title"
            ),
            xaxis=dict(
                title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
                tickfont=dict(size=36, color='black')
            ),
            yaxis=dict(
                title=dict(text=y_axis_metric, font=dict(size=36, color='black')),
                tickfont=dict(size=36, color='black')
            ),
            hoverlabel=dict(font=dict(size=24)),
            font=dict(size=18),
            autosize=True
        )

        if fix_x_axis == True:
            fig_interactive.update_xaxes(range=[x_axis_min, x_axis_max])

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
            df_out = pd.DataFrame({
                x_axis_metric: fig_interactive.data[i].x,
                y_axis_metric: fig_interactive.data[i].y
            })

            buffer = io.StringIO()
            df_out.to_csv(buffer, sep=delimiter_option, index=False)

            base_name = file.name.rsplit(".", 1)[0]
            download_name = f"{base_name}_processed.txt"

            st.download_button(
                label=f"⬇️ Download processed data for {file.name}",
                data=buffer.getvalue(),
                file_name=download_name,
                mime="text/plain"
            )
    else:
        st.info(f"Upload your data file first to see all options.")
st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
import sys


def get_session_memory_usage():
    total_size = 0
    for key in st.session_state:
        try:
            total_size += sys.getsizeof(st.session_state[key])
        except Exception:
            pass
    return total_size / 1024  # in KB


memory_kb = get_session_memory_usage()
st.markdown(f"🧠 Estimated session memory usage: **{memory_kb:.2f} KB**")
st.markdown("""
**The XRDlicious application is open-source and released under the [MIT License](https://github.com/bracerino/prdf-calculator-online/blob/main/LICENCSE).**
""")



def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # in MB


memory_usage = get_memory_usage()
st.write(f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are currently using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. If we will see higher usage of our app and need for a higher memory, we will upgrade to paid server, allowing us to improve the performance. :]")


st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors and maintainers: **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE).**[Pymatgen](https://github.com/materialsproject/pymatgen)** "
" Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE)."
" **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)** Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).
""")
