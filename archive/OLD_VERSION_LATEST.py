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
#import aflow.keywords as K

from pymatgen.io.cif import CifWriter
def get_cod_entries(params):
    response = requests.get('https://www.crystallography.net/cod/result', params=params)
    if response.status_code == 200:
        results = response.json()
        return results  # Returns a list of entries
    else:
        st.error(f"COD search error: {response.status_code}")
        return []


def get_cif_from_cod(entry):
    file_url = entry.get('file')
    if file_url:
        response = requests.get(f"https://www.crystallography.net/cod/{file_url}.cif")
        if response.status_code == 200:
            return response.text
    return None


def get_structure_from_mp(mp_id):
    with MPRester(MP_API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
        return structure

from pymatgen.io.cif import CifParser
def get_structure_from_cif_url(cif_url):
    response = requests.get(f"https://www.crystallography.net/cod/{cif_url}.cif")
    if response.status_code == 200:
        #  writer = CifWriter(response.text, symprec=0.01)
        #  parser = CifParser.from_string(writer)
        #  structure = parser.get_structures(primitive=False)[0]
        return response.text
    else:
        raise ValueError(f"Failed to fetch CIF from URL: {cif_url}")


def get_cod_str(cif_content):
    parser = CifParser.from_str(cif_content)
    structure = parser.get_structures(primitive=False)[0]
    return structure

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"


# import pkg_resources
# installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
# st.subheader("Installed Python Modules")
# for package, version in installed_packages:
#    st.write(f"{package}=={version}")


def get_full_conventional_structure_diffra(structure, symprec=1e-3):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords


    species_list = [site.species for site in structure]
    species_to_type = {}
    type_to_species = {}
    type_index = 1

    types = []
    for sp in species_list:
        sp_tuple = tuple(sorted(sp.items()))  # make it hashable
        if sp_tuple not in species_to_type:
            species_to_type[sp_tuple] = type_index
            type_to_species[type_index] = sp
            type_index += 1
        types.append(species_to_type[sp_tuple])

    cell = (lattice, positions, types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

    new_species_list = [type_to_species[t] for t in std_types]

    conv_structure = Structure(
        lattice=std_lattice,
        species=new_species_list,
        coords=std_positions,
        coords_are_cartesian=False
    )

    return conv_structure

def get_full_conventional_structure(structure, symprec=1e-3):
    # Create the spglib cell tuple: (lattice, fractional coords, atomic numbers)
    cell = (structure.lattice.matrix, structure.frac_coords,
            [max(site.species, key=site.species.get).number for site in structure])

    # Get the symmetry dataset from spglib
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

    # Build the conventional cell as a new Structure object
    conv_structure = Structure(std_lattice, std_types, std_positions)
    return conv_structure


def rgb_color(color_tuple, opacity=0.8):
    r, g, b = [int(255 * x) for x in color_tuple]
    return f"rgba({r},{g},{b},{opacity})"


def load_structure(file_or_name):
    if isinstance(file_or_name, str):
        filename = file_or_name
    else:
        filename = file_or_name.name
        with open(filename, "wb") as f:
            f.write(file_or_name.getbuffer())
    if filename.lower().endswith(".cif"):
        mg_structure = PmgStructure.from_file(filename)
    elif filename.lower().endswith(".data"):
        filename = filename.replace(".data", ".lmp")
        from pymatgen.io.lammps.data import LammpsData
        mg_structure = LammpsData.from_file(filename, atom_style="atomic").structure
    elif filename.lower().endswith(".lmp"):
        from pymatgen.io.lammps.data import LammpsData
        mg_structure = LammpsData.from_file(filename, atom_style="atomic").structure
    else:
        atoms = read(filename)
        mg_structure = AseAtomsAdaptor.get_structure(atoms)
    return mg_structure


def lattice_same_conventional_vs_primitive(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        primitive = analyzer.get_primitive_standard_structure()
        conventional = analyzer.get_conventional_standard_structure()

        lattice_diff = np.abs(primitive.lattice.matrix - conventional.lattice.matrix)
        volume_diff = abs(primitive.lattice.volume - conventional.lattice.volume)

        if np.all(lattice_diff < 1e-3) and volume_diff < 1e-2:
            return True
        else:
            return False
    except Exception as e:
        return None  # Could not determine


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

components.html(
    """
    <head>
        <meta name="description" content="XRDlicious, Online Calculator for Powder XRD/ND Patterns (Diffractograms), Partial Radial Distribution Function (PRDF), and Total RDF from Crystal Structures (CIF, LMP, POSCAR, XSF, ...)">
    </head>
    """,
    height=0,
)

col1, col2 = st.columns([1.25,1])
with col1:
    st.title(
        "🍕 XRDlicious: Online Calculator for Powder XRD/ND Patterns and Partial RDF from Crystal Structures (CIF, LMP, POSCAR, ...)")
    st.info(
    "🌀 Developed by [IMPLANT team](https://implant.fs.cvut.cz/). 📺 [Quick tutorial HERE.](https://youtu.be/ZiRbcgS_cd0)\n\nYou can find crystal structures in CIF format for example at: 📖 [Crystallography Open Database (COD)](https://www.crystallography.net/cod/), "
    "📖 [The Materials Project (MP)](https://next-gen.materialsproject.org/materials), or 📖 [AFLOW Database](http://aflowlib.duke.edu/search/ui/search/?search=Fe).\n\n"
    "Upload **structure files** (e.g., **CIF, LMP, POSCAR, XSF** format) and this tool will calculate either the "
    "**powder X-ray** or **neutron diffraction** (**XRD** or **ND**) patterns or **partial radial distribution function** (**PRDF**) for each **element combination** and **total RDF**. "
    "If **multiple files** are uploaded, the **PRDF** will be **averaged** for corresponding **element combinations** across the structures. For **XRD/ND patterns**, diffraction data from multiple structures are combined into a **single figure**."
    )
from PIL import Image
with col2:
    image = Image.open("images/ts4.png")
    st.image(image)

st.markdown(
    '<a href="https://xrdlicious-peak-match.streamlit.app/" target="_blank" style="font-size:24px;">🔗 Try also: 🍕 XRDlicious - Peak Matcher: Match XRD peaks using structures from Materials Project or AFLOW</a>',
    unsafe_allow_html=True
)

#st.divider()

# Add mode selection at the very beginning
st.sidebar.markdown("## 🍕 XRDlicious")
#mode = st.sidebar.radio("Select Mode", ["Basic", "Advanced"], index=0)
mode = "Basic"
structure_cell_choice = st.sidebar.radio(
    "Structure Cell Type:",
    options=["Conventional Cell", "Primitive Cell (Niggli)", "Primitive Cell (LLL)", "Primitive Cell (no reduction)"],
    index=1,  # default to Conventional
    help="Choose whether to use the crystallographic Primitive Cell or the Conventional Unit Cell for the structures. For Primitive Cell, you can select whether to use Niggli or LLL (Lenstra–Lenstra–Lovász) "
         "lattice basis reduction algorithm to produce less skewed representation of the lattice. The MP database is using Niggli-reduced Primitive Cells."
)

convert_to_conventional = structure_cell_choice == "Conventional Cell"
pymatgen_prim_cell_niggli = structure_cell_choice == "Primitive Cell (Niggli)"
pymatgen_prim_cell_lll = structure_cell_choice == "Primitive Cell (LLL)"
pymatgen_prim_cell_no_reduce = structure_cell_choice == "Primitive Cell (no reduction)"

if mode == "Basic":
    # st.divider()
    st.markdown("""
        <hr style="height:3px;border:none;color:#333;background-color:#333;" />
        """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 24px;'>
        🪧 <strong>Step 1 / 4</strong> Upload Your Crystal Structures (in .cif, .lmp (.data), POSCAR, .xsf, .pw, .cfg, ... Formats) or Fetch Structures from Materials Project/AFLOW Database: 
        <br><span style="font-size: 28px;">⬇️</span>
    </div>
    """, unsafe_allow_html=True)
    # Custom thick black divider
    st.markdown("""
    <hr style="height:3px;border:none;color:#333;background-color:#333;" />
    """, unsafe_allow_html=True)

    # st.divider()

# Initialize session state keys if not already set.
if 'mp_options' not in st.session_state:
    st.session_state['mp_options'] = None
if 'selected_structure' not in st.session_state:
    st.session_state['selected_structure'] = None
if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []  # List to store multiple fetched structures

# Create two columns: one for search and one for structure selection and actions.
#st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)


if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

st.sidebar.subheader("📤 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
        "Upload Structure Files (CIF, POSCAR, XSF, PW, CFG, ...):",
        type=None,
        accept_multiple_files=True,
    key="sidebar_uploader"
    )

if uploaded_files_user_sidebar:
    for file in uploaded_files_user_sidebar:
        # Only add the file if it hasn't been processed before.
        if file.name not in st.session_state.full_structures:
            try:
                # Replace load_structure with your structure-parsing function.
                structure = load_structure(file)
                st.session_state.full_structures[file.name] = structure
            except Exception as e:
                st.error(f"Failed to parse {file.name}: {e}")


# Column 1: Search for structures.
with col1:
    # --- Existing file uploader remains below (optional for user-uploaded structures) ---
    st.subheader("📤 Upload Your Structure Files")
    uploaded_files_user = st.file_uploader(
        "Upload Structure Files (CIF, POSCAR, XSF, PW, CFG, ...):",
        type=None,
        accept_multiple_files=True
    )

if uploaded_files_user:
    for file in uploaded_files_user:
        # Only add the file if it hasn't been processed before.
        if file.name not in st.session_state.full_structures:
            try:
                # Replace load_structure with your structure-parsing function.
                structure = load_structure(file)
                st.session_state.full_structures[file.name] = structure
            except Exception as e:
                st.error(f"Failed to parse {file.name}: {e}")


with col2:
    st.subheader("🔍 or Search for Structures in Databases")
    db_choice = st.radio(
        "Select Database",
         options=["Materials Project", "AFLOW", "COD"],
        index=0,
        help="Choose whether to search for structures in the Materials Project (about 179 000 Materials Entries) or in the AFLOW database (about 60 000 ICSD Entries)."
    )
    if db_choice == "COD":
        # Get COD search query string from sidebar
        cod_search_query = st.text_input(
            "Enter elements separated by spaces (e.g., Sr Ti O):",
            value="Sr Ti O", key='input_cod'
        )

        if st.button("Search COD", key='sidebar_cod_butt'):
            with st.spinner("Searching COD database..."):
                elements = [el.strip() for el in cod_search_query.split() if el.strip()]
                if elements:
                    params = {'format': 'json', 'detail': '1'}
                    for i, el in enumerate(elements, start=1):
                        params[f'el{i}'] = el
                    params['strictmin'] = str(len(elements))
                    params['strictmax'] = str(len(elements))
                    cod_entries = get_cod_entries(params)
                    if cod_entries:
                        st.session_state.cod_options = []
                        st.session_state.full_structures_see_cod = {}
                        for entry in cod_entries:
                            print(entry)
                            cif_content = get_cif_from_cod(entry)
                            if cif_content:
                                try:
                                    structure = get_full_conventional_structure(get_cod_str(cif_content))
                                    cod_id = f"cod_{entry.get('file')}"
                                    st.session_state.full_structures_see_cod[cod_id] = structure
                                    spcs =  entry.get("sg")
                                    st.session_state.cod_options.append(
                                        f"{cod_id}: {structure.composition.reduced_formula} ({spcs}) {structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} {structure.lattice.alpha:.2f}"
                                        f"{structure.lattice.beta:.2f} {structure.lattice.gamma:.2f}"
                                    )
                                except Exception as e:
                                    st.error(f"Error processing COD entry {entry.get('file')}: {e}")
                        if st.session_state.cod_options:
                            st.success(f"Found {len(st.session_state.cod_options)} structures in COD.")
                        else:
                            st.warning("COD: No matching structures found.")
                    else:
                        st.session_state.cod_options = []
                else:
                    st.error("Please enter at least one element for the COD search.")

        # Display search results if available.
        if 'cod_options' in st.session_state and st.session_state.cod_options:
            selected_cod_structure = st.selectbox(
                "Select a structure from COD:",
                st.session_state.cod_options,
                key='sidebar_select_cod'
            )
            cod_id = selected_cod_structure.split(":")[0].strip()
            if cod_id in st.session_state.full_structures_see_cod:
                selected_entry = st.session_state.full_structures_see_cod[cod_id]
                lattice = selected_entry.lattice
                st.write(f"**COD ID:** {cod_id}")
                st.write(f"**Formula:** {selected_entry.composition.reduced_formula}")


                lattice_str = (f"a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, "
                               f"α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")
                st.write(f"**Lattice Parameters:** {lattice_str}")

                # Create a link to the COD website using the file id (remove the 'cod_' prefix).
                cod_url = f"https://www.crystallography.net/cod/{cod_id.split('_')[1]}.html"
                st.write(f"**Link:** {cod_url}")

                # Define file_name now, so it is available for both adding and downloading.
                file_name = f"{selected_entry.composition.reduced_formula}_COD_{cod_id.split('_')[1]}.cif"

                # "Add" structure button
                if st.button("Add Selected Structure (COD)", key="sid_add_btn_cod"):
                    from pymatgen.io.cif import CifWriter

                    cif_writer = CifWriter(selected_entry, symprec=0.01)
                    cif_data = str(cif_writer)
                    st.session_state.full_structures[file_name] = selected_entry
                    import io

                    cif_file = io.BytesIO(cif_data.encode('utf-8'))
                    cif_file.name = file_name
                    if 'uploaded_files' not in st.session_state:
                        st.session_state.uploaded_files = []
                    if all(f.name != file_name for f in st.session_state.uploaded_files):
                        st.session_state.uploaded_files.append(cif_file)
                    st.success("Structure added from COD!")

                # Download button using the defined file_name
                from pymatgen.io.cif import CifWriter

                st.download_button(
                    label="Download COD CIF",
                    data=str(CifWriter(selected_entry, symprec=0.01)),
                    file_name=file_name,
                    mime="chemical/x-cif", type="primary",
                )
    
    if db_choice == "Materials Project":
        mp_search_query = st.text_input("Enter elements separated by spaces (e.g., Sr Ti O):", value="Sr Ti O")
        if st.button("Search Materials Project"):
            with st.spinner("Searching Materials Project database..."):
                elements_list = sorted(set(mp_search_query.split()))
                try:
                    with MPRester(MP_API_KEY) as mpr:
                        docs = mpr.materials.summary.search(
                            elements=elements_list,
                            num_elements=len(elements_list),
                            fields=["material_id", "formula_pretty", "symmetry"]
                        )
                        if docs:
                            st.session_state.mp_options = []
                            st.session_state.full_structures_see = {}  # store full pymatgen Structures
                            for doc in docs:
                                # Retrieve the full structure
                                full_structure = mpr.get_structure_by_material_id(doc.material_id)
                                # (Optionally, convert to conventional cell here)
                                # Retrieve the full structure (including lattice parameters)
                                if convert_to_conventional:
                                    # analyzer = SpacegroupAnalyzer(full_structure)
                                    # structure_to_use = analyzer.get_conventional_standard_structure()
                                    structure_to_use = get_full_conventional_structure(full_structure, symprec=0.1)
                                elif pymatgen_prim_cell_lll:
                                    analyzer = SpacegroupAnalyzer(full_structure)
                                    structure_to_use = analyzer.get_primitive_standard_structure()
                                    structure_to_use = structure_to_use.get_reduced_structure(reduction_algo="LLL")
                                elif pymatgen_prim_cell_no_reduce:
                                    analyzer = SpacegroupAnalyzer(full_structure)
                                    structure_to_use = analyzer.get_primitive_standard_structure()
                                else:
                                    structure_to_use = full_structure
                                st.session_state.full_structures_see[doc.material_id] = structure_to_use
                                lattice = structure_to_use.lattice
                                lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                               f"{lattice.alpha:.2f}, {lattice.beta:.2f}, {lattice.gamma:.2f} °")
                                st.session_state.mp_options.append(
                                    f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol}, {lattice_str})"
                                )
                            st.success(f"Found {len(st.session_state.mp_options)} structures.")
                        else:
                            st.session_state.mp_options = []
                            st.warning("No matching structures found in Materials Project.")
                except Exception as e:
                    st.error(f"An error occurred: {e}.\nThis is likely due to the error within The Materials Project API. Please try again later.")
    if db_choice == "AFLOW": # AFLOW branch
        aflow_elements_input = st.text_input("Enter elements separated by spaces (e.g., Ti O):", value="Ti O")

        # Process user input:
        if aflow_elements_input:
            import re

            # Replace commas with spaces, then split on whitespace.
            elements = re.split(r'[\s,]+', aflow_elements_input.strip())
            elements = [el for el in elements if el]  # Remove any empty strings.

            # Order elements alphabetically.
            ordered_elements = sorted(elements)

            # Create a comma-separated string for the inner search.
            ordered_str = ",".join(ordered_elements)
            # Automatically calculate number of species.
            aflow_nspecies = len(ordered_elements)
        else:
            ordered_str = ""
            aflow_nspecies = 0

        if st.button("Search AFLOW"):
            with st.spinner("Searching AFLOW database..."):
                try:
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
                        st.session_state.aflow_options = []
                        st.session_state.entrys = {}  # store full AFLOW entry objects
                        for entry in results:
                            # Save the full entry object in session state.
                            st.session_state.entrys[entry.auid] = entry
                            # Use the provided geometry string from AFLOW for display.
                            st.session_state.aflow_options.append(
                                f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax} {entry.geometry})"
                            )
                        st.success(f"Found {len(st.session_state.aflow_options)} structures.")
                    else:
                        st.session_state.aflow_options = []
                        st.warning("No matching structures found in AFLOW.")
                except Exception as e:
                    st.warning("No matching structures found in AFLOW.")


# Import AFLOW search functions and keywords

# Column 2: Select structure and add/download CIF.
with col3:
    if db_choice == "Materials Project" and "mp_options" in st.session_state and st.session_state.mp_options:
        st.subheader("Structures Found in Materials Project")
        selected_structure = st.selectbox("Select a structure from MP:", st.session_state.mp_options)
        selected_id = selected_structure.split(":")[0].strip()
        composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
        file_name = f"{selected_id}_{composition}.cif"
        file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

        # Retrieve the corresponding MP structure from session state.
        if selected_id in st.session_state.full_structures_see:
            selected_entry = st.session_state.full_structures_see[selected_id]

            #st.write("### Selected Structure Details")
            st.write(f"**Material ID:** {selected_id}")
            st.write(f"**Formula:** {composition}")

            # Display original lattice parameters
           # lattice = selected_entry.lattice
           # st.write(f"**Primitive Cell Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å")
           # st.write(f"**Primitive Cell Angles:** α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")

            if convert_to_conventional:
                # analyzer = SpacegroupAnalyzer(full_structure)
                # structure_to_use = analyzer.get_conventional_standard_structure()
               converted_structure = get_full_conventional_structure(selected_entry, symprec=0.1)
               conv_lattice = converted_structure.lattice
               st.write(
                    f"**Conventional Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
               st.write(
                    f"**Conventional Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
            elif pymatgen_prim_cell_lll:
                    analyzer = SpacegroupAnalyzer(selected_entry)
                    converted_structure = analyzer.get_primitive_standard_structure()
                    converted_structure = converted_structure.get_reduced_structure(reduction_algo="LLL")
                    conv_lattice = converted_structure.lattice
                    st.write(
                        f"**Primitive Cell (LLL) Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                    st.write(
                        f"**Primitive Cell (LLL) Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
            elif pymatgen_prim_cell_no_reduce:
                analyzer = SpacegroupAnalyzer(selected_entry)
                converted_structure = analyzer.get_primitive_standard_structure()
                conv_lattice = converted_structure.lattice
                st.write(
                    f"**Primitive Cell (No-reduction) Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                st.write(
                    f"**Primitive Cell (No-reduction) Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
            elif pymatgen_prim_cell_niggli:
                analyzer = SpacegroupAnalyzer(selected_entry)
                converted_structure = analyzer.get_primitive_standard_structure()
                converted_structure = converted_structure.get_reduced_structure(reduction_algo="niggli")
                conv_lattice = converted_structure.lattice
                st.write(
                    f"**Primitive Cell (Niggli) Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                st.write(
                    f"**Primitive Cell (Niggli) Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")


            # Optionally, convert to the conventional cell using your defined function.



            # Optionally, show space group using pymatgen's SpacegroupAnalyzer.
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            analyzer = SpacegroupAnalyzer(selected_entry)
            st.write(f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
            mp_url = f"https://materialsproject.org/materials/{selected_id}"
            st.write(f"**Link:** {mp_url}")

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
        st.download_button(
            label="Download MP CIF",
            data=st.session_state.full_structures_see[selected_id].__str__(),
            file_name=file_name,
            type="primary",
            mime="chemical/x-cif"
        )


    elif db_choice == "AFLOW" and "aflow_options" in st.session_state and st.session_state.aflow_options:
        st.subheader("Structures Found in AFLOW")
        selected_structure = st.selectbox("Select a structure from AFLOW:", st.session_state.aflow_options)
        selected_auid = selected_structure.split(": ")[0].strip()
        # Retrieve the corresponding AFLOW entry from session state.
        selected_entry = next(
            (entry for entry in st.session_state.entrys.values() if entry.auid == selected_auid), None)
        if selected_entry:
           # st.write("### Selected Structure Details")
            st.write(f"**AUID:** {selected_entry.auid}")
            st.write(f"**Formula:** {selected_entry.compound}")
            #st.write(f"**Space Group:** ({selected_entry.spacegroup_relax})")




            # Identify a CIF file (choose one ending with '_sprim.cif' or '.cif')

            cif_files = [f for f in selected_entry.files if f.endswith("_sprim.cif") or f.endswith(".cif")]

            if cif_files:

                cif_filename = cif_files[0]

                # Correct the AURL: replace the first ':' with '/'

                host_part, path_part = selected_entry.aurl.split(":", 1)

                corrected_aurl = f"{host_part}/{path_part}"

                file_url = f"http://{corrected_aurl}/{cif_filename}"


                # Fetch the CIF file once.

                response = requests.get(file_url)
                cif_content = response.content


                # "Add" button: store the CIF file in session state.
                structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'), fmt="cif")
                if convert_to_conventional:
                    converted_structure = get_full_conventional_structure(structure_from_aflow, symprec=0.1)
                    conv_lattice = converted_structure.lattice
                    st.write(
                        f"**Conventional Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                    st.write(
                        f"**Conventional Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
                elif pymatgen_prim_cell_lll:
                    analyzer = SpacegroupAnalyzer(structure_from_aflow)
                    converted_structure = analyzer.get_primitive_standard_structure()
                    converted_structure = converted_structure.get_reduced_structure(reduction_algo="LLL")
                    conv_lattice = converted_structure.lattice
                    st.write(
                        f"**Primitive Cell (LLL) Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                    st.write(
                        f"**Primitive Cell (LLL) Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
                elif pymatgen_prim_cell_no_reduce:
                    analyzer = SpacegroupAnalyzer(structure_from_aflow)
                    converted_structure = analyzer.get_primitive_standard_structure()
                    conv_lattice = converted_structure.lattice
                    st.write(
                        f"**Primitive Cell (No-reduction) Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                    st.write(
                        f"**Primitive Cell (No-reduction) Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
                elif pymatgen_prim_cell_niggli:
                    analyzer = SpacegroupAnalyzer(structure_from_aflow)
                    converted_structure = analyzer.get_primitive_standard_structure()
                    converted_structure = converted_structure.get_reduced_structure(reduction_algo="niggli")
                    conv_lattice = converted_structure.lattice
                    st.write(
                        f"**Primitive Cell (Niggli) Lattice:** a = {conv_lattice.a:.3f} Å, b = {conv_lattice.b:.3f} Å, c = {conv_lattice.c:.3f} Å")
                    st.write(
                        f"**Primitive Cell (Niggli) Angles:** α = {conv_lattice.alpha:.2f}°, β = {conv_lattice.beta:.2f}°, γ = {conv_lattice.gamma:.2f}°")
                else:
                    # If no conversion flag is set, display the original lattice.
                    lattice = structure_from_aflow.lattice
                    st.write(
                        f"**Original Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å")
                    st.write(
                        f"**Original Angles:** α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")
                analyzer = SpacegroupAnalyzer(structure_from_aflow)
                st.write(f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

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

#

# Column 2: Select structure and add/download CIF.












#SIDEBAR SEARCH

db_choice = st.sidebar.radio(
        "Select Database",
        options=["Materials Project", "AFLOW"],
        index=0,
        help="Choose whether to search for structures in the Materials Project (about 179 000 Materials Entries) or in the AFLOW database (about 60 000 ICSD Entries).", key ="sidebar_database"
    )

if db_choice == "Materials Project":
    mp_search_query = st.sidebar.text_input("Enter elements separated by spaces (e.g., Sr Ti O):", value="Sr Ti O", key='sidebar_input_mp')
    if st.sidebar.button("Search Materials Project", key='sidebar_mp_butt'):
        with st.spinner("Searching Materials Project database..."):
            elements_list = sorted(set(mp_search_query.split()))
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.materials.summary.search(
                    elements=elements_list,
                    num_elements=len(elements_list),
                    fields=["material_id", "formula_pretty", "symmetry"]
                )
                if docs:
                    st.session_state.mp_options = []
                    st.session_state.full_structures_see = {}  # store full pymatgen Structures
                    for doc in docs:
                        # Retrieve the full structure
                        full_structure = mpr.get_structure_by_material_id(doc.material_id)
                        # (Optionally, convert to conventional cell here)
                        # Retrieve the full structure (including lattice parameters)
                        if convert_to_conventional:
                            # analyzer = SpacegroupAnalyzer(full_structure)
                            # structure_to_use = analyzer.get_conventional_standard_structure()
                            structure_to_use = get_full_conventional_structure(full_structure, symprec=0.1)
                        elif pymatgen_prim_cell_lll:
                            analyzer = SpacegroupAnalyzer(full_structure)
                            structure_to_use = analyzer.get_primitive_standard_structure()
                            structure_to_use = structure_to_use.get_reduced_structure(reduction_algo="LLL")
                        elif pymatgen_prim_cell_no_reduce:
                            analyzer = SpacegroupAnalyzer(full_structure)
                            structure_to_use = analyzer.get_primitive_standard_structure()
                        else:
                            structure_to_use = full_structure
                        st.session_state.full_structures_see[doc.material_id] = structure_to_use
                        lattice = structure_to_use.lattice
                        lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                       f"{lattice.alpha:.2f}, {lattice.beta:.2f}, {lattice.gamma:.2f} °")
                        st.session_state.mp_options.append(
                            f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol}, {lattice_str})"
                        )
                    st.success(f"Found {len(st.session_state.mp_options)} structures.")
                else:
                    st.session_state.mp_options = []
                    st.sidebar.warning("No matching structures found in Materials Project.")
if db_choice == "AFLOW":
    aflow_elements_input = st.sidebar.text_input("Enter elements separated by spaces (e.g., Ti O):", value="Ti O", key='sidebar_AFLOW')

    # Process user input:
    if aflow_elements_input:
        import re

        # Replace commas with spaces, then split on whitespace.
        elements = re.split(r'[\s,]+', aflow_elements_input.strip())
        elements = [el for el in elements if el]  # Remove any empty strings.

        # Order elements alphabetically.
        ordered_elements = sorted(elements)

        # Create a comma-separated string for the inner search.
        ordered_str = ",".join(ordered_elements)
        # Automatically calculate number of species.
        aflow_nspecies = len(ordered_elements)
    else:
        ordered_str = ""
        aflow_nspecies = 0

    if st.sidebar.button("Search AFLOW", key='aflow_but'):
        with st.spinner("Searching AFLOW database..."):
            try:
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
                    st.session_state.aflow_options = []
                    st.session_state.entrys = {}  # store full AFLOW entry objects
                    for entry in results:
                        # Save the full entry object in session state.
                        st.session_state.entrys[entry.auid] = entry
                        # Use the provided geometry string from AFLOW for display.
                        st.session_state.aflow_options.append(
                            f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax} {entry.geometry})"
                        )
                    st.sidebar.success(f"Found {len(st.session_state.aflow_options)} structures.")
                else:
                    st.session_state.aflow_options = []
                    st.sidebar.warning("No matching structures found in AFLOW.")
            except Exception as e:
                st.sidebar.warning("No matching structures found in AFLOW.")

if db_choice == "Materials Project" and "mp_options" in st.session_state and st.session_state.mp_options:
    selected_structure = st.sidebar.selectbox("Select a structure from MP:", st.session_state.mp_options,
                                              key='sidebar_select_mp')
    selected_id = selected_structure.split(":")[0].strip()
    composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
    file_name = f"{selected_id}_{composition}.cif"
    file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

    # Retrieve the corresponding MP structure from session state.
    if selected_id in st.session_state.full_structures_see:
        selected_entry = st.session_state.full_structures_see[selected_id]

    if st.sidebar.button("Add Selected Structure (MP)", key="sid_add_btn_mp"):
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
        st.sidebar.success("Structure added from Materials Project!")


elif db_choice == "AFLOW" and "aflow_options" in st.session_state and st.session_state.aflow_options:
    selected_structure = st.sidebar.selectbox("Select a structure from AFLOW:", st.session_state.aflow_options,
                                              key='option_aflow_sid')
    selected_auid = selected_structure.split(": ")[0].strip()
    # Retrieve the corresponding AFLOW entry from session state.
    selected_entry = next(
        (entry for entry in st.session_state.entrys.values() if entry.auid == selected_auid), None)
    if selected_entry:

        # Identify a CIF file (choose one ending with '_sprim.cif' or '.cif')

        cif_files = [f for f in selected_entry.files if f.endswith("_sprim.cif") or f.endswith(".cif")]

        if cif_files:

            cif_filename = cif_files[0]

            # Correct the AURL: replace the first ':' with '/'

            host_part, path_part = selected_entry.aurl.split(":", 1)

            corrected_aurl = f"{host_part}/{path_part}"

            file_url = f"http://{corrected_aurl}/{cif_filename}"

            # Fetch the CIF file once.

            response = requests.get(file_url)
            cif_content = response.content

            # "Add" button: store the CIF file in session state.
            structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'), fmt="cif")

            if st.sidebar.button("Add Selected Structure (AFLOW)", key="sid_add_btn_aflow"):
                if 'uploaded_files' not in st.session_state:
                    st.session_state.uploaded_files = []
                cif_file = io.BytesIO(cif_content)
                cif_file.name = f"{selected_entry.compound}_{selected_entry.auid}.cif"

                st.session_state.full_structures[cif_file.name] = structure_from_aflow
                if all(f.name != cif_file.name for f in st.session_state.uploaded_files):
                    st.session_state.uploaded_files.append(cif_file)
                st.sidebar.success("Structure added from AFLOW!")

        else:
            st.warning("No CIF file found for this AFLOW entry.")






if st.session_state['mp_options'] is None:
    st.info("Please press the 'Search Materials Project' button to view the available structures.")







st.markdown("---")

if uploaded_files_user or uploaded_files_user_sidebar:
    uploaded_files = st.session_state['uploaded_files'] + uploaded_files_user + uploaded_files_user_sidebar
    if 'full_structures' not in st.session_state:
        st.session_state.full_structures = {}
    for file in uploaded_files_user:
        try:
            structure = load_structure(file)
            # Use file.name as the key (or modify to a unique identifier if needed)
            st.session_state['full_structures'][file.name] = structure
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")
else:
    uploaded_files = st.session_state['uploaded_files']

st.sidebar.markdown("### Final List of Structure Files:")
st.sidebar.write([f.name for f in uploaded_files])


st.sidebar.markdown("### 🗑️ Remove Structure(s) added from MP or AFLOW")

files_to_remove = []
for i, file in enumerate(st.session_state['uploaded_files']):
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(file.name)
    if col2.button("❌", key=f"remove_{i}"):
        files_to_remove.append(file)

if files_to_remove:
    for f in files_to_remove:
        st.session_state['uploaded_files'].remove(f)
    st.rerun()  # 🔁 Force Streamlit to rerun and refresh UI


if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")
else:
    st.warning("📌 Please upload at least one structure file.")

if mode == "Basic" and not uploaded_files:
    st.divider()
    st.markdown("""
    **This application is open-source and released under the [MIT License](https://github.com/bracerino/prdf-calculator-online/blob/main/LICENCSE).**
    """)
    #If used in academic publications, please cite:
    
    st.markdown("""
    ### Acknowledgments
    
    This project utilizes several open-source materials science packages. We gratefully acknowledge their authors and maintainers:
    
    - **[Matminer](https://github.com/hackingmaterials/matminer)**  
      Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE).  
    
    - **[Pymatgen](https://github.com/materialsproject/pymatgen)**  
      Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).  
    
    - **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)**  
      Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER).  
    
    - **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)**  
        Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt).
    
    - **[Materials Project](https://next-gen.materialsproject.org/)**  
      Data from the Materials Project is made available under the  
      [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
    
    - **[AFLOW](http://aflow.org)**  
      Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).   
      When using structures from AFLOW, please cite:  
      Curtarolo et al., *Computational Materials Science*, 58 (2012) 218-226.  
      [DOI: 10.1016/j.commatsci.2012.02.005](https://doi.org/10.1016/j.commatsci.2012.02.005)
    """)
    st.stop()
# --- Detect Atomic Species ---


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
    #st.subheader("📊 Detected Atomic Species")
    #st.write(", ".join(species_list))
else:
    species_list = []



def add_box(view, cell, color='black', linewidth=2):
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corner = i * a + j * b + k * c
                corners.append(corner)
    edges = []
    for idx in range(8):
        i = idx & 1
        j = (idx >> 1) & 1
        k = (idx >> 2) & 1
        if i == 0:
            edges.append((corners[idx], corners[idx + 1]))
        if j == 0:
            edges.append((corners[idx], corners[idx + 2]))
        if k == 0:
            edges.append((corners[idx], corners[idx + 4]))
    for start, end in edges:
        view.addLine({
            'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
            'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
            'color': color,
            'linewidth': linewidth
        })
    arrow_radius = 0.04
    arrow_color = '#000000'
    for vec in [a, b, c]:
        view.addArrow({
            'start': {'x': 0, 'y': 0, 'z': 0},
            'end': {'x': float(vec[0]), 'y': float(vec[1]), 'z': float(vec[2])},
            'color': arrow_color,
            'radius': arrow_radius
        })
    offset = 0.3

    def add_axis_label(vec, label_val):
        norm = np.linalg.norm(vec)
        end = vec + offset * vec / (norm + 1e-6)
        view.addLabel(label_val, {
            'position': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
            'fontSize': 14,
            'fontColor': color,
            'showBackground': False
        })

    a_len = np.linalg.norm(a)
    b_len = np.linalg.norm(b)
    c_len = np.linalg.norm(c)
    add_axis_label(a, f"a = {a_len:.3f} Å")
    add_axis_label(b, f"b = {b_len:.3f} Å")
    add_axis_label(c, f"c = {c_len:.3f} Å")


# --- Structure Visualization ---
jmol_colors = {
    "H": "#FFFFFF",
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Ne": "#B3E3F5",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Ar": "#80D1E3",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "Ge": "#668F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Br": "#A62929",
    "Kr": "#5CB8D1",
    "Rb": "#702EB0",
    "Sr": "#00FF00",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Tc": "#3B9E9E",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",
    "I": "#940094",
    "Xe": "#429EB0",
    "Cs": "#57178F",
    "Ba": "#00C900",
    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Pr": "#D9FFC7",
    "Nd": "#C7FFC7",
    "Pm": "#A3FFC7",
    "Sm": "#8FFFC7",
    "Eu": "#61FFC7",
    "Gd": "#45FFC7",
    "Tb": "#30FFC7",
    "Dy": "#1FFFC7",
    "Ho": "#00FF9C",
    "Er": "#00E675",
    "Tm": "#00D452",
    "Yb": "#00BF38",
    "Lu": "#00AB24",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
    "Tl": "#A6544D",
    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "Po": "#AB5C00",
    "At": "#754F45",
    "Rn": "#428296",
    "Fr": "#420066",
    "Ra": "#007D00",
    "Ac": "#70ABFA",
    "Th": "#00BAFF",
    "Pa": "#00A1FF",
    "U": "#008FFF",
    "Np": "#0080FF",
    "Pu": "#006BFF",
    "Am": "#545CF2",
    "Cm": "#785CE3",
    "Bk": "#8A4FE3",
    "Cf": "#A136D4",
    "Es": "#B31FD4",
    "Fm": "#B31FBA",
    "Md": "#B30DA6",
    "No": "#BD0D87",
    "Lr": "#C70066",
    "Rf": "#CC0059",
    "Db": "#D1004F",
    "Sg": "#D90045",
    "Bh": "#E00038",
    "Hs": "#E6002E",
    "Mt": "#EB0026"
}

st.sidebar.markdown("### Structure Visualization Tool:")
#show_structure = st.sidebar.checkbox("Show Structure Visualization Tool", value=True)
show_structure = True
if uploaded_files:
    if show_structure:
        if mode == "Basic":
            st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <hr style="height:3px;border:none;color:#333;background-color:#333;" />
                """, unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; font-size: 24px;'>
                🪧 <strong>Step 2 / 4 (OPTIONAL):</strong> Visually Inspect Your Crystal Structures and Download CIF File for the Visualized Structure in either Conventional or Primitive Cell Representation, if Needed: 
                <br><span style="font-size: 28px;">⬇️</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <hr style="height:3px;border:none;color:#333;background-color:#333;" />
                """, unsafe_allow_html=True)


        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        col_viz, col_download = st.columns(2)

        with col_viz:
            file_options = [file.name for file in uploaded_files]
            st.subheader("Select Structure for Interactive Visualization:")
            if len(file_options) > 5:
                selected_file = st.selectbox("", file_options)
            else:
                selected_file = st.radio("", file_options)
            
        try:
            structure = read(selected_file)
            mp_struct = AseAtomsAdaptor.get_structure(structure)
        except Exception as e:
            mp_struct = load_structure(selected_file)

        selected_id = selected_file.split("_")[0]  # assumes filename like "mp-1234_FORMULA.cif"
        #print(st.session_state.get('full_structures', {}))
        #if 'full_structures' in st.session_state:
        #mp_struct = st.session_state.get('full_structures', {}).get(selected_file)
        #mp_struct = AseAtomsAdaptor.get_structure(structure)
        #mp_struct = st.session_state.get('uploaded_files', {}).get(selected_file.name)

        if mp_struct:
            if convert_to_conventional:
                # analyzer = SpacegroupAnalyzer(mp_struct)
                # converted_structure = analyzer.get_conventional_standard_structure()
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
            structure = AseAtomsAdaptor.get_atoms(converted_structure)

        # Checkbox option to show atomic positions (labels on structure and list in table)
        show_atomic = st.sidebar.checkbox("Show atomic positions (labels on structure and list in table)", value=True)
        xyz_io = StringIO()
        write(xyz_io, structure, format="xyz")
        xyz_str = xyz_io.getvalue()
        view = py3Dmol.view(width=800, height=600)
        view.addModel(xyz_str, "xyz")
        view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})
        cell = structure.get_cell()  # 3x3 array of lattice vectors
        add_box(view, cell, color='black', linewidth=4)
        view.zoomTo()
        view.zoom(1.2)

        # Download CIF for visualized structure
        if mp_struct:
            visual_pmg_structure = converted_structure
        else:
            visual_pmg_structure = load_structure(selected_file)
        for site in visual_pmg_structure.sites:
            pass
            #print(site.species)  # This will show occupancy info
            # Write CIF content directly using pymatgen:
            # Otherwise, use the chosen conversion
        if convert_to_conventional:
            lattice_info = "conventional"
        elif pymatgen_prim_cell_niggli:
            lattice_info = "primitive_niggli"
        elif pymatgen_prim_cell_lll:
            lattice_info = "primitive_lll"
        elif pymatgen_prim_cell_no_reduce:
            lattice_info = "primitive_no_reduce"
        else:
            lattice_info = "primitive"

        cif_writer_visual = CifWriter(visual_pmg_structure, symprec=0.1, refine_struct=False)

        cif_content_visual = cif_writer_visual.__str__()

        # Prepare a file name (ensure it ends with .cif)
        download_file_name = selected_file.split('.')[0] + '_{}'.format(lattice_info) + '.cif'
        if not download_file_name.lower().endswith('.cif'):
            download_file_name = selected_file.split('.')[0] + '_{}'.format(lattice_info) + '.cif'

        with col_download:
            with st.expander("Download Options", expanded=True):
                file_format = st.radio(
                    "Select file format",
                    ("CIF", "VASP", "LAMMPS", "XYZ",),
                    horizontal=True
                )
                enable_supercell = st.checkbox("Wish to Create Supercell?", value=False)
                if enable_supercell:
                    st.markdown("**Optional: Create Supercell**")
                    # Show how many atoms will be in the resulting supercell
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_a = st.number_input("Repeat along a-axis", min_value=1, max_value=10, value=1, step=1)
                    with col2:
                        n_b = st.number_input("Repeat along b-axis", min_value=1, max_value=10, value=1, step=1)
                    with col3:
                        n_c = st.number_input("Repeat along c-axis", min_value=1, max_value=10, value=1, step=1)

                    base_atoms = len(structure)
                    supercell_multiplier = n_a * n_b * n_c
                    total_atoms = base_atoms * supercell_multiplier
                    st.info(f"Structure will contain **{total_atoms} atoms**.")
                    supercell_structure = structure.copy()  # ASE Atoms object
                    supercell_pmg = visual_pmg_structure.copy()  # pymatgen Structure object

                    if (n_a, n_b, n_c) != (1, 1, 1):
                        supercell_structure = supercell_structure.repeat((n_a, n_b, n_c))
                        from pymatgen.transformations.standard_transformations import SupercellTransformation

                        supercell_matrix = [[n_a, 0, 0], [0, n_b, 0], [0, 0, n_c]]
                        transformer = SupercellTransformation(supercell_matrix)
                        supercell_pmg = transformer.apply_transformation(supercell_pmg)
                else:
                # If supercell not enabled, just use the original structure
                    supercell_structure = structure.copy()
                    supercell_pmg = visual_pmg_structure.copy()

                file_content = None
                download_file_name = None
                mime = "text/plain"

                try:
                    if file_format == "CIF":
                        # Use pymatgen's CifWriter for CIF output.
                        from pymatgen.io.cif import CifWriter

                        cif_writer_visual = CifWriter(supercell_pmg, symprec=0.1, refine_struct=False)
                        file_content = str(cif_writer_visual)

                        if enable_supercell:
                            download_file_name = selected_file.split('.')[0] + '_'+lattice_info+f'_Supercell_{n_a}_{n_b}_{n_c}.cif'
                        else:
                            download_file_name = selected_file.split('.')[0] + '_'+lattice_info+'.cif'
                        mime = "chemical/x-cif"
                    elif file_format == "VASP":
                        out = StringIO()
                        use_fractional = st.checkbox("Output POSCAR with fractional coordinates", value=True,
                                                     key="poscar_fractional")
                        write(out, supercell_structure, format="vasp", direct=use_fractional)
                        file_content = out.getvalue()
                        if enable_supercell:
                             download_file_name = selected_file.split('.')[0] + '_'+lattice_info+f'_Supercell_{n_a}_{n_b}_{n_c}.poscar'
                        else:
                            download_file_name = selected_file.split('.')[0] + '_'+lattice_info+'.poscar'
                    elif file_format == "LAMMPS":
                        st.markdown("**LAMMPS Export Options**")

                        atom_style = st.selectbox("Select atom_style", ["atomic", "charge", "full"], index=0)
                        units = st.selectbox("Select units", ["metal", "real", "si"], index=0)
                        include_masses = st.checkbox("Include atomic masses", value=True)
                        force_skew = st.checkbox("Force triclinic cell (skew)", value=False)

                        out = StringIO()
                        write(
                            out,
                            supercell_structure,
                            format="lammps-data",
                            atom_style=atom_style,
                            units=units,
                            masses=include_masses,
                            force_skew=force_skew
                        )
                        file_content = out.getvalue()
                        if enable_supercell:
                            download_file_name = selected_file.split('.')[0] + '_'+lattice_info+ f'_Supercell_{n_a}_{n_b}_{n_c}'+f'_{atom_style}_{units}.lmp'
                        else:
                            download_file_name = selected_file.split('.')[0] + '_'+lattice_info+f'_{atom_style}_{units}.lmp'
                    elif file_format == "XYZ":
                        out = StringIO()
                        write(out, supercell_structure, format="xyz")
                        file_content = out.getvalue()
                        if enable_supercell:
                            download_file_name = selected_file.split('.')[0] +'_'+lattice_info+f'_Supercell_{n_a}_{n_b}_{n_c}.xyz'
                        else:
                            download_file_name = selected_file.split('.')[0] +'_'+lattice_info+ '.xyz'
                except Exception as e:
                    st.error(f"Error generating {file_format} file: {e}")

                if file_content is not None:
                    st.download_button(
                        label=f"Download {file_format} file",
                        data=file_content,
                        file_name=download_file_name,
                        type="primary",
                        mime=mime
                    )

        offset_distance = 0.3  # distance to offset
        overlay_radius = 0.15  # radius for the overlay spheres


        # Create a new list to store atomic info for the table (labels with occupancy info)
        atomic_info = []
        inv_cell = np.linalg.inv(cell)

        visual_pmg_structure_partial_check = load_structure(selected_file)

        # Check whether any site in the structure has partial occupancy.
        has_partial_occ = any(
            (len(site.species) > 1) or any(occ < 1 for occ in site.species.values())
            for site in visual_pmg_structure_partial_check.sites
        )


        # If partial occupancy is detected, notify the user and offer an enhanced visualization option.
        if has_partial_occ:
            st.info(f"Partial occupancy detected in the uploaded structure. Note that the conversion between cell representions will not be possible now.\n To continue ")
            visualize_partial = st.checkbox("Enable enhanced partial occupancy visualization", value=True)
        else:
            visualize_partial = False


        if visualize_partial:
            visual_pmg_structure = load_structure(selected_file)
            from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

            #try:
            #    structure_with_oxi = visual_pmg_structure.add_oxidation_state_by_guess()
            #except Exception as e:
                # Optionally, handle the exception if oxidation states cannot be assigned.
            #    raise ValueError(f"Could not assign oxidation states to the structure: {e}")

            #Convert to ordered structure
            #ordered_structure = OrderDisorderedStructureTransformation(no_oxi_states=True)
            #ordered_structure = order_trans.apply_transformation(structure_with_oxi)

            from pymatgen.core import Structure

            # Get lattice from original structure
            lattice = visual_pmg_structure.lattice

            # Build new species list: choose the species with highest occupancy at each site
            species = []
            coords = []

            for site in visual_pmg_structure.sites:
                # Pick the species with the highest occupancy
                dominant_specie = max(site.species.items(), key=lambda x: x[1])[0]
                species.append(dominant_specie)
                coords.append(site.frac_coords)

            # Create a new ordered structure
            ordrd = Structure(lattice, species, coords)
            structure = AseAtomsAdaptor.get_atoms(ordrd)


            xyz_io = StringIO()
            write(xyz_io, structure, format="xyz")
            xyz_str = xyz_io.getvalue()
            view = py3Dmol.view(width=800, height=600)
            view.addModel(xyz_str, "xyz")
            view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})
            cell = structure.get_cell()  # 3x3 array of lattice vectors
            add_box(view, cell, color='black', linewidth=4)
            view.zoomTo()
            view.zoom(1.2)
            # Enhanced visualization: iterate over the pymatgen structure to use occupancy info.

            for i, site in enumerate(visual_pmg_structure.sites):
                # Get Cartesian coordinates.
                x, y, z = site.coords

                # Build a string for species and occupancy details.
                species_info = []
                for specie, occ in site.species.items():
                    occ_str = f"({occ * 100:.0f}%)" if occ < 1 else ""
                    species_info.append(f"{specie.symbol}{occ_str}")
                label_text = f"{'/'.join(species_info)}{i}"
                if show_atomic:
                # Add a label with the occupancy info.
                    view.addLabel(label_text, {
                        "position": {"x": x, "y": y, "z": z},
                        "backgroundColor": "white",
                        "fontColor": "black",
                        "fontSize": 10,
                        "borderThickness": 1,
                        "borderColor": "black"
                    })

                frac = np.dot(inv_cell, [x, y, z])
                atomic_info.append({
                    "Atom": label_text,
                    "Elements": "/".join(species_info),
                    "X": round(x, 3),
                    "Y": round(y, 3),
                    "Z": round(z, 3),
                    "Frac X": round(frac[0], 3),
                    "Frac Y": round(frac[1], 3),
                    "Frac Z": round(frac[2], 3)
                })

                        # For sites with partial occupancy, overlay extra spheres.
                species_dict = site.species
                if (len(species_dict) > 1) or any(occ < 1 for occ in species_dict.values()):
                    num_species = len(species_dict)
                    # Distribute offsets around a circle (here in the xy-plane).
                    angles = np.linspace(0, 2 * np.pi, num_species, endpoint=False)
                    offset_distance = 0.3  # adjust as needed
                    overlay_radius = 0.15  # adjust as needed
                    for j, ((specie, occ), angle) in enumerate(zip(species_dict.items(), angles)):
                        dx = offset_distance * np.cos(angle)
                        dy = offset_distance * np.sin(angle)
                        sphere_center = {"x": x + dx, "y": y + dy, "z": z}
                        view.addSphere({
                            "center": sphere_center,
                            "radius": overlay_radius,
                            "color": jmol_colors.get(specie.symbol, "gray"),
                            "opacity": 1.0
                        })
        else:
            if show_atomic:
                # Basic visualization (as before): iterate over ASE atoms.
                for i, atom in enumerate(structure):
                    symbol = atom.symbol
                    x, y, z = atom.position
                    label_text = f"{symbol}{i}"
                    view.addLabel(label_text, {
                        "position": {"x": x, "y": y, "z": z},
                        "backgroundColor": "white",
                        "fontColor": "black",
                        "fontSize": 10,
                        "borderThickness": 1,
                        "borderColor": "black"
                    })
                    frac = np.dot(inv_cell, atom.position)
                    atomic_info.append({
                        "Atom": label_text,
                        "Elements": symbol,
                        "X": round(x, 3),
                        "Y": round(y, 3),
                        "Z": round(z, 3),
                        "Frac X": round(frac[0], 3),
                        "Frac Y": round(frac[1], 3),
                        "Frac Z": round(frac[2], 3)
                    })

        html_str = view._make_html()

        centered_html = f"<div style='display: flex; justify-content: center; position: relative;'>{html_str}</div>"

        unique_elements = sorted(set(structure.get_chemical_symbols()))
        legend_html = "<div style='display: flex; flex-wrap: wrap; align-items: center;justify-content: center;'>"
        for elem in unique_elements:
            color = jmol_colors.get(elem, "#CCCCCC")
            legend_html += (
                f"<div style='margin-right: 15px; display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black;'></div>"
                f"<span>{elem}</span></div>"
            )
        legend_html += "</div>"

        # Get lattice parameters
        cell_params = structure.get_cell_lengths_and_angles()  # (a, b, c, α, β, γ)
        a_para, b_para, c_para = cell_params[:3]
        alpha, beta, gamma = [radians(x) for x in cell_params[3:]]

        volume = a_para * b_para * c_para * sqrt(
            1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 +
            2 * cos(alpha) * cos(beta) * cos(gamma)
        )
        # Get lattice parameters

        lattice_str = (
            f"a = {cell_params[0]:.4f} Å<br>"
            f"b = {cell_params[1]:.4f} Å<br>"
            f"c = {cell_params[2]:.4f} Å<br>"
            f"α = {cell_params[3]:.2f}°<br>"
            f"β = {cell_params[4]:.2f}°<br>"
            f"γ = {cell_params[5]:.2f}°<br>"
            f"Volume = {volume:.2f} Å³"
        )

        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.markdown("<h3 style='text-align: center;'>Interactive Structure Visualization</h3>",
                        unsafe_allow_html=True)

            try:
                mg_structure = AseAtomsAdaptor.get_structure(structure)
                sg_analyzer = SpacegroupAnalyzer(mg_structure)
                spg_symbol = sg_analyzer.get_space_group_symbol()
                spg_number = sg_analyzer.get_space_group_number()
                space_group_str = f"{spg_symbol} ({spg_number})"
            except Exception:
                space_group_str = "Not available"
            try:
                mg_structure = AseAtomsAdaptor.get_structure(structure)
                sg_analyzer = SpacegroupAnalyzer(mg_structure)
                spg_symbol = sg_analyzer.get_space_group_symbol()
                spg_number = sg_analyzer.get_space_group_number()
                space_group_str = f"{spg_symbol} ({spg_number})"

                # New check
                same_lattice = lattice_same_conventional_vs_primitive(mg_structure)
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
            <div style='text-align: center; font-size: 22px; color: {"green" if same_lattice else "gray"}'>
                <strong>{cell_note}</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='text-align: center; font-size: 22px;'>
                <p><strong>Lattice Parameters:</strong><br>{lattice_str}</p>
                <p><strong>Legend:</strong><br>{legend_html}</p>
                <p><strong>Number of Atoms:</strong> {len(structure)}</p>
                <p><strong>Space Group:</strong> {space_group_str}</p>
            </div>
            """, unsafe_allow_html=True)

            # If atomic positions are to be shown, display them as a table.
        if show_atomic:

            df_atoms = pd.DataFrame(atomic_info)
            st.subheader("Atomic Positions")
            st.dataframe(df_atoms)

        with right_col:
            st.components.v1.html(centered_html, height=600)

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

st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
col_settings, col_divider, col_plot = st.columns([1, 0.05, 1])
with col_settings:
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

    # --- Diffraction Calculator Selection ---
    col1, col2, col3 = st.columns(3)
    with col1:
        diffraction_choice = st.radio(
            "Select Diffraction Calculator",
            ["XRD (X-ray)", "ND (Neutron)"],
            index=0,
        )
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

    if diffraction_choice == "ND (Neutron)":
        st.info(
            "🔬 The following neutron diffraction (ND) patterns are for **powder samples**, assuming **randomly oriented crystallites**. "
            "The calculator applies the **Lorentz correction**: `L(θ) = 1  / sin²θ cosθ`. It does not account for other corrections, such as preferred orientation, absorption, "
            "instrumental broadening, or temperature effects (Debye-Waller factors). The main differences in the calculation from the XRD pattern are: "
            " (1) Atomic scattering lengths are constant, and (2) Polarization correction is not necessary."
        )
    else:
        st.info(
            "🔬 The following X-ray diffraction (XRD) patterns are for **powder samples**, assuming **randomly oriented crystallites**. "
            "The calculator applies the **Lorentz-polarization correction**: `LP(θ) = (1 + cos²(2θ)) / (sin²θ cosθ)`. It does not account for other corrections, such as preferred orientation, absorption, "
            "instrumental broadening, or temperature effects (Debye-Waller factors). "
        )


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
    #preset_options = [
    #    'CoKa1', 'CoKa2', 'Co(Ka1+Ka2)', 'Co(Ka1+Ka2+Kb1)', 'CoKb1',
    #    'MoKa1', 'MoKa2', 'Mo(Ka1+Ka2)', 'Mo(Ka1+Ka2+Kb1)', 'MoKb1',
    #    'CuKa1', 'CuKa2', 'Cu(Ka1+Ka2)', 'Cu(Ka1+Ka2+Kb1)', 'CuKb1',
    #    'CrKa1', 'CrKa2', 'Cr(Ka1+Ka2)', 'Cr(Ka1+Ka2+Kb1)', 'CrKb1',
    #    'FeKa1', 'FeKa2', 'Fe(Ka1+Ka2)', 'Fe(Ka1+Ka2+Kb1)', 'FeKb1',
    #    'AgKa1', 'AgKa2', 'Ag(Ka1+Ka2)', 'Ag(Ka1+Ka2+Kb1)', 'AgKb1'
    #]
    preset_options = [
        'CoKa1', 'Co(Ka1+Ka2)', 'Co(Ka1+Ka2+Kb1)',
        'MoKa1', 'Mo(Ka1+Ka2)', 'Mo(Ka1+Ka2+Kb1)',
        'CuKa1',  'Cu(Ka1+Ka2)', 'Cu(Ka1+Ka2+Kb1)',
        'CrKa1',  'Cr(Ka1+Ka2)', 'Cr(Ka1+Ka2+Kb1)',
        'FeKa1',  'Fe(Ka1+Ka2)', 'Fe(Ka1+Ka2+Kb1)',
        'AgKa1', 'Ag(Ka1+Ka2)', 'Ag(Ka1+Ka2+Kb1)',
    ]
    preset_wavelengths = {
        'Cu(Ka1+Ka2)': 0.154,
        'CuKa2': 0.15444,
        'CuKa1': 0.15406,
        'Cu(Ka1+Ka2+Kb1)': 0.153339,
        'CuKb1': 0.13922,
        'Mo(Ka1+Ka2)': 0.071,
        'MoKa2': 0.0711,
        'MoKa1': 0.07093,
        'Mo(Ka1+Ka2+Kb1)': 0.07059119,
        'MoKb1': 0.064,
        'Cr(Ka1+Ka2)': 0.229,
        'CrKa2': 0.22888,
        'CrKa1': 0.22897,
        'Cr(Ka1+Ka2+Kb1)': 0.22775471,
        'CrKb1': 0.208,
        'Fe(Ka1+Ka2)': 0.194,
        'FeKa2': 0.194,
        'FeKa1': 0.19360,
        'Fe(Ka1+Ka2+Kb1)': 0.1927295,
        'FeKb1': 0.176,
        'Co(Ka1+Ka2)': 0.179,
        'CoKa2': 0.17927,
        'CoKa1': 0.17889,
        'Co(Ka1+Ka2+Kb1)': 0.1781100,
        'CoKb1': 0.163,
        'AgKa1': 0.0561,
        'AgKa2': 0.05634,
        'Ag(Ka1+Ka2)': 0.0561,
        'AgKb1': 0.0496,
        'Ag(Ka1+Ka2+Kb1)': 0.0557006
    }
    col1, col2 = st.columns(2)
    preset_options_neutron = ['Thermal Neutrons', 'Cold Neutrons', 'Hot Neutrons']
    preset_wavelengths_neutrons = {
        'Thermal Neutrons': 0.154,
        'Cold Neutrons': 0.475,
        'Hot Neutrons': 0.087
    }

    if diffraction_choice == "XRD (X-ray)":
        with col1:
            preset_choice = st.selectbox(
                "Preset Wavelength",
                options=preset_options,
                index=0,
                help="I_Kalpha2 = 1/2 I_Kalpha1, I_Kbeta = 1/9 I_Kalpha1"
            )

        hide_input_for = [
            'Cu(Ka1+Ka2+Kb1)', 'Cu(Ka1+Ka2)',
            'Mo(Ka1+Ka2+Kb1)','Mo(Ka1+Ka2)',
            'Cr(Ka1+Ka2+Kb1)','Cr(Ka1+Ka2)',
            'Fe(Ka1+Ka2+Kb1)','Fe(Ka1+Ka2)',
            'Co(Ka1+Ka2+Kb1)','Co(Ka1+Ka2)',
            'Ag(Ka1+Ka2+Kb1)','Ag(Ka1+Ka2)'
        ]

        with col2:
            if preset_choice not in hide_input_for:
                wavelength_value = st.number_input(
                    "Wavelength (nm)",
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
    display_metric_min = twotheta_to_metric(st.session_state.two_theta_min, x_axis_metric, wavelength_A, wavelength_nm,
                                            diffraction_choice)
    display_metric_max = twotheta_to_metric(st.session_state.two_theta_max, x_axis_metric, wavelength_A, wavelength_nm,
                                            diffraction_choice)

    if x_axis_metric == "2θ (°)":
        step_val = 1.0
    elif x_axis_metric == "2θ (rad)":
        step_val = 0.0174533
    else:
        step_val = 0.1

    col1, col2 = st.columns(2)

    if x_axis_metric == "d (Å)" or x_axis_metric == "d (nm)":

        min_val = col2.number_input(f"⚙️ Maximum {x_axis_metric}", value=display_metric_min, step=step_val,
                                    key=f"min_val_{x_axis_metric}")
        max_val = col1.number_input(f"⚙️ Minimum {x_axis_metric}", value=display_metric_max, step=step_val,
                                    key=f"max_val_{x_axis_metric}")
    else:
        min_val = col1.number_input(f"⚙️ Minimum {x_axis_metric}", value=display_metric_min, step=step_val,
                                    key=f"min_val_{x_axis_metric}")
        max_val = col2.number_input(f"⚙️ Maximum {x_axis_metric}", value=display_metric_max, step=step_val,
                                    key=f"max_val_{x_axis_metric}")

    # --- Update the canonical two_theta values based on current inputs ---
    st.session_state.two_theta_min = metric_to_twotheta(min_val, x_axis_metric, wavelength_A, wavelength_nm,
                                                        diffraction_choice)
    st.session_state.two_theta_max = metric_to_twotheta(max_val, x_axis_metric, wavelength_A, wavelength_nm,
                                                        diffraction_choice)
    two_theta_display_range = (st.session_state.two_theta_min, st.session_state.two_theta_max)

    if peak_representation != "Delta":
        sigma = st.number_input("⚙️ Gaussian sigma (°) for peak sharpness (smaller = sharper peaks)", min_value=0.01,
                                max_value=1.5, value=0.5, step=0.01)
    else:
        sigma = 0.5
    num_annotate = st.number_input("⚙️ How many highest peaks to annotate (by intensity):", min_value=0, max_value=30,
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

with col_divider:
    st.write("")

# --- XRD Calculation ---
with col_plot:
    if not st.session_state.calc_xrd:
        st.subheader("📊 OUTPUT → Click first on the 'Calculate XRD / ND' button.")

if st.session_state.calc_xrd and uploaded_files:
    include_in_combined = {}
    st.sidebar.markdown("### Structures to include in the Diffraction Plot:")
    for file in uploaded_files:
        include_in_combined[file.name] = st.sidebar.checkbox(f"Include {file.name}", value=True)

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
            "wavelengths": [0.07093, 0.0711, ],
            "factors": [1.0, 1 / 2.0]
        },
        "Mo(Ka1+Ka2+Kb1)": {
            "wavelengths": [0.07093, 0.0711,  0.064],  # in nm: Kα₁, Kα₂, and Kβ (here CuKb1)
            "factors": [1.0, 1 / 2.0, 1 / 9.0]
        },
        "Cr(Ka1+Ka2)": {
            "wavelengths": [0.22897, 0.22888, ],
            "factors": [1.0, 1 / 2.0]
        },
        "Cr(Ka1+Ka2+Kb1)": {
            "wavelengths": [0.22897, 0.22888, 0.208],  # in nm: Kα₁, Kα₂, and Kβ (here CuKb1)
            "factors": [1.0, 1 / 2.0, 1 / 9.0]
        },
        "Fe(Ka1+Ka2)": {
            "wavelengths": [0.19360, 0.194, ],
            "factors": [1.0, 1 / 2.0]
        },
        "Fe(Ka1+Ka2+Kb1)": {
            "wavelengths": [0.19360, 0.194,  0.176],  # in nm: Kα₁, Kα₂, and Kβ (here CuKb1)
            "factors": [1.0, 1 / 2.0, 1 / 9.0]
        },
        "Co(Ka1+Ka2)": {
            "wavelengths": [0.17889, 0.17927, ],
            "factors": [1.0, 1 / 2.0]
        },
        "Co(Ka1+Ka2+Kb1)": {
            "wavelengths": [0.17889, 0.17927, 0.163],  # in nm: Kα₁, Kα₂, and Kβ (here CuKb1)
            "factors": [1.0, 1 / 2.0, 1 / 9.0]
        },
        "Ag(Ka1+Ka2)": {
            "wavelengths": [0.0561, 0.05634, ],
            "factors": [1.0, 1 / 2.0]
        },
        "Ag(Ka1+Ka2+Kb1)": {
            "wavelengths": [0.0561, 0.05634,0.0496],  # in nm: Kα₁, Kα₂, and Kβ (here CuKb1)
            "factors": [1.0, 1 / 2.0, 1 / 9.0]
        }
        # Extend with additional multi-component presets if needed.
    }

    # Check whether the user-selected preset is multi-component.
    is_multi_component = preset_choice in multi_component_presets
    if is_multi_component:
        comp_info = multi_component_presets[preset_choice]
        # Fallback: if "labels" key is missing, assign default labels based on number of wavelengths.
        if "labels" not in comp_info:
            n = len(comp_info["wavelengths"])
            if n == 2:
                comp_info["labels"] = ["Kα1", "Kα2"]
            elif n == 3:
                comp_info["labels"] = ["Kα1", "Kα2", "Kβ"]
            else:
                comp_info["labels"] = ["Kα1"] * n

    with col_plot:
        st.subheader("📊 OUTPUT → Diffraction Patterns")

        # For single-component, we already have wavelength_A (in Å) defined.
        # For multi-component, each component’s wavelength (in nm) will be converted individually.

        # Create the static matplotlib figure.
        fig_combined, ax_combined = plt.subplots(figsize=(6, 4))
        colors = plt.cm.tab10.colors
        pattern_details = {}
        full_range = (0.01, 179.9)

        # Loop over each uploaded file.
        for idx, file in enumerate(uploaded_files):
            mg_structure = load_structure(file)
            mg_structure = get_full_conventional_structure_diffra(mg_structure)

            if is_multi_component:
                # Create a common dense grid.
                num_points = 20000
                x_dense_full = np.linspace(full_range[0], full_range[1], num_points)
                dx = x_dense_full[1] - x_dense_full[0]
                y_dense_total = np.zeros_like(x_dense_full)

                # Lists to collect discrete peaks.
                all_filtered_x = []
                all_filtered_y = []
                all_filtered_hkls = []
                all_peak_types = []  # NEW: store the type of each peak

                comp_info = multi_component_presets[preset_choice]
                for comp_index, (wl, factor) in enumerate(zip(comp_info["wavelengths"], comp_info["factors"])):
                    wavelength_A_comp = wl * 10  # convert nm to Å
                    if diffraction_choice == "ND (Neutron)":
                        diff_calc = NDCalculator(wavelength=wavelength_A_comp)
                    else:
                        diff_calc = XRDCalculator(wavelength=wavelength_A_comp)
                    diff_pattern = diff_calc.get_pattern(mg_structure, two_theta_range=full_range, scaled=False)

                    # Process peaks for this component.
                    filtered_x = []
                    filtered_y = []
                    filtered_hkls = []
                    # Loop over peaks.
                    for x_val, y_val, hkl_group in zip(diff_pattern.x, diff_pattern.y, diff_pattern.hkls):
                        if any(len(h['hkl']) == 3 and tuple(h['hkl'][:3]) == (0, 0, 0) for h in hkl_group):
                            continue
                        if any(len(h['hkl']) == 4 and tuple(h['hkl'][:4]) == (0, 0, 0, 0) for h in hkl_group):
                            continue
                        filtered_x.append(x_val)
                        filtered_y.append(y_val * factor)  # scale intensity
                        filtered_hkls.append(hkl_group)
                        # Append the corresponding label for this component.
                        all_peak_types.append(comp_info["labels"][comp_index])

                    # Create the dense curve for this component.
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

                    # Sum up contributions from this component.
                    y_dense_total += y_dense_comp
                    # Append discrete peaks from this component.
                    all_filtered_x.extend(filtered_x)
                    all_filtered_y.extend(filtered_y)
                    all_filtered_hkls.extend(filtered_hkls)
            else:
                # Single-wavelength branch (as before).
                if diffraction_choice == "ND (Neutron)":
                    diff_calc = NDCalculator(wavelength=wavelength_A)
                else:
                    diff_calc = XRDCalculator(wavelength=wavelength_A)
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
                # For single-wavelength, assign all peaks as "Kα1".
                all_peak_types = ["Kα1"] * len(filtered_x)

            # Intensity scaling.
            if intensity_scale_option == "Normalized":
                norm_factor = np.max(all_filtered_y) if np.max(all_filtered_y) > 0 else 1.0
                y_dense_total = (y_dense_total / np.max(y_dense_total)) * 100
                displayed_intensity_array = (np.array(all_filtered_y) / norm_factor) * 100
            else:
                displayed_intensity_array = np.array(all_filtered_y)

            # Convert discrete peak positions.
            peak_vals = twotheta_to_metric(np.array(all_filtered_x), x_axis_metric, wavelength_A, wavelength_nm,
                                           diffraction_choice)
            ka1_indices = [i for i, pt in enumerate(all_peak_types) if pt == "Kα1"]
            ka1_intensities = [displayed_intensity_array[i] for i in ka1_indices]

            if ka1_intensities:
                sorted_ka1 = sorted(zip(ka1_indices, ka1_intensities), key=lambda x: x[1], reverse=True)
                annotate_indices = set(i for i, _ in sorted_ka1[:num_annotate])
            else:
                annotate_indices = set()

            # Save details (including new 'peak_types') for this file.
            pattern_details[file.name] = {
                "peak_vals": peak_vals,
                "intensities": displayed_intensity_array,
                "hkls": all_filtered_hkls,
                "peak_types": all_peak_types,  # NEW: discrete peak labels
                "annotate_indices": annotate_indices,
                "x_dense_full": x_dense_full,
                "y_dense": y_dense_total
            }
            details = pattern_details[file.name]
            # (Plot the static figure as before.)
            if include_in_combined[file.name]:
                color = colors[idx % len(colors)]
                mask = (x_dense_full >= st.session_state.two_theta_min) & (
                            x_dense_full <= st.session_state.two_theta_max)
                x_dense_plot = twotheta_to_metric(x_dense_full[mask], x_axis_metric, wavelength_A, wavelength_nm,
                                                  diffraction_choice)

                if peak_representation == "Delta":
                    # GROUP BY PEAK TYPE AND PLOT WITH DIFFERENT LINE STYLES
                    peak_groups = {}
                    for i, pt in enumerate(all_peak_types):
                        if pt not in peak_groups:
                            peak_groups[pt] = {"x": [], "y": []}
                        peak_groups[pt]["x"].append(peak_vals[i])
                        peak_groups[pt]["y"].append(displayed_intensity_array[i])

                    plotted_labels = set()  # To avoid legend spam

                    shown_labels = set()

                    shown_labels = set()

                    for pt, data in peak_groups.items():
                        if pt == "Kα1":
                            linestyle = "solid"
                            alpha = 1.0
                        elif pt == "Kα2":
                            linestyle = "dotted"
                            alpha = 0.6
                        elif pt == "Kβ":
                            linestyle = "dashed"
                            alpha = 0.4
                        else:
                            linestyle = "solid"
                            alpha = 0.5

                        # Filter to keep only peaks within range
                        x_vals = []
                        y_vals = []
                        for x, y in zip(data["x"], data["y"]):
                            canonical_twotheta = metric_to_twotheta(x, x_axis_metric, wavelength_A, wavelength_nm,
                                                                    diffraction_choice)
                            if st.session_state.two_theta_min <= canonical_twotheta <= st.session_state.two_theta_max:
                                x_vals.append(x)
                                y_vals.append(y)

                        if not x_vals:
                            continue  # skip if no valid peaks in range

                        label = f"{file.name} - {pt}"
                        label_to_use = label if label not in shown_labels else None
                        shown_labels.add(label)

                        # Now plot all vlines for this group at once (clean legend!)
                        ax_combined.vlines(x_vals, [0] * len(y_vals), y_vals,
                                           color=color,
                                           linestyle=linestyle,
                                           alpha=alpha,
                                           linewidth=2,
                                           label=label_to_use)
                        #plotted_labels.add(label)
                        # After all data has been plotted...
                        current_ylim = ax_combined.get_ylim()
                        new_max = 0

                        # Go through all plotted lines to find the highest Y value
                        for line in ax_combined.get_lines():
                            if len(line.get_ydata()) > 0:
                                new_max = max(new_max, max(line.get_ydata()))

                        # Go through all vlines (they’re stored in ax.collections)
                        for collection in ax_combined.collections:
                            if hasattr(collection, 'get_segments'):
                                for seg in collection.get_segments():
                                    _, y1 = seg[1]  # top of the vertical line
                                    new_max = max(new_max, y1)

                        # Set Y axis limit with +10% buffer
                        ax_combined.set_ylim(0, new_max * 1.2 if new_max > 0 else 1)



                else:
                    ax_combined.plot(x_dense_plot, y_dense_total[mask], label=f"{file.name}", color=color)


                for i, (peak, hkl_group) in enumerate(zip(peak_vals, all_filtered_hkls)):
                    # Convert the peak position to the appropriate metric for the x-axis
                    peak_twotheta = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm,
                                                       diffraction_choice)
                    if st.session_state.two_theta_min <= peak_twotheta <= st.session_state.two_theta_max:
                        closest_index = np.abs(x_dense_full - peak_twotheta).argmin()
                        actual_intensity = y_dense_total[closest_index]
                        # Only annotate if the peak is of type "Kα1"
                        if i in annotate_indices and details["peak_types"][i] == "Kα1":
                            if len(hkl_group[0]['hkl']) == 3:
                                hkl_str = ", ".join([
                                    f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                                    for h in hkl_group])
                            else:
                                hkl_str = ", ".join([
                                    f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                                    for h in hkl_group])
                            ax_combined.annotate(hkl_str, xy=(peak, actual_intensity), xytext=(0, 5),
                                                 textcoords='offset points', fontsize=8, rotation=90,
                                                 ha='center', va='bottom', color=color)
        # Set axis labels and legend.
        ax_combined.set_xlabel(x_axis_metric)
        if intensity_scale_option == "Normalized":
            ax_combined.set_ylabel("Intensity (Normalized, a.u.)")
        else:
            ax_combined.set_ylabel("Intensity (Absolute, a.u.)")
        if ax_combined.get_lines():
            max_intensity = max([np.max(line.get_ydata()) for line in ax_combined.get_lines()])
            ax_combined.set_ylim(0, max_intensity * 1.2)
        ax_combined.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)

        if "placeholder_static" not in st.session_state:
            st.session_state.placeholder_static = st.empty()
        st.session_state.fig_combined = fig_combined
        st.session_state.placeholder_static.pyplot(st.session_state.fig_combined)

    if mode == "Basic":
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        st.markdown("""
                <hr style="height:3px;border:none;color:#333;background-color:#333;" />
                """, unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; font-size: 24px;'>
            🎯 <strong>Results Section 1 / 2:</strong> See the Resulted Diffraction Patterns in Interactive Plot Below ⬇️ or in the Static Plot Above ⬆️.<br>
            🪧 <strong>Step 4 / 4</strong> If Needed, Upload Your Own Diffraction Patterns For Comparison: 
            <br><span style="font-size: 28px;">⬇️</span>
         </div>
        """, unsafe_allow_html=True)
        st.markdown("""
                <hr style="height:3px;border:none;color:#333;background-color:#333;" />
                """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    st.subheader("Interactive Peak Identification and Indexing")
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
    
    # Interactive Plotly figure for peak identification and indexing.
    fig_interactive = go.Figure()

    for idx, (file_name, details) in enumerate(pattern_details.items()):
        if not include_in_combined.get(file_name, False):
            continue

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
                    canonical = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
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
                    # Determine color, dash style, and hover settings based on peak type.
                    if pt == "Kα1":
                        pt_color = base_color
                        dash_type = "solid"
                        hover_info = "text"
                        hover_template = f"<br>{file_name} - {pt}<br><b>{x_axis_metric}: %{{x:.2f}}</b><br>Intensity: %{{y:.2f}}<br><b>%{{text}}</b><extra></extra>"
                    elif pt == "Kα2":
                        pt_color = rgb_color(colors[idx % len(colors)], opacity=0.6)
                        dash_type = "dot"
                        hover_info = "skip"  # Remove hover info for Kα2
                        hover_template = None
                    elif pt == "Kβ":
                        pt_color = rgb_color(colors[idx % len(colors)], opacity=0.4)
                        dash_type = "dash"
                        hover_info = "skip"  # Remove hover info for Kβ
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
                        hoverlabel=dict(bgcolor=pt_color, font=dict(color="white", size=20))
                    ))
            else:
                # Fallback for when no peak_types are defined.
                vertical_x = []
                vertical_y = []
                vertical_hover = []
                for i, peak in enumerate(details["peak_vals"]):
                    canonical = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
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
                        vertical_hover.extend([f"(hkl): {hkl_str}", f"(hkl):: {hkl_str}", None])
                fig_interactive.add_trace(go.Scatter(
                    x=vertical_x,
                    y=vertical_y,
                    mode='lines',
                    name=file_name,
                    showlegend=True,
                    line=dict(color=base_color, width=4, dash="solid"),
                    hoverinfo="text",
                    text=vertical_hover,
                    hovertemplate=f"<br>{file_name}<br><b>{x_axis_metric}: %{{x:.2f}}</b><br>Intensity: %{{y:.2f}}<br><b>%{{text}}</b><extra></extra>",
                    hoverlabel=dict(bgcolor=base_color, font=dict(color="white", size=20))
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
                if (peak_type == "Kα1" and not show_Kalpha1_hover) or (peak_type == "Kα2" and not show_Kalpha2_hover) or (peak_type == "Kβ" and not show_Kbeta_hover):
                    continue
                canonical = metric_to_twotheta(peak, x_axis_metric, wavelength_A, wavelength_nm, diffraction_choice)
                if st.session_state.two_theta_min <= canonical <= st.session_state.two_theta_max:
                    peak_vals_in_range.append(peak)
                    #intensities_in_range.append(details["intensities"][i])
                    gauss = np.exp(-((details["x_dense_full"] - peak) ** 2) / (2 * sigma ** 2))
                    # Calculate the area under the Gaussian.
                    area = np.sum(gauss) * dx
                    # This is the Gaussian that is added to the continuous curve:
                    scaled_gauss = (details["intensities"][i] / area) * gauss
                    # Compute the maximum of the Gaussian.
                    max_gauss = np.max(scaled_gauss)
                    gaussian_max_intensities.append(max_gauss)

                    
                    hkl_group = details["hkls"][i]
                    if len(hkl_group[0]['hkl']) == 3:
                        hkl_str = ", ".join(
                            [f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})"
                             for h in hkl_group])
                    else:
                        hkl_str = ", ".join(
                            [f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})"
                             for h in hkl_group])
                    #  if "peak_types" in details:
                        # hover_text = f"{details['peak_types'][i]}: {hkl_str}"
                  #      hover_text = f"{'(hkl)'}: {hkl_str}"
                  #  else:
                  #      hover_text = f"(hkl): {hkl_str}"


                    # Conditionally include hover text based on the checkboxes:
                    if peak_type == "Kα1":
                        hover_text = f"Kα1 (hkl): {hkl_str}"
                    elif peak_type == "Kα2":
                        hover_text = f"Kα2 (hkl): {hkl_str}"
                    elif peak_type == "Kβ":
                        hover_text = f"Kβ (hkl): {hkl_str}"
                    else:
                        hover_text = f"Kα1 (hkl): {hkl_str}"
                    #peak_hover_texts.append(hover_text)
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

    display_metric_min = twotheta_to_metric(st.session_state.two_theta_min, x_axis_metric, wavelength_A, wavelength_nm,
                                            diffraction_choice)
    display_metric_max = twotheta_to_metric(st.session_state.two_theta_max, x_axis_metric, wavelength_A, wavelength_nm,
                                            diffraction_choice)
    if x_axis_metric in ["d (Å)", "d (nm)"]:
        fig_interactive.update_layout(xaxis=dict(range=[display_metric_max, display_metric_min]))
    else:
        fig_interactive.update_layout(xaxis=dict(range=[display_metric_min, display_metric_max]))

    if peak_representation == "Delta":
        fig_interactive.update_layout(
        height=1000,
        margin=dict(t=80, b=80, l=60, r=30),
        hovermode="x",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            font=dict(size=36)
        ),
        xaxis=dict(
            title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
            tickfont=dict(size=36, color='black')
        ),
        yaxis=dict(
            title=dict(text="Intensity (a.u.)", font=dict(size=36, color='black')),
            tickfont=dict(size=36, color='black'), range=[0, 125]
        ),
        hoverlabel=dict(font=dict(size=30)),
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
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5,
                font=dict(size=36)
            ),
            xaxis=dict(
                title=dict(text=x_axis_metric, font=dict(size=36, color='black'), standoff=20),
                tickfont=dict(size=36, color='black')
            ),
            yaxis=dict(
                title=dict(text="Intensity (a.u.)", font=dict(size=36, color='black')),
                tickfont=dict(size=36, color='black')
            ),
            hoverlabel=dict(font=dict(size=30)),
            font=dict(size=18),
            autosize=True
            )

    if "placeholder_interactive" not in st.session_state:
        st.session_state.placeholder_interactive = st.empty()
    st.session_state.fig_interactive = fig_interactive
    st.session_state.placeholder_interactive.plotly_chart(st.session_state.fig_interactive, use_container_width=True)
    st.subheader("Append Your XRD Pattern Data")
    #
    user_pattern_file = st.file_uploader(
        "Upload additional XRD pattern (2 columns: X-values and Intensity. The first line in the file is skipped, supposed it is a header.)",
        type=["csv", "txt", "xy"],
        key="user_xrd", accept_multiple_files=True
    )

    #  st.session_state.placeholder_interactive.plotly_chart(st.session_state.fig_interactive,
    #                                                       use_container_width=True, )

    # if user_pattern_file is not None and show_user_pattern:

    if user_pattern_file and show_user_pattern:
        # Check if multiple files were uploaded:
        if isinstance(user_pattern_file, list):
            # Get a list of colors from the tab10 colormap
            cmap = plt.cm.Set2
            n_files = len(user_pattern_file)
            static_colors = [cmap(i / n_files) for i in range(n_files)]
            # For Plotly, convert to hex colors:
            interactive_colors = [mcolors.to_hex(c) for c in static_colors]

            user_colorss = ["#000000", "#8B4513", "#808080", "#000000", "#8B4513", "#808080"]
            static_colors = user_colorss
            interactive_colors = user_colorss
            for idx, file in enumerate(user_pattern_file):
                try:
                    df = pd.read_csv(file, sep=r'[,\t; ]+', header=None, skiprows=1 )
                    if df.shape[1] < 2:
                        df = pd.read_csv(file, sep=r'[,\t; ]+', header=None, skiprows=1 )
                    x_user = df.iloc[:, 0].values
                    y_user = df.iloc[:, 1].values
                except Exception as e:
                    st.error(f"Error processing the uploaded file {file.name}: {e}")
                    continue  # Skip this file if there's an error

                if x_user is not None and y_user is not None:
                    if intensity_scale_option == "Normalized":
                        y_user = (y_user / np.max(y_user)) * 100

                    # Filter user data to the current x-axis range
                    mask_user = (x_user >= st.session_state.two_theta_min) & (x_user <= st.session_state.two_theta_max)
                    x_user_filtered = x_user[mask_user]
                    y_user_filtered = y_user[mask_user]

                    # Append to the static matplotlib figure with a unique color
                    ax = st.session_state.fig_combined.gca()
                    ax.plot(x_user_filtered, y_user_filtered, label=file.name,linestyle='--', marker='o', linewidth=0.5, markersize=1,
                            color=static_colors[idx])
                    ax.legend()
                    # Update y-axis range to include new data
                    current_ylim = ax.get_ylim()
                    if len(y_user_filtered) > 0:
                        new_max = np.max(y_user_filtered)
                        ax.set_ylim(0, max(current_ylim[1], new_max * 1.1))
                    st.session_state.placeholder_static.pyplot(st.session_state.fig_combined)

                    # Append to the interactive Plotly figure with the file's name and unique color
                    st.session_state.fig_interactive.add_trace(go.Scatter(
                        x=x_user_filtered,
                        y=y_user_filtered,
                        mode='lines+markers',
                        name=file.name,
                        line=dict(dash='solid', width=1, color=interactive_colors[idx]),
                        marker=dict( color=interactive_colors[idx], size=5)
                    ))
        else:
            # Only one file was uploaded; use the first color from tab10
            static_color = plt.cm.Set2(0)
            interactive_color = mcolors.to_hex(static_color)
            try:
                df = pd.read_csv(user_pattern_file, delim_whitespace=True, header=None)
                if df.shape[1] < 2:
                    df = pd.read_csv(user_pattern_file, sep=",", header=None)
                x_user = df.iloc[:, 0].values
                y_user = df.iloc[:, 1].values
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
                x_user, y_user = None, None

            if x_user is not None and y_user is not None:
                if intensity_scale_option == "Normalized":
                    y_user = (y_user / np.max(y_user)) * 100

                mask_user = (x_user >= st.session_state.two_theta_min) & (x_user <= st.session_state.two_theta_max)
                x_user_filtered = x_user[mask_user]
                y_user_filtered = y_user[mask_user]

                ax = st.session_state.fig_combined.gca()
                ax.plot(x_user_filtered, y_user_filtered, label=user_pattern_file.name, linestyle='--', marker='o',markersize=1,linewidth=0.5,
                        color=static_color)
                ax.legend()
                current_ylim = ax.get_ylim()
                if len(y_user_filtered) > 0:
                    new_max = np.max(y_user_filtered)
                    ax.set_ylim(0, max(current_ylim[1], new_max * 1.1))
                st.session_state.placeholder_static.pyplot(st.session_state.fig_combined)

                st.session_state.fig_interactive.add_trace(go.Scatter(
                    x=x_user_filtered,
                    y=y_user_filtered,
                    mode='lines+markers',
                    name=user_pattern_file.name,
                    line=dict(dash='solid',width=1, color=interactive_color),
                    marker=dict(color=interactive_color, size = 5)
                ))
    # Always update the interactive plot placeholder regardless
    st.session_state.placeholder_interactive.plotly_chart(
        st.session_state.fig_interactive, use_container_width=True, key="interactive_plot_updated"
    )

    if mode == "Basic":
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        st.markdown("""
                <hr style="height:3px;border:none;color:#333;background-color:#333;" />
                """, unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; font-size: 24px;'>
            🎯 <strong>Results Section 2 / 2:</strong>  👉 Extract the Quantitative Data Below. Interactive Table Which Allows Sorting Is Also Available: 
            <br><span style="font-size: 28px;">⬇️</span> ️
         </div>
        """, unsafe_allow_html=True)
        st.markdown("""
                <hr style="height:3px;border:none;color:#333;background-color:#333;" />
                """, unsafe_allow_html=True)


    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
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
                        [f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})" for h in
                         hkl_group])
                else:
                    hkl_str = ", ".join(
                        [f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})" for h in
                         hkl_group])
                table_str += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
            st.code(table_str, language="text")
        with st.expander(f"View Highest Intensity Peaks for XRD Pattern: {file.name}", expanded=True):
            table_str2 = "#X-axis    Intensity    hkl\n"
            for i, (theta, intensity, hkl_group) in enumerate(zip(peak_vals, intensities, hkls)):
                if i in annotate_indices:
                    if len(hkl_group[0]['hkl']) == 3:
                        hkl_str = ", ".join(
                            [f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})" for
                             h in hkl_group])
                    else:
                        hkl_str = ", ".join(
                            [f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3],last=True)})" for
                             h in hkl_group])
                    table_str2 += f"{theta:<12.3f} {intensity:<12.3f} {hkl_str}\n"
            st.code(table_str2, language="text")
        with st.expander(f"View Continuous Curve Data for XRD Pattern: {file.name}"):
            table_str3 = "#X-axis    Y-value\n"
            for x_val, y_val in zip(x_dense_full, y_dense):
                table_str3 += f"{x_val:<12.5f} {y_val:<12.5f}\n"
            st.code(table_str3, language="text")
        st.divider()
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
                        hkl_str = ", ".join([f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][2], last=True)})" for h in hkls[i]])
                    else:
                        hkl_str = ", ".join([f"({format_index(h['hkl'][0], first=True)}{format_index(h['hkl'][1])}{format_index(h['hkl'][3], last=True)})" for h in hkls[i]])
                    data_list.append([peak_vals[i], intensities[i], hkl_str, file_name])
        combined_df = pd.DataFrame(data_list, columns=["{}".format(selected_metric), "Intensity", "(hkl)", "Phase"])
        st.dataframe(combined_df)

# --- RDF (PRDF) Settings and Calculation ---
st.divider()
left_rdf, right_rdf = st.columns(2)
left_rdf, col_divider_rdf, right_rdf = st.columns([1, 0.05, 1])

with left_rdf:
    st.subheader("⚙️ (P)RDF Settings")
    st.info(
        "🔬 **PRDF** describes the atomic element pair distances distribution within a structure, providing insight into **local environments** and **structural disorder**. "
        "It is commonly used in **diffusion studies** to track atomic movement and ion transport, as well as in **phase transition analysis**, revealing changes in atomic ordering during melting or crystallization. "
        "Additionally, PRDF/RDF can be employed as one of the **structural descriptors in machine learning**. "
        "Here, the (P)RDF values are **unitless** (relative PRDF intensity). Peaks = preferred bonding distances. Peak width = disorder. Height = relative likelihood."
    )
    cutoff = st.number_input("⚙️ Cutoff (Å)", min_value=1.0, max_value=50.0, value=10.0, step=1.0, format="%.1f")
    bin_size = st.number_input("⚙️ Bin Size (Å)", min_value=0.05, max_value=5.0, value=0.1, step=0.05, format="%.2f")
    if "calc_rdf" not in st.session_state:
        st.session_state.calc_rdf = False
    if st.button("Calculate RDF"):
        st.session_state.calc_rdf = True

with col_divider_rdf:
    st.write("")

with right_rdf:
    if not st.session_state.calc_rdf:
        st.subheader("📊 OUTPUT → Click first on the 'RDF' button.")
    if st.session_state.calc_rdf and uploaded_files:
        st.subheader("📊 OUTPUT → RDF (PRDF & Total RDF)")
        species_combinations = list(combinations(species_list, 2)) + [(s, s) for s in species_list]
        all_prdf_dict = defaultdict(list)
        all_distance_dict = {}
        global_rdf_list = []
        for file in uploaded_files:
            #structure = read(file.name)
            #mg_structure = AseAtomsAdaptor.get_structure(structure)
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
            fig, ax = plt.subplots(figsize=(6, 4))
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
        st.subheader("Total RDF Plot:")
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
        fig_global, ax_global = plt.subplots(figsize=(6, 4))
        title_global = "Averaged Global RDF" if multi_structures else "Global RDF"
        global_color = colors[len(all_prdf_dict) % len(colors)]
        ax_global.plot(global_bins, global_rdf_avg, label="Global RDF", color=global_color)
        ax_global.set_xlabel("Distance (Å)")
        ax_global.set_ylabel("Total RDF Intensity")
        ax_global.set_title(title_global)
        ax_global.legend()
        ax_global.set_ylim(bottom=0)
        st.pyplot(fig_global)
        with st.expander("View Data for Total RDF"):
            table_str = "#Distance (Å)    Total RDF\n"
            for x, y in zip(global_bins, global_rdf_avg):
                table_str += f"{x:<12.3f} {y:<12.3f}\n"
            st.code(table_str, language="text")
st.divider()

st.markdown("""
**This application is open-source and released under the [MIT License](https://github.com/bracerino/prdf-calculator-online/blob/main/LICENCSE).**
""")
#If used in academic publications, please cite:

st.markdown("""
### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors and maintainers:

- **[Matminer](https://github.com/hackingmaterials/matminer)**  
  Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE).  

- **[Pymatgen](https://github.com/materialsproject/pymatgen)**  
  Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).  

- **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)**  
  Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER).  

- **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)**  
    Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt).

- **[Materials Project](https://next-gen.materialsproject.org/)**  
  Data from the Materials Project is made available under the  
  [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  

- **[AFLOW](http://aflow.org)**  
  Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).   
  When using structures from AFLOW, please cite:  
  Curtarolo et al., *Computational Materials Science*, 58 (2012) 218-226.  
  [DOI: 10.1016/j.commatsci.2012.02.005](https://doi.org/10.1016/j.commatsci.2012.02.005)
""")
