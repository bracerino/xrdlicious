# import pkg_resources
# installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
# st.subheader("Installed Python Modules")
# for package, version in installed_packages:
#    st.write(f"{package}=={version}")

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

# import aflow.keywords as K
from pymatgen.io.cif import CifWriter


def get_formula_type(formula):
    elements = []
    counts = []

    # Parse formula like "Fe2O3"
    import re
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    for element, count in matches:
        elements.append(element)
        counts.append(int(count) if count else 1)

    if len(elements) == 1:
        return "A"

    # Find the GCD to simplify
    from math import gcd
    from functools import reduce

    def find_gcd(numbers):
        return reduce(gcd, numbers)

    divisor = find_gcd(counts)
    counts = [c // divisor for c in counts]

    # Generate formula type
    if len(elements) == 2:
        if counts[0] == 1 and counts[1] == 1:
            return "AB"
        elif counts[0] == 1 and counts[1] == 2:
            return "AB2"
        elif counts[0] == 2 and counts[1] == 1:
            return "A2B"
        else:
            return f"A{counts[0]}B{counts[1]}"
    elif len(elements) == 3:
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1:
            return "ABC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3:
            return "ABC3"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}"
    else:
        return "Complex"

def check_structure_size_and_warn(structure, structure_name="structure"):
    n_atoms = len(structure)

    if n_atoms > 75:
        st.info(f"ℹ️ **Structure Notice**: {structure_name} contains a large number of **{n_atoms} atoms**. "
                f"Calculations may take longer depending on selected parameters. Please be careful to "
                f"not consume much memory, we are hosted on a free server. 😊")
        return "moderate"
    else:
        return "small"

def identify_structure_type(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()

        formula = structure.composition.reduced_formula
        formula_type = get_formula_type(formula)

        if spg_number in STRUCTURE_TYPES and formula_type in STRUCTURE_TYPES[
            spg_number]:
            structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**{structure_type}**"

        pearson = f"{crystal_system[0]}{structure.num_sites}"
        return f"**{crystal_system.capitalize()}** (Formula: {formula_type}, Pearson: {pearson})"

    except Exception as e:
        return f"Error identifying structure: {str(e)}"


STRUCTURE_TYPES = {
    # Cubic Structures
    225: {  # Fm-3m
        "A": "FCC (Face-centered cubic)",
        "AB": "Rock Salt (NaCl)",
        "AB2": "Fluorite (CaF2)",
        "A2B": "Anti-Fluorite",
        "AB3": "Cu3Au (L1₂)",
        "A3B": "AuCu3 type",
        "ABC": "Half-Heusler (C1b)",
        "AB6": "K2PtCl6 (cubic antifluorite)",
        "A2BC4": "Spinel (MgAl2O4)"
    },
    229: {  # Im-3m
        "A": "BCC (Body-centered cubic)",
        "A2B": "Caesium chloride (CsCl, B2)",
        "AB12": "NaZn13 type",
        "A6B": "Tungsten carbide (WC)"
    },
    221: {  # Pm-3m
        "A": "Simple cubic (SC)",
        "AB": "Cesium Chloride (CsCl)",
        "ABC3": "Perovskite (Cubic, ABO3)",
        "AB3": "Cu3Au type",
        "A3B": "Cr3Si (A15)",
        "AB6": "ReO3 type"
    },
    227: {  # Fd-3m
        "A": "Diamond cubic",
        "A2B": "Pyrite (FeS2)",
        "AB2": "Fluorite-like",
        "A2B4C": "Normal spinel",
        "AB4C2": "Inverse spinel",
        "AB2O4": "Spinel"
    },
    216: {  # F-43m
        "AB": "Zinc Blende (Sphalerite)",
        "A2B": "Antifluorite"
    },
    215: {  # P-43m
        "ABC3": "Inverse-perovskite",
        "AB4": "Half-anti-fluorite"
    },
    223: {  # Pm-3n
        "AB": "α-Mn structure",
        "A2B": "Cr3Si-type"
    },
    230: {  # Ia-3d
        "A2B3": "Garnet structure ((Ca,Mg,Fe)3(Al,Fe)2(SiO4)3)",
        "AB2": "Pyrochlore"
    },
    217: {  # I-43m
        "A12B": "α-Mn structure"
    },
    219: {  # F-43c
        "AB": "Sodium thallide"
    },
    205: {  # Pa-3
        "AB2": "Cuprite (Cu2O)",
        "AB6": "ReO3 structure"
    },

    # Hexagonal Structures
    194: {  # P6_3/mmc
        "A": "HCP (Hexagonal close-packed)",
        "AB": "Wurtzite (high-T)",
        "A2B": "AlB2 type (hexagonal)",
        "AB2": "CdI2 type",
        "AB3": "Ni3Sn type",
        "A3B": "DO19 structure (Ni3Sn-type)"
    },
    186: {  # P6_3mc
        "AB": "Wurtzite (ZnS)",
        "A2B": "Marcasite"
    },
    191: {  # P6/mmm
        "AB": "Graphite (hexagonal)",
        "AB2": "MoS2 type",
        "A2B": "AlB2 type",
        "AB5": "CaCu5 type",
        "A2B17": "Th2Ni17 type"
    },
    193: {  # P6_3/mcm
        "AB3": "Na3As structure",
        "A2B": "ZrBeSi structure"
    },
    187: {  # P-6m2
        "AB": "Nickeline (NiAs)",
        "AB2": "CdI2 type"
    },
    164: {  # P-3m1
        "AB2": "CdI2 type",
        "A": "Graphene layers"
    },
    166: {  # R-3m
        "A": "Rhombohedral",
        "AB": "Calcite/Dolomite",
        "AB2": "Corundum (Al2O3)",
        "A2B3": "α-Al2O3 type"
    },
    160: {  # R3m
        "A2X3": "Binary tetradymite",
        "AX2": "Delafossite"
    },

    # Tetragonal Structures
    139: {  # I4/mmm
        "A": "Body-centered tetragonal",
        "AB": "β-Tin",
        "A2B": "CuAu (L10)",
        "AB2": "MoSi2 type",
        "A3B": "Ni3Ti structure"
    },
    136: {  # P4_2/mnm
        "AB2": "Rutile (TiO2)",
        "A2B": "MoSi2 type"
    },
    123: {  # P4/mmm
        "AB": "γ-CuTi",
        "A2B": "CuAu (L10)"
    },
    140: {  # I4/mcm
        "AB2": "Anatase (TiO2)",
        "A15": "β-W structure"
    },
    141: {  # I41/amd
        "AB2": "Anatase (TiO2)",
        "A2": "α-Sn structure"
    },
    115: {  # P-4m2
        "ABC2": "Chalcopyrite (CuFeS2)"
    },
    129: {  # P4/nmm
        "AB": "PbO structure"
    },

    # Orthorhombic Structures
    62: {  # Pnma
        "AB": "MnP structure",
        "AB2": "Cotunnite (PbCl2)",
        "ABX3": "Perovskite (orthorhombic)",
        "A2B": "Fe2P type",
        "ABO3": "GdFeO3-type distorted perovskite",
        "A2BX4": "Olivine ((Mg,Fe)2SiO4)"
    },
    63: {  # Cmcm
        "A": "α-U structure",
        "AB": "CrB structure",
        "AB2": "HgBr2 type"
    },
    74: {  # Imma
        "AB": "TlI structure",
        "A2B": "Marcasite"
    },
    64: {  # Cmca
        "A": "α-Ga structure"
    },
    65: {  # Cmmm
        "AB2": "η-Fe2C structure"
    },
    70: {  # Fddd
        "A": "Orthorhombic unit cell"
    },

    # Monoclinic Structures
    14: {  # P21/c
        "AB": "Monoclinic structure",
        "AB2": "Baddeleyite (ZrO2)",
        "ABO3": "Monazite (CePO4)"
    },
    12: {  # C2/m
        "AB2": "Thortveitite (Sc2Si2O7)",
        "A2B3": "Bixbyite"
    },
    15: {  # C2/c
        "ABO4": "Scheelite (CaWO4)"
    },

    # Triclinic Structures
    2: {  # P-1
        "AB": "Triclinic structure",
        "AB3": "Wollastonite (CaSiO3)",
        "ABO4": "Kaolinite"
    },

    # Other important structures
    99: {  # P4mm
        "ABCD3": "Tetragonal perovskite"
    },
    167: {  # R-3c
        "AB": "Calcite (CaCO3)",
        "A2B3": "Corundum (Al2O3)"
    },
    176: {  # P6_3/m
        "A10B6C2X31": "Apatite (Ca10(PO4)6(OH)2)"
    },
    58: {  # Pnnm
        "AB2": "Marcasite (FeS2)"
    },
    11: {  # P21/m
        "A2B": "ThSi2 type"
    },
    72: {  # Ibam
        "A2B": "MoSi2 type"
    },
    198: {  # P213
        "AB": "FeSi structure",
        "A12": "β-Mn structure"
    },
    88: {  # I41/a
        "ABO4": "Scheelite (CaWO4)"
    },
    33: {  # Pna21
        "AB": "FeAs structure"
    },
    130: {  # P4/ncc
        "AB2": "Cristobalite (SiO2)"
    },
    152: {  # P3121
        "AB2": "Quartz (SiO2)"
    },
    200: {  # Pm-3
        "A6B": "Fe3W3C"
    },
    224: {  # Pn-3m
        "AB": "Pyrochlore-related"
    },
    127: {  # P4/mbm
        "A2B": "σ-phase structure",
        "AB5": "CaCu5 type"
    },
    148: {  # R-3
        "AB3": "Calcite (CaCO3)"
    },
    69: {  # Fmmm
        "A15": "β-W structure"
    },
    128: {  # P4/mnc
        "A15": "Cr3Si (A15)"
    },
    206: {  # Ia-3
        "A2B": "Pyrite derivative",
        "AB2": "Pyrochlore (defective)"
    },
    212: {  # P4_3 32
        "AB": "β-quartz (SiO2)",
        "A4B3": "Mn4Si3 type"
    },
    226: {  # Fm-3c
        "AX2": "BiF3 type"
    },
    196: {  # F23
        "AB": "FeS2 type"
    },
    227: {  # Fd-3m
        "A8B": "Gamma-brass"
    }
}


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

    std_lattice = dataset.std_lattice
    std_positions = dataset.std_positions
    std_types = dataset.std_types

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


def get_cod_entries(params):
    try:
        response = requests.get('https://www.crystallography.net/cod/result', params=params)
        if response.status_code == 200:
            results = response.json()
            return results  # Returns a list of entries
        else:
            st.error(f"COD search error: {response.status_code}")
            return []
    except Exception as e:
        st.write(
            "Error during connection to COD database. Probably reason is that the COD database server is currently down.")


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

def apply_y_scale(y_values, scale_type):
    if scale_type == "Logarithmic":
        # Add 1 to avoid log(0) and return 0 for 0 values
        return np.log10(y_values + 1)
    elif scale_type == "Square Root":
        return np.sqrt(y_values)
    else:  # Linear
        return y_values


def convert_intensity_scale(intensity_values, scale_type):
    if intensity_values is None or len(intensity_values) == 0:
        return intensity_values

    converted = np.copy(intensity_values)
    min_positive = 1

    if scale_type == "Square Root":
        converted[converted < 0] = 0
        converted = np.sqrt(converted)
    elif scale_type == "Logarithmic":
        converted[converted <= 1] = 1
        converted = np.log10(converted)
    return converted
