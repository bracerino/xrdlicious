#import pkg_resources
#import streamlit as st
#installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
#st.subheader("Installed Python Modules")
#for package, version in installed_packages:
#    st.write(f"{package}=={version}")

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"gcd is deprecated.*",
    category=FutureWarning,
)

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from collections import defaultdict
from itertools import combinations
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

from pymatgen.ext.optimade import OptimadeRester

def search_mc3d_optimade(query_params, limit=300):
    import requests
    from pymatgen.core import Structure, Lattice, Composition
    import streamlit as st

    # st.write("=" * 50)
    # st.write("🔍 **MC3D SEARCH DEBUG INFO**")
    # st.write(f"📝 Query parameters: {query_params}")
    # st.write(f"🔢 Limit: {limit}")

    endpoints = [
        "https://optimade.materialscloud.org/main/mc3d-pbesol-v2/v1/structures",
        "https://optimade.materialscloud.org/main/mc3d-pbe-v1/v1/structures",
    ]

    filter_parts = []
    strict_elements = None
    target_composition = None
    check_composition = False

    if 'elements' in query_params:
        elements = query_params['elements']
        strict_elements = set(elements)
        # st.write(f"🧪 Searching for elements: {elements} (STRICT - only these elements)")
        for el in elements:
            filter_parts.append(f'elements HAS "{el}"')

    if 'formula' in query_params:
        formula_input = query_params['formula'].replace(' ', '')
        # st.write(f"⚗️ Formula input from user: {formula_input}")

        try:
            target_composition = Composition(formula_input)
            check_composition = True
            elements_from_formula = sorted([str(el) for el in target_composition.elements])

            # st.write(f"✨ User's formula: {formula_input}")
            # st.write(f"✨ Pymatgen reduced formula: {target_composition.reduced_formula}")
            # st.write(f"✨ MC3D likely stores as: {''.join([el + str(int(target_composition[el])) if target_composition[el] != 1 else el for el in elements_from_formula])}")
            # st.write(f"🧪 Searching by elements: {elements_from_formula} (order-independent)")

            for el in elements_from_formula:
                filter_parts.append(f'elements HAS "{el}"')

            strict_elements = set(elements_from_formula)

        except Exception as e:
            st.warning(f"⚠️ Could not parse formula: {e}")

    filter_str = " AND ".join(filter_parts) if filter_parts else None

    params = {
        'page_limit': min(limit, 100)
    }
    if filter_str:
        params['filter'] = filter_str
        # st.write(f"🔍 **OPTIMADE Filter**: `{filter_str}`")
    # else:
    #     st.write("⚠️ No filter applied - fetching first results")

    for idx, endpoint in enumerate(endpoints):
        # st.write("-" * 50)
        # st.write(f"🌐 **Endpoint {idx + 1}/{len(endpoints)}**: {endpoint}")

        try:
            # st.write(f"📤 Sending request with params: {params}")
            response = requests.get(endpoint, params=params, timeout=30)
            # st.write(f"📡 **Response Status Code**: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # st.write(f"📦 Response keys: {list(data.keys())}")

                entries = data.get('data', [])
                # st.write(f"📊 **Number of entries in response**: {len(entries)}")

                if not entries:
                    # st.warning(f"⚠️ No entries returned from this endpoint")
                    # if 'meta' in data:
                    #     st.write(f"ℹ️ Meta info: {data['meta']}")
                    continue

                # if entries:
                #     st.write(f"🔬 First entry ID: {entries[0].get('id', 'N/A')}")
                #     st.write(f"🔬 First entry attributes keys: {list(entries[0].get('attributes', {}).keys())}")

                structures = []
                parse_errors = 0
                filtered_out = 0

                for entry_idx, entry in enumerate(entries[:limit]):
                    try:
                        attrs = entry['attributes']
                        entry_id = attrs.get('_mcloud_mc3d_id', entry['id'])

                        lattice_vectors = attrs.get('lattice_vectors')
                        if not lattice_vectors:
                            parse_errors += 1
                            continue

                        lattice = Lattice(lattice_vectors)

                        species = attrs.get('species_at_sites', [])
                        if not species:
                            parse_errors += 1
                            continue

                        if 'cartesian_site_positions' in attrs:
                            coords = attrs['cartesian_site_positions']
                            coords_are_cartesian = True
                        elif 'fractional_site_positions' in attrs:
                            coords = attrs['fractional_site_positions']
                            coords_are_cartesian = False
                        else:
                            parse_errors += 1
                            continue

                        structure = Structure(
                            lattice,
                            species,
                            coords,
                            coords_are_cartesian=coords_are_cartesian
                        )

                        if strict_elements:
                            structure_elements = set([str(el) for el in structure.composition.elements])
                            if structure_elements != strict_elements:
                                filtered_out += 1
                                continue

                            if check_composition and target_composition:
                                structure_comp = structure.composition.reduced_composition
                                target_comp = target_composition.reduced_composition

                                # if entry_idx < 3:
                                #     st.write(f"🔬 Entry #{entry_idx + 1} - {entry_id}:")
                                #     st.write(f"   - Target composition: {target_comp}")
                                #     st.write(f"   - Structure composition: {structure_comp}")
                                #     st.write(f"   - Match: {structure_comp == target_comp}")

                                if structure_comp != target_comp:
                                    filtered_out += 1
                                    continue

                        formula = attrs.get('chemical_formula_reduced', structure.composition.reduced_formula)

                        structures.append({
                            'id': entry_id,
                            'structure': structure,
                            'formula': formula
                        })

                        # if (entry_idx + 1) % 10 == 0:
                        #     st.write(f"✅ Parsed {entry_idx + 1}/{len(entries[:limit])} entries...")

                    except Exception as e:
                        parse_errors += 1
                        continue

                # st.write(f"📈 **Parsing Summary**:")
                # st.write(f"   - Successfully parsed: {len(structures)}")
                # st.write(f"   - Failed to parse: {parse_errors}")
                # if strict_elements:
                #     st.write(f"   - Filtered out (wrong elements): {filtered_out}")

                if structures:
                   #st.success(f"✅ Found {len(structures)} structures in MC3D via OPTIMADE.")
                    # st.write("=" * 50)
                    return structures
                # else:
                #     st.error("❌ No structures could be parsed from this endpoint")

            # elif response.status_code == 404:
            #     st.warning(f"⚠️ Endpoint not found (404)")
            # elif response.status_code == 500:
            #     st.error(f"❌ Server error (500)")
            #     try:
            #         error_data = response.json()
            #         st.write(f"Error details: {error_data}")
            #     except:
            #         st.write(f"Raw response: {response.text[:500]}")
            # else:
            #     st.warning(f"⚠️ Unexpected status code: {response.status_code}")
            #     st.write(f"Response text: {response.text[:500]}")

        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out for {endpoint}")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Connection error for {endpoint}")
        except Exception as e:
            st.error(f"❌ Unexpected error with endpoint {endpoint}: {str(e)}")
            import traceback
            st.write(f"Traceback: {traceback.format_exc()}")
            continue

    st.error("❌ Could not retrieve structures from any MC3D endpoint")
    return []


def get_mc3d_structure_by_id(mc3d_id):

    import requests
    from pymatgen.core import Structure, Lattice
    import streamlit as st

    endpoints = [
        "https://optimade.materialscloud.org/main/mc3d-pbesol-v2/v1/structures",
        "https://optimade.materialscloud.org/main/mc3d-pbe-v1/v1/structures",
    ]

    params = {
        'filter': f'_mcloud_mc3d_id="{mc3d_id}"'
    }

    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                entries = data.get('data', [])

                if not entries:
                    continue

                entry = entries[0]
                attrs = entry['attributes']

                lattice_vectors = attrs.get('lattice_vectors')
                if not lattice_vectors:
                    continue

                lattice = Lattice(lattice_vectors)
                species = attrs.get('species_at_sites', [])

                if not species:
                    continue

                if 'cartesian_site_positions' in attrs:
                    coords = attrs['cartesian_site_positions']
                    coords_are_cartesian = True
                elif 'fractional_site_positions' in attrs:
                    coords = attrs['fractional_site_positions']
                    coords_are_cartesian = False
                else:
                    continue

                structure = Structure(
                    lattice,
                    species,
                    coords,
                    coords_are_cartesian=coords_are_cartesian
                )

                return structure

        except Exception as e:
            continue

    st.warning(f"Could not fetch structure {mc3d_id} from MC3D")
    return None

ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

def get_formula_type(formula):
    elements = []
    counts = []

    import re
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    for element, count in matches:
        elements.append(element)
        counts.append(int(count) if count else 1)

    if len(elements) == 1:
        return "A"

    elif len(elements) == 2:
        # Binary compounds
        if counts[0] == 1 and counts[1] == 1:
            return "AB"
        elif counts[0] == 1 and counts[1] == 2:
            return "AB2"
        elif counts[0] == 2 and counts[1] == 1:
            return "A2B"
        elif counts[0] == 1 and counts[1] == 3:
            return "AB3"
        elif counts[0] == 3 and counts[1] == 1:
            return "A3B"
        elif counts[0] == 1 and counts[1] == 4:
            return "AB4"
        elif counts[0] == 4 and counts[1] == 1:
            return "A4B"
        elif counts[0] == 1 and counts[1] == 5:
            return "AB5"
        elif counts[0] == 5 and counts[1] == 1:
            return "A5B"
        elif counts[0] == 1 and counts[1] == 6:
            return "AB6"
        elif counts[0] == 6 and counts[1] == 1:
            return "A6B"
        elif counts[0] == 2 and counts[1] == 3:
            return "A2B3"
        elif counts[0] == 3 and counts[1] == 2:
            return "A3B2"
        elif counts[0] == 2 and counts[1] == 5:
            return "A2B5"
        elif counts[0] == 5 and counts[1] == 2:
            return "A5B2"
        elif counts[0] == 1 and counts[1] == 12:
            return "AB12"
        elif counts[0] == 12 and counts[1] == 1:
            return "A12B"
        elif counts[0] == 2 and counts[1] == 17:
            return "A2B17"
        elif counts[0] == 17 and counts[1] == 2:
            return "A17B2"
        elif counts[0] == 3 and counts[1] == 4:
            return "A3B4"
        else:
            return f"A{counts[0]}B{counts[1]}"

    elif len(elements) == 3:
        # Ternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1:
            return "ABC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3:
            return "ABC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1:
            return "AB3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1:
            return "A3BC"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4:
            return "AB2C4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4:
            return "A2BC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 2:
            return "AB4C2"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1:
            return "A2B4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 2:
            return "A4BC2"
        elif counts[0] == 4 and counts[1] == 2 and counts[2] == 1:
            return "A4B2C"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1:
            return "AB2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1:
            return "A2BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2:
            return "ABC2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4:
            return "ABC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1:
            return "AB4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1:
            return "A4BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 5:
            return "ABC5"
        elif counts[0] == 1 and counts[1] == 5 and counts[2] == 1:
            return "AB5C"
        elif counts[0] == 5 and counts[1] == 1 and counts[2] == 1:
            return "A5BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 6:
            return "ABC6"
        elif counts[0] == 1 and counts[1] == 6 and counts[2] == 1:
            return "AB6C"
        elif counts[0] == 6 and counts[1] == 1 and counts[2] == 1:
            return "A6BC"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 1:
            return "A2B2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 2:
            return "A2BC2"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 2:
            return "AB2C2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1:
            return "A3B2C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2:
            return "A3BC2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2:
            return "AB3C2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1:
            return "A2B3C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3:
            return "A2BC3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3:
            return "AB2C3"
        elif counts[0] == 3 and counts[1] == 3 and counts[2] == 1:
            return "A3B3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 3:
            return "A3BC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 3:
            return "AB3C3"
        elif counts[0] == 4 and counts[1] == 3 and counts[2] == 1:
            return "A4B3C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 3:
            return "A4BC3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 3:
            return "AB4C3"
        elif counts[0] == 3 and counts[1] == 4 and counts[2] == 1:
            return "A3B4C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 4:
            return "A3BC4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "AB3C4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "ABC6"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 7:
            return "A2B2C7"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}"

    elif len(elements) == 4:
        # Quaternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "ABCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "ABCD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "ABC3D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "AB3CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A3BCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "ABCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "ABC4D"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "AB4CD"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A4BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 4:
            return "AB2CD4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "A2BCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 4:
            return "ABC2D4"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4 and counts[3] == 1:
            return "AB2C4D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "A2BC4D"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "A2B4CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A2BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "AB2CD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "ABC2D"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "ABCD2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "A3B2CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "A3BC2D"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "A3BCD2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2 and counts[3] == 1:
            return "AB3C2D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 2:
            return "AB3CD2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 2:
            return "ABC3D2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "A2B3CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "A2BC3D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "A2BCD3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3 and counts[3] == 1:
            return "AB2C3D"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 3:
            return "AB2CD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 3:
            return "ABC2D3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 6:
            return "A1B4C1D6"
        elif counts[0] == 5 and counts[1] == 3 and counts[2] == 1 and counts[3] == 13:
            return "A5B3C1D13"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 4 and counts[3] == 9:
            return "A2B2C4D9"

        elif counts == [3, 2, 1, 4]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C1D4"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}"

    elif len(elements) == 5:
        # Five-element compounds (complex minerals like apatite)
        if counts == [1, 1, 1, 1, 1]:
            return "ABCDE"
        elif counts == [10, 6, 2, 31, 1]:  # Apatite-like: Ca10(PO4)6(OH)2
            return "A10B6C2D31E"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13DE"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13"
        elif counts == [3, 2, 3, 12, 1]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C3D12E"

        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}E{counts[4]}"

    elif len(elements) == 6:
        # Six-element compounds (very complex minerals)
        if counts == [1, 1, 1, 1, 1, 1]:
            return "ABCDEF"
        elif counts == [1, 1, 2, 6, 1, 1]:  # Complex silicate-like
            return "ABC2D6EF"
        else:
            # For 6+ elements, use a more compact notation
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, ...
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)

    else:
        if len(elements) <= 10:
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, G, H, I, J
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)
        else:
            return "Complex"
import time

LARGE_STRUCTURE_ATOM_THRESHOLD = 75


def _detect_running_locally():
    try:
        host = st.context.headers.get("host", "")
        return "localhost" in host or "127.0.0.1" in host
    except Exception:
        return False


def check_structure_size_and_warn(structure, structure_name="structure",
                                  is_local=None):
    n_atoms = len(structure)
    if is_local is None:
        is_local = _detect_running_locally()
    if n_atoms > LARGE_STRUCTURE_ATOM_THRESHOLD:
        if not is_local:
            st.info(
                f"ℹ️ **Structure Notice**: {structure_name} contains a large "
                f"number of **{n_atoms} atoms**. Calculations may take longer "
                f"depending on selected parameters. Please be careful to "
                f"not consume much memory, we are hosted on a free server. 😊"
            )
        return "moderate"
    return "small"


def report_large_structures(items, is_local=None):
    if is_local is None:
        is_local = _detect_running_locally()
    if is_local:
        return
    large = [(name, n) for name, n in items
             if n > LARGE_STRUCTURE_ATOM_THRESHOLD]
    if not large:
        return
    bullets = "\n".join(f"- **{name}** — {n} atoms" for name, n in large)
    st.info(
        "ℹ️ **Structure Notice**: the following structure(s) contain many "
        "atoms — calculations may take longer and use more memory. We are "
        "hosted on a free server, so please mind the memory budget 😊.\n\n"
        f"{bullets}"
    )



SPACE_GROUP_SYMBOLS = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P21", 5: "C2", 6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc", 10: "P2/m",
    11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c", 15: "C2/c", 16: "P222", 17: "P2221", 18: "P21212", 19: "P212121", 20: "C2221",
    21: "C222", 22: "F222", 23: "I222", 24: "I212121", 25: "Pmm2", 26: "Pmc21", 27: "Pcc2", 28: "Pma2", 29: "Pca21", 30: "Pnc2",
    31: "Pmn21", 32: "Pba2", 33: "Pna21", 34: "Pnn2", 35: "Cmm2", 36: "Cmc21", 37: "Ccc2", 38: "Amm2", 39: "Aem2", 40: "Ama2",
    41: "Aea2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2", 45: "Iba2", 46: "Ima2", 47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban",
    51: "Pmma", 52: "Pnna", 53: "Pmna", 54: "Pcca", 55: "Pbam", 56: "Pccn", 57: "Pbcm", 58: "Pnnm", 59: "Pmmn", 60: "Pbcn",
    61: "Pbca", 62: "Pnma", 63: "Cmcm", 64: "Cmca", 65: "Cmmm", 66: "Cccm", 67: "Cmma", 68: "Ccca", 69: "Fmmm", 70: "Fddd",
    71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma", 75: "P4", 76: "P41", 77: "P42", 78: "P43", 79: "I4", 80: "I41",
    81: "P-4", 82: "I-4", 83: "P4/m", 84: "P42/m", 85: "P4/n", 86: "P42/n", 87: "I4/m", 88: "I41/a", 89: "P422", 90: "P4212",
    91: "P4122", 92: "P41212", 93: "P4222", 94: "P42212", 95: "P4322", 96: "P43212", 97: "I422", 98: "I4122", 99: "P4mm", 100: "P4bm",
    101: "P42cm", 102: "P42nm", 103: "P4cc", 104: "P4nc", 105: "P42mc", 106: "P42bc", 107: "P42mm", 108: "P42cm", 109: "I4mm", 110: "I4cm",
    111: "I41md", 112: "I41cd", 113: "P-42m", 114: "P-42c", 115: "P-421m", 116: "P-421c", 117: "P-4m2", 118: "P-4c2", 119: "P-4b2", 120: "P-4n2",
    121: "I-4m2", 122: "I-4c2", 123: "I-42m", 124: "I-42d", 125: "P4/mmm", 126: "P4/mcc", 127: "P4/nbm", 128: "P4/nnc", 129: "P4/mbm", 130: "P4/mnc",
    131: "P4/nmm", 132: "P4/ncc", 133: "P42/mmc", 134: "P42/mcm", 135: "P42/nbc", 136: "P42/mnm", 137: "P42/mbc", 138: "P42/mnm", 139: "I4/mmm", 140: "I4/mcm",
    141: "I41/amd", 142: "I41/acd", 143: "P3", 144: "P31", 145: "P32", 146: "R3", 147: "P-3", 148: "R-3", 149: "P312", 150: "P321",
    151: "P3112", 152: "P3121", 153: "P3212", 154: "P3221", 155: "R32", 156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c", 160: "R3m",
    161: "R3c", 162: "P-31m", 163: "P-31c", 164: "P-3m1", 165: "P-3c1", 166: "R-3m", 167: "R-3c", 168: "P6", 169: "P61", 170: "P65",
    171: "P62", 172: "P64", 173: "P63", 174: "P-6", 175: "P6/m", 176: "P63/m", 177: "P622", 178: "P6122", 179: "P6522", 180: "P6222",
    181: "P6422", 182: "P6322", 183: "P6mm", 184: "P6cc", 185: "P63cm", 186: "P63mc", 187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
    191: "P6/mmm", 192: "P6/mcc", 193: "P63/mcm", 194: "P63/mmc", 195: "P23", 196: "F23", 197: "I23", 198: "P213", 199: "I213", 200: "Pm-3",
    201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3", 205: "Pa-3", 206: "Ia-3", 207: "P432", 208: "P4232", 209: "F432", 210: "F4132",
    211: "I432", 212: "P4332", 213: "P4132", 214: "I4132", 215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c", 220: "I-43d",
    221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m", 225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c", 229: "Im-3m", 230: "Ia-3d"
}


def identify_structure_type(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()

        formula = structure.composition.reduced_formula
        formula_type = get_formula_type(formula)
        # No logging here: this runs on every rerun (the structure info panels
        # call it), which filled the console during ordinary use. Files are
        # reported once, when they are first read — see app.py.
        if spg_number in STRUCTURE_TYPES and spg_number == 62 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Aragonite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==167 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
          #  print("YES")
          # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Calcite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==227 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "SiO2":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**β - Cristobalite (SiO2)**"
        elif formula == "C" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Graphite**"
        elif formula == "MoS2" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**MoS2 Type**"
        elif formula == "NiAs" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Nickeline (NiAs)**"
        elif formula == "ReO3" and spg_number in STRUCTURE_TYPES and spg_number ==221 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**ReO3 type**"
        elif formula == "TlI" and spg_number in STRUCTURE_TYPES and spg_number ==63 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**TlI structure**"
        elif spg_number in STRUCTURE_TYPES and formula_type in STRUCTURE_TYPES[
            spg_number]:
           # print("YES")
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
    },
    92: {
        "AB2": "α-Cristobalite (SiO2)"
    },
    229: {  # Im-3m
        "A": "BCC (Body-centered cubic)",
        "AB12": "NaZn13 type",
        "AB": "Tungsten carbide (WC)"
    },
    221: {  # Pm-3m
        "A": "Simple cubic (SC)",
        "AB": "Cesium Chloride (CsCl)",
        "ABC3": "Perovskite (Cubic, ABO3)",
        "AB3": "Cu3Au type",
        "A3B": "Cr3Si (A15)",
        #"AB6": "ReO3 type"
    },
    227: {  # Fd-3m
        "A": "Diamond cubic",

        "AB2": "Fluorite-like",
        "AB2C4": "Normal spinel",
        "A3B4": "Inverse spinel",
        "AB2C4": "Spinel",
        "A8B": "Gamma-brass",
        "AB2": "β - Cristobalite (SiO2)",
        "A2B2C7": "Pyrochlore"
    },
    55: {  # Pbca
        "AB2": "Brookite (TiO₂ polymorph)"
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
        "A3B": "Cr3Si-type"
    },
    230: {  # Ia-3d
        "A3B2C1D4": "Garnet structure ((Ca,Mg,Fe)3(Al,Fe)2(SiO4)3)",
        "AB2": "Pyrochlore"
    },
    217: {  # I-43m
        "A12B": "α-Mn structure"
    },
    219: {  # F-43c
        "AB": "Sodium thallide"
    },
    205: {  # Pa-3
        "A2B": "Cuprite (Cu2O)",
        "AB6": "ReO3 structure",
        "AB2": "Pyrite (FeS2)",
    },
    156: {
        "AB2": "CdI2 type",
    },
    # Hexagonal Structures
    194: {  # P6_3/mmc
        "AB": "Wurtzite (high-T)",
        "AB2": "AlB2 type (hexagonal)",
        "A3B": "Ni3Sn type",
        "A3B": "DO19 structure (Ni3Sn-type)",
        "A": "Graphite (hexagonal)",
        "A": "HCP (Hexagonal close-packed)",
        #"AB2": "MoS2 type",
    },
    186: {  # P6_3mc
        "AB": "Wurtzite (ZnS)",
    },
    191: {  # P6/mmm


        "AB2": "AlB2 type",
        "AB5": "CaCu5 type",
        "A2B17": "Th2Ni17 type"
    },
    193: {  # P6_3/mcm
        "A3B": "Na3As structure",
        "ABC": "ZrBeSi structure"
    },
   # 187: {  # P-6m2
#
 #   },
    164: {  # P-3m1
        "AB2": "CdI2 type",
        "A": "Graphene layers"
    },
    166: {  # R-3m
        "A": "Rhombohedral",
        "A2B3": "α-Al2O3 type",
        "ABC2": "Delafossite (CuAlO2)"
    },
    160: {  # R3m
        "A2B3": "Binary tetradymite",
        "AB2": "Delafossite"
    },

    # Tetragonal Structures
    139: {  # I4/mmm
        "A": "Body-centered tetragonal",
        "AB": "β-Tin",
        "A2B": "MoSi2 type",
        "A3B": "Ni3Ti structure"
    },
    136: {  # P4_2/mnm
        "AB2": "Rutile (TiO2)"
    },
    123: {  # P4/mmm
        "AB": "γ-CuTi",
        "AB": "CuAu (L10)"
    },
    140: {  # I4/mcm
        "AB2": "Anatase (TiO2)",
        "A": "β-W structure"
    },
    141: {  # I41/amd
        "AB2": "Anatase (TiO₂)",
        "A": "α-Sn structure",
        "ABC4": "Zircon (ZrSiO₄)"
    },
    122: {  # P-4m2
        "ABC2": "Chalcopyrite (CuFeS2)"
    },
    129: {  # P4/nmm
        "AB": "PbO structure"
    },

    # Orthorhombic Structures
    62: {  # Pnma
        "ABC3": "Aragonite (CaCO₃)",
        "AB2": "Cotunnite (PbCl2)",
        "ABC3": "Perovskite (orthorhombic)",
        "A2B": "Fe2P type",
        "ABC3": "GdFeO3-type distorted perovskite",
        "A2BC4": "Olivine ((Mg,Fe)2SiO4)",
        "ABC4": "Barite (BaSO₄)"
    },
    63: {  # Cmcm
        "A": "α-U structure",
        "AB": "CrB structure",
        "AB2": "HgBr2 type"
    },
    74: {  # Imma
        "AB": "TlI structure",
    },
    64: {  # Cmca
        "A": "α-Ga structure"
    },
    65: {  # Cmmm
        "A2B": "η-Fe2C structure"
    },
    70: {  # Fddd
        "A": "Orthorhombic unit cell"
    },

    # Monoclinic Structures
    14: {  # P21/c
        "AB": "Monoclinic structure",
        "AB2": "Baddeleyite (ZrO2)",
        "ABC4": "Monazite (CePO4)"
    },
    12: {  # C2/m
        "A2B2C7": "Thortveitite (Sc2Si2O7)"
    },
    15: {  # C2/c
        "A1B4C1D6": "Gypsum (CaH4O6S)",
        "ABC6": "Gypsum (CaH4O6S)",
        "ABC4": "Scheelite (CaWO₄)",
        "ABC5": "Sphene (CaTiSiO₅)"
    },
    1: {
        "A2B2C4D9": "Kaolinite"
    },
    # Triclinic Structures
    2: {  # P-1
        "AB": "Triclinic structure",
        "ABC3": "Wollastonite (CaSiO3)",
    },

    # Other important structures
    99: {  # P4mm
        "ABCD3": "Tetragonal perovskite"
    },
    167: {  # R-3c
        "ABC3": "Calcite (CaCO3)",
        "A2B3": "Corundum (Al2O3)"
    },
    176: {  # P6_3/m
        "A10B6C2D31E": "Apatite (Ca10(PO4)6(OH)2)",
        "A5B3C1D13": "Apatite (Ca5(PO4)3OH",
        "A5B3C13": "Apatite (Ca5(PO4)3OH"
    },
    58: {  # Pnnm
        "AB2": "Marcasite (FeS2)"
    },
    11: {  # P21/m
        "A2B": "ThSi2 type"
    },
    72: {  # Ibam
        "AB2": "MoSi2 type"
    },
    198: {  # P213
        "AB": "FeSi structure",
        "A12": "β-Mn structure"
    },
    88: {  # I41/a
        "ABC4": "Scheelite (CaWO4)"
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
        "A3B3C": "Fe3W3C"
    },
    224: {  # Pn-3m
        "AB": "Pyrochlore-related",
        "A2B": "Cuprite (Cu2O)"
    },
    127: {  # P4/mbm
        "AB": "σ-phase structure",
        "AB5": "CaCu5 type"
    },
    148: {  # R-3
        "ABC3": "Calcite (CaCO₃)",
        "ABC3": "Ilmenite (FeTiO₃)",
        "ABCD3": "Dolomite",
    },
    69: {  # Fmmm
        "A": "β-W structure"
    },
    128: {  # P4/mnc
        "A3B": "Cr3Si (A15)"
    },
    206: {  # Ia-3
        "AB2": "Pyrite derivative",
        "AB2": "Pyrochlore (defective)",
        "A2B3": "Bixbyite"
    },
    212: {  # P4_3 32

        "A4B3": "Mn4Si3 type"
    },
    180: {
        "AB2": "β-quartz (SiO2)",
    },
    226: {  # Fm-3c
        "AB2": "BiF3 type"
    },
    196: {  # F23
        "AB2": "FeS2 type"
    },
    96: {
        "AB2": "α-Cristobalite (SiO2)"
    }

}


STRUCTURES_AS_IS_KEY = "use_structures_as_is"
PENDING_AS_IS_KEY = "_pending_as_is"


def use_structures_as_is():
    """Global toggle set in the Diffraction Settings ("No symmetry search").

    When it is on, uploaded structures are used exactly as they were read
    from the file — no symmetry search, no conversion to a standardized
    conventional cell and no symmetry constraints in the structure editor.
    """
    try:
        # Request made by the symmetry pop-up. It is resolved on the first
        # read of the run — the structure editor renders before the
        # diffraction settings, so it must not show the standardized cell for
        # one render before the setting takes effect.
        if st.session_state.pop(PENDING_AS_IS_KEY, False):
            st.session_state[STRUCTURES_AS_IS_KEY] = True
    except Exception:
        pass
    try:
        return bool(st.session_state.get(STRUCTURES_AS_IS_KEY, False))
    except Exception:
        return False


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
    cell = (structure.lattice.matrix, structure.frac_coords, [site.specie.number for site in structure])
            #[max(site.species, key=site.species.get).number for site in structure])

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

    conv_structure = Structure(std_lattice, std_types, std_positions)
    return conv_structure


def rgb_color(color_tuple, opacity=0.8):
    r, g, b = [int(255 * x) for x in color_tuple]
    return f"rgba({r},{g},{b},{opacity})"


_LAMMPS_ATOM_STYLES = (
    "atomic", "charge", "full", "molecular", "bond", "angle",
)


def _detect_lammps_atom_style(filename):
    try:
        with open(filename, "r", errors="ignore") as f:
            in_atoms_block = False
            for line in f:
                s = line.strip()
                if not in_atoms_block:
                    if s.lower().startswith("atom_style"):
                        parts = s.split()
                        if len(parts) >= 2 and parts[1].lower() in _LAMMPS_ATOM_STYLES:
                            return parts[1].lower()
                    if s.startswith("Atoms"):
                        if "#" in s:
                            hint = s.split("#", 1)[1].strip().split()
                            if hint and hint[0].lower() in _LAMMPS_ATOM_STYLES:
                                return hint[0].lower()
                        in_atoms_block = True
                        continue
                else:
                    if not s or s.startswith("#"):
                        continue
                    cols = s.split()
                    n = len(cols)
                    has_images = n in (8, 9, 10) and all(
                        c.lstrip("-").isdigit() for c in cols[-3:]
                    )
                    n_data = n - (3 if has_images else 0)
                    if n_data == 5:
                        return "atomic"
                    if n_data == 6:
                        try:
                            float(cols[2])
                            if "." in cols[2] or "e" in cols[2].lower():
                                return "charge"
                        except Exception:
                            pass
                        return "molecular"
                    if n_data == 7:
                        return "full"
                    return None
    except Exception:
        return None
    return None


def _load_lammps_data(filename):
    from pymatgen.io.lammps.data import LammpsData

    detected = _detect_lammps_atom_style(filename)
    candidates = []
    if detected:
        candidates.append(detected)
    for s in _LAMMPS_ATOM_STYLES:
        if s not in candidates:
            candidates.append(s)

    last_err = None
    for style in candidates:
        try:
            struct = LammpsData.from_file(filename, atom_style=style).structure
            if len(struct) > 0:
                return struct
        except Exception as exc:
            last_err = exc
            continue
    raise ValueError(
        "Could not parse LAMMPS data file with any of the common atom_style "
        f"options ({', '.join(_LAMMPS_ATOM_STYLES)}). "
        f"Last error: {last_err}"
    )


_CIF_SG_SYMBOL_TAGS = (
    "_space_group_name_H-M_alt",
    "_symmetry_space_group_name_H-M",
    "_space_group_name_H-M",
)
_CIF_SG_NUMBER_TAGS = (
    "_space_group_IT_number",
    "_symmetry_Int_Tables_number",
)


def _clean_cif_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()


def extract_cif_space_group(cif_text):
    """Read the space group declared in a CIF file (symbol and/or IT number).

    Returns a dict with 'symbol' and 'number' keys (either may be None), or
    None when the CIF declares nothing or only the trivial P1 group.
    """
    symbol, number = None, None
    for line in cif_text.splitlines():
        s = line.strip()
        if not s.startswith("_"):
            continue
        parts = s.split(None, 1)
        tag = parts[0]
        value = _clean_cif_value(parts[1]) if len(parts) > 1 else ""
        if not value or value in ("?", "."):
            continue
        if symbol is None and tag in _CIF_SG_SYMBOL_TAGS:
            symbol = value
        elif number is None and tag in _CIF_SG_NUMBER_TAGS:
            try:
                number = int(float(value))
            except ValueError:
                pass
        if symbol is not None and number is not None:
            break

    if symbol is None and number is None:
        return None
    # P1 (#1) carries no symmetry information, so it is not reported.
    # P-1 (#2) is a real space group and is kept.
    if number == 1:
        return None
    if number is None and symbol is not None and symbol.replace(" ", "").upper() == "P1":
        return None
    return {"symbol": symbol, "number": number}


def format_cif_space_group(sg_info):
    if not sg_info:
        return None
    symbol, number = sg_info.get("symbol"), sg_info.get("number")
    if symbol and number:
        return f"{symbol} (#{number})"
    if symbol:
        return str(symbol)
    return f"#{number}"


# ---------------------------------------------------------------------------
# CIF repair
#
# Some CIF files (typically written by GUI programs after a "make P1" step)
# list *all* atoms of the unit cell but keep the symmetry operations of the
# original space group in the file. Applying those operations then puts several
# copies of the same atom on the same position, pymatgen adds their
# occupancies up and refuses the file with "Occupancy 4.0 exceeded tolerance."
# Other programs simply ignore the redundant operations; the helpers below do
# the same, but only after checking that the atom list really is complete.
# ---------------------------------------------------------------------------
import os
from pymatgen.io.cif import CifFile, CifParser as _CifParser, str2float
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import find_in_coord_list_pbc

CIF_REPAIR_NOTES_KEY = "cif_repair_notes"

_CIF_SYMOP_LOOP_TAGS = (
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop_id",
    "_symmetry_equiv_pos_site_id",
)
_CIF_SG_DROP_TAGS = (
    "_symmetry_space_group_name_hall",
    "_space_group_name_hall",
    "_symmetry_cell_setting",
)


def _cif_as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _cif_symmetry_is_redundant(cif_content, tol=1e-3):
    """True when the CIF atom list already contains every symmetry image.

    In that case the symmetry operations only duplicate atoms that are written
    in the file anyway, and can safely be dropped. Returns False whenever the
    operations still generate new atoms (a normal CIF holding only the
    asymmetric unit), so that such files are never silently truncated.
    """
    try:
        blocks = list(CifFile.from_str(cif_content).data.values())
    except Exception:
        return False

    for block in blocks:
        data = block.data
        ops_raw = None
        for tag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
            if data.get(tag):
                ops_raw = _cif_as_list(data[tag])
                break
        if not ops_raw or len(ops_raw) < 2:
            continue
        try:
            ops = [SymmOp.from_xyz_str(_clean_cif_value(op)) for op in ops_raw]
        except Exception:
            continue

        try:
            coords = np.array([
                [str2float(x), str2float(y), str2float(z)]
                for x, y, z in zip(_cif_as_list(data["_atom_site_fract_x"]),
                                   _cif_as_list(data["_atom_site_fract_y"]),
                                   _cif_as_list(data["_atom_site_fract_z"]))
            ], dtype=float)
        except Exception:
            continue
        if len(coords) < 2:
            continue
        # The check below is quadratic in the number of sites; huge files are
        # left alone rather than spending seconds on them.
        if len(coords) * len(ops) > 50000:
            continue

        symbols = _cif_as_list(data.get("_atom_site_type_symbol")) or \
            _cif_as_list(data.get("_atom_site_label"))
        if len(symbols) != len(coords):
            symbols = [""] * len(coords)

        overlapping = False
        for idx, coord in enumerate(coords):
            for op in ops:
                image = op.operate(coord)
                matches = find_in_coord_list_pbc(coords, image, atol=tol)
                # The image of a listed atom must again be a listed atom of the
                # same element, otherwise the atom list is not complete.
                matches = [m for m in matches if symbols[m] == symbols[idx]]
                if not len(matches):
                    return False
                if any(int(m) != idx for m in matches):
                    overlapping = True
        if overlapping:
            return True
    return False


def _cif_content_without_symmetry(cif_content):
    """Copy of the CIF text with the symmetry operations reduced to P1."""
    lines = cif_content.splitlines()
    out, idx = [], 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        low = stripped.lower()

        if low.startswith("loop_"):
            header_end = idx + 1
            headers = []
            while header_end < len(lines) and lines[header_end].strip().startswith("_"):
                headers.append(lines[header_end].strip().split()[0].lower())
                header_end += 1
            if any(h in _CIF_SYMOP_LOOP_TAGS for h in headers):
                data_end = header_end
                while data_end < len(lines):
                    row = lines[data_end].strip()
                    if not row or row.startswith("_") or \
                            row.lower().startswith(("loop_", "data_")):
                        break
                    data_end += 1
                out.append("loop_")
                out.append("_space_group_symop_operation_xyz")
                out.append("   'x, y, z'")
                idx = data_end
                continue

        tag = low.split()[0] if stripped.startswith("_") else ""
        if tag in ("_symmetry_space_group_name_h-m", "_space_group_name_h-m",
                   "_space_group_name_h-m_alt"):
            out.append("_symmetry_space_group_name_H-M    'P 1'")
            idx += 1
            continue
        if tag in ("_symmetry_int_tables_number", "_space_group_it_number"):
            out.append("_symmetry_Int_Tables_number       1")
            idx += 1
            continue
        if tag in _CIF_SG_DROP_TAGS:
            idx += 1
            continue

        out.append(line)
        idx += 1
    return "\n".join(out) + "\n"


def parse_cif_content(cif_content, primitive=False):
    """Parse CIF text, repairing files pymatgen rejects. Returns (structure, notes).

    'notes' is a list of short messages describing what had to be fixed; it is
    empty for a file that parses as written.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = _CifParser.from_str(cif_content).parse_structures(
                primitive=primitive)
        if structures:
            return structures[0], []
        first_error = "no structure could be read from the file"
    except Exception as exc:
        first_error = str(exc)

    if _cif_symmetry_is_redundant(cif_content):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = _CifParser.from_str(
                _cif_content_without_symmetry(cif_content)
            ).parse_structures(primitive=primitive)
        if structures:
            note = (
                f"all {len(structures[0])} atoms of the cell are listed "
                "together with the space-group operations, which duplicated "
                "them; the operations were ignored."
            )
            return structures[0], [note]

    raise ValueError(first_error)


def record_cif_repair(filename, notes):
    """Remember what had to be fixed in an uploaded CIF file."""
    if not notes:
        return
    name = os.path.basename(str(filename))
    try:
        st.session_state.setdefault(CIF_REPAIR_NOTES_KEY, {})[name] = list(notes)
    except Exception:
        pass


def render_cif_repair_notes(container=None):
    """Subtle note about CIF files that had to be repaired while loading."""
    store = st.session_state.get(CIF_REPAIR_NOTES_KEY) or {}
    if not store:
        return
    # Only files that are still loaded are worth commenting on.
    loaded = st.session_state.get("full_structures") or {}
    if loaded:
        store = {name: notes for name, notes in store.items()
                 if name in {os.path.basename(str(k)) for k in loaded}}
    target = st if container is None else container
    for name, notes in store.items():
        for note in notes:
            target.caption(f"ℹ️ **{name}**: {note}")


def load_structure(file_or_name):
    if isinstance(file_or_name, str):
        filename = file_or_name
    else:
        filename = file_or_name.name
        with open(filename, "wb") as f:
            f.write(file_or_name.getbuffer())
    if filename.lower().endswith(".cif"):
        with open(filename, "r", errors="ignore") as f:
            cif_content = f.read()
        mg_structure, cif_notes = parse_cif_content(cif_content)
        record_cif_repair(filename, cif_notes)
        try:
            sg_info = extract_cif_space_group(cif_content)
            if sg_info:
                mg_structure.properties["cif_space_group"] = sg_info
                st.session_state.setdefault("cif_space_groups", {})[filename] = sg_info
        except Exception:
            pass
    elif filename.lower().endswith(".data"):
        filename = filename.replace(".data", ".lmp")
        mg_structure = _load_lammps_data(filename)
    elif filename.lower().endswith(".lmp"):
        mg_structure = _load_lammps_data(filename)
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
    structure = parser.parse_structures(primitive=False)[0]
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


def convert_to_hill_notation(formula_input):
    import re
    formula_parts = formula_input.strip().split()
    elements_dict = {}

    for part in formula_parts:
        match = re.match(r'([A-Z][a-z]?)(\d*)', part)
        if match:
            element = match.group(1)
            count = match.group(2) if match.group(2) else ""
            elements_dict[element] = count

    hill_order = []
    if 'C' in elements_dict:
        if elements_dict['C']:
            hill_order.append(f"C{elements_dict['C']}")
        else:
            hill_order.append("C")
        del elements_dict['C']
    if 'H' in elements_dict:
        if elements_dict['H']:
            hill_order.append(f"H{elements_dict['H']}")
        else:
            hill_order.append("H")
        del elements_dict['H']

    for element in sorted(elements_dict.keys()):
        if elements_dict[element]:
            hill_order.append(f"{element}{elements_dict[element]}")
        else:
            hill_order.append(element)

    return " ".join(hill_order)

def sort_formula_alphabetically(formula_input):
    formula_parts = formula_input.strip().split()
    return " ".join(sorted(formula_parts))

MINERALS = {
    # Cubic structures
    225: {  # Fm-3m
        "Rock Salt (NaCl)": "Na Cl",
        "Fluorite (CaF2)": "Ca F2",
        "Anti-Fluorite (Li2O)": "Li2 O",
    },
    229: {  # Im-3m
        "BCC Iron": "Fe",
    },
    221: {  # Pm-3m
        "Perovskite (SrTiO3)": "Sr Ti O3",
        "ReO3 type": "Re O3",
        "Inverse-perovskite (Ca3TiN)": "Ca3 Ti N",
        "Cesium chloride (CsCl)": "Cs Cl"
    },
    227: {  # Fd-3m
        "Diamond": "C",

        "Normal spinel (MgAl2O4)": "Mg Al2 O4",
        "Inverse spinel (Fe3O4)": "Fe3 O4",
        "Pyrochlore (Ca2NbO7)": "Ca2 Nb2 O7",
        "β-Cristobalite (SiO2)": "Si O2"

    },
    216: {  # F-43m
        "Zinc Blende (ZnS)": "Zn S",
        "Half-anti-fluorite (Li4Ti)": "Li4 Ti"
    },
    215: {  # P-43m


    },
    230: {  # Ia-3d
        "Garnet (Ca3Al2Si3O12)": "Ca3 Al2 Si3 O12",
    },
    205: {  # Pa-3
        "Pyrite (FeS2)": "Fe S2",
    },
    224:{
        "Cuprite (Cu2O)": "Cu2 O",
    },
    # Hexagonal structures
    194: {  # P6_3/mmc
        "HCP Magnesium": "Mg",
        "Ni3Sn type": "Ni3 Sn",
        "Graphite": "C",
        "MoS2 type": "Mo S2",
        "Nickeline (NiAs)": "Ni As",
    },
    186: {  # P6_3mc
        "Wurtzite (ZnS)": "Zn S"
    },
    191: {  # P6/mmm


        "AlB2 type": "Al B2",
        "CaCu5 type": "Ca Cu5"
    },
    #187: {  # P-6m2
#
 #   },
    156: {
        "CdI2 type": "Cd I2",
    },
    164: {
    "CdI2 type": "Cd I2",
    },
    166: {  # R-3m
    "Delafossite (CuAlO2)": "Cu Al O2"
    },
    # Tetragonal structures
    139: {  # I4/mmm
        "β-Tin (Sn)": "Sn",
        "MoSi2 type": "Mo Si2"
    },
    136: {  # P4_2/mnm
        "Rutile (TiO2)": "Ti O2"
    },
    123: {  # P4/mmm
        "CuAu (L10)": "Cu Au"
    },
    141: {  # I41/amd
        "Anatase (TiO2)": "Ti O2",
        "Zircon (ZrSiO4)": "Zr Si O4"
    },
    122: {  # P-4m2
        "Chalcopyrite (CuFeS2)": "Cu Fe S2"
    },
    129: {  # P4/nmm
        "PbO structure": "Pb O"
    },

    # Orthorhombic structures
    62: {  # Pnma
        "Aragonite (CaCO3)": "Ca C O3",
        "Cotunnite (PbCl2)": "Pb Cl2",
        "Olivine (Mg2SiO4)": "Mg2 Si O4",
        "Barite (BaSO4)": "Ba S O4",
        "Perovskite (GdFeO3)": "Gd Fe O3"
    },
    63: {  # Cmcm
        "α-Uranium": "U",
        "CrB structure": "Cr B",
        "TlI structure": "Tl I",
    },
   # 74: {  # Imma
   #
   # },
    64: {  # Cmca
        "α-Gallium": "Ga"
    },

    # Monoclinic structures
    14: {  # P21/c
        "Baddeleyite (ZrO2)": "Zr O2",
        "Monazite (CePO4)": "Ce P O4"
    },
    206: {  # C2/m
        "Bixbyite (Mn2O3)": "Mn2 O3"
    },
    15: {  # C2/c
        "Gypsum (CaSO4·2H2O)": "Ca S H4 O6",
        "Scheelite (CaWO4)": "Ca W O4"
    },

    1: {
        "Kaolinite": "Al2 Si2 O9 H4"

    },
    # Triclinic structures
    2: {  # P-1
        "Wollastonite (CaSiO3)": "Ca Si O3",
        #"Kaolinite": "Al2 Si2 O5"
    },

    # Other important structures
    167: {  # R-3c
        "Calcite (CaCO3)": "Ca C O3",
        "Corundum (Al2O3)": "Al2 O3"
    },
    176: {  # P6_3/m
        "Apatite (Ca5(PO4)3OH)": "Ca5 P3 O13 H"
    },
    58: {  # Pnnm
        "Marcasite (FeS2)": "Fe S2"
    },
    198: {  # P213
        "FeSi structure": "Fe Si"
    },
    88: {  # I41/a
        "Scheelite (CaWO4)": "Ca W O4"
    },
    33: {  # Pna21
        "FeAs structure": "Fe As"
    },
    96: {  # P4/ncc
        "α-Cristobalite (SiO2)": "Si O2"
    },
    92: {
        "α-Cristobalite (SiO2)": "Si O2"
    },
    152: {  # P3121
        "Quartz (SiO2)": "Si O2"
    },
    148: {  # R-3
        "Ilmenite (FeTiO3)": "Fe Ti O3",
        "Dolomite (CaMgC2O6)": "Ca Mg C2 O6",
    },
    180: {  # P4_3 32
        "β-quartz (SiO2)": "Si O2"
    }
}

def show_xrdlicious_roadmap():
    st.markdown("""
### Roadmap

* The XRDlicious will be regularly updated. The planned features are listed below. If you spot a bug or have a feature idea, please let us know at: lebedmi2@cvut.cz and we will gladly consider it.
-------------------------------------------------------------------------------------------------------------------

#### Code optimization 
* ⏳ Optimizing the code for better performance. ⏳Separate critical parameters (such as wavelength, new file, debye-waller factors) for diffraction patterns complete recalculations from non-critical (such as intensity scale or x-axis units) ✅ Optimized search in COD database.

#### Wavelength Input: Energy Specification
* ⏳ Allow direct input of X-ray energy (keV) for synchrotron measurements, converting to wavelength automatically.

#### Improved Database Search
* ✅ Add search by keywords, space groups, ids... in database queries.

#### Adding More Databases
* ⏳Potentially add additional databases such as NOMAD, or implement the OPTIMADE aggregator 

#### Expanded Correction Factors & Peak Shapes
* ⏳ Add more peak shape functions (Lorentzian, Pseudo-Voigt).
* ⏳ Introduce preferred orientation and basic absorption corrections.
* ⏳ Instrumental broadening - introduce Caglioti formula.
* ⏳ Calculate and apply peak shifts due to sample displacement error.

#### Enhanced Structure Visualization 
* ✅ Allow to change structure visualization style between Plotly and Py3Dmol
* ✅ Added option to change between perspective and orthogonal view

#### Enhanced Background Subtraction (Experimental Data)
* ⏳ Improve tools for background removal on uploaded experimental patterns.

#### Enhanced XRD Data Conversion
* ⏳ More accessible conversion interface - not hidden within the interactive plot.
* ⏳ Batch operations on multiple files at once (e.g., FDS/VDS, wavelength).
* ✅ Add conversion from PANalytical .xrdml and Rigaku .ras diffraction pattern file format to .xy file format 

#### Basic Peak Fitting (Experimental Data)
* ⏳ Fitting: Advanced goal for fitting profiles or full patterns to refine parameters.

#### Machine Learning 
* ⏳ Outlook for ML models for structure-properties correlations
""")


DEFAULT_TWO_THETA_MAX_FOR_PRESET = {
    'Copper (CuKa1)': 120.0,
    'Cu(Ka1+Ka2)': 120.0,
    'CuKa2': 120.0,
    'Cu(Ka1+Ka2+Kb1)': 120.0,
    'CuKb1': 120.0,
    'Molybdenum (MoKa1)': 60.0,
    'Mo(Ka1+Ka2)': 60.0,
    'MoKa2': 60.0,
    'Mo(Ka1+Ka2+Kb1)': 60.0,
    'MoKb1': 70.0,
    'Cobalt (CoKa1)': 130.0,
    'Co(Ka1+Ka2)': 130.0,
    'CoKa2': 130.0,
    'Co(Ka1+Ka2+Kb1)': 130.0,
    'CoKb1': 130.0,
    'Chromium (CrKa1)': 150.0,
    'Cr(Ka1+Ka2)': 150.0,
    'CrKa2': 150.0,
    'Cr(Ka1+Ka2+Kb1)': 150.0,
    'CrKb1': 150.0,
    'Iron (FeKa1)': 140.0,
    'Fe(Ka1+Ka2)': 140.0,
    'FeKa2': 140.0,
    'Fe(Ka1+Ka2+Kb1)': 140.0,
    'FeKb1': 140.0,
    'Silver (AgKa1)': 50.0,
    'Ag(Ka1+Ka2)': 50.0,
    'AgKa2': 50.0,
    'Ag(Ka1+Ka2+Kb1)': 50.0,
    'AgKb1': 50.0,
}

DEFAULT_TWO_THETA_MAX_FOR_NEUTRON_PRESET = {
    'Thermal Neutrons': 150.0,
    'Cold Neutrons': 160.0,
    'Hot Neutrons': 120.0,
    'Custom': 165.0
}


def add_box(view, cell, color='black', linewidth=1.5):
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corner = i * cell[0] + j * cell[1] + k * cell[2]
                corners.append(corner)

    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        start = corners[edge[0]]
        end = corners[edge[1]]
        view.addLine({
            'start': {'x': start[0], 'y': start[1], 'z': start[2]},
            'end': {'x': end[0], 'y': end[1], 'z': end[2]},
            'color': color,
            'linewidth': linewidth
        })


import concurrent.futures
import requests
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure


def fetch_and_parse_cod_cif(entry):
    file_id = entry.get('file')
    if not file_id:
        return None, None, None, "Missing file ID in entry"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        cif_url = f"https://www.crystallography.net/cod/{file_id}.cif"
        response = requests.get(cif_url, timeout=15, headers=headers)
        response.raise_for_status()
        cif_content = response.text
        parser = CifParser.from_str(cif_content)

        structure = parser.parse_structures(primitive=False)[0]
        cod_id = f"cod_{file_id}"
        return cod_id, structure, entry, None

    except Exception as e:
        return None, None, None, str(e)

# ---------------------------------------------------------------------------
# Experimental pattern readers (PANalytical .xrdml / Rigaku .ras)
#
# These instrument formats are not two-column text, so every place that shows
# uploaded experimental data (powder diffraction, lattice fitting, interactive
# data plot) reads them through the helpers below instead of pd.read_csv.
# ---------------------------------------------------------------------------

INSTRUMENT_PATTERN_EXTENSIONS = ("xrdml", "xml", "ras", "rasx", "raw")


def _decode_pattern_bytes(raw):
    if not isinstance(raw, bytes):
        return str(raw)
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_xrdml_pattern(content):
    """(x, y) arrays from a PANalytical XRDML file, or None if not parsable."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(content)
    ns_uri = root.tag.split("}")[0][1:] if "}" in root.tag else ""
    ns = {"x": ns_uri} if ns_uri else {}
    prefix = "x:" if ns_uri else ""

    # Scans can sit at different depths depending on the XRDML version, so the
    # data points are searched for anywhere in the tree.
    for dp in root.findall(f".//{prefix}dataPoints", ns):
        ints_el = dp.find(f"{prefix}intensities", ns)
        if ints_el is None or not (ints_el.text or "").strip():
            # Some files store counts per second instead of raw counts.
            ints_el = dp.find(f"{prefix}counts", ns)
        if ints_el is None or not (ints_el.text or "").strip():
            continue
        intensities = np.array(ints_el.text.split(), dtype=float)

        positions = dp.findall(f"{prefix}positions", ns)
        # Prefer the scan axis, fall back to whatever axis is present.
        ordered = ([p for p in positions if p.attrib.get("axis") == "2Theta"]
                   + [p for p in positions if p.attrib.get("axis") != "2Theta"])
        for pos in ordered:
            lst = pos.find(f"{prefix}listPositions", ns)
            if lst is not None and (lst.text or "").strip():
                x = np.array(lst.text.split(), dtype=float)
                if len(x) == len(intensities):
                    return x, intensities
            start = pos.find(f"{prefix}startPosition", ns)
            end = pos.find(f"{prefix}endPosition", ns)
            if start is not None and end is not None:
                x = np.linspace(float(start.text), float(end.text),
                                len(intensities))
                return x, intensities
    return None


def parse_ras_pattern(content):
    """(x, y) arrays from a Rigaku RAS file, or None if no data block found."""
    angles, intensities = [], []
    in_data = False
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("*RAS_INT_START"):
            in_data = True
            continue
        if line.startswith("*RAS_INT_END"):
            break
        if not in_data or not line or line.startswith("*"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                angles.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError:
                continue
    if angles:
        return np.array(angles), np.array(intensities)
    return None


def _decode_rasx_text(raw_bytes):
    """RASX parts are UTF-8 with a BOM; older exports can be Shift-JIS."""
    for encoding in ("utf-8-sig", "shift_jis"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def parse_rasx_pattern(raw_bytes):
    """(x, y) from a Rigaku SmartLab .rasx archive, or None.

    A .rasx is a ZIP holding one Profile<N>.txt per scan with tab-separated
    angle / intensity / attenuator rows.
    """
    import zipfile

    with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
        names = zf.namelist()

        def pick(predicate):
            return next(iter(sorted(f for f in names
                                    if predicate(f.lower()))), None)

        profile_name = (
            pick(lambda n: "profile" in n and n.endswith(".txt"))
            or pick(lambda n: n.endswith(".asc"))
            or pick(lambda n: n.endswith(".txt"))
        )
        if not profile_name:
            return None
        profile_text = _decode_rasx_text(zf.read(profile_name))

    angles, intensities = [], []
    for line in profile_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            angle = float(parts[0])
            intensity = float(parts[1])
        except ValueError:
            continue
        # The third column is the attenuator factor the intensity was measured
        # through; the true count rate is the product of the two.
        if len(parts) >= 3:
            try:
                intensity *= float(parts[2])
            except ValueError:
                pass
        angles.append(angle)
        intensities.append(intensity)

    if angles:
        return np.array(angles), np.array(intensities)
    return None


def _parse_raw_v1(data):
    """Bruker RAW1.01 (DIFFRACplus). Only the first range is read."""
    import struct

    file_size = len(data)
    file_header_size = 712

    range_cnt = struct.unpack_from("<i", data, 12)[0]
    if range_cnt < 1 or range_cnt > 1000:
        range_cnt = 1

    cur = file_header_size
    header_len = struct.unpack_from("<i", data, cur)[0]
    if header_len < 304 or cur + header_len > file_size:
        header_len = 304

    num_points = struct.unpack_from("<i", data, cur + 4)[0]
    start_2theta = struct.unpack_from("<d", data, cur + 16)[0]
    step_size = struct.unpack_from("<d", data, cur + 176)[0]

    data_offset = cur + header_len
    available = (file_size - data_offset) // 4
    if num_points <= 0 or num_points > available:
        num_points = available
    if num_points <= 0:
        return None
    # A single range whose data does not reach the end of the file: the
    # intensity block is the trailing num_points float32 values.
    if range_cnt == 1 and (file_size - data_offset) != num_points * 4:
        data_offset = file_size - num_points * 4

    if not (np.isfinite(step_size) and 0 < abs(step_size) < 50):
        step_size = 0.0
    if not np.isfinite(start_2theta):
        start_2theta = 0.0

    intensities = np.frombuffer(data, dtype=np.float32, count=num_points,
                                offset=data_offset).astype(float)
    angles = np.arange(num_points) * step_size + start_2theta
    return angles, intensities


def _parse_raw4(data):
    """Bruker RAW4.00: the scan block is located by its signature."""
    import struct

    fs = len(data)
    for off in range(8, fs - 20):
        try:
            start = struct.unpack_from("<d", data, off)[0]
            step = struct.unpack_from("<d", data, off + 8)[0]
            n = struct.unpack_from("<i", data, off + 16)[0]
        except struct.error:
            continue
        if not (-180.0 <= start <= 180.0):
            continue
        if not (1e-6 < step < 5.0):
            continue
        if not (2 <= n <= 50_000_000):
            continue
        if n * 4 > fs - 20:
            continue
        # The intensity block (float32) runs to the end of the file.
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=fs - n * 4)
        if not np.all(np.isfinite(arr)):
            continue
        if arr.min() < 0 or arr.max() <= 0:
            continue
        return np.arange(n) * step + start, arr.astype(float)
    return None


def _parse_raw_v2v3(data):
    """Older Bruker RAW (v2/v3): fixed header offsets, data at 2600."""
    import struct

    file_size = len(data)
    data_offset = 2600

    start_angle = struct.unpack_from("<f", data, 136)[0]
    step_size = struct.unpack_from("<f", data, 140)[0]
    try:
        num_points = struct.unpack_from("<i", data, 148)[0]
    except struct.error:
        num_points = 0

    if not num_points or num_points <= 0:
        num_points = (file_size - data_offset) // 4
    if num_points <= 0:
        return None
    if not (np.isfinite(start_angle) and np.isfinite(step_size)):
        return None

    intensities = np.frombuffer(data, dtype=np.float32, count=num_points,
                                offset=data_offset).astype(float)
    angles = np.arange(num_points) * float(step_size) + float(start_angle)
    return angles, intensities


def parse_raw_pattern(raw_bytes):
    """(x, y) from a Bruker .raw file of any of the three known versions."""
    if raw_bytes.startswith(b"RAW1.01"):
        return _parse_raw_v1(raw_bytes)
    if raw_bytes.startswith(b"RAW4"):
        return _parse_raw4(raw_bytes)
    return _parse_raw_v2v3(raw_bytes)


def parse_instrument_pattern(file_obj, warn=False):
    """(x, y) from an .xrdml/.ras/.rasx/.raw upload; None for other formats."""
    name = getattr(file_obj, "name", "")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext not in INSTRUMENT_PATTERN_EXTENSIONS:
        return None

    try:
        file_obj.seek(0)
        raw = file_obj.read()
    finally:
        try:
            file_obj.seek(0)
        except Exception:
            pass
    if not isinstance(raw, bytes):
        raw = str(raw).encode("utf-8", errors="replace")

    try:
        if ext == "rasx":
            result = parse_rasx_pattern(raw)
        elif ext == "raw":
            result = parse_raw_pattern(raw)
        elif ext in ("xrdml", "xml"):
            result = parse_xrdml_pattern(_decode_pattern_bytes(raw))
        else:
            result = parse_ras_pattern(_decode_pattern_bytes(raw))
    except Exception as exc:
        if warn:
            st.warning(f"Could not parse '{name}' as {ext.upper()}: {exc}")
        return None

    if result is None and warn:
        st.warning(f"No diffraction data found in '{name}'.")
    return result


def read_pattern_dataframe(file_obj, has_header=False, skip_header=True):
    """Two-column DataFrame from any supported experimental pattern file.

    Instrument formats (.xrdml/.ras) are decoded with their own parsers; all
    other files keep the previous plain-text behaviour (optional header row,
    comment lines skipped).
    """
    parsed = parse_instrument_pattern(file_obj)
    if parsed is not None:
        x, y = parsed
        return pd.DataFrame({"2Theta": x, "Intensity": y})

    file_obj.seek(0)
    if has_header:
        return pd.read_csv(file_obj, sep=r'\s+|,|;', engine='python', header=0)
    if skip_header:
        content = _decode_pattern_bytes(file_obj.read())
        lines = content.splitlines()
        comment_line_indices = [idx for idx, line in enumerate(lines)
                                if line.strip().startswith('#')]
        lines_to_skip = sorted(set([0] + comment_line_indices))
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=r'\s+|,|;', engine='python',
                           header=None, skiprows=lines_to_skip)
    return pd.read_csv(file_obj, sep=r'\s+|,|;', engine='python', header=None)


def auto_normalize_and_stack_plots(files, skip_header, has_header, offset_gap):
    if not files:
        st.warning("Please upload files before trying to normalize and stack.")
        return

    st.session_state.auto_stack_enabled = True
    st.session_state.normalized_intensity = True

    min_adjustments = []
    y_offset_accumulator = 0

    total_shift = 100 + offset_gap

    for i, file in enumerate(files):
        try:
            df = read_pattern_dataframe(file, has_header=has_header,
                                        skip_header=skip_header)

            y_data = df.iloc[:, 1].values
            min_adjustments.append(np.min(y_data))

            st.session_state[f'y_offset_{i}'] = y_offset_accumulator
            y_offset_accumulator += total_shift

        except Exception as e:
            st.error(f"Failed to process {file.name} for auto-stacking: {e}")
            min_adjustments.append(0)

    st.session_state.min_adjustments = min_adjustments


def reset_layout(files):
    st.session_state.auto_stack_enabled = False
    st.session_state.normalized_intensity = False
    if files:
        for i in range(len(files)):
            st.session_state[f'y_offset_{i}'] = 0.0
            st.session_state[f'y_scale_{i}'] = 1.0
            
def extract_space_group_number(selected_option):
    if selected_option:
        return int(selected_option.split(' ')[0])
    return None
