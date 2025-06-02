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

def check_structure_size_and_warn(structure, structure_name="structure"):
    n_atoms = len(structure)

    if n_atoms > 75:
        st.info(f"ℹ️ **Structure Notice**: {structure_name} contains a large number of **{n_atoms} atoms**. "
                f"Calculations may take longer depending on selected parameters. Please be careful to "
                f"not consume much memory, we are hosted on a free server. 😊")
        return "moderate"
    else:
        return "small"


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
       # print("------")
        print(formula)
       # print(formula_type)
        #print(spg_number)
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

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

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
-------------------------------------------------------------------------------------------------------------------
#### Code optimization 

#### Wavelength Input: Energy Specification
* ⏳ Allow direct input of X-ray energy (keV) for synchrotron measurements, converting to wavelength automatically.

#### Improved Database Search
* ✅ Add search by keywords, space groups, ids... in database queries.

#### Expanded Correction Factors & Peak Shapes
* ⏳ Add more peak shape functions (Lorentzian, Pseudo-Voigt).
* ⏳ Introduce preferred orientation and basic absorption corrections.
* ⏳ Instrumental broadening - introduce Caglioti formula.
* ⏳ Calculate and apply peak shifts due to sample displacement error.

#### Enhanced Background Subtraction (Experimental Data)
* ⏳ Improve tools for background removal on uploaded experimental patterns.

#### Enhanced XRD Data Conversion
* ⏳ More accessible conversion interface - not hidden within the interactive plot.
* ⏳ Batch operations on multiple files at once (e.g., FDS/VDS, wavelength).

#### Basic Peak Fitting (Experimental Data)
* ⏳ Fitting: Advanced goal for fitting profiles or full patterns to refine parameters.
""")
