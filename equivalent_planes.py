import streamlit as st
from pymatgen.symmetry.groups import SpaceGroup
import math
from typing import Union

SPACE_GROUP_OPTIONS = [
        '1 (P1)', '2 (P-1)', '3 (P2)', '4 (P2_1)', '5 (C2)', '6 (Pm)', '7 (Pc)',
        '8 (Cm)', '9 (Cc)', '10 (P2/m)', '11 (P2_1/m)', '12 (C2/m)', '13 (P2/c)',
        '14 (P2_1/c)', '15 (C2/c)', '16 (P222)', '17 (P222_1)', '18 (P2_12_12)',
        '19 (P2_12_12_1)', '20 (C222_1)', '21 (C222)', '22 (F222)', '23 (I222)',
        '24 (I2_12_12_1)', '25 (Pmm2)', '26 (Pmc2_1)', '27 (Pcc2)', '28 (Pma2)',
        '29 (Pca2_1)', '30 (Pnc2)', '31 (Pmn2_1)', '32 (Pba2)', '33 (Pna2_1)',
        '34 (Pnn2)', '35 (Cmm2)', '36 (Cmc2_1)', '37 (Ccc2)', '38 (Amm2)',
        '39 (Aem2)', '40 (Ama2)', '41 (Aea2)', '42 (Fmm2)', '43 (Fdd2)',
        '44 (Imm2)', '45 (Iba2)', '46 (Ima2)', '47 (Pmmm)', '48 (Pnnn)',
        '49 (Pccm)', '50 (Pban)', '51 (Pmma)', '52 (Pnna)', '53 (Pmna)',
        '54 (Pcca)', '55 (Pbam)', '56 (Pccn)', '57 (Pbcm)', '58 (Pnnm)',
        '59 (Pmmn)', '60 (Pbcn)', '61 (Pbca)', '62 (Pnma)', '63 (Cmcm)',
        '64 (Cmce)', '65 (Cmmm)', '66 (Cccm)', '67 (Cmme)', '68 (Ccce)',
        '69 (Fmmm)', '70 (Fddd)', '71 (Immm)', '72 (Ibam)', '73 (Ibca)',
        '74 (Imma)', '75 (P4)', '76 (P4_1)', '77 (P4_2)', '78 (P4_3)', '79 (I4)',
        '80 (I4_1)', '81 (P-4)', '82 (I-4)', '83 (P4/m)', '84 (P4_2/m)',
        '85 (P4/n)', '86 (P4_2/n)', '87 (I4/m)', '88 (I4_1/a)', '89 (P422)',
        '90 (P42_12)', '91 (P4_122)', '92 (P4_12_12)', '93 (P4_222)',
        '94 (P4_22_12)', '95 (P4_322)', '96 (P4_32_12)', '97 (I422)',
        '98 (I4_122)', '99 (P4mm)', '100 (P4bm)', '101 (P4_2cm)', '102 (P4_2nm)',
        '103 (P4cc)', '104 (P4nc)', '105 (P4_2mc)', '106 (P4_2bc)', '107 (I4mm)',
        '108 (I4cm)', '109 (I4_1md)', '110 (I4_1cd)', '111 (P-42m)', '112 (P-42c)',
        '113 (P-42_1m)', '114 (P-42_1c)', '115 (P-4m2)', '116 (P-4c2)',
        '117 (P-4b2)', '118 (P-4n2)', '119 (I-4m2)', '120 (I-4c2)', '121 (I-42m)',
        '122 (I-42d)', '123 (P4/mmm)', '124 (P4/mcc)', '125 (P4/nbm)', '126 (P4/nnc)',
        '127 (P4/mbm)', '128 (P4/mnc)', '129 (P4/nmm)', '130 (P4/ncc)',
        '131 (P4_2/mmc)', '132 (P4_2/mcm)', '133 (P4_2/nbc)', '134 (P4_2/nnm)',
        '135 (P4_2/mbc)', '136 (P4_2/mnm)', '137 (P4_2/nmc)', '138 (P4_2/ncm)',
        '139 (I4/mmm)', '140 (I4/mcm)', '141 (I4_1/amd)', '142 (I4_1/acd)',
        '143 (P3)', '144 (P3_1)', '145 (P3_2)', '146 (R3)', '147 (P-3)',
        '148 (R-3)', '149 (P312)', '150 (P321)', '151 (P3_112)', '152 (P3_121)',
        '153 (P3_212)', '154 (P3_221)', '155 (R32)', '156 (P3m1)', '157 (P31m)',
        '158 (P3c1)', '159 (P31c)', '160 (R3m)', '161 (R3c)', '162 (P-31m)',
        '163 (P-31c)', '164 (P-3m1)', '165 (P-3c1)', '166 (R-3m)', '167 (R-3c)',
        '168 (P6)', '169 (P6_1)', '170 (P6_5)', '171 (P6_2)', '172 (P6_4)',
        '173 (P6_3)', '174 (P-6)', '175 (P6/m)', '176 (P6_3/m)', '177 (P622)',
        '178 (P6_122)', '179 (P6_522)', '180 (P6_222)', '181 (P6_422)', '182 (P6_322)',
        '183 (P6mm)', '184 (P6cc)', '185 (P6_3cm)', '186 (P6_3mc)', '187 (P-6m2)',
        '188 (P-6c2)', '189 (P-62m)', '190 (P-62c)', '191 (P6/mmm)', '192 (P6/mcc)',
        '193 (P6_3/mcm)', '194 (P6_3/mmc)', '195 (P23)', '196 (F23)', '197 (I23)',
        '198 (P2_13)', '199 (I2_13)', '200 (Pm-3)', '201 (Pn-3)', '202 (Fm-3)',
        '203 (Fd-3)', '204 (Im-3)', '205 (Pa-3)', '206 (Ia-3)', '207 (P432)',
        '208 (P4_232)', '209 (F432)', '210 (F4_132)', '211 (I432)', '212 (P4_332)',
        '213 (P4_132)', '214 (I4_132)', '215 (P-43m)', '216 (F-43m)', '217 (I-43m)',
        '218 (P-43n)', '219 (F-43c)', '220 (I-43d)', '221 (Pm-3m)', '222 (Pn-3n)',
        '223 (Pm-3n)', '224 (Pn-3m)', '225 (Fm-3m)', '226 (Fm-3c)', '227 (Fd-3m)',
        '228 (Fd-3c)', '229 (Im-3m)', '230 (Ia-3d)'
    ]
def run_equivalent_hkl_app():
    def get_equivalent_hkl(space_group_symbol: str, h: int, k: int, l: int):
        try:
            sg = SpaceGroup(space_group_symbol)
            
            equivalent_planes = set()
            

            for op in sg.symmetry_ops:
                rotation_matrix = op.rotation_matrix
                
                new_hkl = rotation_matrix.dot([h, k, l])
                h_new, k_new, l_new = [int(round(x)) for x in new_hkl]
                
                gcd_val = math.gcd(math.gcd(abs(h_new), abs(k_new)), abs(l_new))
                if gcd_val > 0:
                    h_reduced = h_new // gcd_val
                    k_reduced = k_new // gcd_val
                    l_reduced = l_new // gcd_val
                    

                    equivalent_planes.add((h_reduced, k_reduced, l_reduced))
            
            unique_families = set()
            for plane in equivalent_planes:
                h_pl, k_pl, l_pl = plane
                neg_plane = (-h_pl, -k_pl, -l_pl)
                canonical = min(plane, neg_plane)
                unique_families.add(canonical)

            return sorted(list(unique_families)), None
        except (ValueError, KeyError):
            return [], f"Error: Invalid space group symbol or number '{space_group_symbol}'. Please check the input and try again."
        except Exception as e:
            return [], f"An unexpected error occurred: {e}"


    st.header("↔️ Equivalent {hkl} Planes Calculator")
    st.write("Enter a space group (number or Hermann-Mauguin (international) symbol) and the Miller indices (h k l) to find all symmetrically equivalent planes (without multiples).")

    col1, col2 = st.columns(2)

    with col1:
        selected_option = st.selectbox(
            "Space Group",
            options=SPACE_GROUP_OPTIONS,
            index=220,
            help="Start typing to search by number or symbol.",
            key="hkl_space_group"
        )

    with col2:
        hkl_input = st.text_input("Miller Indices (h k l)", "1 1 1", help="Enter as space-separated integers, e.g., 1 0 0", key="hkl_indices")

    if st.button("Calculate Equivalent Planes", key="hkl_calculate_button"):
        try:
            h, k, l = map(int, hkl_input.split())
            if selected_option:
                space_group_symbol = selected_option.split('(')[1][:-1]
            else:
                st.error("Please select a space group.")
                return

            with st.spinner("Calculating..."):
                equivalent_hkls, error_message = get_equivalent_hkl(space_group_symbol, h, k, l)

                if error_message:
                    st.error(error_message)
                elif equivalent_hkls:
                    st.success(f"Found **{len(equivalent_hkls)}** unique (without multiples) equivalent planes for **{{{h} {k} {l}}}** in space group **{selected_option}**.")

                    num_columns = 4
                    cols = st.columns(num_columns)
                    #for i, plane in enumerate(equivalent_hkls):
                    #    col_index = i % num_columns
                    #    formatted_plane_latex = f"({ '\\ '.join(map(str, plane)) })"
                    #    cols[col_index].markdown(f"$$ {formatted_plane_latex} $$")
                    
                    for i, plane in enumerate(equivalent_hkls):
                        col_index = i % num_columns
                        def to_latex_overbar(n):
                            if n < 0:
                                return f"\\bar{{{abs(n)}}}"
                            return str(n)
                        latex_hkl = ' '.join(map(to_latex_overbar, plane))
                        formatted_plane_latex = f"({latex_hkl})"
                        cols[col_index].markdown(f"$$ {formatted_plane_latex} $$")
                else:
                    st.warning("No equivalent planes were found. This may be expected for certain special planes and space groups.")

        except ValueError:
            st.error("Invalid input for Miller indices or Space Group. Please check your inputs (e.g., '1 1 1' and '225 (Fm-3m)').")
        except Exception as e:
            st.error(f"An application error occurred: {e}")
