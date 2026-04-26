import io
import numpy as np
import pandas as pd
import streamlit as st
import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure, Lattice as PmgLattice
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from io import StringIO
from ase.io import write
from ase.constraints import FixAtoms

from helpers import load_structure, identify_structure_type, jmol_colors, add_box

ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
    "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
    "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
    "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]


def _vesta_lattice(lattice):
    import math
    a, b, c = lattice.abc
    al, be, ga = [math.radians(x) for x in lattice.angles]
    a_vec = [a, 0.0, 0.0]
    b_vec = [b * math.cos(ga), b * math.sin(ga), 0.0]
    c_x = c * math.cos(be)
    c_y = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    c_z = math.sqrt(max(0.0, c**2 - c_x**2 - c_y**2))
    return PmgLattice(np.array([a_vec, b_vec, [c_x, c_y, c_z]]))


def _draw_supercell_boxes(view, lattice_matrix, sx, sy, sz):
    a, b, c = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]
    edge_pairs = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                tr = i*a + j*b + k*c
                corners = [tr + di*a + dj*b + dk*c for di in [0,1] for dj in [0,1] for dk in [0,1]]
                color = "black" if (i == 0 and j == 0 and k == 0) else "gray"
                for e0, e1 in edge_pairs:
                    s, en = corners[e0], corners[e1]
                    view.addLine({
                        "start": {"x": float(s[0]),  "y": float(s[1]),  "z": float(s[2])},
                        "end":   {"x": float(en[0]), "y": float(en[1]), "z": float(en[2])},
                        "color": color, "linewidth": 1.5,
                    })


def _d_spacing(lattice_matrix, h, k, l):
    recip = 2 * np.pi * np.linalg.inv(lattice_matrix).T
    n = h*recip[0] + k*recip[1] + l*recip[2]
    return 2 * np.pi / np.linalg.norm(n)


def _compute_view_and_up_dirs(lattice_matrix, mode, uvw, hkl):
    lm = lattice_matrix
    a, b, c = lm[0], lm[1], lm[2]
    recip = 2 * np.pi * np.linalg.inv(lm).T
    a_star, b_star, c_star = recip[0], recip[1], recip[2]
    u, v, w = uvw
    h, k, l = hkl
    if h == 0 and k == 0 and l == 0:
        raise ValueError("(h k l) = (0 0 0) is invalid.")
    if u == 0 and v == 0 and w == 0:
        raise ValueError("[u v w] = [0 0 0] is invalid as an upward direction.")
    view_dir = float(h)*a_star + float(k)*b_star + float(l)*c_star
    up_dir   = float(u)*a      + float(v)*b      + float(w)*c
    return np.array(view_dir, dtype=float), np.array(up_dir, dtype=float)


def _rotate_vector_about_axis(vector, axis, angle_deg):
    v  = np.asarray(vector, dtype=float)
    ax = np.asarray(axis,   dtype=float)
    n  = np.linalg.norm(ax)
    if n < 1e-12:
        return v
    ax = ax / n
    th = np.radians(angle_deg)
    return (
        v * np.cos(th)
        + np.cross(ax, v) * np.sin(th)
        + ax * np.dot(ax, v) * (1.0 - np.cos(th))
    )


def _compute_orientation_matrix(view_dir_cart, up_dir_cart):
    vd = np.asarray(view_dir_cart, dtype=float)
    vd /= np.linalg.norm(vd)

    ud_raw = np.asarray(up_dir_cart, dtype=float)
    ud_raw = ud_raw - np.dot(ud_raw, vd) * vd
    if np.linalg.norm(ud_raw) < 1e-10:
        for cand in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]:
            ud_raw = np.array(cand, float) - np.dot(cand, vd) * vd
            if np.linalg.norm(ud_raw) > 1e-10:
                break
    ud = ud_raw / np.linalg.norm(ud_raw)
    xd = np.cross(ud, vd)
    xd /= np.linalg.norm(xd)

    return np.array([xd, ud, vd])


def _orientation_controls(key_suffix, lattice_matrix=None):
    enable = st.checkbox(
        "🔄 Set crystallographic orientation",
        value=False,
        key=f"orient_enable_{key_suffix}",
    )
    if not enable:
        return None

    st.session_state.setdefault(f"orient_u_{key_suffix}", 0)
    st.session_state.setdefault(f"orient_v_{key_suffix}", 0)
    st.session_state.setdefault(f"orient_w_{key_suffix}", 1)
    st.session_state.setdefault(f"orient_h_{key_suffix}", 0)
    st.session_state.setdefault(f"orient_k_{key_suffix}", 0)
    st.session_state.setdefault(f"orient_l_{key_suffix}", 1)
    st.session_state.setdefault(f"orient_roll_deg_{key_suffix}", 0.0)

    PRESETS = [
        ("a* (100)", [0, 0, 1], [1, 0, 0]),
        ("b* (010)", [1, 0, 0], [0, 1, 0]),
        ("c* (001)", [0, 1, 0], [0, 0, 1]),
    ]
    st.caption("Quick presets:")
    preset_cols = st.columns(len(PRESETS))
    _preset_fired = False
    _preset_uvw, _preset_hkl = None, None

    for col, (label, p_uvw, p_hkl) in zip(preset_cols, PRESETS):
        if col.button(label, key=f"orient_preset_{label}_{key_suffix}", use_container_width=True):
            st.session_state[f"orient_u_{key_suffix}"] = p_uvw[0]
            st.session_state[f"orient_v_{key_suffix}"] = p_uvw[1]
            st.session_state[f"orient_w_{key_suffix}"] = p_uvw[2]
            st.session_state[f"orient_h_{key_suffix}"] = p_hkl[0]
            st.session_state[f"orient_k_{key_suffix}"] = p_hkl[1]
            st.session_state[f"orient_l_{key_suffix}"] = p_hkl[2]
            st.session_state[f"orient_roll_deg_{key_suffix}"] = 0.0
            st.session_state["se_stored_orient"] = {
                "active": True, "mode": "hkl",
                "uvw": p_uvw, "hkl": p_hkl, "roll_deg": 0.0,
            }
            _preset_fired = True
            _preset_uvw, _preset_hkl = p_uvw, p_hkl

    st.markdown(
        "<div style='font-weight:600;font-size:0.90rem;margin-top:8px;'>"
        "Project along the normal to (h k l)</div>",
        unsafe_allow_html=True,
    )

    col_proj, col_up = st.columns(2)
    with col_proj:
        st.markdown(
            "<div style='font-weight:600;font-size:0.85rem;margin-bottom:4px;'>"
            "Projection plane normal (h k l)</div>",
            unsafe_allow_html=True,
        )
        h = int(st.number_input("h", step=1, format="%d", key=f"orient_h_{key_suffix}"))
        k = int(st.number_input("k", step=1, format="%d", key=f"orient_k_{key_suffix}"))
        l = int(st.number_input("l", step=1, format="%d", key=f"orient_l_{key_suffix}"))
    with col_up:
        st.markdown(
            "<div style='font-weight:600;font-size:0.85rem;margin-bottom:4px;'>"
            "Upward direction [u v w]</div>",
            unsafe_allow_html=True,
        )
        u = int(st.number_input("u", step=1, format="%d", key=f"orient_u_{key_suffix}"))
        v = int(st.number_input("v", step=1, format="%d", key=f"orient_v_{key_suffix}"))
        w = int(st.number_input("w", step=1, format="%d", key=f"orient_w_{key_suffix}"))

    uvw = [u, v, w] if not _preset_fired else _preset_uvw
    hkl = [h, k, l] if not _preset_fired else _preset_hkl
    mode = "hkl"

    if all(x == 0 for x in hkl):
        st.warning("⚠️ (h k l) = (0 0 0) is not a valid projection plane.")
        return None
    if all(x == 0 for x in uvw):
        st.warning("⚠️ [u v w] = [0 0 0] is not a valid upward direction.")
        return None

    roll_key = f"orient_roll_deg_{key_suffix}"

    if lattice_matrix is not None:
        try:
            dot_cond = hkl[0]*uvw[0] + hkl[1]*uvw[1] + hkl[2]*uvw[2]
            if dot_cond != 0:
                st.caption(
                    f"⚠️ hu+kv+lw = {dot_cond} ≠ 0: upward direction is not strictly "
                    f"in the projection plane. It will be projected onto the screen plane, "
                    f"as in VESTA."
                )
            view_dir, up_dir = _compute_view_and_up_dirs(lattice_matrix, mode, uvw, hkl)
            M = _compute_orientation_matrix(view_dir, up_dir)
            st.markdown(
                "<div style='font-weight:600;font-size:0.85rem;margin-top:8px;'>"
                "Orientation matrix</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.80rem;background:#f4f4f4;"
                f"padding:8px 10px;border-radius:6px;line-height:1.7;'>"
                f"{M[0,0]:+.6f}  {M[0,1]:+.6f}  {M[0,2]:+.6f}<br>"
                f"{M[1,0]:+.6f}  {M[1,1]:+.6f}  {M[1,2]:+.6f}<br>"
                f"{M[2,0]:+.6f}  {M[2,1]:+.6f}  {M[2,2]:+.6f}"
                f"</div>",
                unsafe_allow_html=True,
            )
            try:
                d = _d_spacing(lattice_matrix, hkl[0], hkl[1], hkl[2])
                st.caption(f"d-spacing ({hkl[0]} {hkl[1]} {hkl[2]}): {d:.4f} Å")
            except Exception:
                pass
            roll_val = float(st.session_state.get(roll_key, 0.0))
            if abs(roll_val) > 1e-9:
                st.caption(f"Roll around projection axis: {roll_val:.1f}°")
        except Exception as e:
            st.caption(f"⚠️ Could not calculate orientation: {e}")

    apply = st.button(
        "Apply orientation",
        key=f"orient_apply_{key_suffix}",
        type="primary",
        use_container_width=True,
    )
    st.caption("Projection normal points out of the screen · [u v w] is the upward direction.")

    cx, cy, cz = st.columns(3)
    sx = int(cx.number_input("Repeat X", min_value=1, max_value=5, value=1, step=1, key=f"orient_sx_{key_suffix}"))
    sy = int(cy.number_input("Repeat Y", min_value=1, max_value=5, value=1, step=1, key=f"orient_sy_{key_suffix}"))
    sz = int(cz.number_input("Repeat Z", min_value=1, max_value=5, value=1, step=1, key=f"orient_sz_{key_suffix}"))

    return mode, uvw, hkl, (apply or _preset_fired), sx, sy, sz, float(st.session_state.get(roll_key, 0.0))


def _make_supercell_xyz(atoms, structure, sx, sy, sz, norm_lat=None):
    lat = norm_lat if norm_lat is not None else structure.lattice
    lm = lat.matrix
    rows = []
    for ai in range(sx):
        for bi in range(sy):
            for ci in range(sz):
                tr = ai*lm[0] + bi*lm[1] + ci*lm[2]
                for atom in atoms:
                    cart = lat.get_cartesian_coords([atom["x"], atom["y"], atom["z"]]) + tr
                    rows.append(f"{atom['element']} {cart[0]:.6f} {cart[1]:.6f} {cart[2]:.6f}")
    return "\n".join([str(len(rows)), f"Supercell {sx}x{sy}x{sz}"] + rows)


def _get_wyckoffs(structure):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return sga.get_symmetry_dataset().wyckoffs
    except Exception:
        return ["-"] * len(structure.sites)


def _load_atoms_from_structure(structure):
    wyckoffs = _get_wyckoffs(structure)
    atoms = []
    for i, site in enumerate(structure.sites):
        frac = site.frac_coords
        for sp, occ in site.species.items():
            atoms.append({
                "element": sp.symbol,
                "occ":      round(float(occ), 4),
                "x":        round(float(frac[0]), 5),
                "y":        round(float(frac[1]), 5),
                "z":        round(float(frac[2]), 5),
                "wyckoff":  wyckoffs[i],
                "site_idx": i,
            })
    return atoms


def _rebuild_structure_from_atoms(atoms, lattice):
    site_groups = {}
    for atom in atoms:
        key = (round(atom["x"], 5), round(atom["y"], 5), round(atom["z"], 5))
        site_groups.setdefault(key, []).append(atom)
    new_struct = Structure(lattice, [], [])
    for (x, y, z), group in site_groups.items():
        species_dict = {}
        for atom in group:
            el = atom["element"]
            species_dict[el] = species_dict.get(el, 0.0) + atom["occ"]
        new_struct.append(
            species=species_dict,
            coords=(x, y, z),
            coords_are_cartesian=False,
            properties={"wyckoff": group[0].get("wyckoff", "-")},
        )
    return new_struct


def _get_unique_wyckoff_atoms(atoms):
    seen = {}
    unique = []
    for atom in atoms:
        key = (atom["wyckoff"], atom["element"])
        if key not in seen:
            seen[key] = {"atom": atom, "count": 1}
        else:
            seen[key]["count"] += 1
    for val in seen.values():
        entry = dict(val["atom"])
        entry["wyckoff_count"] = val["count"]
        unique.append(entry)
    return unique


def _propagate_wyckoff_change(atoms, original_wyckoff, original_element, changed_fields):
    result = []
    for atom in atoms:
        a = dict(atom)
        if a["wyckoff"] == original_wyckoff and a["element"] == original_element:
            a.update(changed_fields)
        result.append(a)
    return result


def _get_ordered_structure(export_struct):
    return Structure(
        export_struct.lattice,
        [max(site.species.items(), key=lambda x: x[1])[0] for site in export_struct],
        [site.frac_coords for site in export_struct],
    )



_COVALENT_RADII = {
    "H":0.31,"He":0.28,"Li":1.28,"Be":0.96,"B":0.84,"C":0.76,"N":0.71,"O":0.66,
    "F":0.57,"Ne":0.58,"Na":1.66,"Mg":1.41,"Al":1.21,"Si":1.11,"P":1.07,"S":1.05,
    "Cl":1.02,"Ar":1.06,"K":2.03,"Ca":1.76,"Sc":1.70,"Ti":1.60,"V":1.53,"Cr":1.39,
    "Mn":1.61,"Fe":1.52,"Co":1.50,"Ni":1.24,"Cu":1.32,"Zn":1.22,"Ga":1.22,"Ge":1.20,
    "As":1.19,"Se":1.20,"Br":1.20,"Kr":1.16,"Rb":2.20,"Sr":1.95,"Y":1.90,"Zr":1.75,
    "Nb":1.64,"Mo":1.54,"Tc":1.47,"Ru":1.46,"Rh":1.42,"Pd":1.39,"Ag":1.45,"Cd":1.44,
    "In":1.42,"Sn":1.39,"Sb":1.39,"Te":1.38,"I":1.39,"Xe":1.40,"Cs":2.44,"Ba":2.15,
    "La":2.07,"Ce":2.04,"Pr":2.03,"Nd":2.01,"Pm":1.99,"Sm":1.98,"Eu":1.98,"Gd":1.96,
    "Tb":1.94,"Dy":1.92,"Ho":1.92,"Er":1.89,"Tm":1.90,"Yb":1.87,"Lu":1.87,"Hf":1.75,
    "Ta":1.70,"W":1.62,"Re":1.51,"Os":1.44,"Ir":1.41,"Pt":1.36,"Au":1.36,"Hg":1.32,
    "Tl":1.45,"Pb":1.46,"Bi":1.48,"Po":1.40,"At":1.50,"Rn":1.50,
}


def _compute_bonds_cartesian(cart_positions, elements, lattice_matrix, tolerance=1.15, include_pbc=True):
    n = len(cart_positions)
    if n == 0:
        return [], []
    carts = np.array(cart_positions, dtype=float)

    image_offsets = [(0, 0, 0)]
    if include_pbc:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if (di, dj, dk) != (0, 0, 0):
                        image_offsets.append((di, dj, dk))

    bonds = []
    image_atoms_dict = {}
    for i in range(n):
        ri = _COVALENT_RADII.get(elements[i], 1.5)
        for j in range(n):
            rj = _COVALENT_RADII.get(elements[j], 1.5)
            r_max = (ri + rj) * tolerance
            for di, dj, dk in image_offsets:
                if di == 0 and dj == 0 and dk == 0 and j <= i:
                    continue
                offset = (di * lattice_matrix[0] +
                          dj * lattice_matrix[1] +
                          dk * lattice_matrix[2])
                p2 = carts[j] + offset
                dist = np.linalg.norm(p2 - carts[i])
                if 0.1 < dist < r_max:
                    bonds.append((carts[i].copy(), p2.copy(), elements[i], elements[j]))
                    if (di, dj, dk) != (0, 0, 0):
                        key = tuple(np.round(p2, 3))
                        if key not in image_atoms_dict:
                            image_atoms_dict[key] = (p2.copy(), elements[j])
    return bonds, list(image_atoms_dict.values())


def _add_bonds_to_view(view, bonds, bond_radius=0.08):
    for p1, p2, el1, el2 in bonds:
        mid = (p1 + p2) * 0.5
        c1 = jmol_colors.get(el1, "#CCCCCC")
        c2 = jmol_colors.get(el2, "#CCCCCC")
        view.addCylinder({
            "start": {"x": float(p1[0]),  "y": float(p1[1]),  "z": float(p1[2])},
            "end":   {"x": float(mid[0]), "y": float(mid[1]), "z": float(mid[2])},
            "radius": bond_radius, "fromCap": False, "toCap": False, "color": c1,
        })
        view.addCylinder({
            "start": {"x": float(mid[0]), "y": float(mid[1]), "z": float(mid[2])},
            "end":   {"x": float(p2[0]),  "y": float(p2[1]),  "z": float(p2[2])},
            "radius": bond_radius, "fromCap": False, "toCap": False, "color": c2,
        })


def _render_py3dmol(atoms, structure, base_atom_size, show_lattice_vectors,
                    use_orthographic, show_atom_labels, orientation_result,
                    show_bonds=False, bonds_pbc=True, bond_tolerance=1.15,
                    bond_radius=0.08, roll_key=None):

    active_orient = None
    show_success  = False
    sx = sy = sz  = 1

    if orientation_result is not None:
        if len(orientation_result) == 8:
            mode, uvw, hkl, apply_btn, sx, sy, sz, roll_deg = orientation_result
        else:
            mode, uvw, hkl, apply_btn, sx, sy, sz = orientation_result
            roll_deg = 0.0

        if apply_btn:
            active_orient = {
                "active": True, "mode": "hkl",
                "uvw": uvw, "hkl": hkl, "roll_deg": roll_deg,
                "sx": sx, "sy": sy, "sz": sz,
            }
            st.session_state["se_stored_orient"] = active_orient
            show_success = True
        else:
            stored = st.session_state.get("se_stored_orient", {})
            if stored.get("active"):
                active_orient = stored
                sx = int(stored.get("sx", 1))
                sy = int(stored.get("sy", 1))
                sz = int(stored.get("sz", 1))
    else:
        stored = st.session_state.get("se_stored_orient", {})
        if stored.get("active"):
            active_orient = stored
            sx = int(stored.get("sx", 1))
            sy = int(stored.get("sy", 1))
            sz = int(stored.get("sz", 1))

    norm_lat = _vesta_lattice(structure.lattice)
    raw_lm   = norm_lat.matrix

    orientation_applied = False
    orient_matrix = np.eye(3)
    display_lm    = raw_lm.copy()

    if active_orient is not None and active_orient.get("active"):
        try:
            uvw_a  = active_orient["uvw"]
            hkl_a  = active_orient["hkl"]
            roll_a = float(active_orient.get("roll_deg", 0.0))

            view_dir, up_dir = _compute_view_and_up_dirs(raw_lm, "hkl", uvw_a, hkl_a)

            if abs(roll_a) > 1e-12:
                up_dir = _rotate_vector_about_axis(up_dir, view_dir, roll_a)

            orient_matrix = _compute_orientation_matrix(view_dir, up_dir)
            display_lm    = np.array([orient_matrix @ v for v in raw_lm])
            orientation_applied = True

            if show_success:
                st.success("Orientation applied.")
        except Exception as e:
            orientation_applied = False
            orient_matrix = np.eye(3)
            display_lm    = raw_lm.copy()
            if show_success:
                st.error(f"Orientation failed: {e}")

    def _dc(cart_raw):
        return orient_matrix @ np.asarray(cart_raw, dtype=float)

    rows = []
    for ai in range(sx):
        for bi in range(sy):
            for ci in range(sz):
                tr_raw = ai*raw_lm[0] + bi*raw_lm[1] + ci*raw_lm[2]
                for atom in atoms:
                    cart_raw = norm_lat.get_cartesian_coords(
                        [atom["x"], atom["y"], atom["z"]]
                    ) + tr_raw
                    cart = _dc(cart_raw)
                    rows.append(f"{atom['element']} {cart[0]:.6f} {cart[1]:.6f} {cart[2]:.6f}")

    xyz_str = "\n".join([str(len(rows)), f"py3Dmol {sx}x{sy}x{sz}"] + rows)

    view = py3Dmol.view(width=900, height=700)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"model": 0}, {"sphere": {"radius": base_atom_size / 30, "colorscheme": "Jmol"}})

    if show_bonds:
        bond_carts_raw = []
        bond_els       = []
        if sx == 1 and sy == 1 and sz == 1:
            for atom in atoms:
                bond_carts_raw.append(norm_lat.get_cartesian_coords(
                    [atom["x"], atom["y"], atom["z"]]
                ))
                bond_els.append(atom["element"])
            bond_lm = raw_lm
        else:
            for ai in range(sx):
                for bi in range(sy):
                    for ci in range(sz):
                        tr = ai*raw_lm[0] + bi*raw_lm[1] + ci*raw_lm[2]
                        for atom in atoms:
                            bond_carts_raw.append(
                                norm_lat.get_cartesian_coords(
                                    [atom["x"], atom["y"], atom["z"]]
                                ) + tr
                            )
                            bond_els.append(atom["element"])
            bond_lm = np.array([sx*raw_lm[0], sy*raw_lm[1], sz*raw_lm[2]])

        bonds_raw, image_atoms_raw = _compute_bonds_cartesian(
            bond_carts_raw, bond_els, bond_lm,
            tolerance=bond_tolerance, include_pbc=bonds_pbc,
        )

        bonds_display = [(_dc(p1), _dc(p2), el1, el2) for p1, p2, el1, el2 in bonds_raw]
        _add_bonds_to_view(view, bonds_display, bond_radius=bond_radius)

        if bonds_pbc and image_atoms_raw:
            ghost_lines = [str(len(image_atoms_raw)), "ghost atoms"]
            for cart_raw, el in image_atoms_raw:
                cart = _dc(cart_raw)
                ghost_lines.append(f"{el} {float(cart[0]):.6f} {float(cart[1]):.6f} {float(cart[2]):.6f}")
            ghost_xyz = "\n".join(ghost_lines)
            view.addModel(ghost_xyz, "xyz")
            view.setStyle(
                {"model": 1},
                {"sphere": {"radius": base_atom_size / 30, "colorscheme": "Jmol", "opacity": 0.45}},
            )

    if np.linalg.det(raw_lm) > 1e-6 and show_lattice_vectors:
        if sx == 1 and sy == 1 and sz == 1:
            add_box(view, display_lm, color="black", linewidth=2)
        else:
            _draw_supercell_boxes(view, display_lm, sx, sy, sz)
        for vec, raw_vec, col, lbl in zip(
            [display_lm[0], display_lm[1], display_lm[2]],
            [raw_lm[0],     raw_lm[1],     raw_lm[2]],
            ["red", "green", "blue"],
            [f"a={np.linalg.norm(raw_lm[0]):.3f}Å",
             f"b={np.linalg.norm(raw_lm[1]):.3f}Å",
             f"c={np.linalg.norm(raw_lm[2]):.3f}Å"],
        ):
            view.addArrow({
                "start": {"x": 0.0, "y": 0.0, "z": 0.0},
                "end":   {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])},
                "color": col, "radius": 0.1,
            })
            view.addLabel(lbl, {
                "position": {"x": float(vec[0])*1.1, "y": float(vec[1])*1.1, "z": float(vec[2])*1.1},
                "backgroundColor": col, "fontColor": "white", "fontSize": 12,
            })

    if show_atom_labels:
        if len(atoms) <= 200:
            for idx, atom in enumerate(atoms):
                cart = _dc(norm_lat.get_cartesian_coords([atom["x"], atom["y"], atom["z"]]))
                view.addLabel(f"{atom['element']}{idx+1}", {
                    "position": {"x": float(cart[0]), "y": float(cart[1]), "z": float(cart[2])},
                    "backgroundColor": "white", "fontColor": "black", "fontSize": 12,
                    "borderThickness": 1, "borderColor": "grey",
                })
        else:
            st.warning("Too many atoms for labeling (>200). Labels disabled.")

    if use_orthographic:
        view.setProjection("orthogonal")
        view.setCameraParameters({"orthographic": True})
    else:
        view.setProjection("perspective")
        view.setCameraParameters({"orthographic": False})

    view.zoomTo()
    if orientation_applied:
        total = sx * sy * sz
        zoom = 1.1 if total == 1 else (0.9 if total <= 8 else (0.8 if total <= 27 else 0.7))
        view.zoom(zoom)
    else:
        view.zoom(1.1)
        view.rotate(10, "x")

    html_content = view._make_html()
    components.html(
        f"<div style='display:flex;justify-content:center;border:2px solid #333;"
        f"border-radius:10px;overflow:hidden;background-color:#f8f9fa;'>{html_content}</div>",
        height=720,
    )

    elements_present = sorted(set(a["element"] for a in atoms))
    legend_items = [
        f"<div style='margin-right:15px;display:flex;align-items:center;'>"
        f"<div style='width:16px;height:16px;background-color:{jmol_colors.get(e, '#CCCCCC')};"
        f"margin-right:8px;border:2px solid black;border-radius:50%;'></div>"
        f"<span style='font-weight:bold;font-size:14px;'>{e}</span></div>"
        for e in elements_present
    ]
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;"
        f"margin-top:12px;padding:10px;background-color:#f0f2f6;border-radius:10px;'>"
        f"{''.join(legend_items)}</div>",
        unsafe_allow_html=True,
    )
    st.info("🖱️ Left click + drag to rotate · Scroll to zoom · Middle click + drag to pan")

    if roll_key is not None and orientation_result is not None:
        col_roll_btn, col_roll_step, col_roll_reset = st.columns([2, 1, 1])
        with col_roll_step:
            roll_step = st.number_input(
                "Roll step (°)",
                min_value=0.1, max_value=180.0, value=1.0, step=0.5,
                format="%.1f",
                key=f"{roll_key}_step",
                label_visibility="collapsed",
            )
        with col_roll_btn:
            if st.button(
                f"↻ Roll {roll_step:.1f}°",
                key=f"{roll_key}_btn",
                use_container_width=True,
            ):
                new_roll = float(st.session_state.get(roll_key, 0.0)) + roll_step
                st.session_state[roll_key] = new_roll
                stored = dict(st.session_state.get("se_stored_orient", {}))
                stored["roll_deg"] = new_roll
                st.session_state["se_stored_orient"] = stored
                st.rerun()
        with col_roll_reset:
            if st.button(
                "Reset roll",
                key=f"{roll_key}_reset",
                use_container_width=True,
            ):
                st.session_state[roll_key] = 0.0
                stored = dict(st.session_state.get("se_stored_orient", {}))
                stored["roll_deg"] = 0.0
                st.session_state["se_stored_orient"] = stored
                st.rerun()
        current_roll = float(st.session_state.get(roll_key, 0.0))
        if abs(current_roll) > 1e-9:
            st.caption(f"Current roll: {current_roll:.1f}°")


def _lattice_section(structure, selected_file):
    lat_key = f"se_lat_{selected_file}"
    lattice = structure.lattice

    if lat_key not in st.session_state:
        st.session_state[lat_key] = {
            "a": lattice.a, "b": lattice.b, "c": lattice.c,
            "alpha": lattice.alpha, "beta": lattice.beta, "gamma": lattice.gamma,
        }

    try:
        sga = SpacegroupAnalyzer(structure)
        crystal_system = sga.get_crystal_system()
        spg_symbol = sga.get_space_group_symbol()
        spg_number = sga.get_space_group_number()
        st.markdown(
            f"<div style='background:#e8f4fd;border-radius:8px;padding:8px 14px;margin-bottom:10px;"
            f"font-size:0.92rem;'><b>Crystal system:</b> {crystal_system.upper()} &nbsp;·&nbsp; "
            f"<b>Space group:</b> {spg_symbol} (#{spg_number})</div>",
            unsafe_allow_html=True,
        )
        override = st.checkbox("Override symmetry constraints (allow editing all parameters)",
                               value=False, key=f"se_override_{selected_file}")
        if override:
            crystal_system = "triclinic"
    except Exception:
        crystal_system = "triclinic"
        st.warning("Could not determine crystal system — all parameters editable.")

    params_info = {
        "cubic":        {"modifiable": ["a"],                                      "hint": "b = a, c = a, α = β = γ = 90°"},
        "tetragonal":   {"modifiable": ["a", "c"],                                 "hint": "b = a, α = β = γ = 90°"},
        "orthorhombic": {"modifiable": ["a", "b", "c"],                            "hint": "α = β = γ = 90°"},
        "hexagonal":    {"modifiable": ["a", "c"],                                 "hint": "b = a, α = β = 90°, γ = 120°"},
        "trigonal":     {"modifiable": ["a", "c", "alpha"],                        "hint": "b = a"},
        "monoclinic":   {"modifiable": ["a", "b", "c", "beta"],                    "hint": "α = γ = 90°"},
        "triclinic":    {"modifiable": ["a", "b", "c", "alpha", "beta", "gamma"],  "hint": "All free"},
        "unknown":      {"modifiable": ["a", "b", "c", "alpha", "beta", "gamma"],  "hint": "All free"},
    }
    info = params_info.get(crystal_system, params_info["triclinic"])
    modifiable = info["modifiable"]
    st.caption(f"Constraints: {info['hint']}")

    stored = st.session_state[lat_key]
    col_a, col_b, col_c = st.columns(3)
    col_al, col_be, col_ga = st.columns(3)

    with col_a:
        new_a = st.number_input("a (Å)", value=float(stored["a"]), min_value=0.1, max_value=200.0,
                                step=0.001, format="%.5f", key=f"se_a_{selected_file}")
    with col_b:
        if "b" in modifiable:
            new_b = st.number_input("b (Å)", value=float(stored["b"]), min_value=0.1, max_value=200.0,
                                    step=0.001, format="%.5f", key=f"se_b_{selected_file}")
        else:
            st.text_input("b (Å)", value=f"{new_a:.5f}", disabled=True)
            new_b = new_a
    with col_c:
        if "c" in modifiable:
            new_c = st.number_input("c (Å)", value=float(stored["c"]), min_value=0.1, max_value=200.0,
                                    step=0.001, format="%.5f", key=f"se_c_{selected_file}")
        else:
            fixed_c = new_a if crystal_system == "cubic" else stored["c"]
            st.text_input("c (Å)", value=f"{fixed_c:.5f}", disabled=True)
            new_c = fixed_c

    with col_al:
        if "alpha" in modifiable:
            new_alpha = st.number_input("α (°)", value=float(stored["alpha"]), min_value=0.1, max_value=179.9,
                                        step=0.01, format="%.5f", key=f"se_alpha_{selected_file}")
        else:
            fixed_al = 90.0 if crystal_system in ["cubic","tetragonal","orthorhombic","hexagonal","monoclinic"] else stored["alpha"]
            st.text_input("α (°)", value=f"{fixed_al:.5f}", disabled=True)
            new_alpha = fixed_al
    with col_be:
        if "beta" in modifiable:
            new_beta = st.number_input("β (°)", value=float(stored["beta"]), min_value=0.1, max_value=179.9,
                                       step=0.01, format="%.5f", key=f"se_beta_{selected_file}")
        else:
            if crystal_system in ["cubic","tetragonal","orthorhombic","hexagonal"]:
                fixed_be = 90.0
            elif crystal_system == "trigonal" and "alpha" in modifiable:
                fixed_be = new_alpha
            else:
                fixed_be = stored["beta"]
            st.text_input("β (°)", value=f"{fixed_be:.5f}", disabled=True)
            new_beta = fixed_be
    with col_ga:
        if "gamma" in modifiable:
            new_gamma = st.number_input("γ (°)", value=float(stored["gamma"]), min_value=0.1, max_value=179.9,
                                        step=0.01, format="%.5f", key=f"se_gamma_{selected_file}")
        else:
            if crystal_system in ["cubic","tetragonal","orthorhombic","monoclinic"]:
                fixed_ga = 90.0
            elif crystal_system in ["hexagonal","trigonal"]:
                fixed_ga = 120.0
            else:
                fixed_ga = stored["gamma"]
            st.text_input("γ (°)", value=f"{fixed_ga:.5f}", disabled=True)
            new_gamma = fixed_ga

    return new_a, new_b, new_c, new_alpha, new_beta, new_gamma, lat_key


def _atoms_section(atoms, structure, selected_file):
    add_pending_key    = f"se_add_pending_{selected_file}"
    remove_pending_key = f"se_remove_pending_{selected_file}"

    atoms = list(st.session_state.get("se_atoms") or atoms)
    updated_atoms = list(atoms)

    use_unique = st.checkbox(
        "✨ Show & edit unique Wyckoff positions only (changes propagate to all equivalent atoms)",
        value=False, key=f"se_unique_{selected_file}",
    )

    show_add = st.checkbox("➕ Add new atomic site", value=False, key=f"se_show_add_{selected_file}")
    if show_add:
        with st.container(border=True):
            nc1, nc2, nc3, nc4, nc5 = st.columns(5)
            new_site_el  = nc1.selectbox("Element", ELEMENTS, index=7, key=f"se_new_el_{selected_file}")
            new_site_x   = nc2.number_input("Frac X", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"se_new_x_{selected_file}")
            new_site_y   = nc3.number_input("Frac Y", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"se_new_y_{selected_file}")
            new_site_z   = nc4.number_input("Frac Z", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"se_new_z_{selected_file}")
            new_site_occ = nc5.number_input("Occupancy", value=1.0, min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key=f"se_new_occ_{selected_file}")
            if st.button("➕ Add site (double-click)", key=f"se_add_site_{selected_file}", type="primary"):
                if st.session_state.get(add_pending_key):
                    new_idx = max((a["site_idx"] for a in updated_atoms), default=-1) + 1
                    updated_atoms.append({"element": new_site_el, "occ": new_site_occ,
                                          "x": new_site_x, "y": new_site_y, "z": new_site_z,
                                          "wyckoff": "-", "site_idx": new_idx})
                    st.session_state["se_atoms"] = updated_atoms
                    st.session_state[add_pending_key] = False
                    st.success(f"✅ Site {new_site_el} added.")
                else:
                    st.session_state[add_pending_key] = True
                    st.info("Click once more to confirm.")

    show_remove = st.checkbox("🗑️ Remove an atomic site", value=False, key=f"se_show_remove_{selected_file}")
    if show_remove:
        with st.container(border=True):
            if use_unique:
                st.info("Switch off Wyckoff mode to remove individual sites.")
            else:
                atom_labels = [f"{i+1}: {a['element']} @ ({a['x']:.4f}, {a['y']:.4f}, {a['z']:.4f})  occ={a['occ']}"
                               for i, a in enumerate(updated_atoms)]
                if atom_labels:
                    to_remove = st.selectbox("Select site to remove", atom_labels, key=f"se_remove_sel_{selected_file}")
                    remove_idx = int(to_remove.split(":")[0]) - 1
                    if st.button("🗑️ Remove site (double-click)", key=f"se_remove_btn_{selected_file}", type="primary"):
                        if st.session_state.get(remove_pending_key) == remove_idx:
                            updated_atoms = [a for i, a in enumerate(updated_atoms) if i != remove_idx]
                            st.session_state["se_atoms"] = updated_atoms
                            st.session_state[remove_pending_key] = None
                            st.success(f"✅ Site {remove_idx+1} removed.")
                        else:
                            st.session_state[remove_pending_key] = remove_idx
                            st.info("Click once more to confirm.")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    display_atoms = _get_unique_wyckoff_atoms(updated_atoms) if use_unique else updated_atoms

    hdr_cols = st.columns([0.35, 1.1, 0.85, 0.85, 0.85, 0.75, 0.5])
    for col, label in zip(hdr_cols, ["#", "Element", "Frac X", "Frac Y", "Frac Z", "Occupancy", "Wyckoff"]):
        col.markdown(f"<span style='font-weight:700;font-size:0.82rem;color:#1e3a8a;'>{label}</span>",
                     unsafe_allow_html=True)
    st.markdown("<div style='height:2px;background:#e2e8f0;border-radius:2px;margin-bottom:6px;'></div>",
                unsafe_allow_html=True)

    for row_idx, atom in enumerate(display_atoms):
        original_element = atom["element"]
        original_wyckoff = atom["wyckoff"]
        cols = st.columns([0.35, 1.1, 0.85, 0.85, 0.85, 0.75, 0.5])

        with cols[0]:
            if use_unique:
                count = atom.get("wyckoff_count", 1)
                st.markdown(f"<div style='padding-top:6px;font-size:0.82rem;color:#888;'>{count}×</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding-top:6px;font-size:0.82rem;color:#888;'>{row_idx+1}</div>", unsafe_allow_html=True)
        with cols[1]:
            new_element = st.selectbox(f"el_{row_idx}", options=ELEMENTS,
                index=ELEMENTS.index(atom["element"]) if atom["element"] in ELEMENTS else 0,
                key=f"se_el_{selected_file}_{row_idx}_{use_unique}", label_visibility="collapsed")
        with cols[2]:
            new_x = st.number_input(f"x_{row_idx}", value=float(atom["x"]), min_value=-1.0, max_value=2.0,
                step=0.001, format="%.5f", key=f"se_x_{selected_file}_{row_idx}_{use_unique}", label_visibility="collapsed")
        with cols[3]:
            new_y = st.number_input(f"y_{row_idx}", value=float(atom["y"]), min_value=-1.0, max_value=2.0,
                step=0.001, format="%.5f", key=f"se_y_{selected_file}_{row_idx}_{use_unique}", label_visibility="collapsed")
        with cols[4]:
            new_z = st.number_input(f"z_{row_idx}", value=float(atom["z"]), min_value=-1.0, max_value=2.0,
                step=0.001, format="%.5f", key=f"se_z_{selected_file}_{row_idx}_{use_unique}", label_visibility="collapsed")
        with cols[5]:
            new_occ = st.number_input(f"occ_{row_idx}", value=float(atom["occ"]), min_value=0.001, max_value=1.0,
                step=0.001, format="%.3f", key=f"se_occ_{selected_file}_{row_idx}_{use_unique}", label_visibility="collapsed")
        with cols[6]:
            st.markdown(f"<div style='padding-top:6px;font-size:0.82rem;color:#666;'>{atom['wyckoff']}</div>", unsafe_allow_html=True)

        changed_fields = {}
        if new_element != original_element: changed_fields["element"] = new_element
        if abs(new_x - atom["x"]) > 1e-7:  changed_fields["x"] = new_x
        if abs(new_y - atom["y"]) > 1e-7:  changed_fields["y"] = new_y
        if abs(new_z - atom["z"]) > 1e-7:  changed_fields["z"] = new_z
        if abs(new_occ - atom["occ"]) > 1e-5: changed_fields["occ"] = new_occ

        if changed_fields:
            if use_unique:
                updated_atoms = _propagate_wyckoff_change(updated_atoms, original_wyckoff, original_element, changed_fields)
            else:
                for i, a in enumerate(updated_atoms):
                    if (a["site_idx"] == atom["site_idx"] and a["element"] == original_element
                            and abs(a["x"] - atom["x"]) < 1e-5
                            and abs(a["y"] - atom["y"]) < 1e-5
                            and abs(a["z"] - atom["z"]) < 1e-5):
                        updated_atoms[i] = {**a, **changed_fields}
                        break

    return updated_atoms


def _build_download_content(export_struct, dl_format, selected_file):
    ordered = _get_ordered_structure(export_struct)
    ase_atoms = AseAtomsAdaptor.get_atoms(ordered)

    if dl_format == "CIF":
        symprec_cif = st.selectbox("Symmetry precision (symprec)",
            options=[0.001, 0.01, 0.1, 1.0], index=1, format_func=str,
            key=f"se_cif_symprec_{selected_file}")
        file_content = CifWriter(export_struct, symprec=symprec_cif, write_site_properties=True).__str__()
        dl_name, mime = "structure.cif", "chemical/x-cif"

    elif dl_format == "VASP (POSCAR)":
        col_v1, col_v2 = st.columns(2)
        use_fractional = col_v1.checkbox("Fractional coordinates", value=True, key=f"se_vasp_frac_{selected_file}")
        use_sd = col_v2.checkbox("Selective Dynamics (all atoms free)", value=False, key=f"se_vasp_sd_{selected_file}")
        if use_sd:
            ase_atoms.set_constraint(FixAtoms(indices=[]))
        out = StringIO()
        write(out, ase_atoms, format="vasp", direct=use_fractional, sort=True)
        file_content, dl_name, mime = out.getvalue(), "POSCAR", "text/plain"

    elif dl_format == "LAMMPS":
        col_l1, col_l2 = st.columns(2)
        atom_style = col_l1.selectbox("atom_style", ["atomic","charge","full"], index=0, key=f"se_lmp_style_{selected_file}")
        units       = col_l2.selectbox("units", ["metal","real","si"], index=0, key=f"se_lmp_units_{selected_file}")
        col_l3, col_l4 = st.columns(2)
        inc_masses  = col_l3.checkbox("Include masses", value=True, key=f"se_lmp_masses_{selected_file}")
        force_skew  = col_l4.checkbox("Force triclinic cell", value=False, key=f"se_lmp_skew_{selected_file}")
        out = StringIO()
        write(out, ase_atoms, format="lammps-data", atom_style=atom_style, units=units,
              masses=inc_masses, force_skew=force_skew)
        file_content, dl_name, mime = out.getvalue(), "structure.lmp", "text/plain"

    else:
        col_x1, _ = st.columns(2)
        xyz_type = col_x1.radio("Coordinate type",
            ["Cartesian (extended XYZ)", "Fractional"], index=0, key=f"se_xyz_type_{selected_file}")
        if xyz_type == "Fractional":
            lv = ordered.lattice.matrix
            lat_str = " ".join(f"{x:.6f}" for row in lv for x in row)
            lines = [str(len(ordered)), f'Lattice="{lat_str}" Properties=species:S:1:pos:R:3 fractional=T']
            for site in ordered:
                frac = site.frac_coords
                lines.append(f"{site.specie.symbol} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f}")
        else:
            lv = ordered.lattice.matrix
            lat_str = " ".join(f"{x:.6f}" for row in lv for x in row)
            lines = [str(len(ordered)), f'Lattice="{lat_str}" Properties=species:S:1:pos:R:3']
            for site in ordered:
                cart = ordered.lattice.get_cartesian_coords(site.frac_coords)
                lines.append(f"{site.specie.symbol} {cart[0]:.6f} {cart[1]:.6f} {cart[2]:.6f}")
        file_content, dl_name, mime = "\n".join(lines), "structure.xyz", "text/plain"

    return file_content, dl_name, mime


def _merge_uploaded_files(original, session_key="uploaded_files"):
    merged = list(original) if original else []
    existing_names = {f.name for f in merged}
    for f in st.session_state.get(session_key, []):
        if f.name not in existing_names:
            merged.append(f)
            existing_names.add(f.name)
    return merged


def run_structure_editor(uploaded_files):
    if not uploaded_files:
        st.info("Upload structure files from the sidebar to begin.")
        return _merge_uploaded_files(uploaded_files)

    for key in ["se_selected_file", "se_current_structure", "se_atoms", "se_atoms_loaded"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "se_stored_orient" not in st.session_state:
        st.session_state["se_stored_orient"] = {"active": False}

    file_options = [f.name for f in uploaded_files]
    selected_file = (
        st.selectbox("Select structure", file_options)
        if len(file_options) > 1
        else file_options[0]
    )

    if selected_file != st.session_state["se_selected_file"]:
        st.session_state["se_selected_file"]     = selected_file
        st.session_state["se_current_structure"] = None
        st.session_state["se_atoms"]             = None
        st.session_state["se_atoms_loaded"]      = False
        st.session_state["se_stored_orient"]     = {"active": False}
        for k in list(st.session_state.keys()):
            if k.startswith(f"se_lat_{selected_file}"):
                del st.session_state[k]
        try:
            structure = load_structure(selected_file)
        except Exception:
            try:
                from ase.io import read as ase_read
                structure = AseAtomsAdaptor.get_structure(ase_read(selected_file))
            except Exception as e:
                st.error(f"Could not load {selected_file}: {e}")
                return _merge_uploaded_files(uploaded_files)
        st.session_state["se_current_structure"] = structure

    structure = st.session_state["se_current_structure"]
    if structure is None:
        return _merge_uploaded_files(uploaded_files)

    atoms   = st.session_state["se_atoms"]
    lat     = structure.lattice
    formula = structure.composition.reduced_formula

    def _build_export_struct():
        _lat_stored = st.session_state.get(f"se_lat_{selected_file}", {})
        _la  = _lat_stored.get("a",     structure.lattice.a)
        _lb  = _lat_stored.get("b",     structure.lattice.b)
        _lc  = _lat_stored.get("c",     structure.lattice.c)
        _lal = _lat_stored.get("alpha", structure.lattice.alpha)
        _lbe = _lat_stored.get("beta",  structure.lattice.beta)
        _lga = _lat_stored.get("gamma", structure.lattice.gamma)
        current_atoms = st.session_state.get("se_atoms") or _load_atoms_from_structure(structure)
        return _rebuild_structure_from_atoms(
            current_atoms,
            PmgLattice.from_parameters(_la, _lb, _lc, _lal, _lbe, _lga),
        )

    def _render_add_section(tab_suffix):
        st.markdown(
            "<hr style='border:none;height:2px;background:linear-gradient(to right,#8b0000,#e57373);"
            "border-radius:4px;margin:20px 0 14px 0;'>",
            unsafe_allow_html=True,
        )
        st.markdown("#### 📦 Add to Calculator")
        col_name, col_fmt = st.columns([2, 1])
        with col_name:
            custom_name = st.text_input(
                "Name for the modified structure:",
                value=f"MODIFIED_{formula}",
                key=f"se_name_{selected_file}_{tab_suffix}",
            )
            custom_name_cif = custom_name if custom_name.endswith(".cif") else custom_name + ".cif"
        with col_fmt:
            dl_fmt_add = st.selectbox(
                "Format (download only)",
                ["CIF", "VASP (POSCAR)", "LAMMPS", "XYZ"],
                key=f"se_dl_fmt_{selected_file}_{tab_suffix}",
            )
        build_ok, export_struct = False, None
        try:
            export_struct = _build_export_struct()
            lat_new = export_struct.lattice
            build_ok = True
            st.markdown(
                f"<div style='background:#f0fff4;border-radius:8px;padding:8px 14px;margin:8px 0;"
                f"font-size:0.88rem;color:#065f46;'>"
                f"<b>Preview:</b> {export_struct.composition.reduced_formula} · "
                f"{len(export_struct)} sites · "
                f"a={lat_new.a:.4f} b={lat_new.b:.4f} c={lat_new.c:.4f} Å · "
                f"V={lat_new.volume:.2f} Å³</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Cannot build structure: {e}")
        col_add, col_dl = st.columns([1, 1])
        with col_add:
            if st.button("➕ Add to Calculator", type="primary",
                         key=f"se_add_{selected_file}_{tab_suffix}", disabled=not build_ok):
                try:
                    cif_content = CifWriter(export_struct, symprec=0.1, write_site_properties=True).__str__()
                    cif_file = io.BytesIO(cif_content.encode("utf-8"))
                    cif_file.name = custom_name_cif
                    if "uploaded_files" not in st.session_state:
                        st.session_state["uploaded_files"] = []
                    st.session_state["uploaded_files"] = [
                        f for f in st.session_state["uploaded_files"] if f.name != custom_name_cif]
                    st.session_state["uploaded_files"].append(cif_file)
                    if "full_structures" not in st.session_state:
                        st.session_state["full_structures"] = {}
                    st.session_state["full_structures"][custom_name_cif] = export_struct
                    st.success(f"\u2705 '{custom_name_cif}' added to the calculator!")
                except Exception as e:
                    st.error(f"Error adding structure: {e}")
        with col_dl:
            if build_ok and export_struct is not None:
                try:
                    fc, dn, mime = _build_download_content(
                        export_struct, dl_fmt_add, f"{selected_file}_{tab_suffix}"
                    )
                    if fc:
                        st.download_button(
                            label=f"\u2b07\ufe0f Download {dl_fmt_add}",
                            data=fc, file_name=dn, mime=mime,
                            key=f"se_dl_{selected_file}_{tab_suffix}_{dl_fmt_add}",
                        )
                except Exception as e:
                    st.error(f"Error preparing download: {e}")

    tab_viz, tab_lattice, tab_atoms, tab_export = st.tabs([
        "\U0001f52c Visualization",
        "\U0001f537 Lattice Parameters",
        "\u269b\ufe0f Atomic Sites",
        "\U0001f4be Export Structure",
    ])

    with tab_viz:
        render_atoms = atoms if atoms is not None else _load_atoms_from_structure(structure)
        stored_lat = st.session_state.get(f"se_lat_{selected_file}")
        try:
            if atoms is not None and stored_lat:
                preview_lattice = PmgLattice.from_parameters(
                    stored_lat["a"], stored_lat["b"], stored_lat["c"],
                    stored_lat["alpha"], stored_lat["beta"], stored_lat["gamma"])
                preview_struct = _rebuild_structure_from_atoms(render_atoms, preview_lattice)
            else:
                preview_struct = structure
        except Exception:
            preview_struct = structure

        viz_left, viz_right = st.columns([1, 3])
        with viz_left:
            base_atom_size = st.slider("Atom size", 1, 30, 10, key=f"se_atom_size_{selected_file}")
            show_lv     = st.checkbox("Show lattice vectors & cell", value=True,  key=f"se_show_lv_{selected_file}")
            use_ortho   = st.checkbox("Orthographic projection",     value=False, key=f"se_ortho_{selected_file}")
            show_labels = st.checkbox("Show atom labels",            value=False, key=f"se_labels_{selected_file}")
            show_asym   = st.checkbox("Show asymmetric unit only",   value=False, key=f"se_asym_{selected_file}")

            st.markdown(
                "<div style='height:1px;background:#e2e8f0;margin:8px 0;'></div>",
                unsafe_allow_html=True,
            )
            show_bonds = st.checkbox("🔗 Show bonds", value=False, key=f"se_bonds_{selected_file}")
            if show_bonds:
                bonds_pbc = st.checkbox(
                    "Extend bonds across periodic boundaries",
                    value=True, key=f"se_bonds_pbc_{selected_file}",
                )
                bond_tol = st.slider(
                    "Bond tolerance (× covalent radii sum)",
                    min_value=0.80, max_value=1.50, value=1.15, step=0.01,
                    key=f"se_bond_tol_{selected_file}",
                )
                bond_r = st.slider(
                    "Bond radius (Å)",
                    min_value=0.02, max_value=0.25, value=0.08, step=0.01,
                    key=f"se_bond_r_{selected_file}",
                )
            else:
                bonds_pbc, bond_tol, bond_r = True, 1.15, 0.08
            st.markdown(
                "<div style='height:1px;background:#e2e8f0;margin:8px 0;'></div>",
                unsafe_allow_html=True,
            )

            orientation_result = _orientation_controls(
                key_suffix=f"se_{selected_file}",
                lattice_matrix=_vesta_lattice(preview_struct.lattice).matrix,
            )
            _pl      = preview_struct.lattice
            _density = float(str(preview_struct.density).split()[0])
            st.markdown(
                f"<div style='background:#f0f4ff;border-radius:8px;padding:9px 12px;margin-top:10px;"
                f"font-size:0.82rem;color:#333;line-height:1.7;'>"
                f"<b style='color:#1e3a8a;font-size:0.95rem;'>{preview_struct.composition.reduced_formula}</b> "
                f"{identify_structure_type(preview_struct)}<br>"
                f"{len(preview_struct)} sites<br>"
                f"a={_pl.a:.4f} b={_pl.b:.4f} c={_pl.c:.4f} \u00c5<br>"
                f"\u03b1={_pl.alpha:.2f}\u00b0 \u03b2={_pl.beta:.2f}\u00b0 \u03b3={_pl.gamma:.2f}\u00b0<br>"
                f"V={_pl.volume:.2f} \u00c5\u00b3 &nbsp; \u03c1={_density:.3f} g/cm\u00b3"
                f"</div>",
                unsafe_allow_html=True,
            )

        if show_asym:
            try:
                sga = SpacegroupAnalyzer(preview_struct, symprec=0.1)
                sym_data = sga.get_symmetry_dataset()
                equiv = sym_data.equivalent_atoms
                seen = set()
                asym_atoms = []
                for atom in render_atoms:
                    si = atom["site_idx"]
                    rep = int(equiv[si]) if si < len(equiv) else si
                    if rep not in seen:
                        seen.add(rep)
                        asym_atoms.append(atom)
                viz_atoms = asym_atoms
                st.caption(f"Asymmetric unit: {len(viz_atoms)} of {len(render_atoms)} sites shown.")
            except Exception as e:
                st.caption(f"⚠️ Could not determine asymmetric unit: {e}")
                viz_atoms = render_atoms
        else:
            viz_atoms = render_atoms

        with viz_right:
            _render_py3dmol(
                atoms=viz_atoms, structure=preview_struct,
                base_atom_size=base_atom_size, show_lattice_vectors=show_lv,
                use_orthographic=use_ortho, show_atom_labels=show_labels,
                orientation_result=orientation_result,
                show_bonds=show_bonds, bonds_pbc=bonds_pbc,
                bond_tolerance=bond_tol, bond_radius=bond_r,
                roll_key=f"orient_roll_deg_se_{selected_file}",
            )

    with tab_lattice:
        with st.container(border=True):
            new_a, new_b, new_c, new_alpha, new_beta, new_gamma, lat_key = _lattice_section(structure, selected_file)
        st.session_state[lat_key] = {
            "a": new_a, "b": new_b, "c": new_c,
            "alpha": new_alpha, "beta": new_beta, "gamma": new_gamma,
        }
        _render_add_section("lat")
        st.markdown("<br><br>", unsafe_allow_html=True)

    with tab_atoms:
        load_atoms_check = st.checkbox(
            "\U0001f4e5 Load & edit atomic positions",
            value=bool(st.session_state.get("se_atoms_loaded")),
            key=f"se_load_atoms_cb_{selected_file}",
        )
        if load_atoms_check and not st.session_state.get("se_atoms_loaded"):
            st.session_state["se_atoms"]        = _load_atoms_from_structure(structure)
            st.session_state["se_atoms_loaded"] = True
        if not load_atoms_check and st.session_state.get("se_atoms_loaded"):
            st.session_state["se_atoms_loaded"] = False

        if st.session_state.get("se_atoms_loaded"):
            atoms = st.session_state["se_atoms"]
            if atoms is None:
                atoms = _load_atoms_from_structure(structure)
                st.session_state["se_atoms"] = atoms
            with st.container(border=True):
                updated_atoms = _atoms_section(atoms, structure, selected_file)
            st.session_state["se_atoms"] = updated_atoms
            _render_add_section("atoms")
            st.markdown("<br><br>", unsafe_allow_html=True)

    with tab_export:
        st.markdown(
            "<div style='background:#fff8e1;border-left:4px solid #f59e0b;border-radius:6px;"
            "padding:9px 14px;margin-bottom:16px;font-size:0.9rem;'>"
            "Download the current structure (combining any lattice and atomic edits) in your chosen format.</div>",
            unsafe_allow_html=True,
        )
        try:
            export_struct_exp = _build_export_struct()
            _ln = export_struct_exp.lattice
            st.markdown(
                f"<div style='background:#f0fff4;border-radius:8px;padding:8px 14px;margin-bottom:14px;"
                f"font-size:0.88rem;color:#065f46;'>"
                f"<b>Structure to export:</b> {export_struct_exp.composition.reduced_formula} · "
                f"{len(export_struct_exp)} sites · "
                f"a={_ln.a:.4f} b={_ln.b:.4f} c={_ln.c:.4f} \u00c5 · "
                f"V={_ln.volume:.2f} \u00c5\u00b3</div>",
                unsafe_allow_html=True,
            )
            dl_format_exp = st.selectbox(
                "Download format", ["CIF", "VASP (POSCAR)", "LAMMPS", "XYZ"],
                key=f"se_dl_fmt_exp_{selected_file}",
            )
            try:
                fc_exp, dn_exp, mime_exp = _build_download_content(
                    export_struct_exp, dl_format_exp, f"{selected_file}_exp"
                )
                if fc_exp:
                    st.download_button(
                        label=f"\u2b07\ufe0f Download {dl_format_exp}",
                        data=fc_exp, file_name=dn_exp, mime=mime_exp,
                        key=f"se_dl_exp_{selected_file}_{dl_format_exp}",
                        type="primary",
                    )
            except Exception as e:
                st.error(f"Error preparing download: {e}")
        except Exception as e:
            st.error(f"Cannot build structure for export: {e}")
        st.markdown("<br><br>", unsafe_allow_html=True)

    return _merge_uploaded_files(uploaded_files)
