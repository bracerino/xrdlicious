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
    """
    Returns (view_dir_cart, up_dir_cart) in Cartesian coordinates.

    Mode 'uvw': look along real-space direction [u v w].
                up is along the plane normal of (h k l) in reciprocal space.
    Mode 'hkl': look along reciprocal-space direction = normal to (h k l) plane.
                up is along real-space direction [u v w].
    """
    lm = lattice_matrix
    a, b, c = lm[0], lm[1], lm[2]
    recip = 2 * np.pi * np.linalg.inv(lm).T
    a_star, b_star, c_star = recip[0], recip[1], recip[2]

    u, v, w = uvw
    h, k, l = hkl

    if mode == "uvw":
        view_dir = float(u)*a + float(v)*b + float(w)*c
        if h == 0 and k == 0 and l == 0:
            up_dir = np.array([0.0, 1.0, 0.0])
        else:
            up_dir = float(h)*a_star + float(k)*b_star + float(l)*c_star
    else:
        if h == 0 and k == 0 and l == 0:
            raise ValueError("(h k l) = (0 0 0) is invalid.")
        view_dir = float(h)*a_star + float(k)*b_star + float(l)*c_star
        if u == 0 and v == 0 and w == 0:
            up_dir = np.array([0.0, 1.0, 0.0])
        else:
            up_dir = float(u)*a + float(v)*b + float(w)*c

    return np.array(view_dir, dtype=float), np.array(up_dir, dtype=float)


def _compute_view_rotations(view_dir_cart, up_dir_cart):
    """
    Compute three sequential world-axis rotations (ZYZ) for py3Dmol that orient
    the scene so view_dir_cart faces the camera (+z) and up_dir_cart is screen-up (+y).

    py3Dmol convention: camera at +z looking toward -z, y is screen-up.
    view.rotate(angle, axis) rotates the scene around the world axis.

    Step 1 – rotate scene around world-Z by rot_z1:
             brings the view_dir's xy projection onto the +x half-plane.
    Step 2 – rotate scene around world-Y by rot_y:
             tilts view_dir up to align with world +z (→ toward camera).
    Step 3 – rotate scene around world-Z by rot_z2:
             spins so that up_dir ends up pointing along world +y (screen-up).
    """
    vd = np.asarray(view_dir_cart, dtype=float)
    norm_vd = np.linalg.norm(vd)
    if norm_vd < 1e-10:
        return []
    vd /= norm_vd

    dx, dy, dz = vd

    rot_z1_deg = -np.degrees(np.arctan2(dy, dx))
    rot_z1_rad = np.radians(rot_z1_deg)

    xy_mag = np.sqrt(dx**2 + dy**2)
    rot_y_deg = np.degrees(np.arctan2(xy_mag, dz))
    rot_y_rad = np.radians(rot_y_deg)

    Rz1 = np.array([
        [ np.cos(rot_z1_rad), -np.sin(rot_z1_rad), 0.0],
        [ np.sin(rot_z1_rad),  np.cos(rot_z1_rad), 0.0],
        [ 0.0,                  0.0,                1.0],
    ])
    Ry = np.array([
        [ np.cos(rot_y_rad), 0.0, np.sin(rot_y_rad)],
        [ 0.0,               1.0, 0.0               ],
        [-np.sin(rot_y_rad), 0.0, np.cos(rot_y_rad)],
    ])

    ud_raw = np.asarray(up_dir_cart, dtype=float)
    ud_raw = ud_raw - np.dot(ud_raw, vd) * vd
    if np.linalg.norm(ud_raw) < 1e-10:
        for cand in [[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]]:
            ud_raw = np.array(cand) - np.dot(cand, vd) * vd
            if np.linalg.norm(ud_raw) > 1e-10:
                break
    ud = ud_raw / np.linalg.norm(ud_raw)

    ud_rot = Ry @ (Rz1 @ ud)
    rot_z2_deg = np.degrees(np.arctan2(ud_rot[0], ud_rot[1]))

    rotations = []
    if abs(rot_z1_deg) > 0.01:
        rotations.append((rot_z1_deg, "z"))
    if abs(rot_y_deg) > 0.01:
        rotations.append((rot_y_deg, "y"))
    if abs(rot_z2_deg) > 0.01:
        rotations.append((rot_z2_deg, "z"))
    return rotations


def _compute_orientation_matrix(view_dir_cart, up_dir_cart):
    """
    Returns the 3x3 rotation matrix matching VESTA's convention:
      Row 0 = screen-right axis  (xd) as Cartesian unit vector
      Row 1 = screen-up axis     (ud) as Cartesian unit vector
      Row 2 = into-screen axis   (vd) as Cartesian unit vector
    Values are dimensionless (pure rotation matrix), matching VESTA's display.
    """
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
    """
    VESTA-style orientation control matching the dialog layout:
    - Mode radio (project along [uvw] | project along normal to (hkl))
    - Orientation matrix display
    - Projection vector u,v,w  |  Upward vector h,k,l
    - Apply button
    - Supercell repeats
    """
    enable = st.checkbox(
        "🔄 Set crystallographic orientation",
        value=False,
        key=f"orient_enable_{key_suffix}",
    )
    if not enable:
        return None

    PRESETS = [
        ("a",  "uvw", [1,0,0], [0,0,1]),
        ("b",  "uvw", [0,1,0], [1,0,0]),
        ("c",  "uvw", [0,0,1], [0,1,0]),
        ("a*", "hkl", [1,0,0], [0,0,1]),
        ("b*", "hkl", [0,1,0], [1,0,0]),
        ("c*", "hkl", [0,0,1], [0,1,0]),
    ]
    st.caption("Quick presets:")
    preset_cols = st.columns(len(PRESETS))
    _preset_fired = False
    _preset_mode, _preset_uvw, _preset_hkl = None, None, None
    for col, (label, p_mode, p_uvw, p_hkl) in zip(preset_cols, PRESETS):
        if col.button(label, key=f"orient_preset_{label}_{key_suffix}", use_container_width=True):
            _mode_str = "Project along [u v w]" if p_mode == "uvw" else "Project along the normal to (h k l)"
            st.session_state[f"orient_mode_{key_suffix}"] = _mode_str
            st.session_state[f"orient_u_{key_suffix}"]    = p_uvw[0]
            st.session_state[f"orient_v_{key_suffix}"]    = p_uvw[1]
            st.session_state[f"orient_w_{key_suffix}"]    = p_uvw[2]
            st.session_state[f"orient_h_{key_suffix}"]    = p_hkl[0]
            st.session_state[f"orient_k_{key_suffix}"]    = p_hkl[1]
            st.session_state[f"orient_l_{key_suffix}"]    = p_hkl[2]
            st.session_state["se_stored_orient"] = {
                "active": True, "mode": p_mode, "uvw": p_uvw, "hkl": p_hkl
            }
            _preset_fired = True
            _preset_mode, _preset_uvw, _preset_hkl = p_mode, p_uvw, p_hkl

    mode_label = st.radio(
        "Projection mode:",
        options=["Project along [u v w]", "Project along the normal to (h k l)"],
        key=f"orient_mode_{key_suffix}",
        horizontal=False,
    )
    mode = "uvw" if "u v w" in mode_label else "hkl"

    col_proj, col_up = st.columns(2)

    with col_proj:
        st.markdown("<div style='font-weight:600;font-size:0.85rem;margin-bottom:4px;'>Projection vector</div>", unsafe_allow_html=True)
        u = int(st.number_input("u", value=0, step=1, format="%d", key=f"orient_u_{key_suffix}"))
        v = int(st.number_input("v", value=0, step=1, format="%d", key=f"orient_v_{key_suffix}"))
        w = int(st.number_input("w", value=1, step=1, format="%d", key=f"orient_w_{key_suffix}"))

    with col_up:
        st.markdown("<div style='font-weight:600;font-size:0.85rem;margin-bottom:4px;'>Upward vector</div>", unsafe_allow_html=True)
        h = int(st.number_input("h", value=0, step=1, format="%d", key=f"orient_h_{key_suffix}"))
        k = int(st.number_input("k", value=1, step=1, format="%d", key=f"orient_k_{key_suffix}"))
        l = int(st.number_input("l", value=0, step=1, format="%d", key=f"orient_l_{key_suffix}"))

    uvw = [u, v, w] if not _preset_fired else _preset_uvw
    hkl = [h, k, l] if not _preset_fired else _preset_hkl
    mode = mode if not _preset_fired else _preset_mode

    if mode == "uvw" and all(x == 0 for x in uvw):
        st.warning("⚠️ Projection vector [u v w] = [0 0 0] is not a valid direction.")
        return None
    if mode == "hkl" and all(x == 0 for x in hkl):
        st.warning("⚠️ (h k l) = (0 0 0) is not a valid plane.")
        return None

    if lattice_matrix is not None:
        try:
            dot_cond = hkl[0]*uvw[0] + hkl[1]*uvw[1] + hkl[2]*uvw[2]
            if dot_cond != 0:
                st.caption(
                    f"⚠️ hu+kv+lw = {dot_cond} ≠ 0: upward vector not strictly in projection plane. "
                    f"Up direction will be auto-adjusted."
                )
            view_dir, up_dir = _compute_view_and_up_dirs(lattice_matrix, mode, uvw, hkl)
            M = _compute_orientation_matrix(view_dir, up_dir)
            st.markdown("<div style='font-weight:600;font-size:0.85rem;margin-top:8px;'>Orientation matrix</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.80rem;background:#f4f4f4;"
                f"padding:8px 10px;border-radius:6px;line-height:1.7;'>"
                f"{M[0,0]:+.6f}  {M[0,1]:+.6f}  {M[0,2]:+.6f}<br>"
                f"{M[1,0]:+.6f}  {M[1,1]:+.6f}  {M[1,2]:+.6f}<br>"
                f"{M[2,0]:+.6f}  {M[2,1]:+.6f}  {M[2,2]:+.6f}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if mode == "hkl":
                try:
                    d = _d_spacing(lattice_matrix, hkl[0], hkl[1], hkl[2])
                    st.caption(f"d-spacing ({hkl[0]} {hkl[1]} {hkl[2]}): {d:.4f} Å")
                except Exception:
                    pass
        except Exception:
            pass

    apply = st.button("Apply orientation", key=f"orient_apply_{key_suffix}", type="primary")
    st.caption("Affects 3D preview only.")

    cx, cy, cz = st.columns(3)
    sx = int(cx.number_input("Repeat X", min_value=1, max_value=5, value=1, step=1, key=f"orient_sx_{key_suffix}"))
    sy = int(cy.number_input("Repeat Y", min_value=1, max_value=5, value=1, step=1, key=f"orient_sy_{key_suffix}"))
    sz = int(cz.number_input("Repeat Z", min_value=1, max_value=5, value=1, step=1, key=f"orient_sz_{key_suffix}"))

    return mode, uvw, hkl, (apply or _preset_fired), sx, sy, sz


def _apply_orientation_to_view(view, lattice_matrix, mode, uvw, hkl, sx=1, sy=1, sz=1):
    try:
        view_dir, up_dir = _compute_view_and_up_dirs(lattice_matrix, mode, uvw, hkl)
        rotations = _compute_view_rotations(view_dir, up_dir)
        view.zoomTo()
        for angle, axis in rotations:
            view.rotate(angle, axis)
        total = sx * sy * sz
        zoom = 1.1 if total == 1 else (0.9 if total <= 8 else (0.8 if total <= 27 else 0.7))
        view.zoom(zoom)
        return True, f"Orientation applied."
    except Exception as e:
        return False, f"Orientation failed: {e}"


def _make_supercell_xyz(atoms, structure, sx, sy, sz):
    lm = structure.lattice.matrix
    rows = []
    for ai in range(sx):
        for bi in range(sy):
            for ci in range(sz):
                tr = ai*lm[0] + bi*lm[1] + ci*lm[2]
                for atom in atoms:
                    cart = structure.lattice.get_cartesian_coords([atom["x"], atom["y"], atom["z"]]) + tr
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


def _render_py3dmol(atoms, structure, base_atom_size, show_lattice_vectors,
                    use_orthographic, show_atom_labels, orientation_result):
    sx, sy, sz = 1, 1, 1
    if orientation_result is not None:
        _, _, _, _, sx, sy, sz = orientation_result

    lm = structure.lattice.matrix

    if sx == 1 and sy == 1 and sz == 1:
        lines = [str(len(atoms)), "py3Dmol"]
        for atom in atoms:
            cart = structure.lattice.get_cartesian_coords([atom["x"], atom["y"], atom["z"]])
            lines.append(f"{atom['element']} {cart[0]:.6f} {cart[1]:.6f} {cart[2]:.6f}")
        xyz_str = "\n".join(lines)
    else:
        xyz_str = _make_supercell_xyz(atoms, structure, sx, sy, sz)

    view = py3Dmol.view(width=900, height=700)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"model": 0}, {"sphere": {"radius": base_atom_size / 30, "colorscheme": "Jmol"}})

    if np.linalg.det(lm) > 1e-6 and show_lattice_vectors:
        if sx == 1 and sy == 1 and sz == 1:
            add_box(view, lm, color="black", linewidth=2)
        else:
            _draw_supercell_boxes(view, lm, sx, sy, sz)
        for vec, col, lbl in zip(
            [lm[0], lm[1], lm[2]], ["red", "green", "blue"],
            [f"a={np.linalg.norm(lm[0]):.3f}Å", f"b={np.linalg.norm(lm[1]):.3f}Å", f"c={np.linalg.norm(lm[2]):.3f}Å"],
        ):
            view.addArrow({"start": {"x": 0.0, "y": 0.0, "z": 0.0},
                           "end":   {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])},
                           "color": col, "radius": 0.1})
            view.addLabel(lbl, {"position": {"x": float(vec[0])*1.1, "y": float(vec[1])*1.1, "z": float(vec[2])*1.1},
                                "backgroundColor": col, "fontColor": "white", "fontSize": 12})

    if show_atom_labels:
        if len(atoms) <= 200:
            for idx, atom in enumerate(atoms):
                cart = structure.lattice.get_cartesian_coords([atom["x"], atom["y"], atom["z"]])
                view.addLabel(f"{atom['element']}{idx+1}", {
                    "position": {"x": float(cart[0]), "y": float(cart[1]), "z": float(cart[2])},
                    "backgroundColor": "white", "fontColor": "black", "fontSize": 12,
                    "borderThickness": 1, "borderColor": "grey"})
        else:
            st.warning("Too many atoms for labeling (>200). Labels disabled.")

    should_apply = False
    show_success = False
    stored_orient = None

    if orientation_result is not None:
        mode, uvw, hkl, apply_btn, sx, sy, sz = orientation_result
        if apply_btn:
            should_apply = True
            show_success = True
            st.session_state["se_stored_orient"] = {
                "active": True, "mode": mode, "uvw": uvw, "hkl": hkl
            }

    stored = st.session_state.get("se_stored_orient", {})
    if stored.get("active") and not should_apply:
        should_apply = True
        stored_orient = stored

    if should_apply:
        if stored_orient:
            mode_a, uvw_a, hkl_a = stored_orient["mode"], stored_orient["uvw"], stored_orient["hkl"]
        else:
            mode_a, uvw_a, hkl_a = orientation_result[0], orientation_result[1], orientation_result[2]

        success, message = _apply_orientation_to_view(view, lm, mode_a, uvw_a, hkl_a, sx, sy, sz)
        if show_success:
            (st.success if success else st.error)(message)
    else:
        if use_orthographic:
            view.setProjection("orthogonal")
            view.setCameraParameters({"orthographic": True})
            view.zoomTo()
        else:
            view.setProjection("perspective")
            view.setCameraParameters({"orthographic": False})
            view.zoomTo()
            view.zoom(1.1)
            view.rotate(10, "x")

    if use_orthographic:
        view.setProjection("orthogonal")
        view.setCameraParameters({"orthographic": True})

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
            orientation_result = _orientation_controls(
                key_suffix=f"se_{selected_file}",
                lattice_matrix=preview_struct.lattice.matrix,
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
        with viz_right:
            _render_py3dmol(
                atoms=render_atoms, structure=preview_struct,
                base_atom_size=base_atom_size, show_lattice_vectors=show_lv,
                use_orthographic=use_ortho, show_atom_labels=show_labels,
                orientation_result=orientation_result,
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
