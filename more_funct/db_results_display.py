from __future__ import annotations

import io
import re
from collections import Counter

import requests
import streamlit as st
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from helpers import (
    SPACE_GROUP_SYMBOLS,
    check_structure_size_and_warn,
    get_full_conventional_structure,
    identify_structure_type,
)

_DB_KEYS: dict[str, str] = {
    "Materials Project": "mp_options",
    "AFLOW":             "aflow_options",
    "COD":               "cod_options",
    "MC3D":              "mc3d_options",
}

_DB_COLORS: dict[str, str] = {
    "Materials Project": "#1565C0",   # blue
    "AFLOW":             "#E65100",   # orange
    "COD":               "#2E7D32",   # green
    "MC3D":              "#6A1B9A",   # purple
}

_DB_ICONS: dict[str, str] = {
    "Materials Project": "🔵",
    "AFLOW":             "🟠",
    "COD":               "🟢",
    "MC3D":              "🟣",
}

_CRYSTAL_SYSTEMS: list[str] = [
    "Triclinic",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Hexagonal",
    "Cubic",
]

_CRYSTAL_SYSTEM_ICONS: dict[str, str] = {
    "Triclinic":    "🔷",
    "Monoclinic":   "🔶",
    "Orthorhombic": "🟦",
    "Tetragonal":   "🟧",
    "Trigonal":     "🔺",
    "Hexagonal":    "🔸",
    "Cubic":        "⬛",
}

_CRYSTAL_SYSTEM_COLORS: dict[str, str] = {
    "Triclinic":    "#78909C",
    "Monoclinic":   "#8D6E63",
    "Orthorhombic": "#1976D2",
    "Tetragonal":   "#F57C00",
    "Trigonal":     "#7B1FA2",
    "Hexagonal":    "#C62828",
    "Cubic":        "#2E7D32",
}


_CS_ORDER = {cs: i for i, cs in enumerate(_CRYSTAL_SYSTEMS)}


def _sg_to_crystal_system(sg_number: int) -> str:
    if 1   <= sg_number <= 2:   return "Triclinic"
    if 3   <= sg_number <= 15:  return "Monoclinic"
    if 16  <= sg_number <= 74:  return "Orthorhombic"
    if 75  <= sg_number <= 142: return "Tetragonal"
    if 143 <= sg_number <= 167: return "Trigonal"
    if 168 <= sg_number <= 194: return "Hexagonal"
    if 195 <= sg_number <= 230: return "Cubic"
    return "Unknown"


def _sg_number(opt: str) -> int:
    m = re.search(r'#(\d+)', opt)
    if m:
        return int(m.group(1))
    m = re.search(r'\((\d+)\)', opt)
    if m:
        return int(m.group(1))
    return 9999


def _crystal_system_from_option(opt: str) -> str:
    return _sg_to_crystal_system(_sg_number(opt))


def _natoms(opt: str) -> int:
    m = re.search(r',\s*(\d+)\s*atoms', opt)
    return int(m.group(1)) if m else 0


def _volume(opt: str) -> float:
    m = re.search(r'([\d.]+)\s*Å³', opt)
    return float(m.group(1)) if m else 0.0


def _formula_key(opt: str) -> str:
    return opt.split()[0].lower()



_SORT_OPTIONS: list[str] = [
    "Space Group ↑",
    "Space Group ↓",
    "Formula (A→Z)",
    "Formula (Z→A)",
    "N. of Atoms ↑",
    "N. of Atoms ↓",
    "Volume ↑",
    "Volume ↓",
    "Crystal System",
]


def sort_structure_options(
    options: list[str],
    sort_by: str,
    crystal_system_filter: str = "All",
) -> list[str]:
    if not options:
        return options

    # 1. filter
    if crystal_system_filter != "All":
        options = [o for o in options
                   if _crystal_system_from_option(o) == crystal_system_filter]

    if not options:
        return options

    # 2. sort
    key_map = {
        "Space Group ↑": (lambda o: _sg_number(o),                                   False),
        "Space Group ↓": (lambda o: _sg_number(o),                                   True),
        "Formula (A→Z)": (lambda o: _formula_key(o),                                 False),
        "Formula (Z→A)": (lambda o: _formula_key(o),                                 True),
        "N. of Atoms ↑": (lambda o: _natoms(o),                                      False),
        "N. of Atoms ↓": (lambda o: _natoms(o),                                      True),
        "Volume ↑":       (lambda o: _volume(o),                                     False),
        "Volume ↓":       (lambda o: _volume(o),                                     True),
        "Crystal System": (lambda o: (_CS_ORDER.get(_crystal_system_from_option(o), 99),
                                      _sg_number(o)),                                False),
    }
    if sort_by in key_map:
        keyfn, rev = key_map[sort_by]
        options = sorted(options, key=keyfn, reverse=rev)

    return options


def _render_crystal_system_distribution(all_options: list[str]) -> None:
    counts: Counter[str] = Counter()
    for opt in all_options:
        cs = _crystal_system_from_option(opt)
        if cs != "Unknown":
            counts[cs] += 1

    total = sum(counts.values())
    if total == 0:
        st.info("No structures with identifiable space groups found.")
        return

    rows = []
    for cs in _CRYSTAL_SYSTEMS:
        n   = counts.get(cs, 0)
        pct = n / total * 100
        col = _CRYSTAL_SYSTEM_COLORS[cs]
        ico = _CRYSTAL_SYSTEM_ICONS[cs]
        row = (
            f'<div style="display:flex;align-items:center;margin:4px 0;gap:8px;">'
            f'<span style="width:115px;font-size:0.82rem;color:#333;font-weight:500;">{ico} {cs}</span>'
            f'<div style="flex:1;background:#e9ecef;border-radius:6px;height:17px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;background:{col};height:100%;border-radius:6px;"></div>'
            f'</div>'
            f'<span style="width:70px;font-size:0.82rem;color:#555;text-align:right;font-weight:600;">'
            f'{n}&nbsp;<span style="font-weight:400;">({pct:.0f}%)</span></span>'
            f'</div>'
        )
        rows.append(row)

    html = (
        '<p style="font-size:0.85rem;font-weight:600;color:#555;margin:6px 0 8px 0;">'
        'Distribution across all databases</p>'
        '<div style="background:#f8f9fa;border-radius:10px;padding:12px 16px;">'
        + "".join(rows)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_structure_card(
    structure,
    db_name: str,
    title: str,
    link_md: str | None = None,
) -> None:
    color    = _DB_COLORS.get(db_name, "#1e3a8a")
    lattice  = structure.lattice
    volume   = lattice.volume
    n_atoms  = len(structure)
    density  = float(str(structure.density).split()[0])
    atomic_d = n_atoms / volume

    try:
        sga    = SpacegroupAnalyzer(structure)
        sg_sym = sga.get_space_group_symbol()
        sg_num = sga.get_space_group_number()
        cs     = _sg_to_crystal_system(sg_num)
    except Exception:
        sg_sym, sg_num, cs = "?", "?", "?"

    proto   = identify_structure_type(structure)
    cs_col  = _CRYSTAL_SYSTEM_COLORS.get(cs, "#555")
    cs_icon = _CRYSTAL_SYSTEM_ICONS.get(cs, "")

    db_icon = _DB_ICONS.get(db_name, "🧬")
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{color}18,{color}08);border-left:5px solid {color};border-radius:10px;padding:12px 18px 10px 18px;margin:10px 0 12px 0;display:flex;align-items:center;gap:14px;">'
        f'<span style="font-size:1.08rem;font-weight:700;color:{color};flex:1;">{db_icon} {title}</span>'
        f'<span style="background:{cs_col};color:white;font-size:0.8rem;font-weight:600;padding:3px 11px;border-radius:14px;white-space:nowrap;">{cs_icon} {cs}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Space Group", sg_sym)
    m2.metric("SG Number",   sg_num)
    m3.metric("N. of Atoms", n_atoms)
    m4.metric("Volume",      f"{volume:.2f} Å³")
    m5.metric("Density",     f"{density:.3f} g/cm³")

    lat_str = (
        f"**a** = {lattice.a:.4f} Å,  **b** = {lattice.b:.4f} Å,  "
        f"**c** = {lattice.c:.4f} Å  |  "
        f"**α** = {lattice.alpha:.2f}°,  **β** = {lattice.beta:.2f}°,  "
        f"**γ** = {lattice.gamma:.2f}°"
    )
    st.markdown(f"**Prototype:** {proto}  \n**Lattice:** {lat_str}")
    st.markdown(f"**Atomic density:** {atomic_d:.5f} Å⁻³")

    if link_md:
        st.markdown(f"**Link:** {link_md}")


def _render_mp(sort_by: str, cs_filter: str) -> None:
    raw     = st.session_state.get("mp_options", [])
    options = sort_structure_options(raw, sort_by, cs_filter)
    structs = st.session_state.get("full_structures_see", {})

    _count_badge(len(options), len(raw), "Materials Project", cs_filter)
    if not options:
        st.info("No structures match the selected crystal system filter.")
        return

    selected    = st.selectbox("Select a structure:", options, key="mp_sel_disp")
    selected_id = selected.split(",")[-1].replace(":", "").strip()
    composition = selected.split("(")[0].strip()
    file_name   = re.sub(r'[\\/:"*?<>|]+', '_', f"{selected_id}_{composition}.cif")

    structure = structs.get(selected_id)
    if structure is None:
        st.warning("Structure data not found in session – please search again.")
        return

    _render_structure_card(
        structure, "Materials Project",
        title   = f"{composition} · {selected_id}",
        link_md = f"[Open on Materials Project ↗](https://materialsproject.org/materials/{selected_id})",
    )
    st.info("ℹ️ If H is missing from the CIF it will also be absent from the formula.")

    col_add, col_dl, _ = st.columns([1, 1, 2])
    with col_add:
        if st.button("➕ Add to workspace", key="mp_add", use_container_width=True):
            check_structure_size_and_warn(structure, f"MP {selected_id}")
            st.session_state.full_structures[file_name] = structure
            _push_cif_to_uploads(file_name, str(CifWriter(structure)))
            st.success("✅ Added from Materials Project!")
    with col_dl:
        st.download_button(
            "💾 Download CIF", type="primary",
            data=str(CifWriter(structure, symprec=0.01)),
            file_name=file_name, mime="chemical/x-cif",
            key="mp_dl", use_container_width=True,
        )


def _render_aflow(sort_by: str, cs_filter: str) -> None:
    raw     = st.session_state.get("aflow_options", [])
    options = sort_structure_options(raw, sort_by, cs_filter)
    entrys  = st.session_state.get("entrys", {})

    st.warning(
        "⚠️ AFLOW does not expose atomic occupancies and returns only the "
        "primitive cell via API. Volume and atom count are therefore "
        "omitted from the dropdown list."
    )
    _count_badge(len(options), len(raw), "AFLOW", cs_filter)
    if not options:
        st.info("No structures match the selected crystal system filter.")
        return

    selected      = st.selectbox("Select a structure:", options, key="aflow_sel_disp")
    selected_auid = selected.split(",")[-1].strip()
    entry = next((e for e in entrys.values() if e.auid == selected_auid), None)

    if entry is None:
        st.warning("Entry not found in session – please search again.")
        return

    cif_files = [f for f in entry.files
                 if f.endswith("_sprim.cif") or f.endswith(".cif")]
    if not cif_files:
        st.warning("No CIF file available for this AFLOW entry.")
        return

    host_part, path_part = entry.aurl.split(":", 1)
    file_url = f"http://{host_part}/{path_part}/{cif_files[0]}"
    try:
        cif_content = requests.get(file_url, timeout=15).content
    except Exception as exc:
        st.error(f"Could not fetch CIF from AFLOW: {exc}")
        return

    structure_raw  = Structure.from_str(cif_content.decode("utf-8"), fmt="cif")
    structure_conv = get_full_conventional_structure(structure_raw, symprec=0.1)

    _render_structure_card(
        structure_conv, "AFLOW",
        title   = f"{entry.compound} · {entry.auid}",
        link_md = f"[Open on AFLOW ↗](https://aflowlib.duke.edu/search/ui/material/?id={entry.auid})",
    )
    st.info("ℹ️ If H is missing from the CIF it will also be absent from the formula.")

    fname = f"{entry.compound}_{entry.auid}.cif"
    col_add, col_dl, _ = st.columns([1, 1, 2])
    with col_add:
        if st.button("➕ Add to workspace", key="aflow_add", use_container_width=True):
            _push_bytes_to_uploads(fname, cif_content)
            st.session_state.full_structures[fname] = structure_raw
            check_structure_size_and_warn(structure_raw, fname)
            st.success("✅ Added from AFLOW!")
    with col_dl:
        st.download_button(
            "💾 Download CIF", type="primary",
            data=cif_content, file_name=fname, mime="chemical/x-cif",
            key="aflow_dl", use_container_width=True,
        )


def _render_cod(sort_by: str, cs_filter: str) -> None:
    raw     = st.session_state.get("cod_options", [])
    options = sort_structure_options(raw, sort_by, cs_filter)
    structs = st.session_state.get("full_structures_see_cod", {})

    _count_badge(len(options), len(raw), "COD", cs_filter)
    if not options:
        st.info("No structures match the selected crystal system filter.")
        return

    selected  = st.selectbox("Select a structure:", options, key="cod_sel_disp")
    cod_id    = selected.split(",")[-1].strip()
    structure = structs.get(cod_id)

    if structure is None:
        st.warning("Structure data not found in session – please search again.")
        return

    numeric_id = cod_id.removeprefix("cod_")
    _render_structure_card(
        structure, "COD",
        title   = f"{structure.composition.reduced_formula} · COD {numeric_id}",
        link_md = f"[Open on COD ↗](https://www.crystallography.net/cod/{numeric_id}.html)",
    )
    st.info("ℹ️ If H is missing from the CIF it will also be absent from the formula.")

    file_name = f"{structure.composition.reduced_formula}_COD_{numeric_id}.cif"
    col_add, col_dl, _ = st.columns([1, 1, 2])
    with col_add:
        if st.button("➕ Add to workspace", key="cod_add", use_container_width=True):
            st.session_state.full_structures[file_name] = structure
            _push_cif_to_uploads(file_name, str(CifWriter(structure, symprec=0.01)))
            check_structure_size_and_warn(structure, file_name)
            st.success("✅ Added from COD!")
    with col_dl:
        st.download_button(
            "💾 Download CIF", type="primary",
            data=str(CifWriter(structure, symprec=0.01)),
            file_name=file_name, mime="chemical/x-cif",
            key="cod_dl", use_container_width=True,
        )


def _render_mc3d(sort_by: str, cs_filter: str) -> None:
    raw     = st.session_state.get("mc3d_options", [])
    options = sort_structure_options(raw, sort_by, cs_filter)
    structs = st.session_state.get("mc3d_structures", {})

    st.info("ℹ️ MC3D structures accessed via the OPTIMADE API from Materials Cloud.")
    _count_badge(len(options), len(raw), "MC3D", cs_filter)
    if not options:
        st.info("No structures match the selected crystal system filter.")
        return

    selected  = st.selectbox("Select a structure:", options, key="mc3d_sel_disp")
    mc3d_id   = selected.split(",")[-1].strip()
    structure = structs.get(mc3d_id)

    if structure is None:
        st.warning("Structure data not found in session – please search again.")
        return

    formula = structure.composition.reduced_formula
    _render_structure_card(
        structure, "MC3D",
        title   = f"{formula} · {mc3d_id}",
        link_md = f"[Open on Materials Cloud ↗](https://mc3d.materialscloud.org/#/details/{mc3d_id}/pbesol-v2)",
    )

    file_name = re.sub(r'[\\/:"*?<>|]+', '_', f"{mc3d_id}_{formula}.cif")
    col_add, col_dl, _ = st.columns([1, 1, 2])
    with col_add:
        if st.button("➕ Add to workspace", key="mc3d_add", use_container_width=True):
            st.session_state.full_structures[file_name] = structure
            _push_cif_to_uploads(file_name, str(CifWriter(structure, symprec=0.01)))
            check_structure_size_and_warn(structure, file_name)
            st.success("✅ Added from MC3D!")
    with col_dl:
        st.download_button(
            "💾 Download CIF", type="primary",
            data=str(CifWriter(structure, symprec=0.01)),
            file_name=file_name, mime="chemical/x-cif",
            key="mc3d_dl", use_container_width=True,
        )



def _count_badge(n_filtered: int, n_total: int, db_name: str, cs_filter: str) -> None:
    color  = _DB_COLORS.get(db_name, "#1e3a8a")
    icon   = _DB_ICONS.get(db_name, "")
    label  = f"{n_filtered} of {n_total}" if cs_filter != "All" else str(n_total)
    suffix = f" \u2014 filtered to {cs_filter}" if cs_filter != "All" else ""
    plural = "s" if n_total != 1 else ""
    st.markdown(
        f'<div style="display:inline-block;background:{color};color:white;font-size:0.88rem;font-weight:600;padding:3px 14px;border-radius:20px;margin-bottom:8px;">'
        f'{icon} {label} structure{plural} in {db_name}{suffix}</div>',
        unsafe_allow_html=True,
    )


def _push_cif_to_uploads(file_name: str, cif_text: str) -> None:
    buf = io.BytesIO(cif_text.encode("utf-8"))
    buf.name = file_name
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if all(f.name != file_name for f in st.session_state.uploaded_files):
        st.session_state.uploaded_files.append(buf)


def _push_bytes_to_uploads(file_name: str, data: bytes) -> None:
    buf = io.BytesIO(data)
    buf.name = file_name
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if all(f.name != file_name for f in st.session_state.uploaded_files):
        st.session_state.uploaded_files.append(buf)


_RENDERERS = {
    "Materials Project": _render_mp,
    "AFLOW":             _render_aflow,
    "COD":               _render_cod,
    "MC3D":              _render_mc3d,
}


def show_database_results() -> None:
    active = [db for db, key in _DB_KEYS.items()
              if st.session_state.get(key)]

    if not active:
        return

    all_options: list[str] = []
    for db in active:
        all_options.extend(st.session_state.get(_DB_KEYS[db], []))

    st.markdown(
        '<hr style="border:none;height:4px;background:linear-gradient(to right,#1565C0,#E65100,#2E7D32,#6A1B9A);border-radius:4px;margin:22px 0 16px 0;">',
        unsafe_allow_html=True,
    )

    st.markdown("### 🗂️ Retrieved Structures")

    badges_html = "&ensp;".join(
        f'<span style="background:{_DB_COLORS[db]};color:white;font-size:0.8rem;font-weight:600;padding:2px 10px;border-radius:12px;">'
        f'{_DB_ICONS[db]} {db}: {len(st.session_state.get(_DB_KEYS[db], []))}</span>'
        for db in active
    )
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    ctrl_sort, ctrl_cs, ctrl_info = st.columns([2, 2, 3])

    with ctrl_sort:
        sort_by = st.selectbox(
            "📑 Sort results by",
            options=_SORT_OPTIONS,
            index=0,
            key="db_results_sort_by",
            help=(
                "Applies to all database tabs simultaneously.\n\n"
                "Note: Volume and atom-count sorting is unavailable for AFLOW "
                "entries because that info is not included in the AFLOW API response."
            ),
        )

    with ctrl_cs:
        cs_filter = st.selectbox(
            "🔷 Filter by crystal system",
            options=["All"] + _CRYSTAL_SYSTEMS,
            index=0,
            key="db_results_cs_filter",
            help=(
                "Narrows every database tab to the chosen crystal system.\n"
                "Derived from the space-group number already stored in each "
                "option string — no extra API call needed."
            ),
        )

    with ctrl_info:
        if cs_filter != "All":
            n_match = sum(
                1 for o in all_options
                if _crystal_system_from_option(o) == cs_filter
            )
            cs_col  = _CRYSTAL_SYSTEM_COLORS.get(cs_filter, "#555")
            cs_ico  = _CRYSTAL_SYSTEM_ICONS.get(cs_filter, "")
            pct     = n_match / len(all_options) * 100 if all_options else 0
            st.markdown(
                f'<div style="background:{cs_col}18;border:1.5px solid {cs_col};border-radius:10px;padding:10px 14px;margin-top:22px;">'
                f'<span style="font-size:1.0rem;font-weight:700;color:{cs_col};">{cs_ico} {cs_filter}</span><br>'
                f'<span style="font-size:0.9rem;color:#333;"><b>{n_match}</b> of <b>{len(all_options)}</b> total structures ({pct:.0f}%) match.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    auto_open = (sort_by == "Crystal System" or cs_filter != "All")
    show_dist = st.checkbox(
        "📊 Show crystal system distribution across all retrieved structures",
        value=auto_open,
        key="db_show_cs_distribution",
    )
    if show_dist:
        with st.container():
            _render_crystal_system_distribution(all_options)

    tabs = st.tabs([f"{_DB_ICONS[db]} {db}" for db in active])
    for tab, db_name in zip(tabs, active):
        with tab:
            _RENDERERS[db_name](sort_by, cs_filter)
