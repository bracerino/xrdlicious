"""Loading a limited number of frames from (ext)XYZ trajectories.

An uploaded ``.xyz`` / ``.extxyz`` file may hold a whole MD trajectory with
hundreds or thousands of frames. Turning every frame into a structure would
flood the app (and the memory of the online server), so only a few frames are
picked — evenly spaced by default — and each one is handed to the rest of the
app as if it had been uploaded as its own single-frame file
(``<name>_frame<i>.xyz``). From that point on the frames go through the
ordinary code paths: they can be visualised, compared, removed, etc.

The file is only scanned for frame boundaries (no parsing of the atoms), and
the result is cached in session state, so large trajectories are not re-read on
every rerun of the script.
"""

import io
import os

import numpy as np
import streamlit as st

TRAJECTORY_EXTENSIONS = (".xyz", ".extxyz")
DEFAULT_MAX_FRAMES = 10
MAX_FRAMES_ONLINE = 20
MAX_FRAMES_LOCAL = 1000

FRAME_SELECTION_MODES = ("Evenly spaced", "Last frames", "First frames")

_SCAN_CACHE_KEY = "_trajectory_scan_cache"


def _frame_offsets(raw):
    """Byte offsets ``[(start, end), ...]`` of the frames of an XYZ file.

    Each XYZ frame is: a line with the number of atoms, a comment line (the
    extxyz header with Lattice=..., Properties=...), then one line per atom.
    Returns None when the content does not follow that layout.
    """
    lines = raw.splitlines(keepends=True)
    offsets, pos, i, n_lines = [], 0, 0, len(lines)
    while i < n_lines:
        first = lines[i].strip()
        if not first:                      # blank lines between/after frames
            pos += len(lines[i])
            i += 1
            continue
        try:
            n_atoms = int(first.split()[0])
        except ValueError:
            return None
        if n_atoms < 0 or i + n_atoms + 2 > n_lines:
            return None
        start = pos
        pos += sum(len(line) for line in lines[i:i + n_atoms + 2])
        offsets.append((start, pos))
        i += n_atoms + 2
    return offsets or None


def select_frame_indices(n_total, max_frames, mode="Evenly spaced"):
    """Indices of the frames to load (always sorted, without duplicates)."""
    max_frames = max(1, int(max_frames))
    if n_total <= max_frames:
        return list(range(n_total))
    if mode == "Last frames":
        return list(range(n_total - max_frames, n_total))
    if mode == "First frames":
        return list(range(max_frames))
    # Evenly spaced, always including the first and the last frame
    idx = np.unique(np.round(np.linspace(0, n_total - 1, max_frames)).astype(int))
    return [int(i) for i in idx]


def _is_trajectory_candidate(file):
    return os.path.splitext(file.name)[1].lower() in TRAJECTORY_EXTENSIONS


def _scan(file):
    """(raw bytes, frame offsets or None), cached per uploaded file."""
    cache = st.session_state.setdefault(_SCAN_CACHE_KEY, {})
    raw = file.getvalue()
    key = (file.name, len(raw))
    if key not in cache:
        cache[key] = _frame_offsets(raw)
    return raw, cache[key]


def _frame_file(raw, span, name):
    """A single-frame file object that behaves like st.file_uploader's."""
    buffer = io.BytesIO(raw[span[0]:span[1]])
    buffer.name = name
    return buffer


def frame_file_name(file_name, index, n_total):
    stem = os.path.splitext(file_name)[0]
    width = max(1, len(str(n_total - 1)))
    return f"{stem}_frame{index:0{width}d}.xyz"


def expand_trajectories(files, container=None, is_local=False,
                        key_prefix="traj"):
    """Replace multi-frame XYZ files by a limited set of single-frame files.

    Files that are not trajectories (CIF, POSCAR, single-frame XYZ, ...) are
    passed through unchanged. When at least one trajectory is present, the
    frame-selection controls are drawn into ``container`` (the sidebar).
    """
    if not files:
        return files
    target = st.sidebar if container is None else container

    scanned = []
    for f in files:
        if _is_trajectory_candidate(f):
            try:
                raw, offsets = _scan(f)
            except Exception:
                raw, offsets = None, None
            if offsets and len(offsets) > 1:
                scanned.append((f, raw, offsets))
                continue
        scanned.append((f, None, None))

    trajectories = [s for s in scanned if s[2] is not None]
    if not trajectories:
        return files

    limit = MAX_FRAMES_LOCAL if is_local else MAX_FRAMES_ONLINE
    box = target.container(key=f"{key_prefix}_settings_box")
    box.markdown("**🎞️ Trajectory frames**")
    c1, c2 = box.columns([1, 1.4])
    max_frames = int(c1.number_input(
        "Max. frames",
        min_value=1, max_value=limit,
        value=min(DEFAULT_MAX_FRAMES, limit), step=1,
        key=f"{key_prefix}_max_frames",
        help=(
            "Only this many frames are loaded from each uploaded XYZ/extXYZ "
            "trajectory, so that large trajectories do not overload the app. "
            f"Up to {limit} frames per trajectory"
            + ("." if is_local else " on the online version (run XRDlicious "
               f"locally for up to {MAX_FRAMES_LOCAL}).")
        ),
    ))
    mode = c2.selectbox(
        "Which frames",
        FRAME_SELECTION_MODES,
        key=f"{key_prefix}_frame_mode",
        help="Evenly spaced frames always include the first and the last one.",
    )

    expanded = []
    for f, raw, offsets in scanned:
        if offsets is None:
            expanded.append(f)
            continue
        n_total = len(offsets)
        chosen = select_frame_indices(n_total, max_frames, mode)
        expanded.extend(
            _frame_file(raw, offsets[i], frame_file_name(f.name, i, n_total))
            for i in chosen
        )
        if len(chosen) <= 8:
            shown = ", ".join(str(i) for i in chosen)
        else:
            shown = (", ".join(str(i) for i in chosen[:4]) + ", …, "
                     + ", ".join(str(i) for i in chosen[-2:]))
        box.caption(f"**{f.name}**: {len(chosen)} of {n_total} frames "
                    f"loaded (frames {shown}; numbered from 0).")
    return expanded
