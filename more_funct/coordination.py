"""Coordination polyhedra: which ideal shape surrounds a chosen element.

For every site of a chosen *central* element, the neighbouring sites of a
chosen *ligand* element (or of any element) are found — periodic images
included — and the environment is compared with ideal reference shapes
(octahedron, tetrahedron, square plane, trigonal bipyramid, cuboctahedron, …)
using pymatgen's ``LocalStructOrderParams``. Each shape gets a score between 0
and 1 (1 = ideal), the best one is reported, together with the bond lengths and
the usual distortion measures:

* bond-length distortion index  Δ = 1/n Σ |dᵢ − ⟨d⟩| / ⟨d⟩          (Baur 1974)
* quadratic elongation ⟨λ⟩ = 1/n Σ (dᵢ / d₀)², d₀ = centre–vertex distance of
  the regular polyhedron of the same volume               (Robinson et al. 1971)
* bond-angle variance σ² = Σ (θᵢ − θ₀)² / (m − 1)          (Robinson et al. 1971)

(⟨λ⟩ and σ² are only defined for octahedra and tetrahedra.)

The geometry of each polyhedron (convex hull of the ligands, split into
triangles with outward normals, plus its outline edges) is computed here as
well, so the viewer only has to translate it to every place the centre is
drawn.
"""

from collections import Counter
from functools import lru_cache

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams, cn_opt_params

NEIGHBOR_METHODS = ("Nearest shell", "Fixed cutoff", "Automatic (CrystalNN)")
ANY_ELEMENT = "Any element"
MAX_ANALYZED_CENTERS = 1000
DEFAULT_SHELL_WINDOW = 1.5   # nearest shell: candidates up to 1.5 × shortest distance
SHELL_GAP = 1.12             # … cut at the largest jump in distance, if ≥ 12 %

_SHAPE_RENAME = {"body-centered cubic": "cubic"}
_TET_ANGLE = 109.4712206


# --------------------------------------------------------------------------
# reference shapes
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _order_params_for_cn(cn):
    """(shape names, LocalStructOrderParams) for a coordination number."""
    shapes = cn_opt_params.get(cn)
    if not shapes:
        return None
    names, types, params = [], [], []
    for name, spec in shapes.items():
        names.append(_SHAPE_RENAME.get(name, name))
        types.append(spec[0])
        params.append(dict(spec[1]) if len(spec) > 1 and spec[1] else None)
    return tuple(names), LocalStructOrderParams(types, parameters=params)


def _shape_scores(vectors):
    """{shape name: score} of the neighbour vectors around a centre."""
    cn = len(vectors)
    entry = _order_params_for_cn(cn)
    if entry is None:
        return {}
    names, lsop = entry
    box = 60.0 + 2.0 * float(np.max(np.abs(vectors)))
    centre = np.full(3, box / 2.0)
    mol = Structure(Lattice.cubic(box), ["X0+"] + ["X0+"] * cn,
                    [centre] + [centre + v for v in vectors],
                    coords_are_cartesian=True)
    try:
        values = lsop.get_order_parameters(mol, 0, indices_neighs=list(range(1, cn + 1)))
    except Exception:
        return {}
    return {n: float(v) for n, v in zip(names, values) if v is not None and np.isfinite(v)}


# --------------------------------------------------------------------------
# polyhedron geometry
# --------------------------------------------------------------------------
def _planar_mesh(points):
    """Flat polygon (coplanar ligands): fan triangles, drawn from both sides."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[2]
    u, w = vt[0], vt[1]
    rel = points - centroid
    order = np.argsort(np.arctan2(rel @ w, rel @ u))
    ring = points[order]
    tris, edges = [], []
    for k in range(len(ring)):
        edges.append((ring[k], ring[(k + 1) % len(ring)]))
    for k in range(1, len(ring) - 1):
        a, b, c = ring[0], ring[k], ring[k + 1]
        tris.append((a, b, c, normal))
        tris.append((a, c, b, -normal))
    return tris, edges


def polyhedron_mesh(vectors):
    """Triangles ``[(a, b, c, outward normal)]`` and outline edges ``[(p, q)]``
    of the convex hull of the ligand vectors (relative to the centre).
    Returns ([], []) for fewer than three ligands."""
    pts = np.asarray(vectors, dtype=float)
    if len(pts) < 3:
        return [], []
    if len(pts) == 3:
        return _planar_mesh(pts)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # all ligands in one plane (square planar, trigonal planar, …)
        return _planar_mesh(pts)
    tris = []
    edge_faces = {}
    for f_idx, (simplex, eq) in enumerate(zip(hull.simplices, hull.equations)):
        n = eq[:3]
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        if np.dot(np.cross(b - a, c - a), n) < 0:
            b, c = c, b
        tris.append((a, b, c, n))
        for i, j in ((0, 1), (1, 2), (0, 2)):
            key = tuple(sorted((int(simplex[i]), int(simplex[j]))))
            edge_faces.setdefault(key, []).append(n)
    edges = []
    for (i, j), normals in edge_faces.items():
        # an edge between two coplanar triangles is only a diagonal of a
        # flat face (e.g. the squares of a cuboctahedron) — not drawn
        if len(normals) == 2 and np.dot(normals[0], normals[1]) > 0.999:
            continue
        edges.append((pts[i], pts[j]))
    return tris, edges


def _hull_volume(pts):
    if len(pts) < 4:
        return None
    from scipy.spatial import ConvexHull
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return None


def _angles(vectors):
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    out = []
    for i in range(len(unit)):
        for j in range(i + 1, len(unit)):
            out.append(np.degrees(np.arccos(np.clip(unit[i] @ unit[j], -1.0, 1.0))))
    return np.array(out)


def _distortion(vectors, shape):
    """Volume, quadratic elongation and bond-angle variance."""
    d = np.linalg.norm(vectors, axis=1)
    vol = _hull_volume(vectors)
    quad_el = ang_var = None
    if shape == "octahedral" and len(d) == 6 and vol:
        d0 = (3.0 * vol / 4.0) ** (1.0 / 3.0)
        quad_el = float(np.mean((d / d0) ** 2))
        cis = np.sort(_angles(vectors))[:12]          # the 3 trans angles ≈ 180° left out
        ang_var = float(np.sum((cis - 90.0) ** 2) / 11.0)
    elif shape == "tetrahedral" and len(d) == 4 and vol:
        d0 = (9.0 * np.sqrt(3.0) * vol / 8.0) ** (1.0 / 3.0)
        quad_el = float(np.mean((d / d0) ** 2))
        ang_var = float(np.sum((_angles(vectors) - _TET_ANGLE) ** 2) / 5.0)
    return vol, quad_el, ang_var


# --------------------------------------------------------------------------
# neighbour search
# --------------------------------------------------------------------------
def _neighbor_vectors(structure, idx, ligand, method, value, cnn=None):
    """Cartesian vectors centre → ligand (periodic images included) and the
    ligand elements."""
    site = structure[idx]

    def _ok(el):
        return ligand == ANY_ELEMENT or el == ligand

    if method == "Automatic (CrystalNN)":
        import warnings
        with warnings.catch_warnings():
            # "no oxidation states" / "covalent radii will be used" notes
            warnings.simplefilter("ignore")
            info = cnn.get_nn_info(structure, idx)
        out = [(np.asarray(n["site"].coords) - site.coords, n["site"].specie.symbol)
               for n in info if _ok(n["site"].specie.symbol)]
    else:
        r = float(value) if method == "Fixed cutoff" else 4.0
        neigh = []
        while True:
            neigh = [n for n in structure.get_neighbors(site, r)
                     if _ok(n.specie.symbol) and n.nn_distance > 0.1]
            if neigh or method == "Fixed cutoff" or r >= 8.0:
                break
            r += 2.0
        if method == "Nearest shell" and neigh:
            # candidates up to `value` × the shortest distance; the shell
            # ends at the largest jump in the sorted distances (≥ 12 %), so a
            # distorted polyhedron keeps its long bonds while the next shell
            # is left out
            d_min = min(n.nn_distance for n in neigh)
            neigh = sorted((n for n in neigh if n.nn_distance <= d_min * float(value) + 1e-6),
                           key=lambda n: n.nn_distance)
            d = np.array([n.nn_distance for n in neigh])
            if len(d) > 1:
                ratios = d[1:] / d[:-1]
                k = int(np.argmax(ratios))
                if ratios[k] > SHELL_GAP:
                    neigh = neigh[:k + 1]
        out = [(np.asarray(n.coords) - site.coords, n.specie.symbol) for n in neigh]
    if not out:
        return np.zeros((0, 3)), []
    vecs, els = zip(*out)
    return np.array(vecs, dtype=float), list(els)


def _is_centrosymmetric(vectors, tol=0.35, max_unpaired=3):
    """True when (nearly) every neighbour vector has an opposite partner −v.

    In the anticuboctahedron (hcp) 6 of the 12 neighbours miss their partner
    by ~0.58 × the bond length, while thermal noise in a cuboctahedron stays
    below ~0.2 ×; a few unpaired neighbours are tolerated.
    """
    unpaired = sum(
        np.min(np.linalg.norm(vectors + v, axis=1)) > tol * np.linalg.norm(v)
        for v in vectors)
    return unpaired <= max_unpaired


def _equivalence(structure):
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return list(SpacegroupAnalyzer(structure, symprec=0.1)
                    .get_symmetry_dataset().equivalent_atoms)
    except Exception:
        return list(range(len(structure)))


def _ligand_summary(dists):
    """'2×1.952, 4×2.010' — bond lengths grouped to 0.001 Å."""
    groups = Counter(f"{d:.3f}" for d in sorted(dists))
    return ", ".join(f"{n}×{d}" if n > 1 else d for d, n in groups.items())


def analyze_coordination(sites, lattice_matrix, center, ligand,
                         method="Nearest shell", value=DEFAULT_SHELL_WINDOW,
                         max_centers=MAX_ANALYZED_CENTERS):
    """Coordination environments of all ``center`` sites.

    ``sites`` is a tuple of (element, fx, fy, fz) — one entry per site, with
    the dominant element of a disordered site. Returns a dict with
    ``envs`` (one per centre site: frac, cn, shape, score, scores, vectors,
    mesh, distances, group, …) and ``skipped`` (centres beyond the limit).
    """
    lattice = Lattice(np.asarray(lattice_matrix, dtype=float))
    structure = Structure(lattice, [s[0] for s in sites], [s[1:] for s in sites])
    centers = [i for i, s in enumerate(sites) if s[0] == center]
    skipped = max(0, len(centers) - max_centers)
    centers = centers[:max_centers]
    equiv = _equivalence(structure)
    cnn = CrystalNN() if method == "Automatic (CrystalNN)" else None

    envs = []
    for idx in centers:
        vecs, els = _neighbor_vectors(structure, idx, ligand, method, value, cnn)
        d = np.linalg.norm(vecs, axis=1) if len(vecs) else np.zeros(0)
        scores = _shape_scores(vecs) if len(vecs) >= 2 else {}
        if scores:
            shape = max(scores, key=scores.get)
            score = scores[shape]
            # pymatgen's cuboctahedral parameter does not tell the
            # cuboctahedron (fcc) from the anticuboctahedron (hcp); only the
            # former has a centre of symmetry
            if shape == "cuboctahedral" and not _is_centrosymmetric(vecs):
                shape = "anticuboctahedral (hcp-like)"
        else:
            shape, score = (f"CN {len(vecs)} (no reference shape)" if len(vecs) else "no neighbours"), None
        vol, quad_el, ang_var = _distortion(vecs, shape) if len(vecs) else (None, None, None)
        mean_d = float(d.mean()) if len(d) else None
        envs.append({
            "site_idx": idx,
            "frac": tuple(float(x) for x in sites[idx][1:]),
            "group": int(equiv[idx]),
            "cn": int(len(vecs)),
            "ligand_elements": els,
            "shape": shape, "score": score, "scores": scores,
            "vectors": vecs,
            "mesh": polyhedron_mesh(vecs),
            "distances": d,
            "mean_d": mean_d,
            "bl_distortion": (float(np.mean(np.abs(d - mean_d)) / mean_d) if mean_d else None),
            "volume": vol, "quad_elongation": quad_el, "angle_variance": ang_var,
        })
    return {"envs": envs, "skipped": skipped}


def summary_rows(envs, center):
    """Table rows, one per group of symmetry-equivalent centres."""
    groups = {}
    for env in envs:
        groups.setdefault(env["group"], []).append(env)
    rows = []
    for k, (_, members) in enumerate(groups.items(), start=1):
        e = members[0]
        second = None
        if e["scores"] and len(e["scores"]) > 1:
            ordered = sorted(e["scores"].items(), key=lambda kv: kv[1], reverse=True)
            second = f"{ordered[1][0]} ({ordered[1][1]:.2f})"
        rows.append({
            "Site": f"{center}{k}",
            "Count": len(members),
            "CN": e["cn"],
            "Geometry": e["shape"],
            "Match (0–1)": None if e["score"] is None else round(e["score"], 3),
            "Next best": second or "–",
            "⟨d⟩ (Å)": None if e["mean_d"] is None else round(e["mean_d"], 4),
            "Bond lengths (Å)": _ligand_summary(e["distances"]) if len(e["distances"]) else "–",
            "Δ (distortion index)": None if e["bl_distortion"] is None else round(e["bl_distortion"], 5),
            "Volume (Å³)": None if e["volume"] is None else round(e["volume"], 3),
            "⟨λ⟩ quad. elongation": None if e["quad_elongation"] is None else round(e["quad_elongation"], 5),
            "σ² angle variance (deg²)": None if e["angle_variance"] is None else round(e["angle_variance"], 3),
        })
    return rows
