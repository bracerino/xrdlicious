import numpy as np
import streamlit as st
import py3Dmol


def calculate_hkl_normal_and_rotation(lattice_matrix, h, k, l):
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

    """
    Calculate the normal vector to the (h k l) plane and determine rotation angles
    to orient the view perpendicular to this plane.

    Parameters:
    -----------
    lattice_matrix : numpy.ndarray
        3x3 matrix where rows are lattice vectors [a, b, c]
    h, k, l : int
        Miller indices of the crystallographic plane

    Returns:
    --------
    tuple: (normal_vector, rotation_x, rotation_y, rotation_z)
        normal_vector: normalized normal to the (h k l) plane
        rotation_x, rotation_y, rotation_z: rotation angles in degrees
    """

    # Get reciprocal lattice vectors
    # reciprocal lattice matrix = 2π * (real lattice matrix)^(-T)
    reciprocal_matrix = 2 * np.pi * np.linalg.inv(lattice_matrix).T

    # Calculate normal vector in reciprocal space
    # Normal to (h k l) plane = h*a* + k*b* + l*c*
    normal_reciprocal = h * reciprocal_matrix[0] + k * reciprocal_matrix[1] + l * reciprocal_matrix[2]

    # Normalize the normal vector
    normal_length = np.linalg.norm(normal_reciprocal)
    if normal_length == 0:
        raise ValueError("Invalid Miller indices: (0 0 0) plane does not exist")

    normal_vector = normal_reciprocal / normal_length

    # Calculate rotation angles to align normal with z-axis
    # We want to rotate so that the normal vector points along +z direction

    # Current normal vector components
    nx, ny, nz = normal_vector

    # Calculate rotation angles
    # First rotate around z-axis to align projection in xy-plane with x-axis
    rotation_z = -np.degrees(np.arctan2(ny, nx))

    # Then rotate around y-axis to align with z-axis
    xy_magnitude = np.sqrt(nx ** 2 + ny ** 2)
    rotation_y = np.degrees(np.arctan2(xy_magnitude, nz))

    # No rotation around x-axis needed for this approach
    rotation_x = 0

    return normal_vector, rotation_x, rotation_y, rotation_z


def add_hkl_plane_controls(key_suffix=""):


    enable_hkl = st.checkbox(
        "🔄 Orient view perpendicular to crystallographic plane",
        value=False,
        key=f"enable_hkl_{key_suffix}"
    )

    if not enable_hkl:
        return None

    # Create input controls for Miller indices
    col_h, col_k, col_l, col_apply = st.columns([1, 1, 1, 2])

    with col_h:
        h = st.number_input(
            "h",
            value=1,
            step=1,
            format="%d",
            key=f"hkl_h_{key_suffix}"
        )

    with col_k:
        k = st.number_input(
            "k",
            value=0,
            step=1,
            format="%d",
            key=f"hkl_k_{key_suffix}"
        )

    with col_l:
        l = st.number_input(
            "l",
            value=0,
            step=1,
            format="%d",
            key=f"hkl_l_{key_suffix}"
        )

    #with col_apply:
    apply_orientation = st.button(
        f"Apply ({h} {k} {l}) orientation",
        help="Reorient the structure to view the specified crystallographic plane perpendicular to the screen",
        key=f"apply_hkl_{key_suffix}", type ='primary'
    )

    if h == 0 and k == 0 and l == 0:
        st.warning("⚠️ Invalid Miller indices: (0 0 0) plane does not exist")
        return None

    st.caption("Note: This only affects the 3D preview, not the calculator")

    col_x, col_y, col_z = st.columns(3)

    with col_x:
        supercell_x = st.number_input(
            "Repeat X",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Number of unit cells to repeat along X direction (1-5)",
            key=f"supercell_x_{key_suffix}"
        )

    with col_y:
        supercell_y = st.number_input(
            "Repeat Y",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Number of unit cells to repeat along Y direction (1-5)",
            key=f"supercell_y_{key_suffix}"
        )

    with col_z:
        supercell_z = st.number_input(
            "Repeat Z",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Number of unit cells to repeat along Z direction (1-5)",
            key=f"supercell_z_{key_suffix}"
        )

    return h, k, l, apply_orientation, supercell_x, supercell_y, supercell_z


def apply_hkl_orientation_to_py3dmol(view, lattice_matrix, h, k, l, supercell_x=1, supercell_y=1, supercell_z=1):
    try:
        normal_vector, rot_x, rot_y, rot_z = calculate_hkl_normal_and_rotation(lattice_matrix, h, k, l)

        view.zoomTo()

        if abs(rot_z) > 0.1:
            view.rotate(rot_z, 'z')
        if abs(rot_y) > 0.1:
            view.rotate(rot_y, 'y')
        if abs(rot_x) > 0.1:
            view.rotate(rot_x, 'x')

        total_cells = supercell_x * supercell_y * supercell_z
        if total_cells == 1:
            zoom_factor = 1.1
        elif total_cells <= 8:
            zoom_factor = 0.9
        elif total_cells <= 27:
            zoom_factor = 0.8
        else:
            zoom_factor = 0.7

        view.zoom(zoom_factor)

        return True, f"✅ View oriented perpendicular to ({h} {k} {l}) plane"

    except Exception as e:
        return False, f"❌ Error orienting view: {str(e)}"


def get_hkl_plane_info(lattice_matrix, h, k, l):
    try:
        reciprocal_matrix = 2 * np.pi * np.linalg.inv(lattice_matrix).T
        normal_reciprocal = h * reciprocal_matrix[0] + k * reciprocal_matrix[1] + l * reciprocal_matrix[2]
        d_spacing = 2 * np.pi / np.linalg.norm(normal_reciprocal)

        normal_vector, _, _, _ = calculate_hkl_normal_and_rotation(lattice_matrix, h, k, l)

        return {
            "d_spacing": d_spacing,
            "normal_vector": normal_vector,
            "plane_notation": f"({h} {k} {l})",
            "success": True
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def create_supercell_xyz_for_visualization(df_for_viz, lattice_matrix, supercell_x, supercell_y, supercell_z):

    supercell_atoms = []
    for i in range(supercell_x):
        for j in range(supercell_y):
            for k in range(supercell_z):
                translation = i * lattice_matrix[0] + j * lattice_matrix[1] + k * lattice_matrix[2]

                for _, row in df_for_viz.iterrows():
                    element = row['Element']
                    original_pos = np.array([row['X'], row['Y'], row['Z']])
                    new_pos = original_pos + translation

                    supercell_atoms.append({
                        'element': element,
                        'x': new_pos[0],
                        'y': new_pos[1],
                        'z': new_pos[2]
                    })

    xyz_lines = [str(len(supercell_atoms))]
    xyz_lines.append(f"Supercell {supercell_x}x{supercell_y}x{supercell_z} visualization")

    for atom in supercell_atoms:
        xyz_lines.append(f"{atom['element']} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}")

    return "\n".join(xyz_lines)


def add_supercell_unit_cell_boxes(view, lattice_matrix, supercell_x, supercell_y, supercell_z):

    def add_translated_box(view, lattice_matrix, translation, color='gray', linewidth=1):
        a, b, c = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]
        corners = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    corner = translation + i * a + j * b + k * c
                    corners.append(corner)

        edges = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    start_coord = np.array([i, j, k])
                    start_point = translation + i * a + j * b + k * c

                    for axis in range(3):
                        if start_coord[axis] == 0:
                            neighbor = start_coord.copy()
                            neighbor[axis] = 1
                            end_point = translation + neighbor[0] * a + neighbor[1] * b + neighbor[2] * c
                            edges.append((start_point, end_point))

        for start, end in edges:
            view.addCylinder({
                'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                'color': color,
                'radius': 0.02,
                'opacity': 0.6
            })

    for i in range(supercell_x):
        for j in range(supercell_y):
            for k in range(supercell_z):
                translation = i * lattice_matrix[0] + j * lattice_matrix[1] + k * lattice_matrix[2]

                if i == 0 and j == 0 and k == 0:
                    color = 'black'
                    linewidth = 2
                else:
                    color = 'gray'
                    linewidth = 1

                add_translated_box(view, lattice_matrix, translation, color, linewidth)

    hkl_result = add_hkl_plane_controls()

    if hkl_result is not None:
        h, k, l, apply_orientation = hkl_result
        plane_info = get_hkl_plane_info(visual_pmg_structure.lattice.matrix, h, k, l)

        if plane_info["success"]:
            st.write(f"**d-spacing:** {plane_info['d_spacing']:.4f} Å")
        if apply_orientation:
            success, message = apply_hkl_orientation_to_py3dmol(
                view,
                visual_pmg_structure.lattice.matrix,
                h, k, l
            )

            if success:
                st.success(message)
            else:
                st.error(message)

    return view
