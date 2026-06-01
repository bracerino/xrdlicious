import io

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from helpers import auto_normalize_and_stack_plots, reset_layout


def _displacement_shift(two_theta_deg, displacement_mm, radius_mm):
    """Peak shift (in degrees 2θ) caused by sample displacement.

    Δ2θ = -2·s·cos(θ)/R, where s is the sample displacement from the
    focusing circle and R is the goniometer radius (both in mm).
    """
    theta_rad = np.deg2rad(np.asarray(two_theta_deg, dtype=float) / 2.0)
    return np.rad2deg(-2.0 * displacement_mm * np.cos(theta_rad) / radius_mm)


def render_interactive_data_plot(user_pattern_file):
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'grey']

    st.markdown(
        "#### 📂 Upload your two-column data files in the sidebar to see them in an interactive plot. Multiple files are supported, and your columns can be separated by spaces, tabs, commas, or semicolons."
    )

    colss, colzz, colx, colc, cold = st.columns([1, 1, 1, 1, 1])
    has_header = colss.checkbox("Files contain a header row", value=False)
    skip_header = colzz.checkbox("Skip header row", value=True)
    normalized_intensity = colx.checkbox("Normalized intensity", value=False, key="normalized_intensity")
    x_axis_log = colc.checkbox("Logarithmic X-axis", value=False)
    y_axis_log = cold.checkbox("Logarithmic Y-axis", value=False)

    col_thick, col_size, col_fox, col_xmin, col_xmax, = st.columns([2, 1, 1, 1, 1])
    with col_thick:
        st.info(
            f"ℹ️ You can modify the **graph layout** from the sidebar.️ ℹ️ You can **convert** your **XRD** data below the plot. Enable 'Normalized intensity' to **automatically shift data files in vertical direction**.")
    fix_x_axis = col_fox.checkbox("Fix x-axis range?", value=False)
    if fix_x_axis == True:
        x_axis_min = col_xmin.number_input("X-axis Minimum", value=0.0)
        x_axis_max = col_xmax.number_input("X-axis Maximum", value=10.0)
    x_axis_metric = "X-data"
    y_axis_metric = "Y-data"
    if user_pattern_file:
        files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

        if has_header:
            try:
                sample_file = files[0]
                sample_file.seek(0)
                df_sample = pd.read_csv(
                    sample_file,
                    sep=r'\s+|,|;',
                    engine='python',
                    header=0
                )
                x_axis_metric = df_sample.columns[0]
                y_axis_metric = df_sample.columns[1]
            except Exception as e:
                st.error(f"Error reading header from file {sample_file.name}: {e}")
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"
        else:
            x_axis_metric = "X-data"
            y_axis_metric = "Y-data"

    # ── Stacking controls (always visible once files are uploaded) ──────────
    # Clicking "Stack Plots" automatically enables normalization first
    # (handled inside auto_normalize_and_stack_plots, which sets the
    # 'normalized_intensity' session state before the rerun).
    if user_pattern_file:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col3:
            offset_gap_value = st.number_input(
                "Stacking Gap",
                min_value=0.0,
                value=10.0,
                step=5.0,
                key='stack_offset_gap',
                help="The vertical space to add between stacked, normalized plots."
            )
        with col1:
            st.button(
                "✨ Stack Plots",
                type="primary",
                on_click=auto_normalize_and_stack_plots,
                args=(files, skip_header, has_header, offset_gap_value),
            )
        with col2:
            st.button("🔄 Reset Layout", on_click=reset_layout, args=(files,))

    # ── Sample displacement correction (shown above the plot, per dataset) ──
    # Shifts the 2θ x-values of the selected dataset(s) according to
    # Δ2θ = -2·s·cos(θ)/R (s = sample displacement mm, R = goniometer radius mm).
    displacement_settings = []
    show_original_displacement = False
    if user_pattern_file:
        files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]
        enable_displacement = st.checkbox(
            "⏳ Apply **sample displacement correction** (2θ peak shift)",
            value=False,
            key="enable_displacement",
            help="Shift the 2θ x-values of the selected dataset(s) using "
                 "Δ2θ = -2·s·cos(θ)/R, where s is the sample displacement from "
                 "the focusing circle and R is the goniometer radius."
        )
        if enable_displacement:
            with st.expander("⏳ Sample displacement correction settings", expanded=True):
                st.markdown(
                    "The shift is computed as $\\Delta 2\\theta = -\\frac{2 s \\cos\\theta}{R}$, "
                    "where **s** is the sample displacement from the focusing circle and "
                    "**R** is the goniometer radius. The shift is applied to the 2θ x-values."
                )
                show_original_displacement = st.checkbox(
                    "👁️ Also show the original (uncorrected) pattern",
                    value=False,
                    key="show_original_displacement",
                    help="Overlay the original, uncorrected pattern alongside the shifted one."
                )
                for i, file in enumerate(files):
                    disp_cols = st.columns([1.4, 1, 1])
                    apply_disp = disp_cols[0].checkbox(
                        f"Apply to {file.name}",
                        value=False,
                        key=f"apply_displacement_{i}"
                    )
                    displacement_mm = disp_cols[1].number_input(
                        "Displacement s (mm)",
                        value=0.0,
                        step=0.01,
                        format="%.3f",
                        key=f"displacement_mm_{i}"
                    )
                    goniometer_radius_mm = disp_cols[2].number_input(
                        "Goniometer radius R (mm)",
                        min_value=1.0,
                        value=250.0,
                        step=5.0,
                        format="%.1f",
                        key=f"goniometer_radius_mm_{i}"
                    )
                    displacement_settings.append({
                        "apply": apply_disp,
                        "displacement_mm": displacement_mm,
                        "goniometer_radius_mm": goniometer_radius_mm
                    })

    plot_placeholder = st.empty()
    st.sidebar.markdown("### Interactive Data Plot layout")
    customize_layout = st.sidebar.checkbox(f"Modify the **graph layout**", value=False)
    if customize_layout:
        col_line, col_marker = st.sidebar.columns(2)
        show_lines = col_line.checkbox("Show Lines", value=True, key="show_lines")
        show_markers = col_marker.checkbox("Show Markers", value=False, key="show_markers")

        col_thick, col_size = st.sidebar.columns(2)
        line_thickness = col_thick.number_input("Line Thickness", min_value=0.1, max_value=15.0, value=1.0,
                                                step=0.3,
                                                key="line_thickness2")
        marker_size = col_size.number_input("Marker Size", min_value=0.5, max_value=50.0, value=3.0,
                                            step=1.0,
                                            key="marker_size")

        col_title_font, col_axis_font, col_tick_font = st.sidebar.columns(3)
        title_font_size = col_title_font.number_input("Title Font Size", min_value=10, max_value=50,
                                                      value=36,
                                                      step=2,
                                                      key="title_font_size")
        axis_label_font_size = col_axis_font.number_input("Axis Label Font Size", min_value=10,
                                                          max_value=50,
                                                          value=36,
                                                          step=2, key="axis_font_size")
        tick_font_size = col_tick_font.number_input("Tick Label Font Size", min_value=8, max_value=40,
                                                    value=24,
                                                    step=2,
                                                    key="tick_font_size")

        col_leg_font, col_leg_pos = st.sidebar.columns(2)
        legend_font_size = col_leg_font.number_input("Legend Font Size", min_value=8, max_value=40,
                                                     value=28,
                                                     step=2,
                                                     key="legend_font_size")
        legend_position = col_leg_pos.selectbox(
            "Legend Position",
            options=["Top", "Bottom", "Left", "Right"],
            index=0,
            key="legend_position"
        )

        col_width, col_height = st.sidebar.columns(2)
        graph_width = col_width.number_input("Graph Width (pixels)", min_value=400, max_value=2000,
                                             value=1000,
                                             step=50,
                                             key="graph_width")
        graph_height = col_height.number_input("Graph Height (pixels)", min_value=300, max_value=1500,
                                               value=900,
                                               step=50, key="graph_height")

        st.sidebar.markdown("#### Custom Axis Labels")
        col_x_label, col_y_label = st.sidebar.columns(2)
        custom_x_label = col_x_label.text_input("X-axis Label", value=x_axis_metric, key="custom_x_label")
        custom_y_label = col_y_label.text_input("Y-axis FLabel", value=y_axis_metric, key="custom_y_label")

        if user_pattern_file:
            st.sidebar.markdown("#### Custom Series Names")
            series_names = {}

            if isinstance(user_pattern_file, list):
                for i, file in enumerate(user_pattern_file):
                    series_names[i] = st.sidebar.text_input(f"Label for {file.name}", value=file.name,
                                                            key=f"series_name_{i}")
            else:
                series_names[0] = st.sidebar.text_input(f"Label for {user_pattern_file.name}",
                                                        value=user_pattern_file.name,
                                                        key="series_name_0")
        if user_pattern_file:
            st.sidebar.markdown("#### Custom Series Colors")
            series_colors = {}
            files_for_color = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

            colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#000000', '#7f7f7f']
            for i, file in enumerate(files_for_color):
                default_color = colors[i % len(colors)]
                series_colors[i] = st.sidebar.color_picker(
                    f"Color for {file.name}",
                    value=default_color,
                    key=f"series_color_{i}"
                )
    else:
        series_colors = {}
        show_lines = True
        show_markers = False
        line_thickness = 1.0
        marker_size = 3.0
        title_font_size = 36
        axis_label_font_size = 36
        tick_font_size = 24
        legend_font_size = 28
        legend_position = "Top"
        graph_width = 1000
        graph_height = 900
        custom_x_label = x_axis_metric
        custom_y_label = y_axis_metric
        series_names = {}

    enable_conversion = st.checkbox(f"Enable powder **XRD data conversion**", value=False)

    if user_pattern_file:
        files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

        if has_header:
            try:
                sample_file = files[0]
                sample_file.seek(0)
                df_sample = pd.read_csv(
                    sample_file,
                    sep=r'\s+|,|;',
                    engine='python',
                    header=0
                )
                x_axis_metric = df_sample.columns[0]
                y_axis_metric = df_sample.columns[1]
            except Exception as e:
                st.error(f"Error reading header from file {sample_file.name}: {e}")
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"
        else:
            x_axis_metric = "X-data"
            y_axis_metric = "Y-data"

        file_conversion_settings = []

        if enable_conversion:
            with st.expander("❓ How does Divergence Slit Conversion work?", expanded=False):
                st.markdown("""
                ### Divergence Slit Conversion Explained

                #### Auto Slit
                - The slit **automatically adjusts** with angle (2θ) to **keep the irradiated area constant**.
                - Produces intensity that remains relatively **consistent** across angles.

                #### Fixed Slit
                - The slit has a **fixed opening angle**.
                - As 2θ increases, the **irradiated area is smaller**.
                - Results in **reduced intensity at higher angles**.

                #### Conversion Types

                - **Fixed Slit → Auto Slit**
                  Adjusts for loss of intensity at higher angles by simulating constant irradiated area:

                  $$
                  \\text{Intensity}_{\\text{auto}} = \\text{Intensity}_{\\text{fixed}} \\times \\frac{\\text{Irradiated Length} \\times \\sin(\\theta)}{\\text{Fixed Slit Size}}
                  $$

                - **Auto Slit → Fixed Slit**
                  Simulates reduced illuminated area at higher angles:

                  $$
                  \\text{Intensity}_{\\text{fixed}} = \\text{Intensity}_{\\text{auto}} \\times \\frac{\\text{Fixed Slit Size}}{\\text{Irradiated Length} \\times \\sin(\\theta)}
                  $$

                #### Parameters

                - **Fixed slit size (degrees)**: The opening angle of the slit in degrees.
                - **Irradiated sample length (mm)**: Physical length of sample that is illuminated.
                  - Reflection geometry: *10–20 mm*
                  - Transmission geometry: *1–2 mm*

                """)
            for i, file in enumerate(files):
                with st.expander(f"🔄 Conversion settings for **{file.name}**", expanded=(i == 0)):
                    wave_col, slit_col = st.columns(2)

                    with wave_col:
                        st.markdown("**Diffraction data conversion:**")
                        input_format = st.selectbox(
                            "Convert from:",
                            [
                                "No conversion",
                                "d-spacing (Å)",
                                "2theta (Copper CuKa1)",
                                "2theta (Cobalt CoKa1)",
                                "2theta (Custom)",
                                "q-vector (Å⁻¹)"
                            ],
                            key=f"input_format_{i}",
                            help=f"Copper (CuKa1): 1.5406 Å\n\n"
                                 " Molybdenum (MoKa1): 0.7093 Å\n\n"
                                 " Chromium (CrKa1): 2.2897 Å\n\n"
                                 " Iron (FeKa1): 1.9360 Å\n\n"
                                 " Cobalt (CoKa1): 1.7889 Å\n\n"
                                 " Silver (AgKa1): 0.5594 Å\n\n"
                                 " q-vector = 4π·sin(θ)/λ\n"
                        )

                        all_output_options = [
                            "No conversion",
                            "d-spacing (Å)",
                            "2theta (Copper CuKa1)",
                            "2theta (Cobalt CoKa1)",
                            "2theta (Custom)",
                            "q-vector (Å⁻¹)"
                        ]

                        if input_format != "No conversion":
                            filtered_options = all_output_options.copy()
                            if input_format in filtered_options and input_format != "2theta (Custom)":
                                filtered_options.remove(input_format)
                        else:
                            filtered_options = all_output_options

                        output_format = st.selectbox(
                            "Convert to:",
                            filtered_options,
                            index=0,
                            key=f"output_format_{i}"
                        )

                        input_custom_wavelength = None
                        output_custom_wavelength = None

                        if input_format == "2theta (Custom)":
                            input_custom_wavelength = st.number_input(
                                "Input custom wavelength (Å)",
                                min_value=0.1,
                                max_value=10.0,
                                value=1.54056,
                                step=0.01,
                                format="%.5f",
                                key=f"input_custom_wl_{i}"
                            )

                        if output_format == "2theta (Custom)":
                            output_custom_wavelength = st.number_input(
                                "Output custom wavelength (Å)",
                                min_value=0.1,
                                max_value=10.0,
                                value=1.54056 if input_custom_wavelength is None else input_custom_wavelength * 0.9,
                                step=0.01,
                                format="%.5f",
                                key=f"output_custom_wl_{i}"
                            )

                        if input_format == "q-vector (Å⁻¹)" and "2theta" in output_format and output_custom_wavelength is None:
                            if "Custom" in output_format:
                                output_custom_wavelength = st.number_input(
                                    "Output wavelength for q-vector to 2theta conversion (Å)",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=1.54056,
                                    step=0.01,
                                    format="%.5f",
                                    key=f"q_to_2theta_wl_{i}"
                                )

                        if "2theta" in input_format and output_format == "q-vector (Å⁻¹)" and input_custom_wavelength is None:
                            if "Custom" in input_format:
                                input_custom_wavelength = st.number_input(
                                    "Input wavelength for 2theta to q-vector conversion (Å)",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=1.54056,
                                    step=0.01,
                                    format="%.5f",
                                    key=f"2theta_to_q_wl_{i}"
                                )

                    with slit_col:

                        st.markdown("**Divergence slit conversion:**")
                        slit_conversion_type = st.selectbox(
                            "Convert slit type:",
                            [
                                "No conversion",
                                "Auto slit to fixed slit",
                                "Fixed slit to auto slit"
                            ],
                            key=f"slit_conv_{i}"
                        )
                        fixed_slit_size = None
                        irradiated_length = None
                        if slit_conversion_type != "No conversion":
                            fixed_slit_size = st.number_input(
                                "Fixed slit size (degrees)",
                                min_value=0.1,
                                max_value=2.0,
                                value=1.0,
                                step=0.1,
                                format="%.2f",
                                key=f"fixed_slit_{i}"
                            )
                            irradiated_length = st.number_input(
                                "Irradiated sample length (mm)",
                                min_value=1.0,
                                max_value=50.0,
                                value=10.0,
                                step=1.0,
                                key=f"irradiated_{i}"
                            )

                            st.info("Typical values: 10-20 mm for reflection, 1-2 mm for transmission geometry.")

                    file_conversion_settings.append({
                        "file_index": i,
                        "conversion_type": "No conversion" if input_format == "No conversion" or not output_format else f"{input_format} to {output_format}",
                        "input_format": input_format,
                        "output_format": output_format,
                        "input_custom_wavelength": input_custom_wavelength,
                        "output_custom_wavelength": output_custom_wavelength,
                        "slit_conversion_type": slit_conversion_type,
                        "fixed_slit_size": fixed_slit_size,
                        "irradiated_length": irradiated_length
                    })

        offset_cols = st.columns(len(files))
        y_offsets = []
        for i, file in enumerate(files):
            offset_val = offset_cols[i].number_input(
                f"Y offset for {file.name}",
                value=0.0,
                key=f"y_offset_{i}"
            )
            y_offsets.append(offset_val)
        scale_cols = st.columns(len(files))
        y_scales = []
        for i, file in enumerate(files):
            scale_val = scale_cols[i].number_input(
                f"Scale factor for {file.name}",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key=f"y_scale_{i}"
            )
            y_scales.append(scale_val)

        fig_interactive = go.Figure()
        # Maps each file index to the index of its main (corrected) trace in
        # fig_interactive.data, since optional "original" overlay traces may be
        # interleaved.
        main_trace_indices = {}
        for i, file in enumerate(files):
            try:
                file.seek(0)
                if has_header:
                    df = pd.read_csv(
                        file,
                        sep=r'\s+|,|;',
                        engine='python',
                        header=0
                    )
                else:
                    if skip_header:
                        file.seek(0)
                        try:
                            file_content = file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            file_content = file.read().decode('latin-1')

                        lines = file_content.splitlines()
                        comment_line_indices = [i for i, line in enumerate(lines) if line.strip().startswith('#')]
                        lines_to_skip = [0] + comment_line_indices
                        lines_to_skip = sorted(set(lines_to_skip))
                        file.seek(0)

                        df = pd.read_csv(
                            file,
                            sep=r'\s+|,|;',
                            engine='python',
                            header=None,
                            skiprows=lines_to_skip
                        )
                    else:
                        df = pd.read_csv(
                            file,
                            sep=r'\s+|,|;',
                            engine='python',
                            header=None
                        )
                    df.columns = [f"Column {j + 1}" for j in range(len(df.columns))]

                x_data = df.iloc[:, 0].values
                y_data = df.iloc[:, 1].values

                if st.session_state.get('auto_stack_enabled', False):
                    min_adjustments = st.session_state.get('min_adjustments', [])
                    if i < len(min_adjustments):
                        y_data = y_data - min_adjustments[i]

                if enable_conversion:
                    settings = file_conversion_settings[i]  # this must match the same index i as files[i]
                    conversion_type = settings.get("conversion_type", "No conversion")
                    if conversion_type == "No conversion":
                        pass
                    else:
                        def convert_data(x_values, conversion_type, input_custom_wavelength=None,
                                         output_custom_wavelength=None):

                            wavelength_map = {
                                "Copper": 1.54056,
                                "CuKa1": 1.54056,
                                "Cobalt": 1.78897,
                                "CoKa1": 1.78897
                            }

                            parts = conversion_type.split(" to ")
                            if len(parts) != 2:
                                print(f"Invalid conversion type format: {conversion_type}")
                                return x_values

                            input_format = parts[0].strip()
                            output_format = parts[1].strip()

                            lambda_in = None
                            if "Copper" in input_format or "CuKa1" in input_format:
                                lambda_in = wavelength_map["Copper"]
                            elif "Cobalt" in input_format or "CoKa1" in input_format:
                                lambda_in = wavelength_map["Cobalt"]
                            elif "Custom" in input_format and input_custom_wavelength is not None:
                                lambda_in = input_custom_wavelength

                            lambda_out = None
                            if "Copper" in output_format or "CuKa1" in output_format:
                                lambda_out = wavelength_map["Copper"]
                            elif "Cobalt" in output_format or "CoKa1" in output_format:
                                lambda_out = wavelength_map["Cobalt"]
                            elif "Custom" in output_format and output_custom_wavelength is not None:
                                lambda_out = output_custom_wavelength

                            if "q-vector" in input_format and "d-spacing" in output_format:
                                valid = x_values > 0
                                d_values = np.zeros_like(x_values)
                                d_values[valid] = 2 * np.pi / x_values[valid]
                                d_values[~valid] = np.nan
                                return d_values

                            elif "d-spacing" in input_format and "q-vector" in output_format:
                                valid = x_values > 0
                                q_values = np.zeros_like(x_values)
                                q_values[valid] = 2 * np.pi / x_values[valid]
                                q_values[~valid] = np.nan
                                return q_values

                            elif "q-vector" in input_format and "2theta" in output_format:
                                if lambda_out is None:
                                    print(f"Missing output wavelength for q-vector to 2theta conversion")
                                    return x_values

                                valid = x_values >= 0
                                sin_arg = (x_values[valid] * lambda_out) / (4 * np.pi)

                                mask = (sin_arg >= -1) & (sin_arg <= 1)
                                sin_arg = sin_arg[mask]

                                theta = np.arcsin(sin_arg)
                                twotheta = 2 * np.degrees(theta)

                                result = np.zeros_like(x_values)
                                result_indices = np.where(valid)[0][mask]
                                result[result_indices] = twotheta
                                result[~valid] = np.nan

                                return result

                            elif "2theta" in input_format and "q-vector" in output_format:
                                if lambda_in is None:
                                    print(f"Missing input wavelength for 2theta to q-vector conversion")
                                    return x_values

                                theta_rad = np.radians(x_values) / 2
                                q_values = (4 * np.pi * np.sin(theta_rad)) / lambda_in

                                return q_values

                            elif ("2theta" in input_format) and ("d-spacing" in output_format):
                                if lambda_in is None:
                                    print(f"Missing input wavelength for conversion: {input_format}")
                                    return x_values

                                theta_rad = np.radians(x_values / 2)
                                valid = np.abs(np.sin(theta_rad)) > 1e-6

                                d = np.zeros_like(x_values)
                                d[valid] = lambda_in / (2 * np.sin(theta_rad[valid]))
                                d[~valid] = np.nan

                                return d

                            elif ("d-spacing" in input_format) and ("2theta" in output_format):
                                if lambda_out is None:
                                    print(f"Missing output wavelength for conversion: {output_format}")
                                    return x_values
                                valid = x_values > 0
                                sin_arg = lambda_out / (2 * x_values[valid])
                                sin_arg = np.clip(sin_arg, 0, 1)

                                theta = np.degrees(np.arcsin(sin_arg))
                                result = np.zeros_like(x_values)
                                result[valid] = 2 * theta
                                result[~valid] = np.nan

                                return result

                            elif ("2theta" in input_format) and ("2theta" in output_format):
                                if lambda_in is None or lambda_out is None:
                                    print(
                                        f"Missing wavelength for 2θ to 2θ conversion. Input λ: {lambda_in}, Output λ: {lambda_out}")
                                    return x_values

                                if abs(lambda_in - lambda_out) < 1e-6:
                                    return x_values

                                theta_rad = np.radians(x_values / 2)
                                valid = np.abs(np.sin(theta_rad)) > 1e-6
                                d = np.zeros_like(x_values)
                                d[valid] = lambda_in / (2 * np.sin(theta_rad[valid]))

                                sin_arg = lambda_out / (2 * d[valid])
                                sin_arg = np.clip(sin_arg, 0, 1)
                                theta_new = np.degrees(np.arcsin(sin_arg))

                                result = np.zeros_like(x_values)
                                result[valid] = 2 * theta_new
                                result[~valid] = np.nan

                                return result

                            else:
                                print(f"No matching conversion logic for: {conversion_type}")
                                return x_values


                        try:
                            x_data = convert_data(
                                x_values=x_data,
                                conversion_type=conversion_type,
                                input_custom_wavelength=settings["input_custom_wavelength"],
                                output_custom_wavelength=settings["output_custom_wavelength"]
                            )

                            if "to q-vector" in conversion_type:
                                x_axis_metric = "q (Å⁻¹)"
                            elif "to d-spacing" in conversion_type:
                                x_axis_metric = "d-spacing (Å)"
                            elif "to 2theta" in conversion_type:
                                if "Copper" in conversion_type:
                                    x_axis_metric = "2θ (Cu Kα, λ=1.54056Å)"
                                elif "Cobalt" in conversion_type:
                                    x_axis_metric = "2θ (Co Kα, λ=1.78897Å)"
                                elif "custom" in conversion_type or "Custom" in conversion_type:
                                    wavelength = settings['output_custom_wavelength']
                                    if wavelength:
                                        x_axis_metric = f"2θ (λ={wavelength}Å)"
                                    else:
                                        x_axis_metric = "2θ (°)"

                            st.success(f"Converted {file.name}: {conversion_type}")

                            valid_mask = ~np.isnan(x_data)
                            if not np.all(valid_mask):
                                x_data = x_data[valid_mask]
                                y_data = y_data[valid_mask]
                                st.warning(f"Some points in {file.name} were invalid for conversion and were removed")

                        except Exception as e:
                            st.error(f"Error in data conversion for {file.name}: {e}")
                            st.warning("Using original data for this file.")


            except Exception as e:
                st.error(
                    f"Error occurred in file processing for {file.name}: {str(e)}")

            # Apply sample displacement shift to the (2θ) x-values.
            # When requested, keep the pre-shift x-values to overlay the
            # original (uncorrected) pattern.
            x_data_original = None
            if i < len(displacement_settings):
                disp = displacement_settings[i]
                if disp["apply"] and disp["goniometer_radius_mm"] > 0 and len(x_data) > 0:
                    if show_original_displacement:
                        x_data_original = np.array(x_data, dtype=float)
                    x_data = x_data + _displacement_shift(
                        x_data, disp["displacement_mm"], disp["goniometer_radius_mm"]
                    )
                    st.success(
                        f"Applied sample displacement shift to {file.name} "
                        f"(s = {disp['displacement_mm']:.3f} mm, R = {disp['goniometer_radius_mm']:.1f} mm)"
                    )

            if enable_conversion and i < len(file_conversion_settings):
                slit_settings = file_conversion_settings[i]

                slit_conversion_type = slit_settings.get("slit_conversion_type", "No conversion")
                fixed_slit_size = slit_settings.get("fixed_slit_size")
                irradiated_length = slit_settings.get("irradiated_length")

                if slit_conversion_type != "No conversion":
                    theta_rad = np.radians(x_data / 2)  # convert 2θ to θ in radians for calculations

                    # Avoid division by zero by creating a valid mask
                    valid_mask = np.abs(np.sin(theta_rad)) > 1e-6

                    if slit_conversion_type == "Auto slit to fixed slit":
                        adjustment_factor = np.zeros_like(y_data)
                        adjustment_factor[valid_mask] = fixed_slit_size / (
                                irradiated_length * np.sin(theta_rad[valid_mask])
                        )

                        y_data = y_data * adjustment_factor
                        st.success(f"Applied Auto slit → Fixed slit conversion for {file.name}")

                    elif slit_conversion_type == "Fixed slit to auto slit":
                        adjustment_factor = np.zeros_like(y_data)
                        adjustment_factor[valid_mask] = (
                                irradiated_length * np.sin(theta_rad[valid_mask]) / fixed_slit_size
                        )

                        y_data = y_data * adjustment_factor
                        st.success(f"Applied Fixed slit → Auto slit conversion for {file.name}")
                    if not np.all(valid_mask):
                        st.warning(
                            f"Some points in {file.name} were omitted during slit conversion due to invalid angles."
                        )
                        x_data = x_data[valid_mask]
                        y_data = y_data[valid_mask]
                        if x_data_original is not None:
                            x_data_original = x_data_original[valid_mask]

            if normalized_intensity and np.max(y_data) > 0:
                y_data = (y_data / np.max(y_data)) * 100

            try:

                if i < len(y_scales):
                    y_data = y_data * y_scales[i]

                y_data = y_data + y_offsets[i]

                mask = np.ones_like(x_data, dtype=bool)
                if x_axis_log:
                    mask &= (x_data > 0)
                if y_axis_log:
                    mask &= (y_data > 0)
                if not np.all(mask):
                    st.warning(
                        f"In file '{file.name}', some data points were omitted because they are not positive and are required for logarithmic scaling.")
                x_data = x_data[mask]
                y_data = y_data[mask]
                if x_data_original is not None:
                    x_data_original = x_data_original[mask]

                if x_axis_log:
                    x_data = np.log10(x_data)
                    if x_data_original is not None:
                        x_data_original = np.log10(x_data_original)
                if y_axis_log:
                    y_data = np.log10(y_data)

                if customize_layout and i in series_colors:
                    color = series_colors[i]
                else:
                    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#000000', '#7f7f7f']
                    color = colors[i % len(colors)]
                mode_str = ""
                if show_lines:
                    mode_str += "lines"
                if show_markers:
                    if mode_str:
                        mode_str += "+markers"
                    else:
                        mode_str = "markers"
                if not mode_str:
                    mode_str = "markers"

                trace_name = series_names.get(i, file.name) if customize_layout else file.name

                if x_data_original is not None:
                    original_name = f"{trace_name} (original)"
                    fig_interactive.add_trace(go.Scatter(
                        x=x_data_original,
                        y=y_data,
                        mode=mode_str,
                        name=original_name,
                        opacity=0.5,
                        line=dict(dash='dot', width=line_thickness, color=color),
                        marker=dict(color=color, size=marker_size, symbol='circle-open'),
                        hovertemplate=(
                            f"<span style='color:{color};'><b>{original_name}</b><br>"
                            "x = %{x:.2f}<br>y = %{y:.2f}</span><extra></extra>"
                        )
                    ))

                fig_interactive.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode_str,
                    name=trace_name,
                    line=dict(dash='solid', width=line_thickness, color=color),
                    marker=dict(color=color, size=marker_size),
                    hovertemplate=(
                        f"<span style='color:{color};'><b>{trace_name}</b><br>"
                        "x = %{x:.2f}<br>y = %{y:.2f}</span><extra></extra>"
                    )
                ))
                main_trace_indices[i] = len(fig_interactive.data) - 1
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

                # Set axis scale
            fig_interactive.update_xaxes(type="linear")
            fig_interactive.update_yaxes(type="linear")

            # Configure legend position based on selection
            legend_config = {
                "font": dict(size=legend_font_size),
                # "title": "Legend Title"
            }

            if legend_position == "Top":
                legend_config.update({
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5
                })
            elif legend_position == "Bottom":
                legend_config.update({
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.2,
                    "xanchor": "center",
                    "x": 0.5
                })
            elif legend_position == "Left":
                legend_config.update({
                    "orientation": "v",
                    "yanchor": "middle",
                    "y": 0.5,
                    "xanchor": "right",
                    "x": -0.1
                })
            elif legend_position == "Right":
                legend_config.update({
                    "orientation": "v",
                    "yanchor": "middle",
                    "y": 0.5,
                    "xanchor": "left",
                    "x": 1.05
                })

            # Update layout with all customized settings
            fig_interactive.update_layout(
                height=graph_height,
                width=graph_width,
                margin=dict(t=80, b=80, l=60, r=30),
                hovermode="closest",
                showlegend=True,
                legend=legend_config,
                xaxis=dict(
                    title=dict(text=custom_x_label, font=dict(size=axis_label_font_size, color='black'), standoff=20),
                    tickfont=dict(size=tick_font_size, color='black'),
                    fixedrange=fix_x_axis
                ),
                yaxis=dict(
                    title=dict(text=custom_y_label, font=dict(size=axis_label_font_size, color='black')),
                    tickfont=dict(size=tick_font_size, color='black')
                ),
                # title=dict(
                #    text="Interactive Data Plot",
                #    font=dict(size=title_font_size, color='black')
                # ),
                hoverlabel=dict(font=dict(size=tick_font_size)),
                font=dict(size=18),
                autosize=False
            )

        if user_pattern_file:
            files = user_pattern_file if isinstance(user_pattern_file, list) else [user_pattern_file]

            if has_header:
                try:
                    sample_file = files[0]
                    sample_file.seek(0)
                    df_sample = pd.read_csv(
                        sample_file,
                        sep=r'\s+|,|;',
                        engine='python',
                        header=0
                    )
                    x_axis_metric = df_sample.columns[0]
                    y_axis_metric = df_sample.columns[1]
                except Exception as e:
                    st.error(f"Error reading header from file {sample_file.name}: {e}")
                    x_axis_metric = "X-data"
                    y_axis_metric = "Y-data"
            else:
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"

        if fix_x_axis == True:
            fig_interactive.update_xaxes(range=[x_axis_min, x_axis_max])

        with plot_placeholder.container():
            st.plotly_chart(fig_interactive)

        st.markdown("### 💾 Download Processed Data")
        delimiter_label_to_value = {
            "Comma (`,`)": ",",
            "Space (` `)": " ",
            "Tab (`\\t`)": "\t",
            "Semicolon (`;`)": ";"
        }

        delimiter_label = st.selectbox("Choose delimiter for download:", list(delimiter_label_to_value.keys()))
        delimiter_option = delimiter_label_to_value[delimiter_label]

        for i, file in enumerate(files):
            if i not in main_trace_indices:
                continue
            trace = fig_interactive.data[main_trace_indices[i]]
            x_data = trace.x
            y_data = trace.y

            if fix_x_axis:
                if x_axis_log:
                    x_values = 10 ** x_data
                else:
                    x_values = x_data

                if x_axis_log:
                    mask = (x_values >= x_axis_min) & (x_values <= x_axis_max)
                else:
                    mask = (x_data >= x_axis_min) & (x_data <= x_axis_max)

                filtered_x = x_data[mask]
                filtered_y = y_data[mask]
            else:
                filtered_x = x_data
                filtered_y = y_data

            df_out = pd.DataFrame({
                x_axis_metric: filtered_x,
                y_axis_metric: filtered_y
            })

            buffer = io.StringIO()
            df_out.to_csv(buffer, sep=delimiter_option, index=False)

            base_name = file.name.rsplit(".", 1)[0]
            download_name = f"{base_name}_processed.xy"

            download_info = ""
            if fix_x_axis:
                download_info = f" (filtered to x-range: {x_axis_min}-{x_axis_max})"

            st.download_button(
                label=f"⬇️ Download processed data for {file.name}{download_info}",
                data=buffer.getvalue(),
                file_name=download_name,
                mime="text/plain",
                key=f"download_btn_{i}_{base_name}"
            )
    else:
        st.info(f"Upload your data file first to see all options.")
