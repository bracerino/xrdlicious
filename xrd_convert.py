import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import xml.etree.ElementTree as ET
from io import StringIO
from datetime import datetime, timedelta
import re


def run_data_converter():
    def parse_xrdml(file_content):
        try:
            root = ET.fromstring(file_content)
            namespace = ''
            if '}' in root.tag:
                namespace = root.tag.split('}')[0][1:]
            ns = {'xrd': namespace}

            def find_text(path, default='N/A'):
                element = root.find(path, ns)
                return element.text if element is not None else default

            def find_attrib(path, attribute, default='N/A'):
                element = root.find(path, ns)
                return element.attrib.get(attribute, default) if element is not None else default

            metadata = {
                'Status': find_attrib('xrd:xrdMeasurement', 'status'),
                'Measurement Type': find_attrib('xrd:xrdMeasurement', 'measurementType'),
                'Start Time': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:startTimeStamp'),
                'End Time': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:endTimeStamp'),
                'Author': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:author/xrd:name'),
                'Anode Material': find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:anodeMaterial'),
                'X-ray Tube Tension': f"{find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:tension')} {find_attrib('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:tension', 'unit')}",
                'X-ray Tube Current': f"{find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:current')} {find_attrib('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:current', 'unit')}",
                'K-Alpha1 Wavelength (Å)': find_text('xrd:xrdMeasurement/xrd:usedWavelength/xrd:kAlpha1'),
                'Detector': find_attrib('xrd:xrdMeasurement/xrd:diffractedBeamPath/xrd:detector', 'name'),
                'Scan Axis': find_attrib('xrd:xrdMeasurement/xrd:scan', 'scanAxis'),
            }
            data_points_path = 'xrd:xrdMeasurement/xrd:scan/xrd:dataPoints'
            start_pos_2theta = float(find_text(f'{data_points_path}/xrd:positions[@axis="2Theta"]/xrd:startPosition'))
            end_pos_2theta = float(find_text(f'{data_points_path}/xrd:positions[@axis="2Theta"]/xrd:endPosition'))
            intensities_str = find_text(f'{data_points_path}/xrd:intensities')
            intensities = np.array(intensities_str.split(), dtype=float)
            two_theta_array = np.linspace(start_pos_2theta, end_pos_2theta, len(intensities))
            data_df = pd.DataFrame({'2Theta': two_theta_array, 'Intensity': intensities})
            return metadata, data_df
        except Exception as e:
            st.error(f"Failed to parse XRDML file. Error: {e}")
            return None, None

    def parse_ras(file_content):
        try:
            metadata = {}
            data_lines = []
            in_header_section = False
            in_data_section = False

            for line in file_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line == '*RAS_HEADER_START':
                    in_header_section = True
                    continue
                if line == '*RAS_HEADER_END':
                    in_header_section = False
                    continue
                if line == '*RAS_INT_START':
                    in_data_section = True
                    continue
                if line == '*RAS_INT_END':
                    in_data_section = False
                    break

                if in_header_section and line.startswith('*'):
                    parts = line[1:].split(None, 1)
                    if len(parts) == 2:
                        key, value = parts
                        metadata[key] = value.strip('"')

                if in_data_section:
                    data_parts = line.split()
                    if len(data_parts) >= 2:
                        try:
                            angle = float(data_parts[0])
                            intensity = float(data_parts[1])
                            data_lines.append([angle, intensity])
                        except ValueError:
                            continue

            if not data_lines:
                st.error("No data points found in the RAS file.")
                return None, None

            data_df = pd.DataFrame(data_lines, columns=['2Theta', 'Intensity'])
            return metadata, data_df

        except Exception as e:
            st.error(f"Failed to parse RAS file. Error: {e}")
            return None, None

    def parse_xy(file_content):
        try:
            first_line = file_content.splitlines()[0]
            has_header = any(char.isalpha() for char in first_line)
            data_io = StringIO(file_content)
            skiprows = 1 if has_header else 0
            df = pd.read_csv(data_io, sep=r'[\s,;]+', engine='python', header=None, skiprows=skiprows,
                             names=['2Theta', 'Intensity'], comment='#')
            return df.dropna().astype(float)
        except Exception as e:
            st.error(f"Failed to parse XY file. Error: {e}")
            return None

    def convert_to_xy(data_df, include_header=False):
        output = StringIO()
        header = ['2Theta', 'Intensity'] if include_header else False
        data_df.to_csv(output, sep='\t', header=header, index=False, float_format='%.6f')
        return output.getvalue()

    def generate_xrdml(metadata_df, data_df):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()

        def split_value_unit(text, default_val, default_unit):
            parts = str(text).split()
            return (parts[0], parts[1]) if len(parts) > 1 else (default_val, default_unit)

        tension, tension_unit = split_value_unit(meta_dict.get('X-ray Tube Tension'), '45', 'kV')
        current, current_unit = split_value_unit(meta_dict.get('X-ray Tube Current'), '40', 'mA')
        start_2theta, end_2theta = data_df['2Theta'].min(), data_df['2Theta'].max()
        intensities_str = ' '.join(map(str, data_df['Intensity'].values.astype(int)))
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.2" status="{meta_dict.get('Status', 'Completed')}">
  <xrdMeasurement measurementType="{meta_dict.get('Measurement Type', 'Scan')}" status="Completed">
    <usedWavelength intended="K-Alpha 1"><kAlpha1 unit="Angstrom">{meta_dict.get('K-Alpha1 Wavelength (Å)', '1.54060')}</kAlpha1></usedWavelength>
    <incidentBeamPath><xRayTube><tension unit="{tension_unit}">{tension}</tension><current unit="{current_unit}">{current}</current><anodeMaterial>{meta_dict.get('Anode Material', 'Cu')}</anodeMaterial></xRayTube></incidentBeamPath>
    <diffractedBeamPath><detector name="{meta_dict.get('Detector', 'Generic Detector')}"></detector></diffractedBeamPath>
    <scan scanAxis="{meta_dict.get('Scan Axis', 'Gonio')}" status="Completed">
      <header><startTimeStamp>{meta_dict.get('Start Time', datetime.now().isoformat())}</startTimeStamp><endTimeStamp>{meta_dict.get('End Time', datetime.now().isoformat())}</endTimeStamp><author><name>{meta_dict.get('Author', 'XRDlicious User')}</name></author></header>
      <dataPoints>
        <positions axis="2Theta" unit="deg"><startPosition>{start_2theta:.6f}</startPosition><endPosition>{end_2theta:.6f}</endPosition></positions>
        <intensities unit="counts">{intensities_str}</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>"""

    def generate_ras(metadata_df, data_df):
        lines = ["*RAS_DATA_START", "*RAS_HEADER_START"]
        for _, row in metadata_df.iterrows():
            lines.append(f"*{row['Parameter']} \"{row['Value']}\"")
        lines.append("*RAS_HEADER_END")
        lines.append("*RAS_INT_START")
        for _, row in data_df.iterrows():
            lines.append(f"{row['2Theta']:.6f} {row['Intensity']:.4f} 1.0000")
        lines.extend(["*RAS_INT_END", "*RAS_DATA_END", "*DSC_DATA_END"])
        return "\n".join(lines)

    def get_default_metadata(format_type='XRDML'):
        now = datetime.now()
        if format_type == 'RAS':
            metadata = {
                'FILE_TYPE': 'RAS_RAW', 'FILE_OPERATOR': 'XRDlicious User',
                'HW_XG_TARGET_NAME': 'Cu', 'HW_XG_WAVE_LENGTH_ALPHA1': '1.540593',
                'MEAS_COND_XG_VOLTAGE': '40', 'MEAS_COND_XG_CURRENT': '30',
                'MEAS_SCAN_AXIS_X': 'TwoThetaTheta', 'MEAS_SCAN_MODE': 'CONTINUOUS',
                'MEAS_SCAN_SPEED': '5.0000', 'MEAS_SCAN_SPEED_UNIT': 'deg/min',
                'MEAS_SCAN_START': '10.0000', 'MEAS_SCAN_STEP': '0.0100', 'MEAS_SCAN_STOP': '90.0000'
            }
        else:
            metadata = {
                'Status': 'Completed', 'Measurement Type': 'Scan',
                'Start Time': now.isoformat(), 'End Time': (now + timedelta(minutes=30)).isoformat(),
                'Author': 'XRDlicious User', 'Anode Material': 'Cu',
                'X-ray Tube Tension': '45 kV', 'X-ray Tube Current': '40 mA',
                'K-Alpha1 Wavelength (Å)': '1.54060', 'Detector': 'Generic Detector',
                'Scan Axis': 'Gonio'
            }
        return pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])

    st.markdown("### 📜 .xrdml (PANalytical) ↔️ .xy ↔️ .ras (Rigaku) File Format Converter")
    st.markdown(
    """
    <div style="background-color:red; padding:10px; border-radius:5px">
        <h3 style="color:white; text-align:center;">🚨 Testing Mode 🚨</h3>
    </div>
    """,
    unsafe_allow_html=True
)
    st.info(
        "Upload an `.xrdml` or `.ras` file to convert to `.xy`, or upload `.xy` text file to convert to `.xrdml` or `.ras`")

    uploaded_file = st.file_uploader("Upload .xrdml, .ras, .xy, .dat, or .txt file",
                                     type=["xrdml", "xml", "ras", "xy", "dat", "txt"])

    if uploaded_file:
        file_content = uploaded_file.getvalue().decode("utf-8")
        file_ext = uploaded_file.name.lower().split('.')[-1]

        if file_ext in ['xrdml', 'xml', 'ras']:
            if file_ext == 'ras':
                metadata, data_df = parse_ras(file_content)
                meta_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])
            else:  # XRDML
                metadata, data_df = parse_xrdml(file_content)
                meta_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])

            if metadata and data_df is not None:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown("#### 📝 Measurement Details")
                    st.dataframe(meta_df, height=400)
                    include_header = st.checkbox("Include header in .xy file", value=False)
                    xy_data = convert_to_xy(data_df, include_header)

                    default_name = uploaded_file.name.rsplit('.', 1)[0] + '.xy'
                    download_filename = st.text_input("Enter filename for download:", default_name)

                    st.download_button("⬇️ Download as .xy File", xy_data, download_filename, "text/plain")
                with col2:
                    st.markdown("#### 📈 Diffraction Pattern")
                    fig = go.Figure(
                        go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
                    fig.update_layout(title=f"Data from {uploaded_file.name}", xaxis_title="2θ (°)",
                                      yaxis_title="Intensity (counts)", height=550, margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)

        elif file_ext in ['xy', 'dat', 'txt']:
            data_df = parse_xy(file_content)
            if data_df is not None:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown("#### 📝 Edit Details for Output File")
                    output_format = st.selectbox("Select Output Format", ['XRDML', 'RAS'])

                    df_state_key = f"meta_df_{output_format}_{uploaded_file.name}"
                    content_state_key = f"content_{output_format}_{uploaded_file.name}"

                    if st.session_state.get('last_file_format_choice') != (uploaded_file.name, output_format):
                        st.session_state[df_state_key] = get_default_metadata(output_format)
                        st.session_state['last_file_format_choice'] = (uploaded_file.name, output_format)

                    edited_df = st.data_editor(st.session_state[df_state_key], num_rows="dynamic", height=425)

                    if st.button(f"Apply Changes & Generate {output_format}"):
                        st.session_state[df_state_key] = edited_df
                        if output_format == 'RAS':
                            st.session_state[content_state_key] = generate_ras(edited_df, data_df)
                        else:
                            st.session_state[content_state_key] = generate_xrdml(edited_df, data_df)
                        st.success(f"{output_format} file is ready for download.")

                    if content_state_key in st.session_state:
                        file_extension = 'ras' if output_format == 'RAS' else 'xrdml'
                        mime_type = 'text/plain' if output_format == 'RAS' else 'application/xml'
                        default_name = uploaded_file.name.rsplit('.', 1)[0] + f'.{file_extension}'
                        download_filename = st.text_input("Enter filename for download:", default_name)
                        st.download_button(f"⬇️ Download as .{file_extension} File",
                                           st.session_state[content_state_key], download_filename, mime_type)

                with col2:
                    st.markdown("#### 📈 Diffraction Pattern")
                    fig = go.Figure(
                        go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
                    fig.update_layout(title=f"Data from {uploaded_file.name}", xaxis_title="2θ (°)",
                                      yaxis_title="Intensity", height=550, margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)
