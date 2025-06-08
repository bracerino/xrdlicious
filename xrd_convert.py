import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import re
import zipfile


def run_data_converter():

    def extract_key_ras_metadata(metadata_dict):
        key_metadata = {
            'X-ray Target': metadata_dict.get('HW_XG_TARGET_NAME', 'N/A'),
            'Voltage (kV)': metadata_dict.get('MEAS_COND_XG_VOLTAGE', 'N/A'),
            'Current (mA)': metadata_dict.get('MEAS_COND_XG_CURRENT', 'N/A'),
            'K-Alpha1 (Å)': metadata_dict.get('HW_XG_WAVE_LENGTH_ALPHA1', 'N/A'),
            'Scan Axis': metadata_dict.get('MEAS_SCAN_AXIS_X', 'N/A'),
            'Start Angle (°)': metadata_dict.get('MEAS_SCAN_START', 'N/A'),
            'Stop Angle (°)': metadata_dict.get('MEAS_SCAN_STOP', 'N/A'),
            'Step Size (°)': metadata_dict.get('MEAS_SCAN_STEP', 'N/A'),
            'Scan Speed': f"{metadata_dict.get('MEAS_SCAN_SPEED', 'N/A')} {metadata_dict.get('MEAS_SCAN_SPEED_UNIT', '')}".strip(),
            'Divergence Slit (DS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-18', 'N/A'),
            'Scattering Slit (SS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-35', 'N/A'),
            'Receiving Slit (RS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-42', 'N/A')
        }
        return key_metadata

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
            full_metadata = {}
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
                        full_metadata[key] = value.strip('"')

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
                return None, None, None

            data_df = pd.DataFrame(data_lines, columns=['2Theta', 'Intensity'])
            key_metadata = extract_key_ras_metadata(full_metadata)
            return full_metadata, key_metadata, data_df

        except Exception as e:
            st.error(f"Failed to parse RAS file. Error: {e}")
            return None, None, None

    def parse_rasx(uploaded_file_object):
        try:
            full_metadata = {}
            data_df = None
            with zipfile.ZipFile(uploaded_file_object, 'r') as zf:
                xml_filename = next((f for f in zf.namelist() if f.lower().endswith('.xml')), None)
                if not xml_filename:
                    st.error("No XML metadata file found in the RASX archive.")
                    return None, None, None

                xml_bytes = zf.read(xml_filename)
                xml_content_str = xml_bytes.decode('shift_jis', errors='replace')
                root = ET.fromstring(xml_content_str)

                for param in root.iter('Parameter'):
                    if 'name' in param.attrib and 'value' in param.attrib:
                        full_metadata[param.attrib['name']] = param.attrib['value']

                bin_filename = next((f for f in zf.namelist() if f.lower().endswith('.bin')), None)
                asc_filename = next((f for f in zf.namelist() if f.lower().endswith('.asc')), None)

                if bin_filename:
                    binary_content = zf.read(bin_filename)
                    intensities = np.frombuffer(binary_content, dtype=np.float32)
                    num_points = len(intensities)
                    start_angle = float(full_metadata.get('Start', 0))
                    stop_angle = float(full_metadata.get('Stop', 90))
                    angles = np.linspace(start_angle, stop_angle, num_points)
                    data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})
                elif asc_filename:
                    asc_content = zf.read(asc_filename).decode('utf-8', errors='replace')
                    data_df = parse_xy(asc_content)
                else:
                    st.error("No data file (.bin or .asc) found in the RASX archive.")
                    return None, None, None

            key_metadata = extract_key_ras_metadata(full_metadata)
            return full_metadata, key_metadata, data_df
        except Exception as e:
            st.error(f"Failed to parse RASX file. Error: {e}")
            return None, None, None

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

    st.markdown("### 📜 .xrdml (PANalytical) ↔️ .xy ↔️ .ras (Rigaku) XRD File Format Converter")
    st.markdown(
        """
        <div style="background-color:#f8d7da; padding:6px 10px; border-radius:4px; border:1px solid #f5c2c7; width: fit-content;">
            <span style="color:#842029; font-size:14px;">🔧 Testing mode</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(
        "Upload an `.xrdml` or `.ras` file to convert to `.xy`, or upload `.xy` text file to convert to `.xrdml` or `.ras`")

    uploaded_file = st.file_uploader("Upload Data File", type=["xrdml", "xml", "ras", "rasx", "xy", "dat", "txt"])

    if uploaded_file:
        file_ext = uploaded_file.name.lower().split('.')[-1]

        if file_ext in ['xrdml', 'xml', 'ras', 'rasx']:
            full_metadata, key_metadata, data_df = None, None, None
            if file_ext == 'ras':
                file_content = uploaded_file.getvalue().decode("utf-8", errors='replace')
                full_metadata, key_metadata, data_df = parse_ras(file_content)
            elif file_ext == 'rasx':
                full_metadata, key_metadata, data_df = parse_rasx(uploaded_file)
            else:
                file_content = uploaded_file.getvalue().decode("utf-8", errors='replace')
                key_metadata, data_df = parse_xrdml(file_content)
                full_metadata = key_metadata

            if data_df is not None:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown("#### 📝 Key Measurement Parameters")
                    key_meta_df = pd.DataFrame(list(key_metadata.items()), columns=['Parameter', 'Value'])
                    st.table(key_meta_df)

                    if full_metadata and file_ext in ['ras', 'rasx']:
                        with st.expander("Show Full Raw Header"):
                            full_meta_df = pd.DataFrame(list(full_metadata.items()), columns=['Parameter', 'Value'])
                            st.dataframe(full_meta_df, height=300)

                    include_header = st.checkbox("Include header in .xy file", value=False)
                    default_name = uploaded_file.name.rsplit('.', 1)[0] + '.xy'
                    download_filename_input = st.text_input("Enter filename for download:", default_name)
                    st.info(f"Please press enter in the filename field when the name was changed.")

                    xy_data = convert_to_xy(data_df, include_header)
                    st.download_button("⬇️ Download as .xy File", xy_data, download_filename_input, "text/plain")
                with col2:
                    st.markdown("#### 📈 Diffraction Pattern")
                    fig = go.Figure(
                        go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
                    fig.update_layout(title=f"Data from {uploaded_file.name}", xaxis_title="2θ (°)",
                                      yaxis_title="Intensity (counts)", height=550, margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)

        elif file_ext in ['xy', 'dat', 'txt']:
            file_content = uploaded_file.getvalue().decode("utf-8", errors='replace')
            data_df = parse_xy(file_content)
            if data_df is not None:
                current_file_name = uploaded_file.name

                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown("#### 📝 Edit Details for Output File")
                    output_format = st.selectbox("Select Output Format", ['XRDML', 'RAS'])

                    df_state_key = f"meta_df_{output_format}_{current_file_name}"

                    if st.session_state.get('last_file_format_choice') != (current_file_name, output_format):
                        st.session_state[df_state_key] = get_default_metadata(output_format)
                        st.session_state['last_file_format_choice'] = (current_file_name, output_format)

                    edited_df = st.data_editor(st.session_state[df_state_key], num_rows="dynamic", height=425)

                    if st.button(f"Apply Changes & Prepare Download"):
                        st.session_state[df_state_key] = edited_df
                        st.success(f"{output_format} file is ready for download below.")

                    download_filename_key = f"download_filename_{output_format}_{current_file_name}"
                    file_extension = 'ras' if output_format == 'RAS' else 'xrdml'
                    default_name = current_file_name.rsplit('.', 1)[0] + f'.{file_extension}'

                    download_filename = st.text_input("Enter filename for download:", default_name,
                                                      key=download_filename_key)
                    st.info(f"Please press enter in the filename field when the name was changed.")

                    if df_state_key in st.session_state:
                        mime_type = 'text/plain' if output_format == 'RAS' else 'application/xml'
                        if output_format == 'RAS':
                            file_content_to_download = generate_ras(st.session_state[df_state_key], data_df)
                        else:
                            file_content_to_download = generate_xrdml(st.session_state[df_state_key], data_df)

                        st.download_button(
                            label=f"⬇️ Download {download_filename}",
                            data=file_content_to_download,
                            file_name=download_filename,
                            mime=mime_type
                        )
                with col2:
                    st.markdown("#### 📈 Diffraction Pattern")
                    fig = go.Figure(
                        go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
                    fig.update_layout(title=f"Data from {uploaded_file.name}", xaxis_title="2θ (°)",
                                      yaxis_title="Intensity", height=550, margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)
